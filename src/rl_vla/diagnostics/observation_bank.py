"""Immutable matched RGB/state/noise banks with simulator instance masks."""

import hashlib
import json
from pathlib import Path

import numpy as np

CAMERAS = {"image": "agentview", "wrist_image": "robot0_eye_in_hand"}
TARGETS = {"object_0": "alphabet_soup", "object_1": "cream_cheese", "object_2": "salad_dressing"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    """Physical state fingerprint excludes simulator clock, preserves all dynamics."""
    value = np.asarray(state, dtype="<f8")
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Expected finite state vector")
    return hashlib.sha256(value[1:].tobytes()).hexdigest()


def save_entry(directory, observation, metadata, masks, sim_state, *, noise_seed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    obs = observation["openpi"]
    for name in CAMERAS:
        image = np.asarray(obs[f"observation/{name}"])
        if image.shape != (224, 224, 3) or image.dtype != np.uint8 or image.std() <= 1:
            raise ValueError(f"Invalid or blank {name} image")
    values = {name: obs[f"observation/{name}"] for name in CAMERAS}
    values.update(state=obs["observation/state"], prompt=np.asarray(obs["prompt"]), sim_state=sim_state)
    np.savez_compressed(directory / "observation.npz", **values)
    np.savez_compressed(directory / "masks.npz", **masks)
    # This is a persisted common diagnostic noise; native evaluation seeds are recorded separately.
    noise = np.random.default_rng(noise_seed).standard_normal((1, 10, 32)).astype(np.float32)
    np.save(directory / "initial_noise.npy", noise, allow_pickle=False)
    record = dict(metadata, prompt=obs["prompt"], noise_seed=noise_seed,
                  entry_dir=str(directory.resolve()),
                  observation_path=str((directory / "observation.npz").resolve()),
                  masks_path=str((directory / "masks.npz").resolve()),
                  noise_generator="numpy PCG64; shared diagnostic noise, not evaluation PRNG",
                  state_sha256=state_digest(sim_state),
                  observation_sha256=digest(directory / "observation.npz"),
                  masks_sha256=digest(directory / "masks.npz"),
                  initial_noise_sha256=digest(directory / "initial_noise.npy"))
    (directory / "metadata.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def simulator_masks(env, observation, *, audit_rgb=True):
    """Render geometry IDs, map via simulator instance ownership, and audit orientation."""
    from PIL import Image
    from robosuite import macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    from rl_vla.envs.libero import _resize_image

    backend = env.backend.env if hasattr(env.backend, "env") else env.backend
    sim = backend.sim
    owners = backend.model.instances_to_ids
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    masks, alignment, locations = {}, {}, {}
    for instance, groups in owners.items():
        geoms = groups.get("geom", [])
        if geoms:
            locations[instance] = np.asarray(sim.data.geom_xpos[geoms]).mean(axis=0).tolist()
    for name, camera in CAMERAS.items():
        # Match render resolution, robosuite convention, OpenPI rotation, then resize.
        alignment[name] = {"convention": convention, "rotation_degrees": 180,
                           "mask_resize": "nearest", "rgb_audited": audit_rgb}
        if audit_rgb:
            rendered = sim.render(camera_name=camera, width=256, height=256, depth=False)
            transformed = _resize_image(np.ascontiguousarray(rendered[::convention][::-1, ::-1]), 224)
            expected = observation["openpi"][f"observation/{name}"]
            difference = np.abs(transformed.astype(int) - expected.astype(int))
            error = int(difference.max())
            alignment[name].update(max_rgb_difference=error, mean_rgb_difference=float(difference.mean()),
                                   fraction_channels_difference_over_2=float((difference > 2).mean()))
            # OSMesa rerenders can differ at isolated rasterization edges. Reject
            # any material scene/orientation mismatch, retaining measured errors.
            if difference.mean() > .25 or (difference > 2).mean() > .005:
                raise ValueError(f"Segmentation/RGB camera alignment failed for {name}: {error}")
        segmentation = sim.render(camera_name=camera, width=256, height=256,
                                  depth=False, segmentation=True)
        geometry = segmentation[::convention, :, 1][::-1, ::-1]
        owned = np.zeros((224, 224), bool)
        for instance, groups in owners.items():
            mask = np.isin(geometry, groups.get("geom", []))
            mask = np.asarray(Image.fromarray(mask).resize((224, 224), Image.Resampling.NEAREST), bool)
            masks[f"{name}__{instance}"] = mask
            owned |= mask
        masks[f"{name}__unowned_background"] = ~owned
    return masks, {"instances": owners, "instance_geom_centroids_xyz": locations,
                   "camera_alignment": alignment, "mask_source": "MuJoCo visible geometry IDs mapped to simulator instances"}


def collect_bank(protocol, task_key, split, output, *, limit=None, resume=False):
    from rl_vla.envs.libero_pro import make_env

    protocol, output = Path(protocol).resolve(), Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    plan = json.loads(protocol.read_text())
    definition = plan["tasks"][task_key]
    ids = definition[f"{split}_state_ids"]
    if limit is not None:
        ids = ids[:limit]
    output.mkdir(parents=True,exist_ok=resume)
    previous=[]
    if resume and (output/"manifest.jsonl").exists():
        previous=[json.loads(line) for line in (output/"manifest.jsonl").read_text().splitlines()]
        if [r["initial_state_id"] for r in previous]!=ids[:len(previous)]:
            raise ValueError("Existing bank is not a prefix of the frozen IDs")
        for record in previous:
            if record["protocol_sha256"]!=digest(protocol):
                raise ValueError("Bank protocol changed")
            for name in ("observation","masks"):
                if digest(record[f"{name}_path"])!=record[f"{name}_sha256"]:
                    raise ValueError(f"Existing immutable bank artifact changed: {name}")
    env = make_env(protocol=str(protocol), task_key=task_key, split=split)
    try:
        # Training bank enumerates IDs deterministically; learner still samples its own RNG.
        env.split = "diagnostic_enumeration"
        env.state_ids = tuple(ids)
        env.evaluation_index=len(previous)
        with (output / "manifest.jsonl").open("a" if resume else "x") as manifest:
            for index in range(len(previous),len(ids)):
                state_id=ids[index]
                observation, reset = env.reset(seed=100000 + index)
                masks, details = simulator_masks(env, observation)
                sim = env.backend.env.sim if hasattr(env.backend, "env") else env.backend.sim
                sim_state = sim.get_state().flatten()
                record = save_entry(output / f"state_{state_id:06d}", observation,
                    {"task": task_key, "split": split, "initial_state_id": state_id,
                     "reset": reset, "evaluation_seed": 100000 + index,
                     "protocol": str(protocol), "protocol_sha256": digest(protocol),
                     "bddl_sha256": definition["source" if split == "source" else "pro"]["bddl"]["sha256"],
                     "target": TARGETS[task_key], "phase": "initial_decision", **details},
                    masks, sim_state, noise_seed=300000 + state_id)
                manifest.write(json.dumps(record, allow_nan=False) + "\n")
                manifest.flush()
                print(json.dumps({"bank": str(output), "completed": index + 1, "total": len(ids)}), flush=True)
    finally:
        env.close()
    summary = {"task": task_key, "split": split, "entries": len(ids),
               "manifest_sha256": digest(output / "manifest.jsonl"),
               "protocol_sha256": digest(protocol), "complete": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def make_ood_env(*, protocol):
    """Evaluate a frozen custom layout without making it a training split."""
    from libero.libero.envs import OffScreenRenderEnv

    from rl_vla.envs.libero_pro import SplitLiberoEnv, checked_file
    from rl_vla.robustness import load_states

    record = json.loads(Path(protocol).read_text())
    states = load_states(checked_file(record["states"]))
    if record["state_ids"] != list(range(len(states))):
        raise ValueError("OOD state IDs do not enumerate the frozen bank")
    backend = OffScreenRenderEnv(bddl_file_name=str(checked_file(record["bddl"])),
                                 camera_heights=256, camera_widths=256,
                                 horizon=max(1000, record["max_steps"]+11))
    return SplitLiberoEnv(state_ids=record["state_ids"], split=record["variant"],
                         task_suite="libero_object", task_id=int(record["task"][-1]),
                         max_steps=record["max_steps"], camera_resolution=256, resize_size=224,
                         settle_steps=10, backend=backend, initial_states=states,
                         task_description=backend.language_instruction)
