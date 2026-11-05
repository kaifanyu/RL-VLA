"""Freeze custom diagnostic object relocations without consulting any policy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.observation_bank import simulator_masks
from rl_vla.envs.libero import LiberoEnv
from rl_vla.envs.libero_pro import make_env

OBJECTS = {"object_0": ("alphabet_soup_1", "butter_1"),
           "object_1": ("cream_cheese_1", "alphabet_soup_1"),
           "object_2": ("salad_dressing_1", "tomato_sauce_1")}
VARIANTS = ("both_new_regions", "continuous_new_offsets", "target_moved_distractor_fixed", "distractor_moved_target_fixed")
CENTERS = {"target": [-.025, -.22], "distractor": [-.025, .035]}


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(protocol, task, output, count=50):
    protocol, output = Path(protocol).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = json.loads(protocol.read_text())
    definition = plan["tasks"][task]
    env = make_env(protocol=str(protocol), task_key=task, split="test")
    backend = env.backend.env
    sim = backend.sim
    target, distractor = OBJECTS[task]
    objects = backend.objects_dict
    joint_positions = {name: sim.model.get_joint_qpos_addr(obj.joints[0])[0]
                       for name, obj in objects.items()}
    base_states = env.initial_states[definition["test_state_ids"]].copy()
    nq = sim.model.nq
    # All generated train, validation, official test and original states are excluded.
    from rl_vla.robustness import load_states
    all_states = np.concatenate([env.initial_states, load_states(definition["source"]["states"]["path"])])
    known_xy = np.concatenate([all_states[:,1+joint_positions[name]:3+joint_positions[name]]
                               for name in (target,distractor)])
    signature = lambda state: np.round(state[1:1+nq], 12).tobytes()
    seen = {signature(row) for row in all_states}
    predeclaration = {"task": task, "variants": list(VARIANTS), "count_per_variant": count,
                      "objects": [target, distractor], "new_center_xy": CENTERS,
                      "generation_seed": 20261031000 + int(task[-1])*100000,
                      "source_protocol": file_record(protocol), "policy_used": False,
                      "scope": "Local custom diagnostic relocations, not official LIBERO-PRO scores",
                      "feasibility": "New centers lie in the original object workspace; reset stability, visibility, no goal and object penetration are checked. Robot reachability is not an IK guarantee."}
    (output / "predeclaration.json").write_text(json.dumps(predeclaration, indent=2) + "\n")
    try:
        for v, variant in enumerate(VARIANTS):
            destination = output / variant
            destination.mkdir()
            accepted, records, rejected = [], [], []
            rng = np.random.default_rng(predeclaration["generation_seed"] + v*10000)
            for attempt in range(count*30):
                if len(accepted) == count:
                    break
                base_id = len(accepted) % len(base_states)
                state = base_states[base_id].copy()
                moved = []
                width = .025 if variant == "continuous_new_offsets" else .01
                planned = {}
                for role, name in (("target", target), ("distractor", distractor)):
                    if (variant == "target_moved_distractor_fixed" and role == "distractor") or (variant == "distractor_moved_target_fixed" and role == "target"):
                        continue
                    xy = np.asarray(CENTERS[role]) + rng.uniform(-width, width, 2)
                    offset = 1 + joint_positions[name]
                    state[offset:offset+2] = xy
                    planned[name] = xy.tolist()
                    moved.append(name)
                try:
                    if signature(state) in seen:
                        raise ValueError("Repeated or known physical configuration")
                    region_distances = {name:float(np.linalg.norm(known_xy-xy,axis=1).min())
                                        for name,xy in planned.items()}
                    if any(distance < .04 for distance in region_distances.values()):
                        raise ValueError("Novel position is within 4cm of a familiar target/distractor position")
                    # Reject object footprint overlap conservatively before reset.
                    for name in moved:
                        start = 1 + joint_positions[name]
                        for other in objects:
                            if other == name:
                                continue
                            other_start = 1 + joint_positions[other]
                            separation = np.linalg.norm(state[start:start+2]-state[other_start:other_start+2])
                            minimum = float(objects[name].horizontal_radius + objects[other].horizontal_radius) + .008
                            if separation < minimum:
                                raise ValueError(f"Footprint overlap {name}/{other}: {separation:.4f} < {minimum:.4f}")
                    env.initial_states = np.asarray([state])
                    observation, _reset = LiberoEnv.reset(env, seed=200000+len(accepted), options={"initial_state_id": 0})
                    sim = backend.sim
                    if not np.isfinite(env.backend.get_sim_state()).all() or env.backend.check_success():
                        raise ValueError("Nonfinite or initially solved reset")
                    locations = {name: sim.data.body_xpos[backend.obj_body_id[name]].copy() for name in objects}
                    for name in moved:
                        if np.linalg.norm(locations[name][:2]-planned[name]) > .035:
                            raise ValueError(f"Relocated object moved excessively while settling: {name}")
                    masks, mask_metadata = simulator_masks(env, observation)
                    if any(max(masks[f"image__{name}"].sum(),masks[f"wrist_image__{name}"].sum()) < 20 for name in (target, distractor)):
                        raise ValueError("Target or distractor insufficiently visible in both policy cameras")
                    seen.add(signature(state))
                    accepted.append(state)
                    records.append({"state_id": len(accepted)-1, "attempt": attempt,
                                    "base_official_test_state_id": definition["test_state_ids"][base_id],
                                    "planned_xy": planned, "settled_xyz": {k:x.tolist() for k,x in locations.items()},
                                    "nearest_familiar_target_or_distractor_xy_distance_m":region_distances,
                                    "external_target_pixels": int(masks[f"image__{target}"].sum()),
                                    "external_distractor_pixels": int(masks[f"image__{distractor}"].sum()),
                                    "wrist_target_pixels":int(masks[f"wrist_image__{target}"].sum()),
                                    "wrist_distractor_pixels":int(masks[f"wrist_image__{distractor}"].sum()),
                                    "qpos_sha256": hashlib.sha256(signature(state)).hexdigest(),
                                    "camera_alignment": mask_metadata["camera_alignment"]})
                    if len(accepted) in (1, count):
                        from PIL import Image
                        frame = np.concatenate([observation["openpi"]["observation/image"], observation["openpi"]["observation/wrist_image"]], axis=1)
                        Image.fromarray(frame).save(destination / f"initial_{len(accepted)-1:03d}.png")
                    print(json.dumps({"task":task,"variant":variant,"accepted":len(accepted),"count":count}),flush=True)
                except (ValueError, RuntimeError) as exc:
                    rejected.append({"attempt": attempt, "error": str(exc)})
                    print(json.dumps({"task":task,"variant":variant,"rejected_attempt":attempt,"reason":str(exc)}),flush=True)
            (destination / "generation.json").write_text(json.dumps({"accepted":records,"rejected":rejected},indent=2)+"\n")
            if len(accepted) != count:
                raise RuntimeError(f"Only {len(accepted)}/{count} valid states for {task}/{variant}")
            np.save(destination / "states.npy", np.asarray(accepted), allow_pickle=False)
            frozen = {"task":task,"variant":variant,"scope":predeclaration["scope"],"state_ids":list(range(count)),
                      "max_steps":definition["max_steps"],"bddl":definition["pro"]["bddl"],
                      "states":file_record(destination / "states.npy"),"source_protocol":file_record(protocol),
                      "generation":file_record(destination / "generation.json"),"predeclaration":file_record(output / "predeclaration.json")}
            (destination / "protocol.json").write_text(json.dumps(frozen,indent=2)+"\n")
            original = (protocol.parent / "test.toml").read_text()
            start, tail = original.split("[env.kwargs]",1)
            _, remaining = tail.split("[train]",1)
            start = start.replace("rl_vla.envs.libero_pro:make_env", "rl_vla.diagnostics.observation_bank:make_ood_env")
            config = start + f'[env.kwargs]\nprotocol = "{destination / "protocol.json"}"\n\n[train]' + remaining
            (destination / "evaluate.toml").write_text(config)
    finally:
        env.close()
    (output / "summary.json").write_text(json.dumps({"complete":True,"task":task,"variants":list(VARIANTS),"states_per_variant":count},indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--task",choices=list(OBJECTS),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--count",type=int,default=50)
    args=parser.parse_args()
    prepare(args.protocol,args.task,args.output,args.count)
