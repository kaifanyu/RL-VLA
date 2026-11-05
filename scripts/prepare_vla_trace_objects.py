"""Prepare official LIBERO-PRO Object 0/1/2 swaps in a new immutable campaign.

Reuses the pinned fresh-constructor generator, official downloader and native
reset adapter. No policy is loaded or consulted during generation or filtering.
"""

import argparse
import ast
import hashlib
import json
import multiprocessing
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from prepare_pro_spatial_grasp import download
from prepare_pro_suite_grasp import (
    DATASET,
    PRO_COMMIT,
    REVISION,
    ROOT,
    generate_attempt,
    record,
    signature,
    verify_record,
    write_json,
)

from rl_vla.envs.libero import LiberoEnv
from rl_vla.envs.libero_pro import make_env, validate_splits
from rl_vla.robustness import load_states

TASKS = {
    f"object_{i}": {"suite": "libero_object", "task_id": i, "max_steps": 280,
                    "generation_seed": 2026310000 + 10000 * i,
                    "split_seed": 2026103100 + i}
    for i in range(3)
}
COUNTS = {"train": 100, "validation": 50, "test": 50, "source": 50}


def immutable_json(path, payload):
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"Refusing to alter frozen artifact: {path}")
    else:
        write_json(path, payload)


def prepare_assets(directory, key):
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "assets.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for field in ("bddl", "published_states", "source_bddl", "source_states"):
            verify_record(manifest[field])
        for item in manifest["screen_states"]:
            verify_record(item)
        return manifest
    pro = ROOT / "third_party/libero_pro"
    commit = subprocess.check_output(["git", "-C", str(pro), "rev-parse", "HEAD"], text=True).strip()
    if commit != PRO_COMMIT:
        raise ValueError("PRO source does not match pinned revision")
    mapping = ast.literal_eval(ast.parse(
        (pro / "libero/libero/benchmark/libero_suite_task_map.py").read_text()).body[0].value)
    spec = TASKS[key] | {"key": key, "condition_suite": "libero_object_swap"}
    name = mapping[spec["condition_suite"]][spec["task_id"]]
    if name != mapping[spec["suite"]][spec["task_id"]]:
        raise ValueError("PRO/source indices disagree")
    spec["name"] = name
    _, bddl = download(f"bddl_files/libero_object_swap/{name}.bddl")
    _, published = download(f"init_files/libero_object_swap/{name}.pruned_init")
    original = ROOT / "third_party/openpi/third_party/libero/libero/libero"
    source_bddl = record(original / f"bddl_files/libero_object/{name}.bddl")
    source_states = record(original / f"init_files/libero_object/{name}.pruned_init")
    if len(load_states(published["path"])) != 50 or len(load_states(source_states["path"])) != 50:
        raise ValueError("Expected 50 published PRO and original states")
    screens = []
    for ready_path in sorted((ROOT / "runs").glob(
            f"benchmark_screen*/cases/pro_libero_object_swap_{spec['task_id']:02d}/ready.json")):
        ready = json.loads(ready_path.read_text())
        if ready["source_commit"] != PRO_COMMIT or ready["dataset_revision"] != REVISION:
            raise ValueError(f"Unrecognized development screen: {ready_path}")
        screens.append(ready["states_record"])
    assets = ROOT / "datasets/benchmark_assets/libero_pro"
    runtime = {"benchmark_root": str(pro / "libero/libero"),
               "bddl_files": str(assets / "bddl_files"), "init_states": str(assets / "init_files"),
               "datasets": str(ROOT / "datasets"), "assets": str(pro / "libero/libero/assets")}
    config = ROOT / ".cache/libero-pro/config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    if config.exists() and yaml.safe_load(config.read_text()) != runtime:
        raise ValueError("Refusing to change PRO runtime paths")
    if not config.exists():
        config.write_text(yaml.safe_dump(runtime))
    manifest = {"spec": spec, "name": name, "pro_source": str(pro), "pro_commit": commit,
                "dataset": DATASET, "dataset_revision": REVISION, "bddl": bddl,
                "published_states": published, "source_bddl": source_bddl,
                "source_states": source_states, "screen_states": screens,
                "condition": "Position/swap", "official_condition": True,
                "screen_inventory": "No prior local screen states found" if not screens else "All found screens excluded"}
    immutable_json(manifest_path, manifest)
    return manifest


def collect(paths, excluded, count=150):
    generated, accepted, rejected = [], [], []
    seen, expected_nq = None, None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            state, nq, seed, error = data["state"], int(data["nq"]), int(data["seed"]), str(data["error"])
        if error:
            rejected.append({"generation_seed": seed, "error": error})
            continue
        if state.shape != excluded.shape[1:] or not np.isfinite(state).all() or nq <= 0:
            raise ValueError(f"Invalid generator artifact: {path}")
        if seen is None:
            expected_nq = nq
            # Exclusion banks may repeat the same screen/published state. This
            # union tolerates repeated exclusions, never repeated accepted states.
            seen = {signature(row, nq) for row in excluded}
        if nq != expected_nq:
            raise ValueError("Simulator dimensions changed")
        physical = signature(state, nq)
        if physical in seen:
            rejected.append({"generation_seed": seed, "error": "Duplicate or excluded physical state"})
            continue
        seen.add(physical)
        generated.append(state)
        accepted.append({"generation_seed": seed, "nq": nq,
                         "qpos_sha256": hashlib.sha256(physical).hexdigest(), "attempt": record(path)})
        if len(generated) == count:
            break
    return generated, accepted, rejected


def freeze(directory, assets, generated, accepted, rejected):
    spec = assets["spec"]
    order = np.random.default_rng(spec["split_seed"]).permutation(150)
    states = np.concatenate([np.asarray(generated)[order], load_states(assets["published_states"]["path"])])
    state_path = directory / "pro_states.npy"
    with state_path.open("xb") as stream:
        np.save(stream, states, allow_pickle=False)
    immutable_json(directory / "generation.json", {
        "method": "Pinned fresh seeded constructor before explicit reset; no policy filtering. "
                  "Reject invalid states, initially solved goals, and duplicate physical qpos "
                  "against published, original retention, development screens and accepted states.",
        "accepted": [accepted[i] for i in order], "rejected": rejected,
        "generated_state_count": 150, "published_test_start": 150,
        "generation_seed": spec["generation_seed"], "split_seed": spec["split_seed"],
        "generator": record(Path(__file__)),
        "reused_generator": record(ROOT / "scripts/prepare_pro_suite_grasp.py"),
        "upstream_generator": record(ROOT / "third_party/libero_pro/notebooks/generate_init_states.py"),
        "excluded_states": [assets["published_states"], assets["source_states"], *assets["screen_states"]],
        "test_origin": "published_PRO"})
    task = {field: spec[field] for field in ("suite", "task_id", "name", "max_steps")}
    task.update(condition="Position/swap", condition_suite="libero_object_swap",
                test_scope="50 published PRO states excluded from training and selection",
                pro_state_count=200, train_state_ids=list(range(100)),
                validation_state_ids=list(range(100, 150)), test_state_ids=list(range(150, 200)),
                source_state_ids=list(range(50)),
                pro={"bddl": assets["bddl"], "states": record(state_path)},
                source={"bddl": assets["source_bddl"], "states": assets["source_states"]})
    validate_splits(task, 2)
    protocol = directory / "protocol.json"
    plan = {"version": 2, "purpose": f"VLA-Trace {spec['key']} official PRO Position/swap adaptation",
            "pro_source": assets["pro_source"], "pro_commit": PRO_COMMIT, "dataset": DATASET,
            "dataset_revision": REVISION, "training_seed": 0, "evaluation_seed": 100000,
            "validation_iterations": [25, 50, 100], "online_validation": True,
            "selection": "Highest validation success count among 25/50/100; ties earliest iteration",
            "scope": "Single RL seed; disjoint states within official PRO swap condition; not a full-suite score",
            "tasks": {spec["key"]: task}}
    tail = "[train]" + (ROOT / "configs/rldt_pi05_libero.toml").read_text().split("[train]", 1)[1]
    for split in COUNTS:
        content = f'''algorithm = "rldt"

[policy]
kind = "pi05"
checkpoint = "{ROOT / 'checkpoints/openpi/pi05_libero'}"
model_config = "pi05_libero"
projection_only = false

[env]
kind = "external"
factory = "rl_vla.envs.libero_pro:make_env"
python = "{ROOT / '.venv-pro/bin/python'}"

[env.kwargs]
protocol = "{protocol}"
task_key = "{spec['key']}"
split = "{split}"

''' + tail + "\n[recording]\nenabled = true\n"
        (directory / f"{split}.toml").write_text(content)
    # Last write commits the full task; subsequent invocations only verify it.
    immutable_json(protocol, plan)


def generate(directory, assets, workers):
    if (directory / "protocol.json").exists():
        plan = json.loads((directory / "protocol.json").read_text())
        task = plan["tasks"][assets["spec"]["key"]]
        validate_splits(task, 2)
        verify_record(task["pro"]["states"])
        return
    attempts = directory / "generation_attempts"
    attempts.mkdir(exist_ok=True)
    immutable_json(attempts / "context.json", {"assets": record(directory / "assets.json"),
                   "generator": record(Path(__file__)),
                   "reused_generator": record(ROOT / "scripts/prepare_pro_suite_grasp.py")})
    excluded = np.concatenate([load_states(item["path"]) for item in
                               [assets["published_states"], assets["source_states"], *assets["screen_states"]]])
    paths = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for start in range(0, 1000, workers):
            batch = [attempts / f"{i:04d}.npz" for i in range(start, min(start + workers, 1000))]
            futures = [pool.submit(generate_attempt, assets["bddl"]["path"], assets["spec"]["generation_seed"] + i,
                                   excluded.shape[1:], str(path))
                       for i, path in enumerate(batch, start=start) if not path.exists()]
            for future in as_completed(futures):
                future.result()
            paths.extend(batch)
            generated, accepted, rejected = collect(paths, excluded)
            write_json(directory / "generation_progress.json", {"accepted": len(generated), "target": 150,
                       "attempts_cached": len(paths), "rejected": rejected})
            print(f"{assets['spec']['key']}: generated {len(generated)}/150, rejected {len(rejected)}", flush=True)
            if len(generated) == 150:
                freeze(directory, assets, generated, accepted, rejected)
                return
            if not generated and len(rejected) >= 20:
                raise RuntimeError("Repeated state-generation failures")
    raise RuntimeError("Insufficient valid unique states")


def split_audit(directory):
    plan = json.loads((directory / "protocol.json").read_text())
    key, = plan["tasks"]
    task = plan["tasks"][key]
    validate_splits(task, 2)
    all_pro = load_states(task["pro"]["states"]["path"])
    source = load_states(task["source"]["states"]["path"])
    generation = json.loads((directory / "generation.json").read_text())
    nq = generation["accepted"][0]["nq"]
    values = {s: source if s == "source" else all_pro[task[f"{s}_state_ids"]] for s in COUNTS}
    hashes = {s: [hashlib.sha256(signature(row, nq)).hexdigest() for row in rows] for s, rows in values.items()}
    pairs = {}
    for i, left in enumerate(COUNTS):
        if len(set(hashes[left])) != COUNTS[left]:
            raise ValueError(f"Duplicate physical state within {left}")
        for right in list(COUNTS)[i + 1:]:
            pairs[f"{left}/{right}"] = len(set(hashes[left]) & set(hashes[right]))
    if any(pairs.values()):
        raise ValueError(f"Physical state overlap: {pairs}")
    report = {"passed": True, "counts": COUNTS, "nq": nq, "qpos_overlap_counts": pairs,
              "qpos_sha256": hashes, "comparison": "qpos rounded to 12 decimals; ignores simulator clock/controller state",
              "state_id_namespace": "train/validation/test share pro state file; source IDs refer to separate source file",
              "protocol": record(directory / "protocol.json")}
    immutable_json(directory / "split_audit.json", report)


def reset_chunk(protocol, key, split, ids, output):
    env = make_env(protocol=protocol, task_key=key, split=split)
    rows = []
    try:
        for state_id in ids:
            env.backend.set_init_state(env.initial_states[state_id].copy())
            if env.backend.check_success():
                raise ValueError("Goal already satisfied before settling")
            obs, info = LiberoEnv.reset(env, seed=100000 + state_id, options={"initial_state_id": state_id})
            if env.backend.check_success():
                raise ValueError("Goal satisfied during reset settling")
            if not np.isfinite(env.backend.get_sim_state()).all() or not np.isfinite(obs["state"]).all():
                raise ValueError("Nonfinite simulator/reset state")
            for camera in ("observation/image", "observation/wrist_image"):
                pixels = obs["openpi"][camera]
                if pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8 or pixels.std() < 1:
                    raise ValueError(f"Invalid camera: {camera}")
            if obs["openpi"]["prompt"] != env.backend.language_instruction or not obs["openpi"]["prompt"]:
                raise ValueError("Instruction mismatch")
            positions = {name: env.backend.sim.data.body_xpos[body_id].tolist()
                         for name, body_id in env.backend.env.obj_body_id.items()}
            target = next(name for name in env.backend.obj_of_interest if name != "basket_1")
            if target not in positions or "basket_1" not in positions or not all(
                    np.isfinite(position).all() for position in positions.values()):
                raise ValueError("Missing/nonfinite target, distractors, basket")
            row = {"split": split, "state_id": state_id, "reset": info,
                   "qualified_state_id": f"{key}/{split}/{state_id}",
                   "prompt": obs["openpi"]["prompt"], "target": target,
                   "positions_world_after_settling": positions, "goal_before_settling": False,
                   "goal_after_settling": False, "camera_std": {
                       k: float(obs["openpi"][k].std()) for k in ("observation/image", "observation/wrist_image")}}
            rows.append(row)
            if state_id == ids[0]:
                import imageio.v3 as iio
                for camera, slug in (("observation/image", "external"), ("observation/wrist_image", "wrist")):
                    iio.imwrite(Path(output) / f"{split}_{state_id:03d}_{slug}.png", obs["openpi"][camera])
    finally:
        env.close()
    return rows


def montage(directory, checks):
    import imageio.v3 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(12, 14), layout="constrained")
    for i, split in enumerate(COUNTS):
        row = min((r for r in checks if r["split"] == split), key=lambda r: r["state_id"])
        for j, camera in enumerate(("external", "wrist")):
            axes[i, j].imshow(iio.imread(directory / "reset_checks" / f"{split}_{row['state_id']:03d}_{camera}.png"))
            axes[i, j].set_title(f"{split} / ID {row['state_id']} / {camera}")
            axes[i, j].axis("off")
        lines = [f"Target: {row['target']}", "Object IDs and world xyz (m):"]
        lines += [f"{name}:\n  " + ", ".join(f"{v:.3f}" for v in p)
                  for name, p in row["positions_world_after_settling"].items() if name != "floor"]
        axes[i, 2].text(0, 1, "\n".join(lines), va="top", fontsize=8, family="monospace")
        axes[i, 2].axis("off")
    fig.suptitle(f"{directory.name}: official PRO Position/swap (train/val/test), original retention\n{checks[0]['prompt']}")
    fig.savefig(directory / "reset_checks/initial_montage.png", dpi=130)
    fig.savefig(directory / "reset_checks/initial_montage.pdf")
    plt.close(fig)


def check(directory, workers):
    protocol = directory / "protocol.json"
    plan = json.loads(protocol.read_text())
    key, = plan["tasks"]
    task = plan["tasks"][key]
    output = directory / "reset_checks"
    output.mkdir(exist_ok=True)
    immutable_json(output / "context.json", {"protocol": record(protocol), "checker": record(Path(__file__))})
    split_audit(directory)
    checks, futures = [], {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for split in COUNTS:
            ids = task[f"{split}_state_ids"]
            for start in range(0, len(ids), 25):
                chunk = ids[start:start + 25]
                path = output / f"{split}_{chunk[0]:03d}_{chunk[-1]:03d}.json"
                if path.exists():
                    checks.extend(json.loads(path.read_text()))
                else:
                    futures[pool.submit(reset_chunk, protocol, key, split, chunk, output)] = path
        for future in as_completed(futures):
            rows = future.result()
            write_json(futures[future], rows)
            checks.extend(rows)
            write_json(output / "progress.json", {"checked": len(checks), "total": 250})
            print(f"{key}: reset-validated {len(checks)}/250", flush=True)
    for split in COUNTS:
        if sorted(r["state_id"] for r in checks if r["split"] == split) != task[f"{split}_state_ids"]:
            raise ValueError(f"Reset coverage differs: {split}")
    checks.sort(key=lambda r: (r["split"], r["state_id"]))
    montage(directory, checks)
    immutable_json(directory / "setup_check.json", {
        "states_checked": len(checks), "checks": checks, "protocol": record(protocol),
        "split_audit": record(directory / "split_audit.json"),
        "method": "Native LiberoEnv reset/settle; pre/post goal false, finite state, instruction, object IDs/world positions, two nonblank 224x224 cameras for every reset"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--phase", choices=("assets", "generate", "check", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Use 1..8 workers")
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".cache/libero-pro")
    for key in args.tasks:
        directory = args.output.resolve() / key
        assets = prepare_assets(directory, key)
        if args.phase in ("all", "generate"):
            generate(directory, assets, args.workers)
        if args.phase in ("all", "check"):
            check(directory, args.workers)


if __name__ == "__main__":
    main()
