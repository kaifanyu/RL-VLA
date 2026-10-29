"""Prepare a deterministic LIBERO-PRO Position pilot, with resumable CPU work.

Each task has an independent protocol compatible with the existing PRO adapter.
No policy outcomes participate in task selection, state generation, or splitting.
"""

import argparse
import ast
import hashlib
import json
import multiprocessing
import os
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from prepare_pro_spatial_grasp import DATASET, PRO_COMMIT, REVISION, ROOT, download, record

from rl_vla.envs.libero import SETTLING_ACTION, LiberoEnv
from rl_vla.envs.libero_pro import make_env, validate_splits
from rl_vla.robustness import load_states

TASKS = {
    "spatial_1": {"suite": "libero_spatial", "task_id": 1, "max_steps": 220,
                  "generation_seed": 2026112700, "split_seed": 20261028},
    "spatial_2": {"suite": "libero_spatial", "task_id": 2, "max_steps": 220,
                  "generation_seed": 2026122700, "split_seed": 20261029},
    "object_0": {"suite": "libero_object", "task_id": 0, "max_steps": 280,
                 "generation_seed": 2026132700, "split_seed": 20261030},
}
TRAIN_COUNT, VALIDATION_COUNT, TEST_COUNT = 100, 50, 50
GENERATED_COUNT = TRAIN_COUNT + VALIDATION_COUNT
MAX_ATTEMPTS = 1000


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_record(item):
    if record(Path(item["path"]))["sha256"] != item["sha256"]:
        raise ValueError(f"Preparation asset changed: {item['path']}")


def task_spec(task_key, mapping):
    spec = dict(TASKS[task_key])
    condition = spec["suite"] + "_swap"
    name = mapping[condition][spec["task_id"]]
    if name != mapping[spec["suite"]][spec["task_id"]]:
        raise ValueError("PRO/source task indices disagree")
    return spec | {"key": task_key, "condition_suite": condition, "name": name}


def prepare_assets(output, keys):
    output.mkdir(parents=True, exist_ok=True)
    pro = ROOT / "third_party/libero_pro"
    commit = subprocess.check_output(["git", "-C", str(pro), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != PRO_COMMIT:
        raise ValueError("Unexpected PRO source revision")
    mapping = ast.literal_eval(ast.parse(
        (pro / "libero/libero/benchmark/libero_suite_task_map.py").read_text()).body[0].value)
    specs = {key: task_spec(key, mapping) for key in keys}
    selection = {
        "version": 1, "tasks": specs, "condition": "Position/swap",
        "selection": "Fixed task IDs Spatial1, Spatial2, Object0; no policy outcome screening.",
        "dataset": DATASET, "dataset_revision": REVISION, "pro_commit": commit,
    }
    selection_path = output / "task_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Refusing to alter an existing suite selection")
    write_json(selection_path, selection)
    config_dir = ROOT / ".cache/libero-pro"
    config_dir.mkdir(exist_ok=True)
    assets = ROOT / "datasets/benchmark_assets/libero_pro"
    expected_config = {
        "benchmark_root": str(pro / "libero/libero"),
        "bddl_files": str(assets / "bddl_files"), "init_states": str(assets / "init_files"),
        "datasets": str(ROOT / "datasets"), "assets": str(pro / "libero/libero/assets"),
    }
    config_path = config_dir / "config.yaml"
    if config_path.exists():
        if yaml.safe_load(config_path.read_text()) != expected_config:
            raise ValueError("Existing PRO runtime asset configuration differs")
    else:
        config_path.write_text(yaml.safe_dump(expected_config))
    for key, spec in specs.items():
        directory = output / key
        directory.mkdir(exist_ok=True)
        manifest_path = directory / "assets.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["spec"] != spec:
                raise ValueError(f"Refusing to alter existing task: {key}")
            for name in ("bddl", "published_states", "source_bddl", "source_states"):
                verify_record(manifest[name])
            continue
        name, condition = spec["name"], spec["condition_suite"]
        _, bddl_record = download(f"bddl_files/{condition}/{name}.bddl")
        states, state_record = download(f"init_files/{condition}/{name}.pruned_init")
        original = ROOT / "third_party/openpi/third_party/libero/libero/libero"
        source_bddl = original / f"bddl_files/{spec['suite']}/{name}.bddl"
        source_states = original / f"init_files/{spec['suite']}/{name}.pruned_init"
        if len(load_states(states)) != TEST_COUNT or len(load_states(source_states)) != TEST_COUNT:
            raise ValueError("Expected 50 published PRO and original states")
        manifest = {"spec": spec, "pro_source": str(pro), "pro_commit": commit,
                    "dataset": DATASET, "dataset_revision": REVISION, "name": name,
                    "bddl": bddl_record, "published_states": state_record,
                    "source_bddl": record(source_bddl), "source_states": record(source_states)}
        write_json(manifest_path, manifest)
        print(f"Pinned assets ready: {key}: {name}", flush=True)


def signature(state, nq):
    """Compare physical configurations, ignoring MuJoCo time and controller state."""
    return np.round(np.asarray(state)[1:1 + nq], 12).tobytes()


def generate_attempt(bddl, seed, shape, path):
    """Capture one freshly constructed simulator, matching the pinned generator."""
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(seed)
    random.seed(seed)
    env, state, nq, error = None, np.empty(0), 0, ""
    try:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        state = np.asarray(env.get_sim_state()).copy()
        nq = env.sim.model.nq
        if state.shape != shape or not np.isfinite(state).all():
            raise ValueError("Generated state has incompatible shape or nonfinite values")
        env.seed(seed)
        env.reset()
        env.set_init_state(state)
        if env.check_success():
            raise ValueError("State already satisfies goal")
        for _ in range(10):
            _, _, done, _ = env.step(SETTLING_ACTION.tolist())
            if done or env.check_success():
                raise ValueError("State ends during settling")
        if not np.isfinite(env.get_sim_state()).all():
            raise ValueError("State becomes nonfinite during settling")
    except Exception as exc:  # noqa: BLE001 - save every failed attempt for deterministic audit.
        error = repr(exc)
    finally:
        if env is not None:
            env.close()
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez(file, state=state, nq=nq, seed=seed, error=error)
    temporary.replace(path)
    return path


def collect_attempts(paths, published, count=GENERATED_COUNT):
    """Consume attempts in seed order, independent of worker completion order."""
    generated, accepted, rejected = [], [], []
    physical, expected_nq = None, None
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            state, nq, seed, error = (saved["state"], int(saved["nq"]),
                                     int(saved["seed"]), str(saved["error"]))
        if error:
            rejected.append({"generation_seed": seed, "error": error})
            continue
        if state.shape != published.shape[1:] or not np.isfinite(state).all() or nq <= 0:
            raise ValueError(f"Invalid cached generation result: {path}")
        if physical is None:
            expected_nq = nq
            physical = {signature(row, nq) for row in published}
            if len(physical) != len(published):
                raise ValueError("Published states have duplicate physical configurations")
        if nq != expected_nq:
            raise ValueError("State dimensions changed between simulator constructors")
        key = signature(state, nq)
        if key in physical:
            rejected.append({"generation_seed": seed, "error": "Duplicate physical configuration"})
            continue
        physical.add(key)
        generated.append(state)
        accepted.append({"generation_seed": seed,
                         "qpos_sha256": hashlib.sha256(key).hexdigest(),
                         "attempt": record(Path(path))})
        if len(generated) == count:
            break
    return generated, accepted, rejected


def freeze_task(directory, assets, generated, accepted, rejected):
    spec = assets["spec"]
    published = load_states(assets["published_states"]["path"])
    order = np.random.default_rng(spec["split_seed"]).permutation(GENERATED_COUNT)
    combined = np.concatenate([np.asarray(generated)[order], published])
    state_path = directory / "pro_states.npy"
    temporary = state_path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.save(file, combined, allow_pickle=False)
    temporary.replace(state_path)
    write_json(directory / "generation.json", {
        "method": "Seed NumPy/Python; construct fresh PRO env; capture before first explicit reset. "
                  "Reject invalid states, duplicate qpos, and goals met before/within ten settling "
                  "actions. Consume attempt seeds in order; no policy filtering.",
        "accepted": [accepted[i] for i in order], "rejected": rejected,
        "generated_state_count": GENERATED_COUNT, "published_test_start": GENERATED_COUNT,
        "generation_seed": spec["generation_seed"], "split_seed": spec["split_seed"],
        "generator": record(Path(__file__)),
        "upstream_generator": record(ROOT / "third_party/libero_pro/notebooks/generate_init_states.py"),
    })
    task = {"suite": spec["suite"], "task_id": spec["task_id"], "name": spec["name"],
            "condition": "Position/swap", "condition_suite": spec["condition_suite"],
            "max_steps": spec["max_steps"], "pro_state_count": len(combined),
            "train_state_ids": list(range(TRAIN_COUNT)),
            "validation_state_ids": list(range(TRAIN_COUNT, GENERATED_COUNT)),
            "test_state_ids": list(range(GENERATED_COUNT, len(combined))),
            "source_state_ids": list(range(TEST_COUNT)),
            "pro": {"bddl": assets["bddl"], "states": record(state_path)},
            "source": {"bddl": assets["source_bddl"], "states": assets["source_states"]}}
    validate_splits(task, 2)
    protocol = directory / "protocol.json"
    plan = {"version": 2, "purpose": f"{spec['key']} PRO Position controlled adaptation",
            "pro_source": assets["pro_source"], "pro_commit": assets["pro_commit"],
            "dataset": DATASET, "dataset_revision": REVISION,
            "training_seed": 0, "evaluation_seed": 100000,
            "validation_iterations": [25, 50, 75, 100],
            "selection": "Highest validation success count; tie: earliest iteration. "
                         "Select among trained checkpoints only, even if baseline scores higher.",
            "scope": "Disjoint initial states in the same published PRO Position condition; "
                     "not disjoint perturbation ranges or a full suite benchmark. Single training seed.",
            "tasks": {spec["key"]: task}}
    tail = "[train]" + (ROOT / "configs/rldt_pi05_libero.toml").read_text().split("[train]", 1)[1]
    for split in ("train", "validation", "test", "source"):
        config = f'''algorithm = "rldt"

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

'''
        path = directory / f"{split}.toml"
        content = config + tail + "\n[recording]\nenabled = true\n"
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Refusing to alter an existing configuration: {path}")
        path.write_text(content)
    # Protocol is the commit marker: configs and state provenance must exist first.
    write_json(protocol, plan)


def generate_task(directory, workers):
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".cache/libero-pro")
    assets = json.loads((directory / "assets.json").read_text())
    for name in ("bddl", "published_states", "source_bddl", "source_states"):
        verify_record(assets[name])
    if (directory / "protocol.json").exists():
        plan = json.loads((directory / "protocol.json").read_text())
        task = plan["tasks"][assets["spec"]["key"]]
        validate_splits(task, 2)
        verify_record(task["pro"]["states"])
        return
    attempts = directory / "generation_attempts"
    attempts.mkdir(exist_ok=True)
    context = {"assets": record(directory / "assets.json"), "generator": record(Path(__file__)),
               "generated_count": GENERATED_COUNT, "max_attempts": MAX_ATTEMPTS}
    context_path = attempts / "context.json"
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("Cannot resume generation after its inputs or generator changed")
    write_json(context_path, context)
    published = load_states(assets["published_states"]["path"])
    base_seed = assets["spec"]["generation_seed"]
    ordered_paths = []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for start in range(0, MAX_ATTEMPTS, workers):
            paths = [attempts / f"{index:04d}.npz"
                     for index in range(start, min(start + workers, MAX_ATTEMPTS))]
            jobs = [pool.submit(generate_attempt, assets["bddl"]["path"], base_seed + index,
                                published.shape[1:], str(path))
                    for index, path in enumerate(paths, start=start) if not path.exists()]
            for future in as_completed(jobs):
                future.result()
            ordered_paths.extend(paths)
            generated, accepted, rejected = collect_attempts(ordered_paths, published)
            write_json(directory / "generation_progress.json", {
                "accepted": len(generated), "rejected": rejected,
                "attempts_cached": len(ordered_paths), "target": GENERATED_COUNT})
            print(f"{assets['spec']['key']}: generated {len(generated)}/{GENERATED_COUNT}; "
                  f"rejected {len(rejected)}", flush=True)
            if len(generated) == GENERATED_COUNT:
                break
            if len(rejected) >= 20 and not generated:
                raise RuntimeError("Repeated state-generation failures; inspect generation_progress.json")
    if len(generated) != GENERATED_COUNT:
        raise RuntimeError("Insufficient valid states")
    freeze_task(directory, assets, generated, accepted, rejected)


def check_chunk(protocol, task_key, split, state_ids, output):
    env = make_env(protocol=protocol, task_key=task_key, split=split)
    checks = []
    try:
        for state_id in state_ids:
            obs, info = LiberoEnv.reset(env, seed=100000 + state_id,
                                       options={"initial_state_id": state_id})
            if not np.isfinite(obs["state"]).all():
                raise ValueError("Nonfinite reset state")
            for camera in ("observation/image", "observation/wrist_image"):
                value = obs["openpi"][camera]
                if value.shape != (224, 224, 3) or value.dtype != np.uint8 or value.std() < 1:
                    raise ValueError(f"Invalid camera observation: {camera}")
            checks.append({"split": split, "state_id": state_id, "reset": info})
        if state_ids:
            import imageio.v3 as iio

            iio.imwrite(Path(output) / f"{split}_{state_ids[-1]:03d}.png",
                        obs["openpi"]["observation/image"])
    finally:
        env.close()
    return checks


def check_task(directory, workers):
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".cache/libero-pro")
    protocol = directory / "protocol.json"
    plan = json.loads(protocol.read_text())
    key, = plan["tasks"]
    task = plan["tasks"][key]
    output = directory / "reset_checks"
    output.mkdir(exist_ok=True)
    context = {"protocol": record(protocol), "checker": record(Path(__file__))}
    context_path = output / "context.json"
    if context_path.exists() and json.loads(context_path.read_text()) != context:
        raise ValueError("Cannot reuse reset checks after their inputs or checker changed")
    write_json(context_path, context)
    futures, checks = {}, []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for split in ("train", "validation", "test", "source"):
            ids = task[f"{split}_state_ids"]
            for start in range(0, len(ids), 25):
                chunk = ids[start:start + 25]
                path = output / f"{split}_{chunk[0]:03d}_{chunk[-1]:03d}.json"
                if path.exists():
                    checks.extend(json.loads(path.read_text()))
                else:
                    future = pool.submit(check_chunk, protocol, key, split, chunk, output)
                    futures[future] = path
        for future in as_completed(futures):
            rows = future.result()
            write_json(futures[future], rows)
            checks.extend(rows)
            write_json(output / "progress.json", {"checked": len(checks), "total": 250})
            print(f"{key}: validated {len(checks)}/250 reset states", flush=True)
    for split in ("train", "validation", "test", "source"):
        actual = sorted(row["state_id"] for row in checks if row["split"] == split)
        if actual != task[f"{split}_state_ids"]:
            raise ValueError(f"Wrong reset validation coverage: {key}/{split}")
    checks.sort(key=lambda row: (row["split"], row["state_id"]))
    write_json(directory / "setup_check.json", {
        "states_checked": len(checks), "checks": checks, "protocol": record(protocol),
        "method": "Same LiberoEnv.reset/settle path; independent simulator processes; "
                  "finite state and two nonblank 224x224 camera checks for every reset."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("assets", "generate", "check", "all"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output.resolve()
    if args.phase in ("assets", "all"):
        prepare_assets(output, args.tasks)
    for key in args.tasks:
        directory = output / key
        if args.phase in ("generate", "all"):
            generate_task(directory, args.workers)
        if args.phase in ("check", "all"):
            check_task(directory, args.workers)


if __name__ == "__main__":
    main()
