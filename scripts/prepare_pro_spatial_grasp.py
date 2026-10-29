"""Freeze a Spatial-0 PRO position experiment with generated train/validation states."""

import argparse
import ast
import hashlib
import json
import os
import random
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from rl_vla.envs.libero import SETTLING_ACTION, LiberoEnv
from rl_vla.envs.libero_pro import validate_splits
from rl_vla.robustness import load_states

ROOT = Path(__file__).resolve().parents[1]
PRO_COMMIT = "eafdb809426b13153aa1e4c42d6601844217dfec"
REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"
DATASET = "zhouxueyang/LIBERO-Pro"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def record(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def download(relative):
    target = ROOT / "datasets/benchmark_assets/libero_pro" / relative
    request = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{DATASET}/paths-info/{REVISION}",
        data=json.dumps({"paths": [relative]}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.load(response)[0]
    if target.exists():
        blob = target.read_bytes()
    else:
        with urllib.request.urlopen(
            f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{relative}",
            timeout=120,
        ) as response:
            blob = response.read()
    actual = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
    if len(blob) != metadata["size"] or actual != metadata["oid"]:
        raise ValueError(f"Pinned dataset integrity failure: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    return target, record(target) | {"git_blob_sha1": actual, "repository_path": relative}


def prepare_assets(output):
    output.mkdir(parents=True, exist_ok=False)
    pro = ROOT / "third_party/libero_pro"
    commit = subprocess.check_output(["git", "-C", str(pro), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != PRO_COMMIT:
        raise ValueError("Unexpected PRO source revision")
    mapping = ast.literal_eval(ast.parse(
        (pro / "libero/libero/benchmark/libero_suite_task_map.py").read_text()).body[0].value)
    name = mapping["libero_spatial_swap"][0]
    _, bddl_record = download(f"bddl_files/libero_spatial_swap/{name}.bddl")
    states, state_record = download(f"init_files/libero_spatial_swap/{name}.pruned_init")
    original = ROOT / "third_party/openpi/third_party/libero/libero/libero"
    source_bddl = original / f"bddl_files/libero_spatial/{name}.bddl"
    source_states = original / f"init_files/libero_spatial/{name}.pruned_init"
    if len(load_states(states)) != 50 or len(load_states(source_states)) != 50:
        raise ValueError("Expected 50 published PRO and original states")
    config_dir = ROOT / ".cache/libero-pro"
    config_dir.mkdir(exist_ok=True)
    assets = ROOT / "datasets/benchmark_assets/libero_pro"
    (config_dir / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(pro / "libero/libero"),
        "bddl_files": str(assets / "bddl_files"), "init_states": str(assets / "init_files"),
        "datasets": str(ROOT / "datasets"), "assets": str(pro / "libero/libero/assets"),
    }))
    manifest = {"pro_source": str(pro), "pro_commit": commit, "dataset": DATASET,
                "dataset_revision": REVISION, "name": name, "bddl": bddl_record,
                "published_states": state_record, "source_bddl": record(source_bddl),
                "source_states": record(source_states)}
    write_json(output / "assets.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def generate(output):
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".cache/libero-pro")
    from libero.libero.envs import OffScreenRenderEnv

    assets = json.loads((output / "assets.json").read_text())
    if (output / "protocol.json").exists():
        raise FileExistsError("Protocol already frozen")
    published = load_states(assets["published_states"]["path"])
    generated, accepted, rejected = [], [], []
    physical = set()
    bddl = assets["bddl"]["path"]
    for attempt in range(1000):
        seed = 2026102600 + attempt
        np.random.seed(seed)
        random.seed(seed)
        env = None
        try:
            # Match the pinned upstream generator: capture immediately after construction.
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            state = np.asarray(env.get_sim_state()).copy()
            nq = env.sim.model.nq
            signature = lambda row, nq=nq: np.round(row[1:1 + nq], 12).tobytes()
            if not physical:
                physical.update(signature(row) for row in published)
                if len(physical) != 50:
                    raise ValueError("Published states have duplicate physical configurations")
            key = signature(state)
            if state.shape != published.shape[1:] or not np.isfinite(state).all():
                raise ValueError("Generated state has incompatible shape or nonfinite values")
            if key in physical:
                raise ValueError("Duplicate physical configuration")
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
            generated.append(state)
            accepted.append({"generation_seed": seed,
                             "qpos_sha256": hashlib.sha256(key).hexdigest()})
            physical.add(key)
        except Exception as exc:
            # No policy is used for filtering; keep every rejection for audit.
            rejected.append({"generation_seed": seed, "error": repr(exc)})
            if len(rejected) >= 20 and not generated:
                raise RuntimeError("Repeated state-generation failures") from exc
        finally:
            if env is not None:
                env.close()
        write_json(output / "generation_progress.json", {
            "accepted": len(generated), "rejected": rejected, "attempts": attempt + 1})
        print(f"Generated {len(generated)}/150, rejected {len(rejected)}", flush=True)
        if len(generated) == 150:
            break
    if len(generated) != 150:
        raise RuntimeError("Insufficient valid states")
    order = np.random.default_rng(20261026).permutation(150)
    combined = np.concatenate([np.asarray(generated)[order], published])
    state_path = output / "pro_states.npy"
    np.save(state_path, combined, allow_pickle=False)
    write_json(output / "generation.json", {
        "method": "Seed NumPy/Python; construct fresh PRO env; capture before first explicit reset, "
                  "matching pinned notebooks/generate_init_states.py. Reject duplicates, "
                  "invalid states and goals satisfied before/within ten settling actions. No policy filtering.",
        "accepted": [accepted[i] for i in order], "rejected": rejected,
        "generated_state_count": 150, "published_test_start": 150,
        "split_seed": 20261026, "generator": record(Path(__file__)),
        "upstream_generator": record(ROOT / "third_party/libero_pro/notebooks/generate_init_states.py"),
    })
    task = {"suite": "libero_spatial", "task_id": 0, "name": assets["name"],
            "max_steps": 220, "pro_state_count": 200,
            "train_state_ids": list(range(100)), "validation_state_ids": list(range(100, 150)),
            "test_state_ids": list(range(150, 200)), "source_state_ids": list(range(50)),
            "pro": {"bddl": assets["bddl"], "states": record(state_path)},
            "source": {"bddl": assets["source_bddl"], "states": assets["source_states"]}}
    validate_splits(task, 2)
    plan = {"version": 2, "purpose": "Spatial task 0 PRO Position RLDT adaptation",
            "pro_source": assets["pro_source"], "pro_commit": assets["pro_commit"],
            "dataset": DATASET, "dataset_revision": REVISION,
            "training_seed": 0, "evaluation_seed": 100000,
            "validation_iterations": [25, 50, 75, 100],
            "selection": "Highest validation success count; tie: earliest iteration. "
                         "Select among RLDT checkpoints only, even if baseline scores higher.",
            "scope": "Disjoint initial states in the same PRO Position condition, not disjoint "
                     "perturbation ranges or a full suite benchmark. Single training seed.",
            "tasks": {"spatial_0": task}}
    protocol = output / "protocol.json"
    write_json(protocol, plan)
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
task_key = "spatial_0"
split = "{split}"

'''
        (output / f"{split}.toml").write_text(config + tail)
    # Check every saved state through the same reset/settle path used during learning.
    from rl_vla.envs.libero_pro import make_env

    checks = []
    for split in ("train", "validation", "test", "source"):
        env = make_env(protocol=protocol, task_key="spatial_0", split=split)
        try:
            for state_id in task[f"{split}_state_ids"]:
                obs, info = LiberoEnv.reset(env, seed=100000 + state_id,
                                           options={"initial_state_id": state_id})
                if not all(np.isfinite(np.asarray(obs[k])).all() for k in ("state",)):
                    raise ValueError("Nonfinite reset observation")
                checks.append({"split": split, "state_id": state_id, "reset": info})
                if state_id == task[f"{split}_state_ids"][0]:
                    import imageio.v3 as iio
                    iio.imwrite(output / f"initial_{split}.png", obs["openpi"]["observation/image"])
        finally:
            env.close()
    write_json(output / "setup_check.json", {"states_checked": len(checks), "checks": checks})
    print(f"Frozen protocol and validated {len(checks)} reset states: {protocol}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("assets", "generate"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (prepare_assets if args.phase == "assets" else generate)(args.output.resolve())
