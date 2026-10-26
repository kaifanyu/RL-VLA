"""Download verified PRO task assets and freeze a two-task RLDT pilot protocol."""

import argparse
import ast
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from rl_vla.robustness import load_states

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"
REPOSITORY = "zhouxueyang/LIBERO-Pro"


def file_record(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new experiment directory: {output}")
    pro = ROOT / "third_party/libero_pro"
    source = ROOT / "third_party/openpi/third_party/libero/libero/libero"
    mapping = ast.literal_eval(ast.parse(
        (pro / "libero/libero/benchmark/libero_suite_task_map.py").read_text()).body[0].value)
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}"
    with urllib.request.urlopen(url + "?recursive=true&limit=1000", timeout=60) as response:
        entries = json.load(response)
    metadata = {item["path"]: item for item in entries if item["type"] == "file"}
    assets = ROOT / "datasets/benchmark_assets/libero_pro"
    plan = {
        "version": 1, "purpose": "Two-task PRO position adaptation pilot; not full benchmark",
        "pro_source": str(pro),
        "pro_commit": subprocess.check_output(["git", "-C", str(pro), "rev-parse", "HEAD"],
                                                text=True).strip(),
        "dataset": REPOSITORY, "dataset_revision": REVISION,
        "split_seed": 20261025, "evaluation_seed": 100000,
        "split_scope": "Disjoint initial states from the same published PRO position condition. "
                       "This is not a held-out perturbation-range experiment.",
        "training": "Independent full action-expert RLDT runs, frozen VLM, seed 0, 100 iterations",
        "tasks": {}, "downloads": [],
    }
    for suite, horizon in (("libero_spatial", 220), ("libero_object", 280)):
        name = mapping[suite + "_swap"][0]
        downloaded = {}
        for folder, extension in (("bddl_files", ".bddl"), ("init_files", ".pruned_init")):
            relative = f"{folder}/{suite}_swap/{name}{extension}"
            target = assets / relative
            if target.exists():
                blob = target.read_bytes()
            else:
                with urllib.request.urlopen(
                    f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{relative}",
                    timeout=60,
                ) as response:
                    blob = response.read()
            meta = metadata[relative]
            oid = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
            if len(blob) != meta["size"] or oid != meta["oid"]:
                raise ValueError(f"Asset does not match pinned Hugging Face metadata: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            downloaded[folder] = target
            plan["downloads"].append(file_record(target) | {"git_blob_sha1": oid})
        states = load_states(downloaded["init_files"])
        if len(states) != 50 or len(np.unique(states, axis=0)) != 50:
            raise ValueError("Expected 50 distinct published PRO states")
        ids = np.random.default_rng(plan["split_seed"]).permutation(50).tolist()
        source_states = source / "init_files" / suite / (name + ".pruned_init")
        if len(load_states(source_states)) < 20:
            raise ValueError("Insufficient original-task retention states")
        key = suite.removeprefix("libero_") + "_0"
        plan["tasks"][key] = {
            "suite": suite, "task_id": 0, "name": name, "max_steps": horizon,
            "pro_state_count": len(states), "train_state_ids": sorted(ids[:30]),
            "heldout_state_ids": sorted(ids[30:]), "source_state_ids": list(range(20)),
            "pro": {"bddl": file_record(downloaded["bddl_files"]),
                    "states": file_record(downloaded["init_files"])},
            "source": {"bddl": file_record(source / "bddl_files" / suite / (name + ".bddl")),
                       "states": file_record(source_states)},
        }
    config_root = ROOT / ".cache/libero-pro"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(pro / "libero/libero"),
        "bddl_files": str(assets / "bddl_files"), "init_states": str(assets / "init_files"),
        "datasets": str(ROOT / "datasets"), "assets": str(pro / "libero/libero/assets"),
    }))
    output.mkdir(parents=True)
    protocol = output / "protocol.json"
    protocol.write_text(json.dumps(plan, indent=2) + "\n")
    tail = "[train]" + (ROOT / "configs/rldt_pi05_libero.toml").read_text().split("[train]", 1)[1]
    for key in plan["tasks"]:
        task_dir = output / key
        task_dir.mkdir()
        for split in ("train", "heldout", "source"):
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
task_key = "{key}"
split = "{split}"

'''
            (task_dir / f"{split}.toml").write_text(config + tail)
    print(protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args().output)
