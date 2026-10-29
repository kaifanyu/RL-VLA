"""Validate all frozen reset states on independent CPU simulator processes."""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from rl_vla.envs.libero import LiberoEnv
from rl_vla.envs.libero_pro import make_env


def check_chunk(protocol, split, state_ids, output):
    env = make_env(protocol=protocol, task_key="spatial_0", split=split)
    checks = []
    try:
        for state_id in state_ids:
            obs, info = LiberoEnv.reset(env, seed=100000 + state_id,
                                       options={"initial_state_id": state_id})
            if not np.isfinite(obs["state"]).all():
                raise ValueError("Nonfinite state")
            for camera in ("observation/image", "observation/wrist_image"):
                image = obs["openpi"][camera]
                if image.shape != (224, 224, 3) or image.dtype != np.uint8 or image.std() < 1:
                    raise ValueError(f"Invalid camera observation: {camera}")
            checks.append({"split": split, "state_id": state_id, "reset": info})
        if state_ids:
            iio.imwrite(Path(output) / f"{split}_{state_ids[-1]:03d}.png",
                        obs["openpi"]["observation/image"])
    finally:
        env.close()
    return checks


def check(experiment, workers):
    protocol = experiment / "protocol.json"
    task = json.loads(protocol.read_text())["tasks"]["spatial_0"]
    output = experiment / "reset_checks"
    output.mkdir(exist_ok=True)
    futures, checks = [], []
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for split in ("train", "validation", "test", "source"):
            ids = task[f"{split}_state_ids"]
            for start in range(0, len(ids), 25):
                futures.append(pool.submit(check_chunk, protocol, split, ids[start:start + 25], output))
        for future in as_completed(futures):
            checks.extend(future.result())
            temporary = output / "progress.tmp"
            temporary.write_text(json.dumps({"checked": len(checks), "total": 250}) + "\n")
            temporary.replace(output / "progress.json")
            print(f"Validated {len(checks)}/250 states", flush=True)
    for split in ("train", "validation", "test", "source"):
        actual = sorted(row["state_id"] for row in checks if row["split"] == split)
        if actual != task[f"{split}_state_ids"]:
            raise ValueError(f"Wrong validation coverage for {split}")
    checks.sort(key=lambda row: (row["split"], row["state_id"]))
    report = {"states_checked": len(checks), "checks": checks,
              "method": "Same LiberoEnv.reset/settle path; independent simulator processes; "
                        "finite state and two nonblank 224x224 camera checks for every reset."}
    temporary = experiment / "setup_check.tmp.json"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(experiment / "setup_check.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    check(args.experiment.resolve(), args.workers)
