"""Run frozen evaluation, independent RLDT adaptation, and matched evaluation."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(experiment, task, phase):
    experiment = experiment.resolve()
    contract = json.loads((experiment / "launch_contract.json").read_text())
    for relative, expected in contract["files"].items():
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source/config changed after launch contract was frozen: {path}")
    plan = json.loads((experiment / "protocol.json").read_text())
    task_plan = plan["tasks"][task]
    out = experiment / task
    status = out / f"{phase}_status.json"
    seed = plan["evaluation_seed"]
    stages = []
    if phase == "baseline":
        stages = [("baseline_heldout", "heldout", None), ("baseline_source", "source", None)]
    elif phase == "train":
        for split in ("heldout", "source"):
            baseline = json.loads((out / f"baseline_{split}.json").read_text())
            if len(baseline["episodes"]) != len(task_plan[f"{split}_state_ids"]):
                raise RuntimeError("Baseline has not completed its fixed evaluation budget")
        stages = [("train", "train", None),
                  ("trained_heldout", "heldout", out / "training/final"),
                  ("trained_source", "source", out / "training/final")]
    else:
        raise ValueError(phase)
    for name, split, checkpoint in stages:
        command = ["bash", str(ROOT / "scripts/run_rldt.sh"),
                   "train" if name == "train" else "evaluate",
                   "--config", str(out / f"{split}.toml"), "--output",
                   str(out / "training" if name == "train" else out / f"{name}.json")]
        if name != "train":
            command += ["--episodes", str(len(task_plan[f"{split}_state_ids"])),
                        "--seed", str(seed)]
        if checkpoint is not None:
            command += ["--checkpoint", str(checkpoint)]
        record = {"task": task, "phase": phase, "stage": name, "state": "running",
                  "job_id": os.environ.get("SLURM_JOB_ID"), "started_unix": time.time(),
                  "command": command}
        write_json(status, record)
        print(f"Starting {task}: {name}", flush=True)
        with (out / f"{name}.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=ROOT, check=False)
        write_json(status, record | {"state": "completed" if result.returncode == 0 else "failed",
                                     "exit_code": result.returncode, "ended_unix": time.time()})
        if result.returncode:
            raise RuntimeError(f"{name} failed: see {out / (name + '.log')}")
    if phase == "train":
        comparison = {"task": task, "protocol": str(experiment / "protocol.json")}
        for split in ("heldout", "source"):
            before = json.loads((out / f"baseline_{split}.json").read_text())
            after = json.loads((out / f"trained_{split}.json").read_text())
            identities = lambda report: [(r["seed"], r["reset"]["initial_state_id"])
                                          for r in report["episodes"]]
            if identities(before) != identities(after):
                raise RuntimeError("Before/after evaluation identities differ")
            comparison[split] = {"episodes": len(before["episodes"]),
                                 "baseline_success_rate": before["success_rate"],
                                 "rldt_success_rate": after["success_rate"],
                                 "change": after["success_rate"] - before["success_rate"]}
        write_json(out / "comparison.json", comparison)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--task", choices=("spatial_0", "object_0"), required=True)
    parser.add_argument("--phase", choices=("baseline", "train"), required=True)
    args = parser.parse_args()
    run(args.experiment, args.task, args.phase)
