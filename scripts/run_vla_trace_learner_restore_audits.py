"""Queue one CPU restore-audit child at a time; retain no native model while waiting."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

TASKS = ("object_0", "object_1", "object_2")
ITERATIONS = (25, 50, 100)
SCOPE = "Native learner serialization restore only; no optimizer/update, policy rollout, replay restoration or environment resume."


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def summary(root):
    directory = root / "learner_restore_audit"
    rows = []
    failed = []
    for task in TASKS:
        for iteration in ITERATIONS:
            path = directory / task / f"iteration_{iteration:06d}.json"
            row = {"task": task, "iteration": iteration, "path": str(path), "passed": None}
            if path.exists():
                value = json.loads(path.read_text())
                if (value["task"], value["iteration"]) != (task, iteration):
                    raise ValueError("Restore artifact identity mismatch")
                row.update(passed=value.get("passed") is True and value.get("complete") is True,
                           sha256=sha(path), checkpoint=value["checkpoint"],
                           expected_checkpoint_files=value["files"])
                if value.get("passed") is False:
                    failed.append({"task": task, "iteration": iteration, "path": str(path), "sha256": sha(path)})
                elif not value.get("complete"):
                    row["passed"] = None
            error = path.with_name(path.stem + ".failed.json")
            if error.exists():
                failed.append({"task": task, "iteration": iteration, "path": str(error), "sha256": sha(error)})
                row["passed"] = False
            rows.append(row)
    complete = all(row["passed"] is True for row in rows) and not failed
    result = {"complete": complete, "passed": True if complete else False if failed else None,
              "scope": SCOPE, "required_checkpoints": 9,
              "completed_checkpoints": sum(row["passed"] is True for row in rows),
              "checkpoints": rows, "failures": failed, "native_models_retained_while_waiting": False,
              "sources_sha256": sha(directory / "sources.json") if (directory / "sources.json").exists() else None}
    write(directory / "summary.json", result)
    return result


def run(root, python, minimum_headroom_gib):
    script = Path(__file__).resolve()
    directory = root / "learner_restore_audit"
    pinned = {"runner_sha256": sha(script), "native_auditor_sha256": sha(script.with_name("audit_vla_trace_learner_restore.py")),
              "python": str(python), "minimum_headroom_gib": minimum_headroom_gib,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "scope": SCOPE}
    plan = directory / "queue_plan.json"
    if plan.exists():
        if json.loads(plan.read_text()) != pinned:
            raise ValueError("Immutable restore queue plan changed")
    else:
        write(plan, pinned)
    while True:
        current = summary(root)
        if current["failures"]:
            raise RuntimeError("An immutable native restore audit reports failure")
        if current["complete"]:
            return
        launched = False
        for row in current["checkpoints"]:
            if row["passed"] is not None:
                continue
            validation = root / "methods/rldt" / row["task"] / f"validation_{row['iteration']:06d}.json"
            if not validation.exists():
                continue
            limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            if limit != "max" and int(limit) - used < minimum_headroom_gib * 1024**3:
                continue
            if sha(script) != pinned["runner_sha256"] or sha(script.with_name("audit_vla_trace_learner_restore.py")) != pinned["native_auditor_sha256"]:
                raise ValueError("Restore queue sources changed")
            command = [str(python), str(script.with_name("audit_vla_trace_learner_restore.py")),
                       "--run", str(root), "--task", row["task"], "--iteration", str(row["iteration"])]
            path = Path(row["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps({"launch": command}), flush=True)
            with path.with_suffix(".log").open("x") as stream:
                child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
            summary(root)
            if child.returncode:
                raise RuntimeError(f"Restore child failed; inspect {path.with_suffix('.log')}")
            launched = True
            break
        if not launched:
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--minimum-headroom-gib", type=int, default=64)
    args = parser.parse_args()
    # Preserve the venv executable symlink: resolving it selects system Python.
    run(args.run.resolve(), args.python.absolute(), args.minimum_headroom_gib)
