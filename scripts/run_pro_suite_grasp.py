"""Sequential RLDT queue; releases the shared GPU lock between stages."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiment_policy import require_active_queue
from run_pro_task_grasp import ROOT, status


def run(suite):
    queue = suite / "rldt_queue"
    require_active_queue(queue)
    queue.mkdir(exist_ok=False)
    tasks = ["spatial_1", "spatial_2", "object_0"]
    failures = []
    for key in tasks:
        task = suite / "tasks" / key
        output = suite / "methods/rldt" / key
        status(queue, key, "waiting_for_preparation", experiment_directory=str(output))
        deadline = time.monotonic() + 3600
        while not (task / "setup_check.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"Preparation did not finish: {key}")
            time.sleep(10)
        status(queue, key, "running", experiment_directory=str(output))
        with (queue / f"{key}.log").open("x") as log:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/run_pro_task_grasp.py"),
                                     "--task", str(task), "--output", str(output)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            failures.append({"task": key, "exit_code": result.returncode})
        (queue / "progress.json").write_text(json.dumps({
            "finished_through": key, "failures": failures}, indent=2) + "\n")
    status(queue, "complete", "completed" if not failures else "completed_with_failures", failures=failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    run(parser.parse_args().suite.resolve())
