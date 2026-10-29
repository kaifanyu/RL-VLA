"""Queue both official pi-RL PPO methods on three fixed PRO tasks."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from experiment_policy import require_active_queue
from run_pi_rl_task_grasp import require_readiness
from run_pro_task_grasp import ROOT, status, write_json


def comparison_matrix(suite):
    from compare_pro_methods import write_comparison

    return write_comparison(suite)


def run(suite, gates):
    require_readiness(gates)
    queue = suite / "pi_rl_queue"
    require_active_queue(queue)
    queue.mkdir(exist_ok=False)
    # Run the condition with the largest observed RLDT gain first so the first
    # completed paired comparison is informative; all six runs are unchanged.
    experiments = [(task, method) for task in ("object_0", "spatial_1", "spatial_2")
                   for method in ("flow_sde", "flow_noise")]
    completed, failures = [], []
    write_json(queue / "queue.json", {"experiments": experiments, "gates": str(gates)})
    for key, method in experiments:
        task, output = suite / "tasks" / key, suite / "methods" / method / key
        label = f"{method}_{key}"
        status(queue, label, "waiting_for_preparation", experiment_directory=str(output))
        deadline = time.monotonic() + 3600
        while not (task / "setup_check.json").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Preparation did not finish: {key}")
            time.sleep(10)
        status(queue, label, "running", experiment_directory=str(output))
        with (queue / f"{label}.log").open("x") as log:
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/run_pi_rl_task_grasp.py"),
                "--task", str(task), "--output", str(output), "--method", method,
                "--gates", str(gates),
            ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        record = {"task": key, "method": method, "exit_code": result.returncode,
                  "experiment_directory": str(output)}
        (completed if result.returncode == 0 else failures).append(record)
        write_json(queue / "progress.json", {"completed": completed, "failures": failures,
                   "remaining": len(experiments) - len(completed) - len(failures)})
        comparison_matrix(suite)
    status(queue, "complete", "completed" if not failures else "completed_with_failures",
           completed=completed, failures=failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.suite.resolve(), args.gates.resolve())
    except BaseException as exc:
        queue = args.suite.resolve() / "pi_rl_queue"
        if queue.exists():
            status(queue, "queue", "failed", error=repr(exc))
        raise
