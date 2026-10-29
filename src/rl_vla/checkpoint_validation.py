"""Synchronous held-out checks without restarting the continuous training state."""

import json
import os
import subprocess
import time
from pathlib import Path


def scheduled_iterations(plan):
    protocol = plan.get("protocol", plan)
    values = protocol.get("validation_iterations", plan.get("iterations", [25, 50, 75, 100]))
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Validation iterations must be a nonempty sequence")
    if any(type(i) is not int or i < 1 or i > 100 for i in values) or list(values) != sorted(set(values)):
        raise ValueError("Validation iterations must be unique, increasing integers in 1..100")
    return tuple(values)


def run_checkpoint_validation(training, checkpoint, iteration, counters, method):
    training, checkpoint = Path(training).resolve(), Path(checkpoint).resolve()
    output = training.parent
    experiment_path = output / "experiment.json"
    if not experiment_path.exists():
        return 0.0
    experiment = json.loads(experiment_path.read_text())
    if not experiment.get("protocol", {}).get("online_validation", False):
        return 0.0
    if method != experiment["method"] or method not in ("rldt", "flow_sde"):
        raise ValueError("Checkpoint validation method disagrees with experiment")
    if iteration not in scheduled_iterations(experiment):
        return 0.0  # Saving a recovery checkpoint does not require evaluating it.
    root = Path(__file__).resolve().parents[2]
    task = Path(experiment["task_directory"])
    destination = output / f"validation_{iteration:06d}.json"
    if destination.exists():
        raise FileExistsError(destination)
    if method == "rldt":
        command = ["bash", str(root / "scripts/run_rldt.sh"), "evaluate", "--config",
                   str(task / "validation.toml"), "--output", str(destination),
                   "--episodes", "50", "--seed", "100000", "--record-episodes", "50"]
    else:
        command = ["bash", str(root / "scripts/run_pi_rl.sh"),
                   "integrations/rlinf_bridge/evaluate.py", "--config", str(training / "config.yaml"),
                   "--protocol", str(task / "protocol.json"), "--task-key", experiment["task_key"],
                   "--split", "validation", "--output", str(destination),
                   "--episodes", "50", "--seed", "100000"]
    command += ["--checkpoint", str(checkpoint)]
    env = os.environ.copy()
    # The learner is paused and retains its optimizer/replay/RNG. The evaluator
    # uses a separate simulator and a bounded JAX pool on the same 96 GB GPU.
    env.update(XLA_PYTHON_CLIENT_MEM_FRACTION="0.20", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    started = time.monotonic()
    progress_path = output / "checkpoint_validation_status.json"
    progress_path.write_text(json.dumps({"state": "running", "iteration": iteration,
                                         "report": str(destination)}, indent=2) + "\n")
    print(f"Pausing training for iteration {iteration} validation on 50 held-out states", flush=True)
    try:
        with destination.with_suffix(".log").open("x") as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    except BaseException as exc:
        progress_path.write_text(json.dumps({"state": "failed", "iteration": iteration,
                                             "error": repr(exc)}, indent=2) + "\n")
        raise
    report = json.loads(destination.read_text())
    definition = experiment["protocol"]["tasks"][experiment["task_key"]]
    if [e["reset"]["initial_state_id"] for e in report["episodes"]] != definition["validation_state_ids"]:
        raise ValueError("Checkpoint validation did not enumerate the frozen split")
    elapsed = time.monotonic() - started
    row = {"iteration": iteration, "checkpoint": str(checkpoint), "method": method,
           "success_rate": report["success_rate"], "successes": sum(e["success"] for e in report["episodes"]),
           "episodes": len(report["episodes"]), "validation_elapsed_seconds": elapsed,
           "training_counters": counters, "report": str(destination)}
    with (output / "checkpoint_validations.jsonl").open("a") as log:
        log.write(json.dumps(row, allow_nan=False) + "\n")
    progress_path.write_text(json.dumps(row | {"state": "completed"}, indent=2, allow_nan=False) + "\n")
    print(f"Iteration {iteration} validation: {row['successes']}/50; resuming continuous training", flush=True)
    return elapsed
