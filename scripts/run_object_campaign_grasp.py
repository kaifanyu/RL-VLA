"""Persistent Object2/Object1 queue: frozen baselines, RLDT and Flow-SDE only."""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from analyze_rldt import load_records
from experiment_policy import enforce, require_active_queue
from run_pi_rl_task_grasp import require_readiness
from run_pro_task_grasp import ROOT, status, write_json

from rl_vla.checkpoint_validation import scheduled_iterations

EXPERIMENTS = [(task, method) for task in ("object_2", "object_1_hard")
               for method in ("rldt", "flow_sde")]


def snapshot(suite):
    rows = []
    iterations = scheduled_iterations(json.loads((suite / "plan.json").read_text()))
    for key, method in EXPERIMENTS:
        directory = suite / "methods" / method / key
        checks = directory / "checkpoint_validations.jsonl"
        online = {r["iteration"]: r for r in load_records(checks)} if checks.exists() else {}
        for iteration in (0, *iterations):
            path = directory / ("baseline_validation.json" if iteration == 0 else f"validation_{iteration:06d}.json")
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            episodes = report["episodes"]
            counters = online.get(iteration, {}).get("training_counters", {})
            rows.append({"task": key, "method": method, "iteration": iteration,
                         "successes": sum(e["success"] for e in episodes), "episodes": len(episodes),
                         "success_rate": report["success_rate"],
                         "training_controller_steps": 0 if iteration == 0 else counters.get("primitive_steps"),
                         "training_decisions": 0 if iteration == 0 else counters.get("decisions"),
                         "training_seconds": 0 if iteration == 0 else counters.get("elapsed_seconds"),
                         "checkpoint_evaluation_seconds": online.get(iteration, {}).get("validation_elapsed_seconds"),
                         "mean_evaluation_controller_steps": sum(e["primitive_steps"] for e in episodes) / len(episodes),
                         "report": str(path)})
    write_json(suite / "validation_results.json", rows)
    if rows:
        temp = suite / "validation_results.csv.tmp"
        with temp.open("w") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(suite / "validation_results.csv")


def finish_existing_baseline(suite):
    """Hand over the live baseline without discarding or rerunning its episodes."""
    handover = json.loads((suite / "schedule_handover.json").read_text())
    old_step = handover["old_step"]
    if old_step != "1470624.97" or os.environ.get("SLURM_STEP_ID") == "97":
        raise RuntimeError("Unexpected baseline handover step")
    baseline = next(p for p in handover["processes"] if p["kind"] == "baseline")
    output = suite / "methods/rldt/object_2"
    report = output / "baseline_validation.json"
    status(suite / "queue", "rldt_object_2", "waiting_for_existing_baseline", experiment_directory=str(output))
    deadline = time.monotonic() + 7200
    while True:
        process = Path(f"/proc/{baseline['pid']}/cmdline")
        try:
            command = process.read_bytes().replace(b"\x00", b" ").decode()
        except FileNotFoundError:
            command = ""
        alive = str(output) in command and "rl_vla.rldt" in command
        if not alive:
            if not report.exists():
                raise RuntimeError("The preserved baseline exited without a completed report")
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Preserved baseline did not complete")
        time.sleep(15)
    # Only obsolete controllers remain in the old step. Keep the allocation.
    subprocess.run(["scancel", "--signal=KILL", old_step], check=True)
    old_status = json.loads((suite / "amendments/remove_validation_75/before" /
                             output.relative_to(ROOT) / "status.json").read_text())
    wall = report.stat().st_mtime - datetime.fromisoformat(old_status["updated"]).timestamp()
    timing = output / "stage_timings.jsonl"
    if timing.exists():
        raise RuntimeError("Unexpected preexisting baseline timing during handover")
    timing.write_text(json.dumps({"stage": "baseline_validation", "wall_seconds": wall,
                      "checkpoint_validation_seconds": 0, "seconds_excluding_checkpoint_validation": wall,
                      "provenance": "Preserved baseline; completion timestamp minus original stage start"}) + "\n")
    handover.update(state="completed", replacement_step=os.environ.get("SLURM_STEP_ID"))
    write_json(suite / "schedule_handover.json", handover)


def run(suite, gates, resume_after_baseline=False):
    if os.environ.get("SLURM_JOB_ID") != "1470624":
        raise RuntimeError("Use the retained GPU allocation 1470624")
    require_readiness(gates)
    queue = suite / "queue"
    require_active_queue(queue)
    queue.mkdir(exist_ok=resume_after_baseline)
    if resume_after_baseline:
        progress = json.loads((queue / "progress.json").read_text())
        if progress["completed"] or progress["failures"]:
            raise RuntimeError("Baseline handover requires an otherwise unstarted queue")
    for _, method in EXPERIMENTS:
        enforce(method, "libero_object")
    write_json(queue / "queue.json", {"experiments": EXPERIMENTS, "gates": str(gates),
               "iterations": 100, "validation_iterations": list(scheduled_iterations(json.loads((suite / "plan.json").read_text()))),
               "baseline": "One frozen validation baseline per task and runtime, plus paired final test/retention baselines",
               "online_validation": True})
    completed, failures = [], []
    write_json(queue / "progress.json", {"completed": [], "failures": [], "remaining": len(EXPERIMENTS)})
    child = None

    def terminate(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    try:
        if resume_after_baseline:
            finish_existing_baseline(suite)
        for key, method in EXPERIMENTS:
            require_active_queue(queue)
            enforce(method, "libero_object")
            task, output = suite / "tasks" / key, suite / "methods" / method / key
            status(queue, f"{method}_{key}", "waiting_for_preparation", experiment_directory=str(output))
            deadline = time.monotonic() + 7200
            while not (task / "setup_check.json").exists():
                require_active_queue(queue)
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Preparation did not finish: {key}")
                time.sleep(15)
            script = "run_pro_task_grasp.py" if method == "rldt" else "run_pi_rl_task_grasp.py"
            command = [sys.executable, str(ROOT / "scripts" / script), "--task", str(task), "--output", str(output)]
            if method == "flow_sde":
                command += ["--method", method, "--gates", str(gates)]
            resuming = resume_after_baseline and key == "object_2" and method == "rldt"
            if resuming:
                command += ["--resume"]
            status(queue, f"{method}_{key}", "running", experiment_directory=str(output))
            with (queue / f"{method}_{key}.log").open("a" if resuming else "x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    snapshot(suite)
                    time.sleep(30)
                result = {"task": key, "method": method, "exit_code": child.returncode,
                          "experiment_directory": str(output)}
            (failures if child.returncode else completed).append(result)
            child = None
            write_json(queue / "progress.json", {"completed": completed, "failures": failures,
                       "remaining": len(EXPERIMENTS) - len(completed) - len(failures)})
            snapshot(suite)
        status(queue, "complete", "completed" if not failures else "completed_with_failures", failures=failures)
    except BaseException as exc:
        status(queue, "interrupted", "cancelled" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
               error=repr(exc), completed=completed, failures=failures)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--resume-after-baseline", action="store_true")
    args = parser.parse_args()
    run(args.suite.resolve(), args.gates.resolve(), resume_after_baseline=args.resume_after_baseline)
