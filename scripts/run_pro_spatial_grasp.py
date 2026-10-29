"""One continuous RLDT run; select on validation before opening final test results."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from analyze_rldt import analyze, compare_evaluations

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_contract(experiment):
    contract = json.loads((experiment / "launch_contract.json").read_text())
    for relative, expected in contract["files"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source/config changed after launch: {relative}")
    for directory, expected in contract["repositories"].items():
        actual = subprocess.check_output(["git", "-C", str(ROOT / directory),
                                          "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise RuntimeError(f"Dependency source changed: {directory}")


def select_checkpoint(reports):
    """Only RLDT validation outcomes enter the predeclared selection rule."""
    if not reports:
        raise ValueError("No validation reports")
    for iteration, report in reports.items():
        if iteration <= 0 or len(report["episodes"]) != 50:
            raise ValueError("Each checkpoint needs exactly 50 validation episodes")
        if any(row["reset"]["experiment_split"] != "validation" for row in report["episodes"]):
            raise ValueError("Checkpoint selection requires validation data")
    best = min(reports, key=lambda i: (-sum(r["success"] for r in reports[i]["episodes"]), i))
    return best


def status(experiment, stage, state, **extra):
    write_json(experiment / "status.json", {
        "stage": stage, "state": state, "job_id": os.environ.get("SLURM_JOB_ID"),
        "step_id": os.environ.get("SLURM_STEP_ID"), "host": socket.gethostname(),
        "updated": datetime.now(ZoneInfo("America/New_York")).isoformat(), **extra})


def run_stage(experiment, stage, split, *, checkpoint=None, train=False):
    verify_contract(experiment)
    output = experiment / ("training" if train else f"{stage}.json")
    command = ["bash", str(ROOT / "scripts/run_rldt.sh"), "train" if train else "evaluate",
               "--config", str(experiment / f"{split}.toml"), "--output", str(output)]
    if not train:
        command += ["--episodes", "50", "--seed", "100000", "--record-episodes", "5"]
    if checkpoint is not None:
        command += ["--checkpoint", str(checkpoint)]
    status(experiment, stage, "running", command=command, started_unix=time.time())
    print(f"Starting {stage}", flush=True)
    with (experiment / f"{stage}.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode:
        raise RuntimeError(f"Stage {stage} exited {result.returncode}; see {stage}.log")
    if train:
        return json.loads((output / "summary.json").read_text())
    report = json.loads(output.read_text())
    expected = json.loads((experiment / "protocol.json").read_text())["tasks"]["spatial_0"]
    state_ids = [row["reset"]["initial_state_id"] for row in report["episodes"]]
    if state_ids != expected[f"{split}_state_ids"]:
        raise RuntimeError(f"Evaluation {stage} did not enumerate its exact frozen split")
    return report


def plot_validation(experiment, baseline, reports, selected):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = sorted(reports)
    rates = [100 * reports[i]["success_rate"] for i in iterations]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(iterations, rates, "o-", label="RLDT validation")
    ax.axhline(100 * baseline["success_rate"], color="gray", linestyle="--", label="Frozen baseline")
    ax.scatter([selected], [100 * reports[selected]["success_rate"]], s=150,
               facecolors="none", edgecolors="black", label="Selected checkpoint")
    ax.set(xlabel="Training iteration", ylabel="Success (%)", ylim=(-2, 102),
           title="PRO Spatial task 0: 50 validation states; training seed 0")
    ax.set_xticks(iterations)
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(experiment / "validation.png", dpi=150)
    plt.close(fig)


def run(experiment):
    verify_contract(experiment)
    plan = json.loads((experiment / "protocol.json").read_text())
    try:
        baseline_validation = run_stage(experiment, "baseline_validation", "validation")
        training = run_stage(experiment, "train", "train", train=True)
        if (training["iterations"], training["actor_updates"], training["critic_updates"]) != (100, 392, 6400):
            raise RuntimeError("Training budget mismatch")
        analyze(experiment)
        validation = {}
        for iteration in plan["validation_iterations"]:
            checkpoint = experiment / "training" / f"iteration_{iteration:06d}"
            validation[iteration] = run_stage(
                experiment, f"validation_{iteration:06d}", "validation", checkpoint=checkpoint)
            compare_evaluations(baseline_validation, validation[iteration])
            write_json(experiment / "validation_progress.json", {
                "baseline_success_rate": baseline_validation["success_rate"],
                "checkpoints": {str(i): r["success_rate"] for i, r in validation.items()}})
        selected = select_checkpoint(validation)
        checkpoint = experiment / "training" / f"iteration_{selected:06d}"
        # Selection is persisted before either final test policy is evaluated.
        write_json(experiment / "selection.json", {
            "iteration": selected, "checkpoint": str(checkpoint), "rule": plan["selection"],
            "validation_success_rates": {str(i): r["success_rate"] for i, r in validation.items()},
            "selected_unix": time.time()})
        plot_validation(experiment, baseline_validation, validation, selected)
        comparisons = {}
        for split in ("test", "source"):
            before = run_stage(experiment, f"baseline_{split}", split)
            after = run_stage(experiment, f"selected_{split}", split, checkpoint=checkpoint)
            comparison = compare_evaluations(before, after)
            comparison["scope"] = ("All 50 published PRO Position states, excluded from training "
                "and checkpoint selection." if split == "test" else
                "All 50 original Spatial task 0 states; retention evaluation.")
            comparison["interval_caveat"] = "One rollout per distinct reset state, one training seed; " \
                "uncertainty across training seeds is not measured."
            changes = np.array([int(b["success"]) - int(a["success"])
                                for a, b in zip(before["episodes"], after["episodes"], strict=True)])
            draws = np.random.default_rng(20261026).choice(changes, size=(20000, len(changes)))
            comparison["paired_bootstrap_95_change_percentage_points"] = (
                np.quantile(100 * draws.mean(axis=1), [.025, .975]).tolist())
            comparisons[split] = comparison
            write_json(experiment / "comparison.json", {
                "selected_iteration": selected, "scope": plan["scope"], **comparisons})
        status(experiment, "complete", "completed", exit_code=0, selected_iteration=selected)
    except BaseException as exc:
        previous = json.loads((experiment / "status.json").read_text()) if (
            experiment / "status.json").exists() else {}
        status(experiment, previous.get("stage", "startup"), "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()
    run(args.experiment.resolve())
