"""Run one frozen PRO task with native RLDT and paired validation/test evaluations."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from analyze_rldt import analyze, compare_evaluations
from experiment_policy import enforce

from rl_vla.checkpoint_validation import scheduled_iterations

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def status(output, stage, state, **extra):
    write_json(output / "status.json", {
        "stage": stage, "state": state,
        "updated": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "step_id": os.environ.get("SLURM_STEP_ID"), **extra})


@contextmanager
def gpu_lock():
    allocation = os.environ.get("SLURM_JOB_ID")
    if not allocation:
        raise RuntimeError("This launcher must run in the assigned Slurm allocation")
    path = ROOT / "runs" / f"gpu_allocation_{allocation}" / "experiment.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def freeze(task, output):
    if output.exists():
        raise FileExistsError(f"Experiment output already exists: {output}")
    if json.loads((task / "setup_check.json").read_text())["states_checked"] != 250:
        raise ValueError("Expected all 250 reset checks")
    plan = json.loads((task / "protocol.json").read_text())
    if len(plan["tasks"]) != 1:
        raise ValueError("Each run must contain exactly one task")
    key, definition = next(iter(plan["tasks"].items()))
    enforce("rldt", definition["suite"])
    output.mkdir(parents=True)
    paths = list((ROOT / "src").rglob("*.py"))
    paths += list((ROOT / "third_party/openpi/src/openpi").rglob("*.py"))
    paths += list((ROOT / "third_party/libero_pro/libero").rglob("*.py"))
    paths += [ROOT / p for p in ["scripts/env.sh", "scripts/run_rldt.sh",
              "scripts/run_pro_task_grasp.py", "scripts/analyze_rldt.py",
              "scripts/prepare_pro_suite_grasp.py", "scripts/prepare_pro_spatial_grasp.py",
              "scripts/experiment_policy.py", "configs/experiment_policy.json",
              "scripts/prepare_object_campaign_grasp.py",
              ".cache/libero-pro/config.yaml", "runs/setup/pi05_download.json",
              "checkpoints/openpi/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"]]
    paths += [task / p for p in ["protocol.json", "pro_states.npy", "generation.json",
                                 "assets.json", "setup_check.json"]]
    paths += list(task.glob("*.toml"))
    contract = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(output / "input_contract.json", contract)
    write_json(output / "experiment.json", {
        "task_key": key, "task_directory": str(task), "protocol": plan,
        "method": "rldt", "training_seed": 0,
        "evaluation_seed": 100000, "iterations": list(scheduled_iterations(plan)),
        "collection_decisions_per_iteration": 32, "maximum_executed_prefix": 5,
        "interaction_accounting": "Log actual controlled steps; exclude evaluation and reset settling",
        "selection": "Highest validation success count; ties earliest iteration",
        "recording": "All training/evaluation trajectories; all evaluation episode videos"})
    subprocess.run(["tar", "--exclude=__pycache__", "--exclude=*.pyc", "-czf",
                    str(output / "source_snapshot.tar.gz"), "src", "scripts", "configs", "tests"],
                   cwd=ROOT, check=True)
    (output / "source_changes.patch").write_bytes(subprocess.check_output(
        ["git", "diff", "--binary"], cwd=ROOT))
    return key, definition


def verify(output):
    for path, expected in json.loads((output / "input_contract.json").read_text()).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen experiment input changed: {path}")


def cached_validation(output, stage, split, checkpoint, definition):
    path = output / f"{stage}.json"
    if split != "validation" or checkpoint is None or not path.exists():
        return None
    report = json.loads(path.read_text())
    if report["checkpoint"] != str(checkpoint):
        raise ValueError("Cached validation checkpoint differs")
    if [e["reset"]["initial_state_id"] for e in report["episodes"]] != definition["validation_state_ids"]:
        raise ValueError("Cached validation split differs")
    if [e["seed"] for e in report["episodes"]] != list(range(100000, 100050)):
        raise ValueError("Cached validation seeds differ")
    return report


def record_stage_time(output, stage, started, report):
    wall = time.monotonic() - started
    validation = report.get("checkpoint_validation_seconds", 0.0)
    with (output / "stage_timings.jsonl").open("a") as log:
        log.write(json.dumps({"stage": stage, "wall_seconds": wall,
                              "checkpoint_validation_seconds": validation,
                              "seconds_excluding_checkpoint_validation": wall - validation}) + "\n")


def run_stage(task, output, definition, stage, split, checkpoint=None, train=False):
    cached = cached_validation(output, stage, split, checkpoint, definition)
    if cached is not None:
        verify(output)
        return cached
    destination = output / ("training" if train else f"{stage}.json")
    command = ["bash", str(ROOT / "scripts/run_rldt.sh"), "train" if train else "evaluate",
               "--config", str(task / f"{split}.toml"), "--output", str(destination)]
    if not train:
        command += ["--episodes", "50", "--seed", "100000", "--record-episodes", "50"]
    if checkpoint:
        command += ["--checkpoint", str(checkpoint)]
    status(output, stage, "waiting_for_gpu", command=command)
    with gpu_lock():
        verify(output)
        status(output, stage, "running", command=command)
        started = time.monotonic()
        with (output / f"{stage}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((destination / "summary.json" if train else destination).read_text())
    record_stage_time(output, stage, started, report)
    if not train:
        identities = [e["reset"]["initial_state_id"] for e in report["episodes"]]
        if identities != definition[f"{split}_state_ids"]:
            raise ValueError(f"{stage} did not enumerate the frozen split")
    return report


def compare(before, after, scope):
    result = compare_evaluations(before, after)
    result["scope"] = scope
    result["interval_caveat"] = "One rollout per distinct state and one training seed"
    changes = np.array([int(b["success"]) - int(a["success"])
                        for a, b in zip(before["episodes"], after["episodes"], strict=True)])
    draws = np.random.default_rng(20261027).choice(changes, size=(20000, len(changes)))
    result["paired_bootstrap_95_change_percentage_points"] = (
        np.quantile(draws.mean(axis=1) * 100, [.025, .975]).tolist())
    result["changed_episodes"] = [
        {"episode": i, "baseline": a, "trained": b}
        for i, (a, b) in enumerate(zip(before["episodes"], after["episodes"], strict=True), 1)
        if a["success"] != b["success"]]
    return result


def validation_plot(output, baseline, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [json.loads(line) for line in (output / "training/metrics.jsonl").read_text().splitlines()]
    points = [{"iteration": 0, "primitive_steps": 0, "success_rate": baseline["success_rate"]}]
    points += [{"iteration": i, "primitive_steps": metrics[i - 1]["primitive_steps"],
                "success_rate": r["success_rate"]} for i, r in sorted(reports.items())]
    write_json(output / "validation_progress.json", {"points": points})
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot([p["primitive_steps"] for p in points], [100 * p["success_rate"] for p in points], "o-")
    ax.axhline(100 * baseline["success_rate"], color="gray", ls="--", label="Frozen baseline")
    for p in points[1:]:
        ax.annotate(str(p["iteration"]), (p["primitive_steps"], 100 * p["success_rate"]),
                    xytext=(0, -14), textcoords="offset points", ha="center")
    ax.set(xlabel="Controlled training environment steps", ylabel="Validation success (%)",
           ylim=(-2, 105), title=f"{output.name}: 50 validation states; one training seed")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(output / "validation_interactions.png", dpi=150)
    plt.close(fig)


def resume_before_training(task, output):
    """Reuse a completed frozen baseline; never restart an existing learner."""
    if (output / "training").exists():
        raise RuntimeError("Cannot use baseline-only resume after training has started")
    verify(output)
    experiment = json.loads((output / "experiment.json").read_text())
    if experiment["method"] != "rldt" or Path(experiment["task_directory"]).resolve() != task.resolve():
        raise ValueError("Resume task/method differs from frozen experiment")
    key = experiment["task_key"]
    definition = experiment["protocol"]["tasks"][key]
    enforce("rldt", definition["suite"])
    baseline = json.loads((output / "baseline_validation.json").read_text())
    if baseline["checkpoint"] is not None:
        raise ValueError("Resume requires an untrained baseline")
    if [e["reset"]["initial_state_id"] for e in baseline["episodes"]] != definition["validation_state_ids"]:
        raise ValueError("Resume baseline states differ")
    if [e["seed"] for e in baseline["episodes"]] != list(range(100000, 100050)):
        raise ValueError("Resume baseline seeds differ")
    return key, definition, baseline


def run(task, output, resume=False):
    if resume:
        key, definition, baseline = resume_before_training(task, output)
    else:
        key, definition = freeze(task, output)
        baseline = None
    experiment_path = output / "experiment.json"
    iterations = scheduled_iterations(json.loads(experiment_path.read_text())) if experiment_path.exists() else (25, 50, 75, 100)
    try:
        if baseline is None:
            baseline = run_stage(task, output, definition, "baseline_validation", "validation")
        summary = run_stage(task, output, definition, "train", "train", train=True)
        if (summary["iterations"], summary["decisions"]) != (100, 3200):
            raise ValueError("Unexpected training budget")
        analyze(output)
        reports, comparisons = {}, {}
        for iteration in iterations:
            reports[iteration] = run_stage(task, output, definition,
                f"validation_{iteration:06d}", "validation",
                checkpoint=output / "training" / f"iteration_{iteration:06d}")
            comparisons[iteration] = compare(baseline, reports[iteration],
                "Validation initial states excluded from training; used to select checkpoint")
            write_json(output / "validation_comparisons.json", comparisons)
            validation_plot(output, baseline, reports)
        selected = min(reports, key=lambda i: (-reports[i]["success_rate"], i))
        checkpoint = output / "training" / f"iteration_{selected:06d}"
        write_json(output / "selection.json", {"iteration": selected, "checkpoint": str(checkpoint),
                   "rule": "Highest validation success count; ties earliest iteration"})
        final = {"task_key": key, "selected_iteration": selected}
        for split in ["test", "source"]:
            before = run_stage(task, output, definition, f"baseline_{split}", split)
            after = run_stage(task, output, definition, f"selected_{split}", split, checkpoint)
            final[split] = compare(before, after,
                definition.get("test_scope", "50 published PRO states excluded from training and selection") if split == "test"
                else "50 original-task states used only for retention evaluation")
            write_json(output / "comparison.json", final)
        status(output, "complete", "completed", selected_iteration=selected)
    except BaseException as exc:
        previous = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
        status(output, previous.get("stage", "startup"), "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse a completed baseline before training has started")
    args = parser.parse_args()
    run(args.task.resolve(), args.output.resolve(), resume=args.resume)
