"""One controlled upstream pi-RL experiment with held-out paired evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from experiment_policy import enforce
from run_pro_task_grasp import (
    ROOT,
    cached_validation,
    compare,
    gpu_lock,
    record_stage_time,
    status,
    write_json,
)

from rl_vla.checkpoint_validation import scheduled_iterations

sys.path.insert(0, str(ROOT / "integrations"))
from rlinf_bridge.configure import configuration

ITERATIONS = (25, 50, 75, 100)
GATES = ("installation_complete", "parity_passed", "flow_sde_smoke_passed", "flow_noise_smoke_passed")


def require_readiness(path):
    gates = json.loads(path.read_text())
    missing = [name for name in GATES if gates.get(name) is not True]
    if missing:
        raise RuntimeError(f"pi-RL readiness checks have not passed: {missing}")
    return gates


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(task, output, method, gates):
    readiness = require_readiness(gates)
    if output.exists():
        raise FileExistsError(output)
    if json.loads((task / "setup_check.json").read_text())["states_checked"] != 250:
        raise ValueError("Expected all 250 reset checks")
    plan = json.loads((task / "protocol.json").read_text())
    if len(plan["tasks"]) != 1:
        raise ValueError("Each experiment must contain exactly one task")
    key, definition = next(iter(plan["tasks"].items()))
    enforce(method, definition["suite"])
    output.mkdir(parents=True)
    training = output / "training"
    training.mkdir()
    cfg = configuration(task, training, method)
    (training / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    simulator_python = ROOT / ".venv-pro/bin/python"
    runtime_command = [str(simulator_python), "-c", (
        "import importlib.metadata as m,json,sys; "
        "print(json.dumps({'python':sys.version,'executable':sys.executable,"
        "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
    )]
    runtime = json.loads(subprocess.check_output(runtime_command, text=True))
    write_json(output / "simulator_runtime.json", runtime)
    paths = list((ROOT / "integrations/pi_rl").rglob("*.py"))
    paths += list((ROOT / "integrations/rlinf_bridge").rglob("*.py"))
    paths += list((ROOT / "third_party/rlinf/rlinf").rglob("*.py"))
    paths += list((ROOT / "third_party/rlinf_openpi/src/openpi").rglob("*.py"))
    paths += list((ROOT / "third_party/libero_pro/libero").rglob("*.py"))
    paths += list((ROOT / "third_party/openpi/packages/openpi-client/src").rglob("*.py"))
    simulator_packages = ROOT / ".venv-pro/lib/python3.11/site-packages"
    paths += list(simulator_packages.glob("*.dist-info/METADATA"))
    paths += list(simulator_packages.glob("*.pth"))
    installed = ROOT / ".venv-rlinf/lib/python3.11/site-packages"
    paths += list((installed / "openpi").rglob("*.py"))
    for module in ("gemma", "paligemma", "siglip"):
        paths += list((installed / "transformers/models" / module).rglob("*.py"))
    paths += [ROOT / p for p in (
        "scripts/run_pi_rl.sh", "scripts/run_pi_rl_task_grasp.py", "scripts/env.sh",
        ".cache/libero-pro/config.yaml",
        "scripts/run_pro_task_grasp.py", "scripts/analyze_rldt.py",
        "src/rl_vla/rldt_env.py", "src/rl_vla/trajectory_recording.py",
        "src/rl_vla/checkpoint_validation.py", "scripts/experiment_policy.py",
        "configs/experiment_policy.json", "scripts/prepare_object_campaign_grasp.py",
        "src/rl_vla/envs/libero.py", "src/rl_vla/envs/libero_pro.py",
        "third_party/rlinf/examples/embodiment/train_embodied_agent.py",
        "checkpoints/openpi/pi05_libero_rlinf/conversion_manifest.json",
        "checkpoints/openpi/pi05_libero_rlinf/physical-intelligence/libero/norm_stats.json",
    )]
    paths += [task / p for p in ("protocol.json", "pro_states.npy", "generation.json",
                                 "assets.json", "setup_check.json")]
    paths += [training / "config.yaml", gates, output / "simulator_runtime.json"]
    write_json(output / "input_contract.json", {str(p.resolve()): sha256(p) for p in paths})
    write_json(output / "experiment.json", {
        "method": method, "framework": "official RLinf v0.3 / pi-RL",
        "task_key": key, "task_directory": str(task), "protocol": plan,
        "training_seed": 0, "evaluation_seed": 100000,
        "iterations": list(scheduled_iterations(plan)), "decisions_per_iteration": 32,
        "maximum_executed_prefix": 5, "training_decision_budget": 3200,
        "interaction_accounting": "Actual controlled steps logged; evaluations and settling excluded",
        "selection": "Highest validation success count; ties earliest iteration",
        "baseline": "Converted original pi05_libero, upstream Torch ODE, 10 denoising steps",
        "rng_note": "Paired Torch baseline and trained policies share per-episode seeds; Torch/JAX seeds differ numerically",
        "recording": "All train/evaluation trajectories; every evaluation controller frame",
        "readiness": readiness,
    })
    subprocess.run(["tar", "--exclude=__pycache__", "--exclude=*.pyc", "-czf",
                    str(output / "source_snapshot.tar.gz"), "integrations/pi_rl", "integrations/rlinf_bridge",
                    "third_party/rlinf/rlinf", "third_party/rlinf_openpi/src/openpi",
                    "scripts/run_pi_rl_task_grasp.py", "scripts/run_pi_rl.sh"], cwd=ROOT, check=True)
    (output / "rlinf_changes.patch").write_bytes(subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/rlinf"), "diff", "--binary"]))
    return key, definition, cfg


def verify(output):
    for path, expected in json.loads((output / "input_contract.json").read_text()).items():
        if sha256(Path(path)) != expected:
            raise RuntimeError(f"Frozen pi-RL input changed: {path}")


def checkpoint_path(output, cfg, iteration):
    return (output / "training" / cfg["runner"]["logger"]["experiment_name"]
            / "checkpoints" / f"global_step_{iteration}")


def run_stage(task, output, key, definition, stage, split=None, checkpoint=None):
    cached = cached_validation(output, stage, split, checkpoint, definition)
    if cached is not None:
        verify(output)
        return cached
    config = output / "training/config.yaml"
    if stage == "train":
        command = ["bash", str(ROOT / "scripts/run_pi_rl.sh"),
                   "integrations/rlinf_bridge/train.py", "--config-path", str(config.parent),
                   "--config-name", "config"]
        destination = output / "training/summary.json"
    else:
        destination = output / f"{stage}.json"
        command = ["bash", str(ROOT / "scripts/run_pi_rl.sh"),
                   "integrations/rlinf_bridge/evaluate.py", "--config", str(config),
                   "--protocol", str(task / "protocol.json"), "--task-key", key,
                   "--split", split, "--output", str(destination), "--episodes", "50",
                   "--seed", "100000"]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
    status(output, stage, "waiting_for_gpu", command=command)
    with gpu_lock():
        verify(output)
        status(output, stage, "running", command=command)
        started = time.monotonic()
        with (output / f"{stage}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads(destination.read_text())
    record_stage_time(output, stage, started, report)
    if split:
        identities = [episode["reset"]["initial_state_id"] for episode in report["episodes"]]
        if identities != definition[f"{split}_state_ids"]:
            raise ValueError(f"{stage} did not enumerate the frozen split")
    return report


def flatten_numeric(value, prefix=""):
    """Aggregate the one-worker (or general multiworker) upstream metric list."""
    values = {}
    if isinstance(value, dict):
        for key, item in value.items():
            for name, numbers in flatten_numeric(item, f"{prefix}/{key}".strip("/")).items():
                values.setdefault(name, []).extend(numbers)
    elif isinstance(value, list):
        for item in value:
            for name, numbers in flatten_numeric(item, prefix).items():
                values.setdefault(name, []).extend(numbers)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        values[prefix] = [float(value)]
    return values


def training_analysis(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(line) for line in (output / "training/iteration_metrics.jsonl").read_text().splitlines()]
    if [int(row["iteration"]) for row in rows] != list(range(1, 101)):
        raise ValueError("Expected one logged row for every iteration 1..100")
    flattened = [{key: float(np.mean(values)) for key, values in
                  flatten_numeric(row.get("actor_training_metrics", {})).items()}
                 for row in rows]
    bad = [{"iteration": row["iteration"], "metric": key, "value": str(value)}
           for row, metrics in zip(rows, flattened, strict=True)
           for key, value in metrics.items() if not math.isfinite(value)]
    write_json(output / "analysis.json", {
        "iterations": len(rows), "training_summary": json.loads((output / "training/summary.json").read_text()),
        "nonfinite_numeric_metrics": bad,
        "metric_keys": sorted({key for metrics in flattened for key in metrics}),
        "metric_note": "Official upstream per-iteration aggregate diagnostics; undefined values remain null",
    })
    desired = ["actor/policy_loss", "critic/value_loss", "actor/approx_kl",
               "actor/clip_fraction", "critic/explained_variance", "actor/grad_norm"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ax, key in zip(axes.flat, desired, strict=True):
        matching = next((name for name in sorted({k for d in flattened for k in d}) if name.endswith(key)), None)
        if matching:
            ax.plot([row["primitive_steps"] for row in rows],
                    [metrics.get(matching, np.nan) for metrics in flattened])
        else:
            ax.text(.5, .5, "Not reported", ha="center", transform=ax.transAxes)
        ax.set(title=key, xlabel="Controlled training steps")
        ax.grid(alpha=.2)
    fig.savefig(output / "actor_critic_diagnostics.png", dpi=150)
    plt.close(fig)
    return rows


def validation_plot(output, baseline, reports, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    interactions = {int(row["iteration"]): row["primitive_steps"] for row in metrics}
    points = [{"iteration": 0, "primitive_steps": 0, "success_rate": baseline["success_rate"]}]
    points += [{"iteration": i, "primitive_steps": interactions[i], "success_rate": report["success_rate"]}
               for i, report in sorted(reports.items())]
    write_json(output / "validation_progress.json", {"points": points})
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot([p["primitive_steps"] for p in points], [100 * p["success_rate"] for p in points], "o-")
    ax.axhline(100 * baseline["success_rate"], color="gray", ls="--", label="Converted frozen Torch baseline")
    for point in points[1:]:
        ax.annotate(str(point["iteration"]), (point["primitive_steps"], 100 * point["success_rate"]),
                    xytext=(0, -14), textcoords="offset points", ha="center")
    ax.set(xlabel="Controlled training environment steps", ylabel="Validation success (%)",
           ylim=(-2, 105), title=f"{output.parent.name}/{output.name}: 50 states, one training seed")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(output / "validation_interactions.png", dpi=150)
    plt.close(fig)


def run(task, output, method, gates):
    key, definition, cfg = freeze(task, output, method, gates)
    experiment_path = output / "experiment.json"
    iterations = scheduled_iterations(json.loads(experiment_path.read_text())) if experiment_path.exists() else ITERATIONS
    try:
        baseline = run_stage(task, output, key, definition, "baseline_validation", "validation")
        summary = run_stage(task, output, key, definition, "train")
        if (summary["iterations"], summary["decisions"]) != (100, 3200):
            raise ValueError(f"Unexpected training budget: {summary}")
        metrics = training_analysis(output)
        reports, comparisons = {}, {}
        for iteration in iterations:
            checkpoint = checkpoint_path(output, cfg, iteration)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            reports[iteration] = run_stage(task, output, key, definition,
                f"validation_{iteration:06d}", "validation", checkpoint)
            comparisons[iteration] = compare(baseline, reports[iteration],
                "Validation states excluded from training; used to select checkpoint")
            write_json(output / "validation_comparisons.json", comparisons)
            validation_plot(output, baseline, reports, metrics)
        selected = min(reports, key=lambda i: (-reports[i]["success_rate"], i))
        checkpoint = checkpoint_path(output, cfg, selected)
        write_json(output / "selection.json", {"iteration": selected, "checkpoint": str(checkpoint),
                   "rule": "Highest validation success count; ties earliest iteration"})
        final = {"task_key": key, "method": method, "selected_iteration": selected}
        for split in ("test", "source"):
            before = run_stage(task, output, key, definition, f"baseline_{split}", split)
            after = run_stage(task, output, key, definition, f"selected_{split}", split, checkpoint)
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
    parser.add_argument("--method", choices=["flow_sde", "flow_noise"], required=True)
    parser.add_argument("--gates", type=Path, required=True)
    args = parser.parse_args()
    run(args.task.resolve(), args.output.resolve(), args.method, args.gates.resolve())
