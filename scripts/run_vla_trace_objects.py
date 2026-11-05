"""Run one official Object swap campaign on an explicitly assigned local GPU.

Uses the unchanged native learner, separate-process online validation, and the
existing evaluation sampler. Never resumes a partially executed learner.
"""

import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import run_pro_task_grasp as legacy

ROOT = Path(__file__).resolve().parents[1]
ITERATIONS = (25, 50, 100)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


@contextmanager
def local_gpu_lock():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("Assign exactly one GPU with CUDA_VISIBLE_DEVICES")
    identifier = subprocess.check_output(
        ["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if not identifier.startswith("GPU-"):
        raise RuntimeError("Cannot identify explicitly assigned GPU")
    directory = ROOT / "runs/.vla_trace_gpu_locks"
    directory.mkdir(exist_ok=True)
    with (directory / f"{identifier}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def freeze(task, output, gate_b):
    plan = json.loads((task / "protocol.json").read_text())
    if plan.get("validation_iterations") != list(ITERATIONS) or not plan.get("online_validation"):
        raise ValueError("Require online validation only at 25, 50, 100")
    preparation = json.loads((task / "preparation_complete.json").read_text())
    if not preparation.get("passed"):
        raise ValueError("All split, reset, and placement preparation audits must pass")
    preparation_records = list(preparation["files"])
    preparation_records += [preparation[name] for name in ("generator", "auditor") if name in preparation]
    for condition in preparation["source_and_task_assets"].values():
        preparation_records.extend(condition.values())
    for record in preparation_records:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Preparation artifact changed after readiness: {record['path']}")
    gate = json.loads(gate_b.read_text())
    if not gate.get("passed", False):
        raise ValueError("Native attention fidelity gate has not passed")
    if gate.get("technical_only", True):
        raise ValueError("Require attention fidelity on a real saved task observation")
    smoke_path = output.parents[2] / "smoke" / task.name / "verification.json"
    smoke = json.loads(smoke_path.read_text())
    if not smoke.get("passed") or not smoke.get("frozen_backbone_identical") or not smoke.get("expert_1_changed"):
        raise ValueError("Task-specific real-model smoke gate has not passed")
    key, definition = legacy.freeze(task, output)
    # Analysis is performed concurrently and is not imported by native training.
    # Freeze all actual runtime modules, plus this launcher and the gate record.
    contract = json.loads((output / "input_contract.json").read_text())
    excluded = [p for p in contract if "/src/rl_vla/diagnostics/" in p]
    contract = {p: v for p, v in contract.items() if p not in excluded}
    contract.update({str(Path(record["path"]).resolve()): record["sha256"]
                     for record in preparation_records})
    for p in (Path(__file__).resolve(), gate_b.resolve(), smoke_path.resolve(), task / "preparation_complete.json",
              ROOT / "scripts/evaluate_vla_trace_sharded.py"):
        contract[str(p)] = sha256(p)
    legacy.write_json(output / "input_contract.json", contract)
    legacy.write_json(output / "contract_scope.json", {
        "excluded_nonruntime_diagnostics": excluded,
        "reason": "Diagnostics are not imported by the unchanged native learner or evaluator; analysis tooling can be developed while training runs.",
    })
    experiment = json.loads((output / "experiment.json").read_text())
    experiment.update({
        "run_uuid": str(uuid.uuid4()), "started": now(), "host": socket.gethostname(),
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "cpuset": os.environ.get("RLDT_CPUSET"),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "start_checkpoint": str(ROOT / "checkpoints/openpi/pi05_libero"),
        "training_init_from": None, "gate_b": str(gate_b),
        "smoke": str(smoke_path), "validation_gpu": os.environ.get("RL_VLA_VALIDATION_GPU"),
        "object_telemetry": "Native read-only object positions and both-fingerpad contact at every controlled step; video review required for failure labels",
        "evaluation_shards": int(os.environ.get("RL_VLA_EVAL_SHARDS", "1")),
        "evaluation_cpuset": os.environ.get("RL_VLA_EVALUATION_CPUSET"),
        "selection_candidates": list(ITERATIONS),
        "exploratory_test_rule": "Evaluate each other requested checkpoint once on the same test/source states only after selection.json is written; never reselect.",
    })
    legacy.write_json(output / "experiment.json", experiment)
    return key, definition


def audit_training(output):
    training = output / "training"
    manifest = json.loads((training / "manifest.json").read_text())
    if manifest["init_from"] is not None or not manifest["frozen_vlm"]:
        raise ValueError("Training did not start from untouched frozen-backbone baseline")
    rows = [json.loads(line) for line in (training / "metrics.jsonl").read_text().splitlines()]
    if [row["iterations"] for row in rows] != list(range(1, 101)):
        raise ValueError("Missing/duplicate/discontinuous training iterations")
    for row in rows:
        i = row["iterations"]
        if (row["decisions"], row["critic_updates"], row["actor_updates"]) != (32*i, 64*i, 4*max(i-2, 0)):
            raise ValueError(f"Unexpected optimizer/interaction counts at {i}")
        if not 0 < row["primitive_steps"] <= row["decisions"] * 5:
            raise ValueError("Invalid primitive training-step counter")
    summary = json.loads((training / "summary.json").read_text())
    if abs(summary["wall_elapsed_seconds"] - summary["elapsed_seconds"] - summary["checkpoint_validation_seconds"]) > .1:
        raise ValueError("Validation time counted as training time")
    checkpoints = {}
    for i in ITERATIONS:
        directory = training / f"iteration_{i:06d}"
        metadata = json.loads((directory / "rldt_pi05.json").read_text())
        for field in ("checkpoint_metadata_sha256", "checkpoint_manifest_sha256", "normalization_assets_sha256"):
            if metadata[field] != manifest["policy_metadata"][field]:
                raise ValueError("Checkpoint frozen base identity changed")
        checkpoints[str(i)] = {
            "files": {name: {"bytes": (directory/name).stat().st_size, "sha256": sha256(directory/name)}
                      for name in ("expert.npz", "learner.npz", "rldt_pi05.json")},
            "counters": rows[i-1],
            "reload_evidence": str(output / f"validation_{i:06d}.json"),
        }
    legacy.write_json(output / "checkpoint_audit.json", {
        "passed": True, "run_uuid": json.loads((output/"experiment.json").read_text())["run_uuid"],
        "continuous_iteration_sequence": True, "init_from": None,
        "training_process": json.loads((output/"training_process.json").read_text()),
        "checkpoints": checkpoints,
        "validation_time_excluded": True,
    })


def run_training(task, output, definition):
    destination = output / "training"
    command = ["bash", str(ROOT/"scripts/run_rldt.sh"), "train", "--config", str(task/"train.toml"), "--output", str(destination)]
    legacy.status(output, "train", "waiting_for_gpu", command=command)
    with local_gpu_lock():
        legacy.verify(output)
        legacy.status(output, "train", "running", command=command)
        started = time.monotonic()
        with (output/"train.log").open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record = {"pid": process.pid, "started": now(), "command": command,
                      "one_continuous_process": True, "init_from": None}
            legacy.write_json(output/"training_process.json", record)
            code = process.wait()
            record.update(finished=now(), exit_code=code)
            legacy.write_json(output/"training_process.json", record)
            if code:
                raise subprocess.CalledProcessError(code, command)
    report = json.loads((destination/"summary.json").read_text())
    legacy.record_stage_time(output, "train", started, report)
    return report


def run_evaluation(task, output, definition, stage, split, checkpoint=None):
    cached = legacy.cached_validation(output, stage, split, checkpoint, definition)
    if cached is not None:
        legacy.verify(output)
        return cached
    destination = output / f"{stage}.json"
    command = [str(ROOT/".venv/bin/python"), str(ROOT/"scripts/evaluate_vla_trace_sharded.py"),
               "--config", str(task/f"{split}.toml"), "--output", str(destination),
               "--episodes", "50", "--seed", "100000", "--record-episodes", "50"]
    if checkpoint is not None:
        command += ["--checkpoint", str(checkpoint)]
    legacy.status(output, stage, "waiting_for_gpu", command=command)
    with local_gpu_lock():
        legacy.verify(output)
        legacy.status(output, stage, "running", command=command)
        started = time.monotonic()
        with (output/f"{stage}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads(destination.read_text())
    if [e["reset"]["initial_state_id"] for e in report["episodes"]] != definition[f"{split}_state_ids"]:
        raise ValueError("Sharded evaluation changed frozen state order")
    if [e["seed"] for e in report["episodes"]] != list(range(100000, 100050)):
        raise ValueError("Sharded evaluation changed per-episode noise seeds")
    legacy.record_stage_time(output, stage, started, report)
    return report


def run(task, output, gate_b):
    os.environ["RL_VLA_OBJECT_TELEMETRY"] = "1"
    legacy.gpu_lock = local_gpu_lock
    key, definition = freeze(task, output, gate_b)
    try:
        baseline = run_evaluation(task, output, definition, "baseline_validation", "validation")
        run_training(task, output, definition)
        reports = {i: run_evaluation(task, output, definition, f"validation_{i:06d}", "validation",
                    output/"training"/f"iteration_{i:06d}") for i in ITERATIONS}
        selected = min(reports, key=lambda i: (-sum(e["success"] for e in reports[i]["episodes"]), i))
        checkpoint = output/"training"/f"iteration_{selected:06d}"
        legacy.write_json(output/"selection.json", {
            "iteration": selected, "checkpoint": str(checkpoint), "locked": now(),
            "rule": "Highest validation success count; ties earliest iteration",
            "validation_counts": {str(i): sum(e["success"] for e in report["episodes"]) for i, report in reports.items()},
            "test_results_seen": False,
        })
        audit_training(output)
        legacy.validation_plot(output, baseline, reports)
        legacy.analyze(output)
        comparison = {"task_key": key, "selected_iteration": selected}
        for split in ("test", "source"):
            before = run_evaluation(task, output, definition, f"baseline_{split}", split)
            selected_report = run_evaluation(task, output, definition, f"selected_{split}", split, checkpoint)
            comparison[split] = legacy.compare(before, selected_report,
                "Locked selection; paired reused published PRO test" if split == "test" else "Original-layout retention")
            legacy.write_json(output/"comparison.json", comparison)
            for i in ITERATIONS:
                if i == selected:
                    continue
                run_evaluation(task, output, definition, f"checkpoint_{i:06d}_{split}", split,
                                 output/"training"/f"iteration_{i:06d}")
        legacy.status(output, "action_evaluations_complete", "completed", selected_iteration=selected,
                      note="Five diagnostic tests and scientific report have independent completion gates.")
    except BaseException as exc:
        previous = json.loads((output/"status.json").read_text()) if (output/"status.json").exists() else {}
        legacy.status(output, previous.get("stage", "startup"), "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-b", type=Path, required=True)
    args = parser.parse_args()
    run(args.task.resolve(), args.output.resolve(), args.gate_b.resolve())
