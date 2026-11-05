"""Execute one immutable next-campaign specification in a fresh attempt directory.

The native learner is unchanged. A narrowly scoped subclass implements E04's
declared actor schedule without modifying the active campaign's source files.
"""

import argparse
import csv
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_iteration(reports):
    if 0 not in reports:
        raise ValueError("Initialization must be a selection candidate")
    return min(reports, key=lambda i: (-sum(bool(e["success"]) for e in reports[i]["episodes"]), i))


def quarter_schedule_parameters(config):
    """The first update in iteration 26 is actor update index 92 at defaults."""
    warmup_iterations = int(config["rldt"].get("critic_warmup_iterations", 2))
    return max(0, 25 - warmup_iterations) * int(config["train"]["actor_steps"])


def install_actor_schedule(config):
    if config.get("campaign", {}).get("actor_schedule") != "quarter_after_iteration_25":
        return None
    import jax.numpy as jnp
    import optax

    from rl_vla import rldt

    original = rldt.Learner
    boundary = quarter_schedule_parameters(config)

    class QuarterAfter25Learner(original):
        def __init__(self, flow, feature_dim, learner_config):
            super().__init__(flow, feature_dim, learner_config)
            warmup = int(self.r.get("actor_warmup_steps", 32))
            lr = float(self.r["actor_lr"])
            native = optax.linear_schedule(lr / max(1, warmup), lr, max(1, warmup))
            schedule = lambda count: native(count) * jnp.where(count >= boundary, .25, 1.)
            self.actor_tx = optax.chain(
                optax.clip_by_global_norm(float(self.r.get("max_grad_norm", 1))),
                optax.adam(schedule),
            )
            self.actor_opt = self.actor_tx.init(self.params)

    rldt.Learner = QuarterAfter25Learner
    return {"actor_schedule": "quarter_after_iteration_25", "first_reduced_update_index": boundary,
            "actor_lr_before": config["rldt"]["actor_lr"],
            "actor_lr_after": config["rldt"]["actor_lr"] / 4,
            "warmup_preserved": True, "optimizer_state_reset_during_training": False}


def native_entry(mode, config_path, output):
    from rl_vla import rldt

    config = rldt.load_config(config_path)
    schedule = install_actor_schedule(config)
    if mode == "train":
        result = rldt.train(config, str(output))
        if schedule:
            write_json(output / "actor_schedule.json", schedule)
        return result
    sys.argv = ["smoke_vla_trace_object.py", "--config", str(config_path), "--output", str(output)]
    runpy.run_path(str(Path(__file__).with_name("smoke_vla_trace_object.py")), run_name="__main__")
    if schedule:
        write_json(output / "actor_schedule.json", schedule)


def child_environment(root):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cache = root / ".cache"
    native = cache / "native/usr/lib/x86_64-linux-gnu"
    env["LD_LIBRARY_PATH"] = str(native) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.setdefault("OPENPI_DATA_HOME", str(cache / "openpi"))
    env.setdefault("LIBERO_CONFIG_PATH", str(cache / "libero"))
    env.setdefault("HF_HOME", str(cache / "huggingface"))
    env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", NUMBA_DISABLE_JIT="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", RL_VLA_OBJECT_TELEMETRY="1",
               RL_VLA_EVAL_SHARDS="1")
    env.setdefault("LP_NUM_THREADS", "1")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")
    # Every job owns one GPU. Paused training and its one evaluator fit in 64 GiB.
    env.pop("RL_VLA_VALIDATION_GPU", None)
    return env


def validate_report(report, spec, split, checkpoint):
    plan = json.loads((Path(spec["task_directory"]) / "protocol.json").read_text())
    expected = plan["tasks"][spec["task"]][f"{split}_state_ids"]
    episodes = report["episodes"]
    if len(episodes) != 50 or [e["reset"]["initial_state_id"] for e in episodes] != expected:
        raise ValueError("Evaluation is incomplete or changes the locked reset order")
    if [e["seed"] for e in episodes] != list(range(100000, 100050)):
        raise ValueError("Evaluation inference seeds differ")
    if report["checkpoint"] != (str(checkpoint) if checkpoint is not None else None):
        raise ValueError("Evaluation checkpoint differs")
    if abs(report["success_rate"] - sum(bool(e["success"]) for e in episodes) / 50) > 1e-12:
        raise ValueError("Evaluation success total differs")
    return report


def reusable_evaluation(spec, checkpoint):
    """Reuse only a complete paired evaluation with equivalent native inputs/code."""
    failures = []
    for candidate in spec.get("reuse_candidates", []):
        path = Path(candidate["path"])
        try:
            if "selection_path" in candidate:
                selection = json.loads(Path(candidate["selection_path"]).read_text())
                if selection["iteration"] != candidate["selected_iteration"]:
                    raise ValueError("Selected policy is a different iteration")
            report = json.loads(path.read_text())
            validate_report(report, spec, spec["split"], checkpoint)
            if report.get("technical_only", False):
                raise ValueError("Technical evaluations cannot be reused as production")
            intended = tomllib.loads((Path(spec["task_directory"]) / f"{spec['split']}.toml").read_text())
            actual = report["config"]
            for section in ("policy", "rldt", "recording"):
                if actual[section] != intended[section]:
                    raise ValueError(f"Evaluation {section} differs")
            left, right = dict(actual["env"]), dict(intended["env"])
            left_kwargs, right_kwargs = dict(left.pop("kwargs")), dict(right.pop("kwargs"))
            left_protocol, right_protocol = left_kwargs.pop("protocol"), right_kwargs.pop("protocol")
            if left != right or left_kwargs != right_kwargs:
                raise ValueError("Simulator contract differs")
            source_task = json.loads(Path(left_protocol).read_text())["tasks"][spec["task"]]
            target_task = json.loads(Path(right_protocol).read_text())["tasks"][spec["task"]]
            if source_task != target_task:
                raise ValueError("Simulator assets/reset definitions differ")
            contract = json.loads(Path(spec["reuse_core_contract"]).read_text())
            source_root, runtime_root = Path(spec["reuse_source_root"]), Path(spec["runtime_root"])
            verified = {}
            required = {"src/rl_vla/rldt.py", "src/rl_vla/rldt_pi05.py", "src/rl_vla/rldt_env.py",
                        "src/rl_vla/envs/libero_pro.py", "scripts/run_rldt.sh", "scripts/evaluate_vla_trace_sharded.py"}
            for original, expected in contract.items():
                try:
                    relative = Path(original).relative_to(source_root)
                except ValueError:
                    continue
                key = relative.as_posix()
                if key.startswith(("src/", "third_party/")) or key in required:
                    if digest(runtime_root / relative) != expected:
                        raise ValueError(f"Native source differs from prior frozen contract: {relative}")
                    verified[key] = expected
            if not required <= verified.keys():
                raise ValueError("Prior frozen contract omits required native runtime files")
            for episode in report["episodes"]:
                if not Path(episode["video"]).is_file() or not Path(episode["trajectory"]).is_dir():
                    raise ValueError("Original videos/telemetry are missing")
            return report, {"reused": True, "report": str(path), "report_sha256": digest(path),
                            "frozen_core_files_verified": verified, "new_policy_evaluation_gpu_seconds": 0,
                            "rule": "Exact effective policy, native code, simulator asset/reset, reward/decoder and inference-noise equivalence"}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"candidate": str(path), "reason": repr(exc)})
    return None, {"reused": False, "rejected_candidates": failures}


def reuse_completed_training(spec, output):
    """Continue scoring a completed training run without executing more updates."""
    recovery = spec["completed_training_reuse"]
    previous = Path(recovery["attempt_directory"])
    original = json.loads((previous / "job_spec.json").read_text())
    if original != {k: v for k, v in spec.items() if k != "completed_training_reuse"}:
        raise ValueError("Recovered training specification differs")
    for name, expected in recovery["sha256"].items():
        if digest(previous / name) != expected:
            raise ValueError(f"Recovered training artifact changed: {name}")
    for name, expected in recovery["file_identity"].items():
        stat = (previous / name).stat()
        if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != expected:
            raise ValueError(f"Recovered checkpoint changed: {name}")
    metrics = [json.loads(line) for line in (previous / "training/metrics.jsonl").read_text().splitlines()]
    summary = json.loads((previous / "training/summary.json").read_text())
    if ([row["iterations"] for row in metrics] != list(range(1, 101))
            or summary["iterations"] != 100 or summary["decisions"] != 3200
            or not 0 < summary["primitive_steps"] <= 16000):
        raise ValueError("Recovered training is incomplete")
    for iteration in (25, 50, 75, 100):
        for filename in ("expert.npz", "learner.npz", "rldt_pi05.json"):
            name = f"training/iteration_{iteration:06d}/{filename}"
            if name not in recovery["file_identity"] or (previous / name).stat().st_size == 0:
                raise ValueError("Recovered training checkpoint missing")
    (output / "training").symlink_to(previous / "training", target_is_directory=True)
    write_json(output / "training_reuse.json", {**recovery,
               "verified_at": now(), "new_training_updates": 0,
               "note": "Completed continuous training preserved; evaluations run in a fresh attempt."})


def run(spec_path, output):
    spec = json.loads(spec_path.read_text())
    root = Path(spec["runtime_root"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Attempt directory must be empty: {output}")
    env = child_environment(root)
    device = env.get("CUDA_VISIBLE_DEVICES", "")
    if not device or "," in device:
        raise ValueError("Exactly one assigned GPU is required")
    for path, expected in spec["input_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Immutable input changed: {path}")
    write_json(output / "job_spec.json", spec)
    started = time.monotonic()
    write_json(output / "status.json", {"state": "running", "started_at": now(), "pid": os.getpid()})

    def command(argv, stage):
        write_json(output / "status.json", {"state": "running", "stage": stage, "pid": os.getpid(),
                                            "command_argv": argv, "updated_at": now()})
        before = time.monotonic()
        with (output / f"{stage}.log").open("x") as log:
            subprocess.run(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with (output / "stage_timings.jsonl").open("a") as stream:
            stream.write(json.dumps({"stage": stage, "elapsed_seconds": time.monotonic() - before}) + "\n")

    def native(mode, destination):
        argv = [str(root / "third_party/openpi/.venv/bin/python"), str(Path(__file__).resolve()),
                "--native", mode, "--config", spec["config_path"], "--output", str(destination)]
        if env.get("RLDT_CPUSET"):
            argv = ["taskset", "-c", env["RLDT_CPUSET"], *argv]
        command(argv, mode)

    def evaluate(stage, split, checkpoint=None):
        path = output / f"{stage}.json"
        if not path.exists():
            argv = ["bash", str(root / "scripts/run_rldt.sh"), "evaluate", "--config",
                    str(Path(spec["task_directory"]) / f"{split}.toml"), "--output", str(path),
                    "--episodes", "50", "--seed", "100000", "--record-episodes", "50"]
            if checkpoint is not None:
                argv += ["--checkpoint", str(checkpoint)]
            command(argv, stage)
        return validate_report(json.loads(path.read_text()), spec, split, checkpoint)

    try:
        if spec["mode"] == "smoke":
            native("smoke", output / "smoke")
            verification = json.loads((output / "smoke/verification.json").read_text())
            if not all(verification.get(k) for k in ("passed", "expert_1_changed", "frozen_backbone_identical")):
                raise ValueError("Native trainable-scope smoke failed")
            result = {"smoke_passed": True, "verification": str(output / "smoke/verification.json")}
        elif spec["mode"] == "evaluation":
            checkpoint = Path(spec["checkpoint"]) if spec.get("checkpoint") else None
            checkpoint_hashes = {}
            if checkpoint is not None:
                for filename in ("expert.npz", "learner.npz", "rldt_pi05.json"):
                    if not (checkpoint / filename).is_file():
                        raise FileNotFoundError(checkpoint / filename)
                    checkpoint_hashes[str(checkpoint / filename)] = digest(checkpoint / filename)
                write_json(output / "checkpoint_lineage.json", {"checkpoint": str(checkpoint), "files": checkpoint_hashes,
                           "source_lineage": spec["source_lineage"]})
            report, reuse = reusable_evaluation(spec, checkpoint)
            write_json(output / "reuse_evidence.json", reuse)
            if report is None:
                report = evaluate("evaluation", spec["split"], checkpoint)
            else:
                write_json(output / "evaluation.json", report)
            for path, expected in checkpoint_hashes.items():
                if digest(path) != expected:
                    raise ValueError(f"Checkpoint changed during evaluation: {path}")
            result = {"successes": sum(bool(e["success"]) for e in report["episodes"]), "episodes": 50,
                      "evaluation_reused": reuse["reused"],
                      "new_policy_evaluation_gpu_seconds": 0 if reuse["reused"] else None}
            write_results(output, spec, [(spec["split"], spec.get("checkpoint_iteration", 0), report)])
        else:
            config = tomllib.loads(Path(spec["config_path"]).read_text())
            if (config["train"]["iterations"], config["train"]["collect_decisions"], config["rldt"]["execute_steps"]) != (100, 32, 5):
                raise ValueError("Production interaction budget differs from frozen protocol")
            protocol = json.loads((Path(spec["task_directory"]) / "protocol.json").read_text())
            write_json(output / "experiment.json", {"method": "rldt", "task_directory": spec["task_directory"],
                       "task_key": spec["task"], "protocol": protocol, "initialization": config["policy"],
                       "trainable_scope": spec["trainable_scope"], "selection_candidates": [0, *spec["validation_iterations"]]})
            write_json(output / "initialization.json", {"checkpoint": config["policy"]["checkpoint"],
                       "initialization_hash": spec["initialization_hash"], "adapter": None,
                       "record": "Read-only complete original checkpoint; no smoke-updated weights reused"})
            deferred_baseline = spec.get("baseline_validation_timing") == "after_training_from_immutable_initialization"
            reports = {} if deferred_baseline else {0: evaluate("baseline_validation", "validation")}
            if spec.get("completed_training_reuse"):
                reuse_completed_training(spec, output)
            else:
                native("train", output / "training")
            metrics = [json.loads(line) for line in (output / "training/metrics.jsonl").read_text().splitlines()]
            summary = json.loads((output / "training/summary.json").read_text())
            if [row["iterations"] for row in metrics] != list(range(1, 101)) or summary["iterations"] != 100:
                raise ValueError("Training did not complete exactly 100 continuous iterations")
            if summary["decisions"] != 3200 or not 0 < summary["primitive_steps"] <= 16000:
                raise ValueError("Training interaction accounting differs")
            for iteration in (25, 50, 75, 100):
                for name in ("expert.npz", "learner.npz", "rldt_pi05.json"):
                    if not (output / "training" / f"iteration_{iteration:06d}" / name).is_file():
                        raise ValueError("Missing complete replayable checkpoint")
            if deferred_baseline:
                reports[0] = evaluate("baseline_validation", "validation")
            for iteration in spec["validation_iterations"]:
                checkpoint = output / "training" / f"iteration_{iteration:06d}"
                reports[iteration] = evaluate(f"validation_{iteration:06d}", "validation", checkpoint)
            selected = select_iteration(reports)
            selected_checkpoint = output / "training" / f"iteration_{selected:06d}" if selected else None
            write_json(output / "selection.json", {"iteration": selected, "checkpoint": str(selected_checkpoint) if selected else None,
                       "locked_at": now(), "rule": "Maximum validation success count; ties earliest; initialization eligible",
                       "validation_counts": {str(i): sum(bool(e["success"]) for e in r["episodes"]) for i, r in reports.items()},
                       "test_results_seen": False})
            rows = [("validation", i, report) for i, report in sorted(reports.items())]
            comparisons = {}
            for split in ("test", "source"):
                before = evaluate(f"baseline_{split}", split)
                after = evaluate(f"selected_{split}", split, selected_checkpoint) if selected else before
                if not selected:
                    write_json(output / f"selected_{split}.json", after)
                comparisons[split] = compare(before, after)
                rows.append((split, 0, before))
                if selected:
                    rows.append((split, selected, after))
            write_json(output / "comparison.json", comparisons)
            write_results(output, spec, rows)
            plot_curve(output, reports, metrics)
            result = {"selected_iteration": selected, "training": summary, "comparison": comparisons}
        result.update(job_id=spec["job_id"], state="completed", ended_at=now(),
                      wall_seconds=time.monotonic() - started, gpu=device,
                      renderer="OSMesa CPU rendering; no renderer GPU allocation")
        write_json(output / "completed.json", result)
        write_json(output / "completion.json", {"completed": True, "job_id": spec["job_id"],
                   "ended_at": result["ended_at"], "result": str(output / "completed.json")})
        write_json(output / "status.json", result)
    except BaseException as exc:
        write_json(output / "status.json", {"state": "failed", "error": repr(exc), "ended_at": now(),
                   "partial_artifacts_preserved": True})
        raise


def compare(before, after):
    left, right = before["episodes"], after["episodes"]
    if [(e["seed"], e["reset"]) for e in left] != [(e["seed"], e["reset"]) for e in right]:
        raise ValueError("Final evaluations are not paired")
    common = [(a["primitive_steps"], b["primitive_steps"]) for a, b in zip(left, right, strict=True) if a["success"] and b["success"]]
    successes = lambda rows: [e["primitive_steps"] for e in rows if e["success"]]
    mean = lambda values: sum(values) / len(values) if values else None
    return {"baseline_successes": len(successes(left)), "selected_successes": len(successes(right)), "episodes": len(left),
            "baseline_success_conditional_steps": mean(successes(left)), "selected_success_conditional_steps": mean(successes(right)),
            "paired_common_success_count": len(common), "paired_common_success_baseline_steps": mean([a for a, _ in common]),
            "paired_common_success_selected_steps": mean([b for _, b in common]),
            "all_episode_baseline_steps": mean([e["primitive_steps"] for e in left]),
            "all_episode_selected_steps": mean([e["primitive_steps"] for e in right]),
            "control_hz": 20, "test_scope": "Previously used published resets; fresh confirmation remains separate"}


def write_results(output, spec, rows):
    fields = ["job_id", "family_id", "task", "variant", "training_seed", "split", "iteration", "successes", "episodes", "trainable_scope", "initialization_hash"]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for split, iteration, report in rows:
            row = {k: spec.get(k) for k in fields}
            row.update(split=split, iteration=iteration, successes=sum(bool(e["success"]) for e in report["episodes"]), episodes=len(report["episodes"]))
            writer.writerow(row)


def plot_curve(output, reports, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = sorted(reports)
    successes = [sum(bool(e["success"]) for e in reports[i]["episodes"]) for i in iterations]
    steps = [metrics[i - 1]["primitive_steps"] if i else 0 for i in iterations]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, x, label in zip(axes, (iterations, steps), ("RL iteration", "Controlled training steps"), strict=True):
        axis.plot(x, successes, "o-")
        axis.set(xlabel=label, ylabel="Validation successes / 50", ylim=(-1, 51))
        axis.grid(alpha=.25)
    figure.savefig(output / "learning_curve.png", dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", choices=("train", "smoke"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.native:
        native_entry(args.native, args.config, args.output.resolve())
    elif args.spec:
        run(args.spec.resolve(), args.output.resolve())
    else:
        parser.error("--spec or --native is required")
