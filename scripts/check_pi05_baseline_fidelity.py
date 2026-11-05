"""Exact baseline diagnostic fidelity against independently cached native pi05_libero.

The parent imports no model libraries and releases its one native child before
waiting for future checkpoints. No environment, episode, or outcome is scored.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("object_0", "object_1", "object_2")
EPISODE_INDICES = (0, 1)
DECISION_INDICES = (0, 1)
EXPECTED_CHECKS = {"native_actions_only", "explicit_native_actions", "intervention_noop_actions",
                   "intervention_noop_path", "captured_actions", "captured_path", "layer_noop_actions", "layer_noop_path"}
EXPECTED_CHECKS.update(f"step{index:02d}_{name}" for index in (0, 5, 9)
                       for name in ("capture_velocity", "projection", "layer_velocity", "euler_path"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def utc():
    return datetime.now(UTC).isoformat()


def evidence_sources():
    return [Path(__file__), ROOT / "src/rl_vla/rldt_pi05.py",
            ROOT / "src/rl_vla/diagnostics/pi05_attention.py",
            ROOT / "src/rl_vla/diagnostics/layer_knockout.py",
            ROOT / "src/rl_vla/diagnostics/compiler_profile_baseline.py",
            ROOT / "src/rl_vla/diagnostics/compiler_dispatch.py",
            ROOT / "scripts/check_pi05_native_baseline_oracle.py"]


def check_inputs(args):
    if os.environ.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES") != "":
        raise ValueError("Automatic per-fusion cache must be disabled in this process")
    expected_flag = "--xla_gpu_load_autotune_results_from="+str(args.autotune.resolve())
    if os.environ.get("XLA_FLAGS") != expected_flag:
        raise ValueError("Only the declared immutable startup autotune flag is permitted")
    cache_root = args.run.resolve() / "compiler_caches" / args.autotune.parent.name
    if not Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).resolve().is_relative_to(cache_root):
        raise ValueError("Native child must use this compiler profile's isolated compilation cache")
    if sha(args.autotune) != args.autotune_sha256:
        raise ValueError("Autotune profile bytes changed")


def ready(run, task):
    method = run / "methods/rldt" / task
    return (method / "baseline_validation.json").exists()


def worker(args):
    check_inputs(args)
    from rl_vla.diagnostics.compiler_dispatch import verify_effective_environment

    compiler_audit = verify_effective_environment(args.autotune.parent / "manifest.json", check_versions=True)
    lock_path = ROOT / "runs/.vla_trace_gpu_locks" / f"{args.gpu_uuid}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        import jax
        import jax.numpy as jnp
        import numpy as np

        from rl_vla.diagnostics import pi05_attention as attention
        from rl_vla.diagnostics.layer_knockout import (
            sample_with_layer_knockout,
            velocity_with_layer_knockout,
        )
        from rl_vla.rldt_pi05 import Pi05Flow

        started = time.monotonic()
        sources = {str(path): sha(path) for path in evidence_sources()+[args.autotune, args.autotune.parent / "manifest.json", args.amendment]}
        from check_pi05_native_baseline_oracle import context_hashes
        flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
        base_files = [flow.checkpoint / "params/_METADATA", flow.checkpoint / "params/manifest.ocdbt"]
        base_files += [path for path in sorted((flow.checkpoint / "assets").rglob("*")) if path.is_file()]
        base_inputs = {str(path): sha(path) for path in base_files}
        native_actions_only = jax.jit(lambda p, c, k, f: flow.sample(p, c, k, 10, frozen=f)[0])
        template = read(args.run / "gate_b_actual/metadata.json")

        def comparison(reference, actual):
            reference, actual = np.asarray(reference), np.asarray(actual)
            if reference.shape != actual.shape:
                raise ValueError(f"Fidelity shape mismatch: {reference.shape} vs {actual.shape}")
            return {"exact": bool(np.array_equal(reference, actual)),
                    "max_abs_difference": float(np.max(np.abs(reference.astype(np.float64)-actual.astype(np.float64)))),
                    "different_elements": int(np.count_nonzero(reference != actual)),
                    "finite": bool(np.isfinite(reference).all() and np.isfinite(actual).all())}

        for task in args.worker_jobs.split(","):
            if task not in TASKS or not ready(args.run, task):
                raise ValueError(f"Invalid or unavailable predetermined baseline task: {task}")
            destination = args.output / task
            if (destination / "summary.json").exists():
                raise FileExistsError("Preserve existing checkpoint proof")
            method = args.run / "methods/rldt" / task
            report_path = method / "baseline_validation.json"
            checkpoint_sources = {**base_inputs, str(report_path): sha(report_path)}
            report = read(report_path)
            params = flow.params
            cases = []
            for episode_index in EPISODE_INDICES:
                episode = report["episodes"][episode_index]
                state_id = episode["reset"]["initial_state_id"]
                if state_id != 100+episode_index or episode["reset"]["experiment_split"] != "validation":
                    raise ValueError("Validation input selection differs from predeclared first two states")
                chunks = sorted(Path(episode["trajectory"]).glob("chunk_*.npz"))
                if len(chunks) < len(DECISION_INDICES):
                    raise ValueError("Predetermined recorded decisions are missing")
                key_stream = jax.random.PRNGKey(episode["seed"])
                for decision_index in DECISION_INDICES:
                    chunk = chunks[decision_index]
                    chunk_hash = sha(chunk)
                    with np.load(chunk, allow_pickle=False) as archive:
                        chunk_metadata = json.loads(str(archive["metadata_json"].item()))
                        raw = {parts[1]: archive[field][0].copy()
                               for field, parts in chunk_metadata["observation_schema"].items() if parts[0] == "openpi"}
                        raw["prompt"] = str(raw["prompt"].item())
                        recorded = archive["normalized_actions"][None].copy()
                    if chunk_metadata["decision"]["decision"] != decision_index:
                        raise ValueError("Recorded decision order differs from predeclaration")
                    key_stream, key = jax.random.split(key_stream)
                    if not np.array_equal(key, chunk_metadata["decision"]["sampling_key"]):
                        raise ValueError("Recorded sampling key differs from native episode seed/split protocol")
                    case_dir = destination / f"state_{state_id:06d}" / f"decision_{decision_index:06d}"
                    case_dir.mkdir(parents=True, exist_ok=False)
                    context = flow.prepare([raw])
                    metadata = attention._token_metadata(attention.transformed_observation(flow, raw),
                                                         template["text_span"][0], flow.horizon)
                    metadata["layers"] = template["layers"]
                    noise = jax.random.normal(key, (1, flow.horizon, flow.action_dim))
                    oracle_dir = args.output / "native_oracle" / task / f"state_{state_id:06d}" / f"decision_{decision_index:06d}"
                    oracle_path = oracle_dir / "summary.json"
                    oracle_summary = read(oracle_path)
                    if not oracle_summary or any(oracle_summary.get(name) is not True for name in ("complete", "passed", "cache_hits_verified")):
                        raise ValueError("Independent original-native cached oracle is required")
                    if any(oracle_summary[name] != value for name, value in
                           (("task", task), ("policy", "pi05_libero"), ("state_id", state_id), ("decision_index", decision_index))):
                        raise ValueError("Independent native oracle case identity differs")
                    if oracle_summary["recorded_chunk_sha256"] != chunk_hash or oracle_summary["context_leaf_sha256"] != context_hashes(context):
                        raise ValueError("Profile input/context differs from independent original-native oracle")
                    if oracle_summary["baseline_model_contract"] != flow.metadata:
                        raise ValueError("Profile base model differs from original-native oracle")
                    if sha(oracle_dir / "arrays.npz") != oracle_summary["arrays_sha256"]:
                        raise ValueError("Independent native oracle arrays changed")
                    with np.load(oracle_dir / "arrays.npz", allow_pickle=False) as oracle_arrays:
                        reference = oracle_arrays["actions"].copy()
                        if not np.array_equal(recorded, oracle_arrays["recorded_actions"]) or not np.array_equal(noise, oracle_arrays["initial_noise"]):
                            raise ValueError("Independent oracle recorded actions/noise differ")
                    oracle_contract = {"summary_path": str(oracle_path), "summary_sha256": sha(oracle_path),
                        "arrays_path": str(oracle_dir / "arrays.npz"), "arrays_sha256": sha(oracle_dir / "arrays.npz"),
                        "native_oracle_source_sha256": sha(ROOT / "scripts/check_pi05_native_baseline_oracle.py"),
                        "cache_hits_verified": True}
                    arrays, checks = {"recorded_actions": recorded, "initial_noise": np.asarray(noise)}, {}
                    arrays["independent_native_oracle_actions"] = reference

                    def check(name, actual, expected, arrays=arrays, checks=checks):
                        jax.block_until_ready(actual)
                        arrays[name] = np.asarray(actual)
                        checks[name] = comparison(expected, actual)

                    check("native_actions_only", native_actions_only(params, context, key, flow.frozen), reference)
                    native, path = attention.native_sample(flow, params, context, 0, initial_noise=noise)
                    check("explicit_native_actions", native, reference)
                    arrays["native_path"] = np.asarray(path)
                    intervened, intervention_path = attention.sample_with_intervention(flow, params, context, key, metadata)
                    check("intervention_noop_actions", intervened, reference)
                    check("intervention_noop_path", intervention_path, path)
                    captured, capture_path, probabilities = attention.capture_sample(flow, params, context, 0,
                                                                                     metadata, initial_noise=noise)
                    check("captured_actions", captured, reference)
                    check("captured_path", capture_path, path)
                    np.savez(case_dir / "sampler_attention.npz", attention=probabilities)
                    layer_actions, layer_path = sample_with_layer_knockout(flow, params, context, key, metadata)
                    check("layer_noop_actions", layer_actions, reference)
                    check("layer_noop_path", layer_path, path)
                    for index in (0, 5, 9):
                        x = path[index]
                        t = jnp.full((1,), 1.0-jnp.asarray(index, jnp.float32)/10, dtype=jnp.float32)
                        velocity = attention.native_velocity(flow, params, context, x, t)
                        captured_velocity, attention_at_step, hidden = attention.capture_velocity(
                            flow, params, context, x, t, metadata, return_hidden=True)
                        projected = attention.project_suffix(flow, params, hidden)
                        layer_velocity = velocity_with_layer_knockout(flow, params, context, x, t, metadata)
                        arrays[f"step{index:02d}_native_velocity"] = velocity
                        arrays[f"step{index:02d}_suffix_hidden"] = hidden
                        arrays[f"step{index:02d}_time"] = np.asarray(t)
                        check(f"step{index:02d}_capture_velocity", captured_velocity, velocity)
                        check(f"step{index:02d}_projection", projected, captured_velocity)
                        check(f"step{index:02d}_layer_velocity", layer_velocity, velocity)
                        expected_next = np.asarray(x-jnp.asarray(velocity)/10)
                        check(f"step{index:02d}_euler_path", native if index == 9 else path[index+1], expected_next)
                        np.savez(case_dir / f"step{index:02d}_attention.npz", attention=attention_at_step)
                    np.savez(case_dir / "arrays.npz", **arrays)
                    if sha(chunk) != chunk_hash:
                        raise ValueError("Actual recorded oracle changed during replay")
                    case = {"complete": True, "passed": all(v["exact"] and v["finite"] for v in checks.values()),
                            "task": task, "policy": "pi05_libero", "state_id": state_id,
                            "episode_index": episode_index, "decision_index": decision_index,
                            "recorded_chunk": str(chunk), "recorded_chunk_sha256": chunk_hash,
                            "oracle": oracle_contract,
                            "context_matches_independent_oracle": True, "noise_matches_independent_oracle": True,
                            "historical_recorded_replay": comparison(recorded, reference),
                            "amendment": {"path": str(args.amendment), "sha256": sha(args.amendment)},
                            "sampling_key": list(map(int, key)), "checks": checks,
                            "arrays_sha256": sha(case_dir / "arrays.npz"),
                            "sampler_attention_sha256": sha(case_dir / "sampler_attention.npz")}
                    write(case_dir / "summary.json", case)
                    cases.append({"path": str(case_dir / "summary.json"), "sha256": sha(case_dir / "summary.json"),
                                  "passed": case["passed"]})
                    print(json.dumps({"task": task, "policy": "pi05_libero", "state": state_id, "decision": decision_index,
                                      "passed": case["passed"], "elapsed_seconds": time.monotonic()-started,
                                      "failed_checks": {k: v for k, v in checks.items() if not v["exact"] or not v["finite"]}}), flush=True)
                    if not case["passed"]:
                        raise ValueError("Exact baseline sampler fidelity failed; preserve evidence and stop")
            for path, expected in {**sources, **checkpoint_sources}.items():
                if sha(path) != expected:
                    raise ValueError(f"Frozen source/input changed: {path}")
            write(destination / "summary.json", {"complete": True, "passed": True, "technical_only": True,
                "task": task, "policy": "pi05_libero", "cases": cases, "cases_expected": 4,
                "baseline_model_contract": flow.metadata,
                "source_input_sha256": {**sources, **checkpoint_sources}, "autotune_sha256": args.autotune_sha256,
                "compiler_profile_audit": compiler_audit,
                "compiler_profile_manifest_sha256": sha(args.autotune.parent / "manifest.json"),
                "amendment": {"path": str(args.amendment), "sha256": sha(args.amendment)},
                "native_oracle_source_sha256": sha(ROOT / "scripts/check_pi05_native_baseline_oracle.py"),
                "compiler_environment": {name: os.environ.get(name) for name in
                    ("XLA_FLAGS", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "JAX_COMPILATION_CACHE_DIR",
                     "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION")},
                "native_runtime_changed": False, "tolerance": "Bit exact; no relaxed threshold", "completed_utc": utc()})
        return 0


def aggregate(args):
    results = {}
    amendment = {"path": str(args.amendment), "sha256": sha(args.amendment)}
    for task in TASKS:
        path = args.output / task / "summary.json"
        proof = read(path)
        if proof is not None:
            if proof.get("complete") is not True or proof.get("passed") is not True or len(proof.get("cases", [])) != 4:
                raise ValueError("Baseline proof is incomplete")
            if proof["task"] != task or proof["policy"] != "pi05_libero" or proof["amendment"] != amendment:
                raise ValueError("Baseline proof identity/protocol differs")
            if proof["autotune_sha256"] != args.autotune_sha256:
                raise ValueError("Baseline proof uses a different compiler profile")
            identities = []
            for item in proof["cases"]:
                case_path = Path(item["path"])
                if sha(case_path) != item["sha256"]:
                    raise ValueError("Baseline case summary changed")
                case = read(case_path)
                if case.get("complete") is not True or case.get("passed") is not True or set(case.get("checks", {})) != EXPECTED_CHECKS:
                    raise ValueError("Baseline case lacks all exact comparisons")
                if case["task"] != task or case["policy"] != "pi05_libero":
                    raise ValueError("Baseline case identity differs")
                if not all(row["exact"] and row["finite"] and row["max_abs_difference"] == 0 for row in case["checks"].values()):
                    raise ValueError("Baseline diagnostic fidelity is not bit exact")
                if sha(case_path.parent / "arrays.npz") != case["arrays_sha256"]:
                    raise ValueError("Baseline raw arrays changed")
                oracle = case["oracle"]
                if sha(oracle["summary_path"]) != oracle["summary_sha256"] or sha(oracle["arrays_path"]) != oracle["arrays_sha256"]:
                    raise ValueError("Baseline native oracle evidence changed")
                native = read(Path(oracle["summary_path"]))
                if any(native.get(name) is not True for name in ("complete", "passed", "cache_hits_verified")):
                    raise ValueError("Baseline native oracle cache proof is missing")
                if any(native[name] != case[name] for name in ("task", "policy", "state_id", "decision_index")):
                    raise ValueError("Baseline oracle case identity differs")
                if native["baseline_model_contract"] != proof["baseline_model_contract"]:
                    raise ValueError("Baseline model provenance differs")
                if native["historical_recorded_replay"] != case["historical_recorded_replay"]:
                    raise ValueError("Baseline historical comparison differs")
                identities.append((case["state_id"], case["decision_index"]))
            if identities != [(100, 0), (100, 1), (101, 0), (101, 1)]:
                raise ValueError("Baseline cases differ from predeclaration")
            results[task] = {"path": str(path), "sha256": sha(path), "passed": True}
        complete = task in results
        readiness = {"complete": complete, "passed": True if complete else None,
            "task": task, "policy": "pi05_libero", "proof": results.get(task),
            "autotune_sha256": args.autotune_sha256,
            "compiler_profile_manifest_sha256": sha(args.autotune.parent / "manifest.json"),
            "auditor_source_sha256": sha(Path(__file__)),
            "native_oracle_source_sha256": sha(ROOT / "scripts/check_pi05_native_baseline_oracle.py"),
            "amendment": amendment, "updated_utc": utc()}
        readiness_path = args.output / "task_readiness" / f"{task}.json"
        previous = read(readiness_path)
        if previous and previous.get("complete") is True:
            if {k: v for k, v in previous.items() if k != "updated_utc"} != {k: v for k, v in readiness.items() if k != "updated_utc"}:
                raise ValueError("Completed baseline task readiness is immutable")
        else:
            write(readiness_path, readiness)
    summary = {"complete": len(results) == 3, "passed": True if len(results) == 3 else None,
        "technical_only": True, "policy": "pi05_libero", "completed_tasks": len(results), "expected_tasks": 3,
        "tasks": results, "cases_per_task": 4, "validation_state_ids": [100, 101], "decision_indices": [0, 1],
        "amendment": amendment, "autotune_sha256": args.autotune_sha256, "updated_utc": utc(),
        "native_oracle_source_sha256": sha(ROOT / "scripts/check_pi05_native_baseline_oracle.py"),
        "reference_protocol": "current_native_cached_baseline_executable_oracle_v1", "native_runtime_changed": False,
        "tolerance": "Bit exact; no relaxed threshold"}
    write(args.run / "baseline_sampler_fidelity.json", summary)
    return summary


def parent(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.autotune_sha256 = sha(args.autotune)
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    source_hashes = {str(path): sha(path) for path in evidence_sources()+[args.autotune, args.autotune.parent / "manifest.json", args.amendment]}
    plan = {"technical_only": True, "source_sha256": source_hashes,
            "validation_state_ids": [100, 101], "episode_indices": list(EPISODE_INDICES),
            "decision_indices": list(DECISION_INDICES), "outcome_filtering": False,
            "tasks": list(TASKS), "policy": "pi05_libero",
            "gpu": args.gpu, "gpu_uuid": gpu_uuid, "cpuset": args.cpuset,
            "autotune_sha256": args.autotune_sha256, "created_utc": utc()}
    plan.update(reference_protocol="current_native_cached_baseline_executable_oracle_v1",
                amendment={"path": str(args.amendment), "sha256": sha(args.amendment)},
                native_oracle_source_sha256=sha(ROOT / "scripts/check_pi05_native_baseline_oracle.py"))
    prior = read(args.output / "plan.json")
    if prior:
        if prior["source_sha256"] != source_hashes or prior["autotune_sha256"] != args.autotune_sha256:
            raise ValueError("Fidelity campaign source/profile changed; preserve previous output")
    else:
        write(args.output / "plan.json", plan)
    batch = len(list((args.output / "batches").glob("*.json"))) if (args.output / "batches").exists() else 0
    while True:
        if any(sha(path) != value for path, value in source_hashes.items()):
            raise ValueError("Source/profile changed while fidelity parent waited")
        summary = aggregate(args)
        if summary["complete"]:
            return 0
        jobs = [task for task in TASKS if task not in summary["tasks"] and ready(args.run, task)]
        memory_stats = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
        headroom = (int(Path("/sys/fs/cgroup/memory.max").read_text())
                    - int(Path("/sys/fs/cgroup/memory.current").read_text()) + int(memory_stats["inactive_file"]))
        if jobs and headroom >= 64*1024**3:
            batch += 1
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=args.gpu, XLA_PYTHON_CLIENT_MEM_FRACTION="0.25",
                       XLA_FLAGS="--xla_gpu_load_autotune_results_from="+str(args.autotune),
                       JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="",
                       JAX_COMPILATION_CACHE_DIR=str(args.run / "compiler_caches" / args.autotune.parent.name / "fidelity_campaign"))
            command = ["taskset", "-c", args.cpuset, str(ROOT / "third_party/openpi/.venv/bin/python"),
                       str(Path(__file__).resolve()), "--run", str(args.run), "--output", str(args.output),
                       "--autotune", str(args.autotune), "--autotune-sha256", args.autotune_sha256,
                       "--gpu-uuid", gpu_uuid, "--worker-jobs", ",".join(jobs), "--amendment", str(args.amendment)]
            oracle_cache = args.run / "compiler_caches" / args.autotune.parent.name / "fidelity_native_oracle"
            if not oracle_cache.exists():
                shutil.copytree(ROOT / ".cache/jax-rl-vla", oracle_cache)
            oracle_env = os.environ.copy()
            oracle_env.update(CUDA_VISIBLE_DEVICES=args.gpu, XLA_PYTHON_CLIENT_MEM_FRACTION="0.25", XLA_FLAGS="",
                              JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="xla_gpu_per_fusion_autotune_cache_dir",
                              JAX_COMPILATION_CACHE_DIR=str(oracle_cache))
            oracle_command = ["taskset", "-c", args.cpuset, str(ROOT / "third_party/openpi/.venv/bin/python"),
                str(ROOT / "scripts/check_pi05_native_baseline_oracle.py"), "--run", str(args.run),
                "--output", str(args.output), "--jobs", ",".join(jobs), "--gpu-uuid", gpu_uuid,
                "--amendment", str(args.amendment), "--profile", str(args.autotune.parent / "manifest.json")]
            oracle_log = args.output / "batches" / f"batch_{batch:04d}_native_oracle.log"
            oracle_log.parent.mkdir(exist_ok=True)
            with oracle_log.open("x") as log:
                process = subprocess.Popen(oracle_command, cwd=ROOT, env=oracle_env, stdout=log, stderr=subprocess.STDOUT)
                write(oracle_log.with_suffix(".json"), {"pid": process.pid, "command": oracle_command,
                    "started_utc": utc(), "jobs": jobs, "environment": {name: oracle_env[name] for name in
                    ("CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_FLAGS",
                     "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "JAX_COMPILATION_CACHE_DIR")}})
                oracle_returncode = process.wait()
            if oracle_returncode == 75:
                time.sleep(30)
                continue
            if oracle_returncode:
                failed = aggregate(args)
                failed.update(passed=False, failure={"returncode": oracle_returncode, "log": str(oracle_log)})
                write(args.run / "baseline_sampler_fidelity.json", failed)
                write(args.output / "status.json", {"state": "failed", "returncode": oracle_returncode, "log": str(oracle_log), "updated_utc": utc()})
                raise RuntimeError(f"Independent native oracle failed; inspect {oracle_log}")
            log_path = args.output / "batches" / f"batch_{batch:04d}.log"
            log_path.parent.mkdir(exist_ok=True)
            with log_path.open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                write(log_path.with_suffix(".json"), {"pid": process.pid, "command": command, "jobs": jobs,
                    "started_utc": utc(), "environment": {name: env[name] for name in
                    ("CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_FLAGS",
                     "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "JAX_COMPILATION_CACHE_DIR")}})
                returncode = process.wait()
            if returncode not in (0, 75):
                write(args.output / "status.json", {"state": "failed", "returncode": returncode, "log": str(log_path), "updated_utc": utc()})
                failed = aggregate(args)
                failed.update(passed=False, failure={"returncode": returncode, "log": str(log_path)})
                write(args.run / "baseline_sampler_fidelity.json", failed)
                raise RuntimeError(f"Fidelity child failed; inspect {log_path}")
            write(args.output / "status.json", {"state": "waiting", "last_returncode": returncode, "updated_utc": utc()})
        elif not args.wait:
            return 0
        if not args.wait:
            aggregate(args)
            return 0
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--autotune", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--cpuset", default="48-55")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--worker-jobs")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--autotune-sha256")
    args = parser.parse_args()
    args.run, args.output, args.autotune = args.run.resolve(), args.output.resolve(), args.autotune.resolve()
    args.amendment = args.amendment.resolve()
    return worker(args) if args.worker_jobs else parent(args)


if __name__ == "__main__":
    sys.exit(main())
