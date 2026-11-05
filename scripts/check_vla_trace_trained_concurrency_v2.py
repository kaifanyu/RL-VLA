"""Profile-bound unscored residency proof: baseline plus three loaded trained experts."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from queue_vla_trace_diagnostics import ROOT, cpu_lanes, file_hash, read, run_jobs, write

from rl_vla.diagnostics.compiler_profile import (
    child_environment,
    load_profile,
    verify_effective_environment,
)

GIB = 1024**3
CONDITIONS = ("none", "no_image", "no_text", "target:background_fill",
              "distractor:background_fill", "random:background_fill",
              "target:mosaic", "distractor:mosaic", "random:mosaic")


def cgroup_memory():
    base = Path("/sys/fs/cgroup")
    current = int((base / "memory.current").read_text())
    maximum = int((base / "memory.max").read_text())
    stats = {name: int(value) for name, value in (line.split() for line in (base / "memory.stat").read_text().splitlines())}
    events = {name: int(value) for name, value in (line.split() for line in (base / "memory.events").read_text().splitlines())}
    return {"current_bytes": current, "maximum_bytes": maximum, "headroom_bytes": maximum-current,
            "anon_bytes": stats["anon"], "kernel_bytes": stats["kernel"], "inactive_file_bytes": stats["inactive_file"],
            "effective_headroom_bytes": maximum-current+stats["inactive_file"], "events": events}


def owned_rss(output):
    status = read(output / "queue_status.json") or {}
    groups = {row["pid"] for row in status.get("active", [])}
    rss = 0
    for folder in Path("/proc").glob("[0-9]*"):
        try:
            if os.getpgid(int(folder.name)) in groups:
                fields = dict(line.split(":", 1) for line in (folder / "status").read_text().splitlines() if ":" in line)
                rss += int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return rss


def run(args):
    args.run, args.output = args.run.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    lanes = cpu_lanes(args.cpuset, 4)
    baseline = args.run / "intervention_concurrency_smoke"
    if not read(baseline / "summary.json").get("passed"):
        raise ValueError("Existing three-worker .25 fidelity proof is required")
    reference_plan = read(baseline / "worker_0/summary.json")["case_plan"]
    historical_guard_source_changes = {}
    for path, expected in reference_plan["source_sha256"].items():
        if Path(path).name == "run_vla_trace_interventions.py":
            historical_guard_source_changes[path] = {"original_sha256": expected, "current_sha256": file_hash(path),
                "scope": "New compiler/profile/readiness guards; numerical runtime/adapter hashes are checked separately"}
            continue
        if file_hash(path) != expected:
            raise ValueError(f"Native numerical source differs from three-worker proof: {path}")
    if file_hash(args.run / "tasks/object_0/validation.toml") != reference_plan["config_sha256"]:
        raise ValueError("Validation configuration differs from three-worker proof")
    if file_hash(args.run / "tasks/object_0/protocol.json") != reference_plan["protocol_sha256"]:
        raise ValueError("Case protocol differs from three-worker proof")
    identity = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not identity.startswith("GPU-"):
        raise ValueError("Cannot resolve physical GPU identity")
    lock_path = ROOT / "runs/.vla_trace_gpu_locks" / f"{identity}.lock"
    lock = lock_path.open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    telemetry, monitor_errors = [], []
    stopped = threading.Event()
    jobs = []
    profile = load_profile(args.run)
    checkpoints = [None] + [args.run / "methods/rldt" / f"object_{i}" / "training/iteration_000025" for i in range(3)]
    checkpoint_files = {}
    for checkpoint in checkpoints[1:]:
        for name in ("expert.npz", "rldt_pi05.json"):
            path = checkpoint / name
            before = path.stat()
            digest = file_hash(path)
            after = path.stat()
            identity_stat = [after.st_size, after.st_mtime_ns, after.st_ino]
            if identity_stat != [before.st_size, before.st_mtime_ns, before.st_ino]:
                raise ValueError("Checkpoint changed while hashing technical input")
            checkpoint_files[str(path)] = {"identity": identity_stat, "sha256": digest}
    for index in range(4):
        destination = args.output / f"worker_{index}"
        command = [str(ROOT / "third_party/openpi/.venv/bin/python"),
            str(ROOT / "scripts/run_vla_trace_interventions.py"),
            "--config", str(args.run / "tasks/object_0/validation.toml"),
            "--token-metadata", str(args.run / "gate_b_actual/metadata.json"),
            "--output", str(destination), "--episodes", "1", "--record-episodes", "1",
            "--seed", "100000", "--technical-smoke", "--smoke-decisions", "2",
            "--conditions", ",".join(CONDITIONS)]
        if checkpoints[index]:
            command += ["--checkpoint", str(checkpoints[index])]
        environment, compiler_contract = child_environment(profile,
            args.run / "compiler_caches/native_trained_v2" / f"trained_four_smoke_worker_{index}",
            base_env={"CUDA_VISIBLE_DEVICES": args.gpu, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20", "RL_VLA_OBJECT_TELEMETRY": "1"})
        compiler_contract_path = args.output / f"worker_{index}_compiler_contract.json"
        write(compiler_contract_path, compiler_contract)
        command += ["--compiler-profile", profile["manifest_path"], "--compiler-contract", str(compiler_contract_path)]
        jobs.append({"name": f"worker_{index}", "destination": str(destination), "command": command,
            "environment": environment, "compiler_contract": compiler_contract})
    paths = [Path(__file__), ROOT / "scripts/queue_vla_trace_diagnostics.py",
        ROOT / "scripts/run_vla_trace_interventions.py", ROOT / "src/rl_vla/diagnostics/pi05_attention.py",
        ROOT / "src/rl_vla/diagnostics/interventions.py", ROOT / "src/rl_vla/diagnostics/compiler_profile.py",
        Path(profile["manifest_path"]), Path(profile["profile_path"]), args.run / "tasks/object_0/validation.toml",
        args.run / "tasks/object_0/protocol.json", args.run / "gate_b_actual/metadata.json"]
    hashes = {str(path): file_hash(path) for path in paths}
    initial_memory = cgroup_memory()
    write(args.output / "plan.json", {"technical_only": True, "workers": 4, "memory_fraction": .20,
        "gpu": args.gpu, "gpu_uuid": identity, "physical_gpu_lock": str(lock_path), "jobs": jobs,
        "lanes": lanes, "source_input_sha256": hashes, "initial_host_memory": initial_memory,
        "reference": str(baseline), "timeout_seconds": 420,
        "historical_guard_source_changes": historical_guard_source_changes,
        "compiler_profile": profile,
        "checkpoint_files": checkpoint_files,
        "loaded_experts": [str(p) if p else "pi05_libero" for p in checkpoints],
        "scope": "Unscored cross-task memory test: all workers use Object0 validationstate100, including Object1/Object2 trained experts; no success or performance comparison",
        "residency": "Native runner retains original flow.params while each trained worker also loads its expert params; frozen source is unchanged",
        "abort_guards": "Effective host headroom (including inactivefile reclaimability)<32GiB, newcgroup OOMevent, deviceusage>63GiB, timeout",
        "native_rollout_queue": "Left untouched; root authorized dedicated idle GPU3 while researchgrid remains held"})

    def guard():
        if monitor_errors:
            raise RuntimeError(monitor_errors[0])
        if time.monotonic()-started > 420:
            raise RuntimeError("Bounded smoke timeout")
        memory = cgroup_memory()
        if memory["effective_headroom_bytes"] < 32*GIB:
            raise RuntimeError("Insufficient remaining host-memory headroom")
        if any(memory["events"].get(name, 0) > initial_memory["events"].get(name, 0) for name in ("oom", "oom_kill")):
            raise RuntimeError("New cgroup OOM event; stop technical smoke")
        for job in jobs:
            verify_effective_environment(profile, job["compiler_contract"], environment=job["environment"])
        if any(file_hash(path) != expected for path, expected in hashes.items()):
            raise RuntimeError("Technical smoke source or input changed")
        for path, expected in checkpoint_files.items():
            value = Path(path).stat()
            if [value.st_size, value.st_mtime_ns, value.st_ino] != expected["identity"]:
                raise RuntimeError("Trained technical checkpoint changed")

    def monitor():
        while not stopped.is_set():
            try:
                gpu = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
                    "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
                used, total, utilization = [int(x.strip()) for x in gpu.split(",")]
                processes = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
                    "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True).strip()
                telemetry.append({"elapsed_seconds": time.monotonic()-started, "gpu_used_mib": used,
                    "gpu_total_mib": total, "gpu_utilization": utilization, "processes": processes,
                    "host_memory": cgroup_memory(), "owned_process_group_rss_bytes": owned_rss(args.output)})
                if used > total-1024:
                    monitor_errors.append("GPU headroom below1GiB; stopping technical smoke")
                    return
            except Exception as exc:  # noqa: BLE001 - Surface monitor failures to the owning scheduler.
                monitor_errors.append(repr(exc))
                return
            stopped.wait(.5)

    def terminate(signum, frame):
        raise SystemExit(f"Stopping owned smoke children on signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    thread = threading.Thread(target=monitor, daemon=True)
    error = None
    try:
        guard()
        if initial_memory["effective_headroom_bytes"] < 96*GIB:
            raise RuntimeError("Four-worker loading requires at least96GiB effective initial host headroom")
        initial_gpu = int(subprocess.check_output(["nvidia-smi", "-i", args.gpu,
            "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
        if initial_gpu > 128:
            raise RuntimeError("Assigned physical GPU is not idle before technical smoke")
        thread.start()
        run_jobs(jobs, lanes, args.output, immutable_check=guard)
    except BaseException as exc:  # noqa: BLE001 - Preserve technical failure evidence after group cleanup.
        error = repr(exc)
    finally:
        stopped.set()
        if thread.is_alive():
            thread.join(timeout=5)
        lock.close()
        write(args.output / "telemetry.json", telemetry)

    rows = []
    if error is None:
        try:
            for condition in CONDITIONS:
                folder = condition.replace(":", "_")
                no_op, action_hashes = [], []
                baseline_reference_exact = None
                for index in range(4):
                    directory = args.output / f"worker_{index}" / folder
                    summary = read(directory / "summary.json")
                    episode = summary["episodes"][0]
                    assert summary["completed"] and summary["technical_smoke"]
                    expected_checkpoint = str(checkpoints[index]) if checkpoints[index] else "pi05_libero"
                    assert episode["checkpoint"] == expected_checkpoint
                    assert episode["initial_state_id"] == 100 and episode["seed"] == 100000
                    assert episode["success"] is None and episode["primitive_steps"] == 10
                    assert Path(episode["video"]).is_file()
                    assert len(read(directory / "episode_000_telemetry.json")) == 10
                    no_op.append(read(directory / "no_op.json"))
                    assert no_op[-1]["passed"]
                    with np.load(directory / "episode_000_actions.npz", allow_pickle=False) as data:
                        actions = data["normalized"].copy()
                    assert len(actions) == 2 and np.isfinite(actions).all()
                    action_hashes.append(file_hash(directory / "episode_000_actions.npz"))
                    if index == 0 and condition in ("none", "no_image", "target:mosaic"):
                        with np.load(baseline / "worker_0" / folder / "episode_000_actions.npz", allow_pickle=False) as previous:
                            baseline_reference_exact = np.array_equal(actions, previous["normalized"])
                        if not baseline_reference_exact:
                            raise ValueError("Baseline actions differ from established .25 reference")
                rows.append({"condition": condition, "four_workers_finite": True,
                    "baseline_bit_exact_to_previous_three_worker_reference": baseline_reference_exact,
                    "trained_worker_reference": "Each loaded expert checked against its canonical native sampler at .20; no cross-policy equality claim",
                    "action_artifact_sha256": action_hashes, "native_no_op": no_op})
            count = max(len(row["processes"].splitlines()) if row["processes"] else 0 for row in telemetry)
            if count != 4:
                raise RuntimeError(f"Expected four simultaneously resident models, saw{count}")
            for path, expected in checkpoint_files.items():
                if file_hash(path) != expected["sha256"]:
                    raise ValueError("Trained checkpoint hash changed during technical smoke")
        except Exception as exc:  # noqa: BLE001 - Record a failed fidelity result without losing telemetry.
            error = repr(exc)
    summary = {"complete": True, "passed": error is None, "technical_only": True, "error": error,
        "workers": 4, "memory_fraction_per_worker": .20, "gpu": args.gpu, "gpu_uuid": identity,
        "peak_gpu_processes": max((len(row["processes"].splitlines()) if row["processes"] else 0 for row in telemetry), default=0),
        "peak_gpu_memory_mib": max((row["gpu_used_mib"] for row in telemetry), default=0),
        "peak_cgroup_host_memory_bytes": max((row["host_memory"]["current_bytes"] for row in telemetry), default=initial_memory["current_bytes"]),
        "peak_cgroup_anon_bytes": max((row["host_memory"]["anon_bytes"] for row in telemetry), default=initial_memory["anon_bytes"]),
        "peak_cgroup_kernel_bytes": max((row["host_memory"]["kernel_bytes"] for row in telemetry), default=initial_memory["kernel_bytes"]),
        "peak_owned_process_group_rss_bytes": max((row["owned_process_group_rss_bytes"] for row in telemetry), default=0),
        "initial_cgroup_host_memory_bytes": initial_memory["current_bytes"],
        "final_host_memory": cgroup_memory(), "elapsed_seconds": time.monotonic()-started,
        "physical_gpu_lock_released": True, "conditions": rows,
        "loaded_experts": [str(p) if p else "pi05_libero" for p in checkpoints],
        "checkpoint_hashes_verified_start_and_final": error is None,
        "compiler_profile": {name: profile[name] for name in ("manifest_path", "manifest_sha256", "profile_sha256", "helper_sha256")},
        "scope": "Unscored cross-task technical residency test only; baseline plus Object0/1/2 iteration25 experts all use Object0 validationstate100, all9conditions, two native decisions; no research queue changes"}
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if error:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--cpuset", default="48-55")
    args = parser.parse_args()

    def terminate(signum, frame):
        raise SystemExit(f"Stopping technical smoke on signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    run(args)


if __name__ == "__main__":
    main()
