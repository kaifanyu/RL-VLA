"""Finite, durable GPU queue gated on completion of the preceding campaign.

One coordinator owns the manifest; UUID file locks arbitrate GPU claims with the
current LIBERO workers. No process is killed to make room. Runtime/config inputs
are hashed and checked before launch. Only identified resource interruptions get
one fresh-output retry, never an implicit partial training resume.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from next_multigpu_gate import inspect_current_gate


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_jobs(path, jobs):
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(j, allow_nan=False) + "\n" for j in jobs))
    temp.replace(path)


def aggregate_results(campaign_dir, jobs):
    """Publish completed attempts only; running and failed partial scores stay local."""
    columns = ["job_id", "family_id", "task", "variant", "training_seed", "split", "iteration",
               "successes", "episodes", "trainable_scope", "initialization_hash"]
    rows = []
    errors = []
    for job in jobs:
        if job.get("state") != "completed" or not job.get("active_output"):
            continue
        path = Path(job["active_output"]) / "results.csv"
        if not path.is_file():
            continue  # Readiness and diagnostic jobs need not produce task scores.
        try:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                records = list(reader)
                if not reader.fieldnames or "job_id" not in reader.fieldnames:
                    raise ValueError("missing job_id column")
                if any(row.get("job_id") != job["job_id"] for row in records):
                    raise ValueError("result job_id mismatch")
                if any(None in row for row in records):
                    raise ValueError("malformed CSV row")
                columns.extend(key for key in reader.fieldnames if key not in columns)
                rows.extend(records)
        except (OSError, ValueError, csv.Error) as exc:
            errors.append({"job_id": job["job_id"], "path": str(path), "error": str(exc)})
    temporary = Path(campaign_dir) / "results.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(Path(campaign_dir) / "results.csv")
    return errors


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def query_gpus():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    busy = {line.split(",")[0].strip() for line in processes.splitlines() if line.strip()}
    rows = []
    for row in csv.reader(output.splitlines()):
        index, uid, total, used, utilization = [s.strip() for s in row]
        rows.append(dict(index=int(index), uuid=uid, total_mib=int(total),
                         used_mib=int(used), utilization=int(utilization),
                         has_compute_process=uid in busy))
    return rows


def allocated_gpus(rows):
    # An explicit allocation, including an empty set, always narrows eligibility.
    for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if name not in os.environ or os.environ[name] == "all":
            continue
        tokens = {x.strip() for x in os.environ[name].split(",") if x.strip()}
        for token in list(tokens):
            parts = token.split("-")
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                tokens.update(str(i) for i in range(int(parts[0]), int(parts[1]) + 1))
        rows = [r for r in rows if str(r["index"]) in tokens or r["uuid"] in tokens]
    if "SLURM_JOB_ID" in os.environ and not any(
        name in os.environ for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES")
    ):
        return []  # Never turn a CPU-only Slurm allocation into whole-node GPU access.
    return rows


def cpu_groups(count):
    allowed = sorted(os.sched_getaffinity(0))
    quota = Path("/sys/fs/cgroup/cpu.max")
    budget = len(allowed)
    if quota.exists():
        cap, period = quota.read_text().split()
        if cap != "max":
            budget = min(budget, max(1, int(cap) // int(period)))
    width = min(8, budget // max(1, count))
    if width < 2:
        raise RuntimeError("Insufficient CPU allocation for independent GPU workers")
    # Group by NUMA node, then physical core, before assigning disjoint CPUs.
    def topology(cpu):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        nodes = list(base.glob("node[0-9]*"))
        node = int(nodes[0].name[4:]) if nodes else 0
        core = int((base / "topology/core_id").read_text())
        package = int((base / "topology/physical_package_id").read_text())
        siblings = []
        for token in (base / "topology/thread_siblings_list").read_text().strip().split(","):
            endpoints = list(map(int, token.split("-")))
            siblings.extend(range(endpoints[0], endpoints[-1] + 1))
        return node, siblings.index(cpu), package, core, cpu
    allowed.sort(key=topology)
    return [allowed[i * width:(i + 1) * width] for i in range(count)]


def worker_cpu_sets(rows, config):
    assignments = config.get("cpu_assignments_by_gpu_index")
    if assignments is None:
        return {g["uuid"]: c for g, c in zip(rows, cpu_groups(len(rows)))}
    allowed = set(os.sched_getaffinity(0))
    used = set()
    result = {}
    for gpu in rows:
        cpus = assignments.get(str(gpu["index"]), [])
        if (not cpus or any(type(cpu) is not int for cpu in cpus)
                or len(set(cpus)) != len(cpus) or not set(cpus) <= allowed
                or set(cpus) & used):
            raise ValueError(f"Invalid or overlapping CPU assignment for GPU {gpu['index']}")
        used.update(cpus)
        result[gpu["uuid"]] = cpus
    return result


def memory_available_gib():
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(info["MemAvailable"].split()[0]) / 1024**2
    root = Path("/sys/fs/cgroup")
    if (root / "memory.max").exists():
        maximum = (root / "memory.max").read_text().strip()
        if maximum != "max":
            current = int((root / "memory.current").read_text())
            stat = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
            reclaimable = int(stat.get("inactive_file", 0))
            available = min(available, (int(maximum) - current + reclaimable) / 1024**3)
    return available


def unresolved(job, jobs):
    reasons = list(job.get("blockers", []))
    by_id = {j["job_id"]: j for j in jobs}
    for key in job.get("dependencies", []):
        if by_id.get(key, {}).get("state") != "completed":
            reasons.append(f"dependency not completed: {key}")
    for key in job.get("required_started_jobs", []):
        other = by_id.get(key, {})
        if not other.get("started_at") or other.get("state") in ("failed", "cancelled"):
            reasons.append(f"mandatory first production run not started: {key}")
    for value in job.get("required_artifacts", []):
        if not Path(value).is_file():
            reasons.append(f"missing artifact: {value}")
    for item in job.get("prerequisites", []):
        path = Path(item["path"])
        if not path.is_file():
            reasons.append(f"missing prerequisite: {path}")
        elif item["kind"] == "json_equals":
            try:
                value = json.loads(path.read_text())
                for part in item["key"].split("."):
                    value = value[part]
                if value != item["value"]:
                    reasons.append(f"prerequisite mismatch: {path}:{item['key']}")
            except (KeyError, ValueError, TypeError):
                reasons.append(f"invalid prerequisite: {path}")
        else:
            reasons.append(f"unsupported prerequisite kind: {item['kind']}")
    if job.get("gpu_count", 1) != 1:
        reasons.append("this dispatcher supports exactly one GPU per job")
    if not job.get("command_argv"):
        reasons.append("no validated executable command")
    return reasons


def verify_input_contract(contract, versions=None):
    """Hash once, then guard every launch against changes to immutable inputs."""
    observed = {}
    for filename, expected in contract.items():
        path = Path(filename)
        try:
            stat = path.stat()
        except OSError as exc:
            raise RuntimeError(f"Immutable input missing: {path}") from exc
        version = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if versions is None:
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"Immutable input hash mismatch: {path}")
            after = path.stat()
            if version != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise RuntimeError(f"Immutable input changed during hashing: {path}")
        elif versions.get(filename) != version:
            raise RuntimeError(f"Immutable input changed after verification: {path}")
        observed[filename] = version
    return observed


def process_tree_private_gib(pid):
    """Measure private resident RAM; unmeasurable workers retain full reservations."""
    processes = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            processes[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    owned = {pid}
    while True:
        children = {child for child, parent in processes.items() if parent in owned}
        if children <= owned:
            break
        owned.update(children)
    kib = 0
    for child in owned:
        try:
            for line in Path(f"/proc/{child}/smaps_rollup").read_text().splitlines():
                if line.startswith(("Private_Clean:", "Private_Dirty:", "Private_Hugetlb:")):
                    kib += int(line.split()[1])
        except (OSError, ValueError):
            continue
    return kib / 1024**2


def transient_failure(log, attempt_directory=None):
    # The orchestrator captures native failures in stage logs, while its own log
    # usually contains only CalledProcessError. Inspect both without losing the
    # original attempt's evidence or retrying a numerical/contract failure.
    paths = [Path(log)]
    if attempt_directory is not None:
        paths.extend(sorted(Path(attempt_directory).glob("*.log")))
    tails = []
    for path in paths:
        with path.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 131072))
            tails.append(f.read().decode(errors="replace"))
    tail = "\n".join(tails)
    if any(s in tail.lower() for s in ("nonfinite", "non-finite", "nan gradient", "hash mismatch",
                                      "immutable input", "assertionerror", "valueerror", "typeerror")):
        return False
    return any(s in tail.lower() for s in (
        "cuda_error_out_of_memory", "resource_exhausted", "cuda out of memory",
        "temporarily unavailable", "connection reset by peer", "resource temporarily unavailable"))


def make_environment(campaign, job, gpu, cpus, attempt):
    env = os.environ.copy()
    for name in ("RL_VLA_VALIDATION_GPU", "RL_VLA_VALIDATION_CPUSET", "RL_VLA_EVALUATION_CPUSET",
                 "RL_VLA_EVAL_SHARDS", "JAX_PLATFORMS", "JAX_PLATFORM_NAME"):
        env.pop(name, None)
    root = Path(campaign["repository"])
    cache = Path(campaign["campaign"]) / "worker_cache" / job["job_id"] / str(attempt)
    for name in ("tmp", "jax", "xdg", "numba", "torch", "triton", "libero"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / ".cache/libero-pro/config.yaml", cache / "libero/config.yaml")
    env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID",
               RLDT_CPUSET=",".join(map(str, cpus)), LP_NUM_THREADS="2",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", NUMBA_DISABLE_JIT="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.80",
               RL_VLA_OBJECT_TELEMETRY="1", RL_VLA_EVAL_SHARDS="1",
               TMPDIR=str(cache / "tmp"), JAX_COMPILATION_CACHE_DIR=str(cache / "jax"),
               XDG_CACHE_HOME=str(cache / "xdg"), NUMBA_CACHE_DIR=str(cache / "numba"),
               TORCHINDUCTOR_CACHE_DIR=str(cache / "torch"), TRITON_CACHE_DIR=str(cache / "triton"),
               LIBERO_CONFIG_PATH=str(cache / "libero"),
               HF_HOME=str(root / ".cache/huggingface"), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OPENPI_DATA_HOME=str(root / ".cache/openpi"),
               NEXT_CAMPAIGN_SOURCE_ROOT=campaign["source_snapshot"],
               PYTHONPATH=":".join(str(Path(campaign["source_snapshot"]) / relative) for relative in (
                   "src", "third_party/openpi/src", "third_party/openpi/packages/openpi-client/src",
                   "third_party/libero_pro")),
               LD_LIBRARY_PATH=str(root / ".cache/native/usr/lib/x86_64-linux-gnu") +
                   (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""))
    return env


def run(campaign_dir, once=False):
    campaign_dir = Path(campaign_dir).resolve()
    config = json.loads((campaign_dir / "queue_config.json").read_text())
    with (campaign_dir / "coordinator.lock").open("a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs_path = campaign_dir / "jobs.jsonl"
        jobs = [json.loads(line) for line in jobs_path.read_text().splitlines() if line]
        if len({j["job_id"] for j in jobs}) != len(jobs):
            raise ValueError("Duplicate job IDs")
        # An orphaned worker must be reconciled, never duplicated after a restart.
        for job in jobs:
            if job.get("state") == "running":
                raise RuntimeError(f"Reconcile prior running job before restart: {job['job_id']}")
        rows = allocated_gpus(query_gpus())
        if not rows:
            raise RuntimeError("No GPUs in this process's allocation")
        cpus_by_uuid = worker_cpu_sets(rows, config)
        atomic_json(campaign_dir / "coordinator.json", dict(pid=os.getpid(), started_at=now(),
                    gpu_uuids=list(cpus_by_uuid), cpus=cpus_by_uuid, command=sys.argv))
        active = {}
        stopping = False

        def stop(_sig, _frame):
            nonlocal stopping
            stopping = True  # Drain active children, leave ready jobs queued.

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        wait_for_current = config.get("wait_for_current_campaign", True)
        if type(wait_for_current) is not bool:
            raise ValueError("wait_for_current_campaign must be boolean")
        gate_open = not wait_for_current
        if gate_open:
            atomic_json(campaign_dir / "dispatch_gate.json", {
                "ready": True, "scope": "independent_jobs_on_unclaimed_gpus",
                "current_campaign_complete": False,
                "authorization": config.get("scheduling_amendment"), "updated_at": now(),
                "note": "Current analysis is not a global launch dependency; job prerequisites and GPU locks remain mandatory.",
            })
        input_versions = None
        contract = None
        tick = 0
        while True:
            tick += 1
            for uid, worker in list(active.items()):
                code = worker["process"].poll()
                if code is None:
                    continue
                job = worker["job"]
                completion = worker["output"] / "completion.json"
                complete = False
                if code == 0 and completion.is_file():
                    try:
                        result = json.loads(completion.read_text())
                        complete = result.get("completed") is True and result.get("job_id") == job["job_id"]
                    except ValueError:
                        pass
                job.update(ended_at=now(), exit_code=code, state="completed" if complete else "failed")
                job["attempt_history"][-1].update(ended_at=job["ended_at"], exit_code=code, completed=complete)
                if not complete:
                    job["failure_reason"] = "worker failure" if code else "missing successful completion manifest"
                    retry_count = job.get("transient_retries_used", 0)
                    if retry_count < int(config.get("maximum_transient_retries", 1)) and transient_failure(worker["log"], worker["output"]):
                        job.update(state="ready", transient_retries_used=retry_count + 1,
                                   retry_reason="one retry after identified transient resource failure")
                worker["handle"].close()
                worker["lock"].close()
                del active[uid]
            if not gate_open:
                gate = inspect_current_gate(Path(config["current_run"]))
                atomic_json(campaign_dir / "current_campaign_gate.json", gate)
                gate_open = gate["ready"]
            if gate_open and input_versions is None:
                contract = json.loads((campaign_dir / "input_hashes.json").read_text())
                input_versions = verify_input_contract(contract)
            for job in jobs:
                if job.get("state") in ("completed", "running", "failed", "cancelled"):
                    continue
                reasons = unresolved(job, jobs)
                if not gate_open:
                    reasons.insert(0, "current LIBERO experiment and computational analysis not complete")
                job["waiting_for"] = reasons
                job["state"] = "blocked_dependency" if reasons else "ready"
            gpus = allocated_gpus(query_gpus()) if gate_open else []
            telemetry = {"updated_at": now(), "gate_open": gate_open, "stopping": stopping,
                         "active_jobs": {k: w["job"]["job_id"] for k, w in active.items()},
                         "gpus": gpus,
                         "memory_available_gib": memory_available_gib(),
                         "disk_available_gib": shutil.disk_usage(campaign_dir).free / 1024**3,
                         "load_average": os.getloadavg(),
                         "states": {s: sum(j["state"] == s for j in jobs) for s in
                                    ("blocked_dependency", "ready", "running", "completed", "failed", "cancelled")}}
            ready = sorted((j for j in jobs if j["state"] == "ready"),
                           key=lambda j: (int(j["priority"]), j["job_id"]))
            if gate_open and ready and not stopping and not once:
                # Budget reservations cover children still loading their models.
                pending_ram = sum(max(0., w["reserved_ram"] - process_tree_private_gib(w["process"].pid))
                                  for w in active.values())
                available_ram = memory_available_gib() - pending_ram
                for gpu in gpus:
                    if not ready:
                        break
                    if gpu["uuid"] in active or gpu["has_compute_process"] or gpu["used_mib"] > 512:
                        continue
                    job = ready[0]
                    required_ram = float(job.get("estimated_host_memory_gb", 32))
                    if available_ram < required_ram + 16 or telemetry["disk_available_gib"] < 150:
                        continue
                    if gpu["total_mib"] < float(job.get("estimated_memory_gb", 48)) * 1024:
                        continue
                    lock_path = Path(config["repository"]) / "runs/.vla_trace_gpu_locks" / (gpu["uuid"] + ".lock")
                    lock_path.parent.mkdir(parents=True, exist_ok=True)
                    lock = lock_path.open("a+")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                        continue
                    # Check again after acquiring the shared advisory lock.
                    check = next(g for g in query_gpus() if g["uuid"] == gpu["uuid"])
                    if check["has_compute_process"] or check["used_mib"] > 512:
                        lock.close()
                        continue
                    attempt = job.get("attempts", 0) + 1
                    verify_input_contract(contract, input_versions)
                    output = Path(job["output_directory"]) / f"attempt_{attempt}"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    argv = list(job["command_argv"])
                    if "--output" not in argv:
                        raise ValueError(f"Worker command requires --output: {job['job_id']}")
                    argv[argv.index("--output") + 1] = str(output)
                    cpus = cpus_by_uuid[gpu["uuid"]]
                    argv = ["taskset", "-c", ",".join(map(str, cpus)), *argv]
                    env = make_environment(config, job, gpu, cpus, attempt)
                    log = output.parent / f"attempt_{attempt}.log"
                    handle = log.open("x")
                    job.update(state="running", attempts=attempt, started_at=now(), ended_at=None,
                               exit_code=None, gpu_uuid=gpu["uuid"], cpu_affinity=cpus,
                               scheduler_id=f"local:{os.getpid()}:{job['job_id']}",
                               active_output=str(output), waiting_for=[])
                    job.setdefault("attempt_history", []).append(dict(attempt=attempt, started_at=job["started_at"],
                        output_directory=str(output), command_argv=argv, gpu_uuid=gpu["uuid"], log=str(log)))
                    save_jobs(jobs_path, jobs)  # Durable claim precedes process launch.
                    try:
                        process = subprocess.Popen(argv, cwd=config["source_snapshot"], env=env,
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(lock.fileno(),))
                    except OSError as exc:
                        job.update(state="failed", ended_at=now(), failure_reason=f"process launch failed: {exc}")
                        job["attempt_history"][-1].update(ended_at=job["ended_at"], completed=False,
                                                         launch_error=repr(exc))
                        handle.close()
                        lock.close()
                        save_jobs(jobs_path, jobs)
                        ready.pop(0)
                        continue
                    job["pid"] = process.pid
                    active[gpu["uuid"]] = dict(process=process, job=job, output=output, lock=lock,
                        log=log, handle=handle, reserved_ram=required_ram, launched=time.monotonic())
                    available_ram -= required_ram
                    ready.pop(0)
                    print(f"{now()} started {job['job_id']} on GPU {gpu['index']} ({gpu['uuid']})", flush=True)
            save_jobs(jobs_path, jobs)
            telemetry["result_aggregation_errors"] = aggregate_results(campaign_dir, jobs)
            telemetry["active_jobs"] = {k: w["job"]["job_id"] for k, w in active.items()}
            telemetry["states"] = {s: sum(j["state"] == s for j in jobs) for s in telemetry["states"]}
            telemetry["queue_state"] = ("draining" if stopping else "waiting_current_campaign" if not gate_open
                else "running" if active else "waiting_resources" if ready
                else "finished" if all(j["state"] == "completed" for j in jobs) else "finished_with_blockers")
            atomic_json(campaign_dir / "queue_status.json", telemetry)
            if not once and not stopping and gate_open and not active and not ready:
                atomic_json(campaign_dir / "final_status.json", {
                    "state": telemetry["queue_state"], "ended_at": now(),
                    "completed_job_ids": [j["job_id"] for j in jobs if j["state"] == "completed"],
                    "failed_jobs": [{"job_id": j["job_id"], "reason": j.get("failure_reason"),
                                     "attempt_history": j.get("attempt_history", [])}
                                    for j in jobs if j["state"] == "failed"],
                    "blocked_jobs": [{"job_id": j["job_id"], "reasons": j.get("waiting_for", [])}
                                     for j in jobs if j["state"] == "blocked_dependency"],
                    "cancelled_job_ids": [j["job_id"] for j in jobs if j["state"] == "cancelled"],
                    "result_aggregation_errors": telemetry["result_aggregation_errors"],
                    "resources_released": True,
                })
            if once or (stopping and not active) or (gate_open and not active and not ready):
                break
            if tick == 1 or tick % 20 == 0:
                print(f"{now()} {telemetry['queue_state']}: {telemetry['states']}", flush=True)
            time.sleep(30)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--once", action="store_true", help="Inspect and record state without launching jobs")
    args = parser.parse_args()
    raise SystemExit(run(args.campaign, args.once))
