"""Durable, gated native attention and closed-loop intervention workers."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from rl_vla.diagnostics.compiler_dispatch import child_environment, load_profile
from rl_vla.diagnostics.trained_fidelity import policy_fidelity_contract, verify_task_fidelity
ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("none", "no_image", "no_text", "target:background_fill",
              "distractor:background_fill", "random:background_fill",
              "target:mosaic", "distractor:mosaic", "random:mosaic")
OVERFLOW_CPU_LANES = {"object_0": "48,49,50", "object_1": "51,52,53", "object_2": "54,55"}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stat(path):
    value = Path(path).stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns, "inode": value.st_ino}


def checkpoint_contract(method, audit):
    if audit.get("passed") is not True or set(audit.get("checkpoints", {})) != {"25", "50", "100"}:
        raise ValueError("Full passed main checkpoint audit is required")
    records = {}
    for iteration in (25, 50, 100):
        for name in ("expert.npz", "rldt_pi05.json"):
            path = method/"training"/f"iteration_{iteration:06d}"/name
            expected = audit["checkpoints"][str(iteration)]["files"][name]
            before = file_stat(path)
            measured = file_hash(path)
            if before != file_stat(path) or before["bytes"] != expected["bytes"] or measured != expected["sha256"]:
                raise ValueError(f"Causal checkpoint differs from passed main audit: {path}")
            records[str(path)] = {**before, "sha256": measured}
    return records


def verify_checkpoint_contract(records, *, full_hash=False):
    for path, record in records.items():
        if file_stat(path) != {key: record[key] for key in ("bytes", "mtime_ns", "inode")}:
            raise ValueError(f"Frozen causal checkpoint stat changed: {path}")
        if full_hash and file_hash(path) != record["sha256"]:
            raise ValueError(f"Frozen causal checkpoint hash changed: {path}")


def wait_file(path, *, failure_path=None):
    while not path.exists():
        failure = read(failure_path) if failure_path else None
        if failure and failure.get("state") == "failed":
            raise RuntimeError(f"Upstream task failed: {failure_path}")
        time.sleep(30)
    return read(path)


TASKS = ("object_0", "object_1", "object_2")


def queue_source_hashes(mode):
    entries = {"grid": ["run_vla_trace_grid.py"], "rollout": ["trace_pi05_rollout_bank.py", "run_vla_trace_grid.py"],
               "interventions": ["run_vla_trace_interventions.py"]}[mode]
    paths = [Path(__file__), *[ROOT/"scripts"/name for name in entries],
             *[ROOT/"src/rl_vla/diagnostics"/name for name in
               ("compiler_dispatch.py", "compiler_profile.py", "compiler_profile_baseline.py", "trained_fidelity.py")]]
    if mode in ("grid", "rollout"):
        paths.append(ROOT/"src/rl_vla/diagnostics/grid_stages.py")
    return {str(path): file_hash(path) for path in paths}


def verify_queue_sources(hashes):
    if any(file_hash(path) != expected for path, expected in hashes.items()):
        raise ValueError("Diagnostic queue source changed after process startup")


@contextmanager
def exclusive_gpu(gpu):
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    path = ROOT/"runs/.vla_trace_gpu_locks"/f"{uuid}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield uuid


def launch_profile_child(command, env, log_path):
    """Wait for an owned fresh process; preserve its actual exit identity."""
    with Path(log_path).open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ticks = int((Path("/proc")/str(process.pid)/"stat").read_text().rsplit(") ", 1)[1].split()[19])
            code = process.wait()
        except BaseException:
            stop_children([process])
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)
    return {"pid": process.pid, "returncode": code, "proc_start_ticks": ticks}


def profiles_for(args):
    profiles = {"baseline": load_profile(args.baseline_compiler_profile),
                "trained": load_profile(args.compiler_profile)}
    if (profiles["baseline"]["manifest"]["profile_id"] != "native_baseline_v1"
            or profiles["trained"]["manifest"]["profile_id"] != "native_trained_v2"):
        raise ValueError("Baseline and trained policies require their respective native compiler profiles")
    return profiles


def stage_environment(args, profile, name):
    env, contract = child_environment(profile, Path(profile["manifest"]["isolated_cache_root"])/name)
    path = args.run/"diagnostic_compiler_contracts"/f"{name}.json"
    write(path, contract)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, XLA_PYTHON_CLIENT_MEM_FRACTION="0.85")
    return env, contract, ["--compiler-profile", profile["manifest_path"], "--compiler-contract", str(path)]


def run_grid(args):
    sources = queue_source_hashes("grid")
    wait_file(args.run/"features_retention/pi05_base/summary.json")
    profiles = profiles_for(args)
    pending, jobs = list(TASKS), []
    while pending:
        verify_queue_sources(sources)
        for task in list(pending):
            method = args.run/"methods/rldt"/task
            if (read(method/"status.json") or {}).get("state") == "failed":
                raise RuntimeError(f"Upstream task failed: {task}")
            if not (method/"selection.json").exists() or not (method/"checkpoint_audit.json").exists():
                continue
            proofs = {kind: policy_fidelity_contract(args.run, task, profile) for kind, profile in profiles.items()}
            if any(proof is None for proof in proofs.values()):
                continue
            with exclusive_gpu(args.gpu):
                for stage in ("baseline", "trained", "baseline_projection", "aggregate"):
                    verify_queue_sources(sources)
                    for proof in proofs.values(): verify_task_fidelity(proof)
                    command = ["taskset", "-c", args.cpuset, str(ROOT/"third_party/openpi/.venv/bin/python"),
                               str(ROOT/"scripts/run_vla_trace_grid.py"), "--run-root", str(args.run),
                               "--output", str(args.run/"traces"), "--count", "50", "--tasks", task, "--stage", stage]
                    contract = None
                    if stage == "aggregate":
                        env = os.environ.copy()
                        env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
                    else:
                        profile = profiles["trained" if stage == "trained" else "baseline"]
                        env, contract, flags = stage_environment(args, profile, f"grid/{task}/{stage}")
                        command += flags
                    child_exit = launch_profile_child(command, env, args.run/f"attention_grid_{task}_{stage}.log")
                    jobs.append({"task": task, "stage": stage, "command": command, "compiler_profile": contract, "child_exit": child_exit})
                result = read(args.run/"traces"/task/"summary.json")
                if not result or result.get("complete", result.get("completed")) is not True:
                    raise ValueError("Task grid stages did not aggregate complete evidence")
            pending.remove(task)
        if pending:
            write(args.run/"grid_queue_status.json", {"complete": False, "pending": pending})
            time.sleep(30)
    verify_queue_sources(sources)
    value = {"complete": True, "completed": True, "tasks": list(TASKS), "count_per_task": 50,
             "input_sha256": sources,
             "jobs": jobs, "child_exits": [job["child_exit"] for job in jobs],
             "child_exit": next(job["child_exit"] for job in reversed(jobs) if job["stage"] != "aggregate")}
    write(args.run/"traces/summary.json", value)
    write(args.run/"attention_grid_queue_complete.json", value)
    write(args.run/"grid_queue_status.json", {"complete": True})


def run_rollout(args):
    """Separate baseline/trained processes, after all oracle work releases GPU5."""
    profiles = profiles_for(args)
    sources = queue_source_hashes("rollout")
    proofs = {}
    while len(proofs) != len(TASKS)*2:
        verify_queue_sources(sources)
        for task in TASKS:
            if (read(args.run/"methods/rldt"/task/"status.json") or {}).get("state") == "failed":
                raise RuntimeError(f"Upstream task failed while awaiting rollout proof: {task}")
            for kind, profile in profiles.items():
                proof = policy_fidelity_contract(args.run, task, profile)
                if proof is not None: proofs[(task, kind)] = proof
        if len(proofs) != len(TASKS)*2:
            time.sleep(30)
    jobs = []
    write(args.run/"rollout_traces/status.json", {"complete": False, "pending_stages": list(profiles)})
    with exclusive_gpu(args.gpu):
        for stage, profile in profiles.items():
            verify_queue_sources(sources)
            for proof in proofs.values(): verify_task_fidelity(proof)
            env, contract, flags = stage_environment(args, profile, f"rollout/{stage}")
            command = ["taskset", "-c", args.cpuset, str(ROOT/"third_party/openpi/.venv/bin/python"),
                       str(ROOT/"scripts/trace_pi05_rollout_bank.py"), "--run", str(args.run),
                       "--output", str(args.run/"rollout_traces"), "--stage", stage, *flags]
            child_exit = launch_profile_child(command, env, args.run/f"rollout_trace_capture_{stage}.log")
            result = read(args.run/"rollout_traces/stages"/stage/"summary.json")
            if not result or result.get("passed") is not True or result.get("complete") is not True or result.get("compiler_profile") != contract:
                raise ValueError("Rollout trace stage did not complete under its parent compiler contract")
            jobs.append({"stage": stage, "command": command, "compiler_profile": contract, "child_exit": child_exit})
        verify_queue_sources(sources)
        value = {"complete": True, "passed": True, "tasks": list(TASKS),
                 "policies": ["pi05_libero", "rldt_000025", "rldt_000050", "rldt_000100"],
                 "jobs": jobs, "child_exit": jobs[-1]["child_exit"], "child_exits": [job["child_exit"] for job in jobs]}
        value["input_sha256"] = sources
        write(args.run/"rollout_traces/summary.json", value)
        write(args.run/"rollout_traces/status.json", {"complete": True, "pending": []})
        write(args.run/"rollout_trace_queue_complete.json", value)


def cpu_lanes(cpuset, workers=3):
    cpus = []
    for part in cpuset.split(","):
        bounds = [int(x) for x in part.strip().split("-")]
        if len(bounds) == 1:
            cpus.append(bounds[0])
        elif len(bounds) == 2 and bounds[0] <= bounds[1]:
            cpus.extend(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError("Invalid CPU range")
    if workers < 1 or len(cpus) < workers or len(cpus) != len(set(cpus)):
        raise ValueError("Each concurrent worker requires a distinct nonempty CPU lane")
    quotient, remainder = divmod(len(cpus), workers)
    lanes, start = [], 0
    for index in range(workers):
        count = quotient + (index < remainder)
        lanes.append(",".join(map(str, cpus[start:start + count])))
        start += count
    return lanes


def stop_children(processes, grace_seconds=10):
    def send(process, signum):
        try:
            os.killpg(process.pid, signum)
            return True
        except ProcessLookupError:
            return False
    for process in processes:
        send(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        for process in processes:
            process.poll()
        if not any(send(process, 0) for process in processes):
            break
        time.sleep(.1)
    for process in processes:
        send(process, signal.SIGKILL)
    for process in processes:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            send(process, signal.SIGKILL)


class GridReleasedOverflow:
    """One optional policy per task after the grid child has exited successfully.

    A shared physical-GPU lock excludes exclusive GPU3 users. Each task also
    holds its own exclusive slot lock, limiting this pool to three models.
    Acquisition is nonblocking, so unavailable overflow never stalls primaries.
    """

    gpu = "3"

    def __init__(self, campaign, task, *, lock_directory=None):
        self.task = task
        self.cpuset = OVERFLOW_CPU_LANES[task]
        self.gate = Path(campaign)/"attention_grid_queue_complete.json"
        self.lock_directory = Path(lock_directory) if lock_directory else ROOT/"runs/.vla_trace_gpu_locks"
        self.uuid = None
        self.used = False
        self.record = None
        self._locks = []

    def try_acquire(self):
        if self.used:
            return False
        try:
            gate_bytes = self.gate.read_bytes()
            gate = json.loads(gate_bytes)
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        if not isinstance(gate, dict) or gate.get("complete") is not True:
            return False
        if self.uuid is None:
            try:
                identity = subprocess.check_output(["nvidia-smi", "-i", self.gpu,
                    "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                return False
            if not identity.startswith("GPU-") or "\n" in identity:
                return False
            self.uuid = identity
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        acquired = False
        try:
            for path, mode in ((self.lock_directory/f"{self.uuid}.lock", fcntl.LOCK_SH),
                               (self.lock_directory/f"{self.uuid}.causal_overflow_{self.task}.lock", fcntl.LOCK_EX)):
                stream = path.open("a+")
                self._locks.append(stream)
                try:
                    fcntl.flock(stream, mode | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
            try:
                if self.gate.read_bytes() != gate_bytes:
                    return False
            except FileNotFoundError:
                return False
            self.record = {"task": self.task, "gpu": self.gpu, "gpu_uuid": self.uuid,
                           "cpuset": self.cpuset, "memory_fraction": "0.25",
                           "grid_completion": str(self.gate),
                           "grid_completion_sha256": hashlib.sha256(gate_bytes).hexdigest()}
            acquired = True
            return True
        finally:
            if not acquired:
                self.release()

    def mark_launched(self, job, process):
        self.used = True
        self.record.update(policy=job["name"], pid=process.pid)

    def verify_gate(self):
        if self.used and file_hash(self.gate) != self.record["grid_completion_sha256"]:
            raise ValueError("Grid completion marker changed after overflow policy launch")

    def release(self):
        for stream in reversed(self._locks):
            stream.close()
        self._locks.clear()


def run_jobs(jobs, lanes, output, *, validate=None, immutable_check=None, overflow=None):
    """One scheduler owns dequeue; prefer free primary lanes over optional overflow."""
    pending, active, completed = list(jobs), {}, []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    primary_count = len(lanes)
    resources = {index: {"cpuset": cpuset, "overflow": False} for index, cpuset in enumerate(lanes)}
    if overflow is not None:
        resources[primary_count] = {"cpuset": overflow.cpuset, "overflow": True, "gpu": overflow.gpu}

    def launch(lane):
        job = pending[0]
        destination = Path(job["destination"])
        if destination.exists():
            raise FileExistsError(f"Preserving existing complete or partial output: {destination}")
        resource = resources[lane]
        env = os.environ.copy()
        env.update(job.get("environment", {}))
        env.update(RLDT_CPUSET=resource["cpuset"], LP_NUM_THREADS=str(len(resource["cpuset"].split(","))))
        if resource["overflow"]:
            env.update(CUDA_VISIBLE_DEVICES=resource["gpu"], XLA_PYTHON_CLIENT_MEM_FRACTION="0.25")
        command = ["taskset", "-c", resource["cpuset"], *job["command"]]
        log = (output/f"{job['name']}.log").open("x")
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        active[lane] = (process, log, job, command)
        pending.pop(0)
        if resource["overflow"]:
            overflow.mark_launched(job, process)
            write(output/"overflow_policy.json", overflow.record)

    def reap():
        for lane, (process, log, job, _) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            if code:
                raise RuntimeError(f"Intervention child {job['name']} exited {code}; preserving its log and outputs")
            if validate:
                validate(job)
            del active[lane]
            completed.append(job["name"])
            if resources[lane]["overflow"]:
                overflow.release()

    def fill_primary_lanes():
        for lane in range(primary_count):
            if pending and lane not in active:
                launch(lane)

    try:
        while pending or active:
            if immutable_check:
                immutable_check()
            if overflow is not None:
                overflow.verify_gate()
            reap()
            fill_primary_lanes()
            if pending and overflow is not None and not overflow.used and overflow.try_acquire():
                # A primary can finish during lock acquisition. Poll it again
                # before consuming the one pending fourth-policy job.
                reap()
                if any(lane not in active for lane in range(primary_count)):
                    overflow.release()
                    fill_primary_lanes()
                else:
                    launch(primary_count)
            write(output/"queue_status.json", {"state": "running", "pending": [j["name"] for j in pending],
                  "completed": completed, "active": [{"name": j["name"], "pid": p.pid,
                  "cpuset": resources[lane]["cpuset"], "overflow": resources[lane]["overflow"],
                  "gpu": resources[lane].get("gpu", j.get("environment", {}).get("CUDA_VISIBLE_DEVICES")),
                  "command": command} for lane, (p, _, j, command) in active.items()]})
            if active:
                time.sleep(1)
        if immutable_check:
            immutable_check()
        if overflow is not None:
            overflow.verify_gate()
        return completed
    finally:
        stop_children([process for process, _, _, _ in active.values()])
        for _, log, _, _ in active.values():
            log.close()
        if overflow is not None:
            overflow.release()


def policy_jobs(args, policies, selection, output, *, compiler_profile=None, baseline_profile=None):
    jobs = []
    for policy, checkpoint in policies.items():
        destination = output/policy
        command = [str(ROOT/"third_party/openpi/.venv/bin/python"),
            str(ROOT/"scripts/run_vla_trace_interventions.py"),
            "--config", str(args.run/"tasks"/args.task/"test.toml"), "--selection", str(selection),
            "--token-metadata", str(args.run/"gate_b_actual/metadata.json"),
            "--output", str(destination), "--episodes", "10", "--record-episodes", "10",
            "--seed", "100000", "--conditions", ",".join(CONDITIONS)]
        if checkpoint:
            command += ["--checkpoint", str(checkpoint)]
        compiler_contract = None
        environment = {"CUDA_VISIBLE_DEVICES": str(args.gpu),
                       "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20" if getattr(args, "workers", 3) == 4 else "0.25", "RL_VLA_OBJECT_TELEMETRY": "1"}
        if compiler_profile is not None:
            profile = compiler_profile if checkpoint else baseline_profile
            if profile is None:
                raise ValueError("The untouched baseline requires its separate native compiler profile")
            env, compiler_contract = child_environment(profile,
                Path(profile["manifest"]["isolated_cache_root"])/"causal"/args.task/policy)
            environment.update({key: env[key] for key in compiler_contract["environment"]})
            contract_path = output/"compiler_contracts"/f"{policy}.json"
            write(contract_path, compiler_contract)
            command += ["--compiler-profile", profile["manifest_path"],
                        "--compiler-contract", str(contract_path)]
        jobs.append({"name": policy, "checkpoint": str(checkpoint) if checkpoint else "pi05_libero",
                     "destination": str(destination), "command": command,
                     "environment": environment, "compiler_profile": compiler_contract})
    return jobs


def validate_policy(job, expected_ids, expected_seeds, selection_sha):
    destination = Path(job["destination"])
    summary = read(destination/"summary.json")
    if not summary or not summary.get("completed") or summary["case_plan"]["technical_smoke"]:
        raise ValueError("Intervention policy did not complete research evaluation")
    plan = summary["case_plan"]
    if job.get("compiler_profile") is not None and plan.get("compiler_profile") != job["compiler_profile"]:
        raise ValueError("Intervention child compiler profile differs from parent contract")
    if (plan["conditions"] != list(CONDITIONS) or plan["state_ids"] != expected_ids or
            plan["seeds"] != expected_seeds or plan["checkpoint"] != job["checkpoint"] or
            plan["selection_sha256"] != selection_sha):
        raise ValueError("Intervention policy plan differs from frozen cases/checkpoint/selection")
    if len(summary["conditions"]) != len(CONDITIONS):
        raise ValueError("Missing intervention condition summaries")
    for condition in CONDITIONS:
        folder = destination/condition.replace(":", "_")
        report = read(folder/"summary.json")
        no_op = read(folder/"no_op.json")
        if not report or report.get("technical_smoke") or not report.get("completed") or not no_op or no_op.get("passed") is not True:
            raise ValueError("Intervention condition incomplete or native no-op failed")
        episodes = report["episodes"]
        if [e["initial_state_id"] for e in episodes] != expected_ids or [e["seed"] for e in episodes] != expected_seeds:
            raise ValueError("Intervention states/noise pairing differs")
        for episode in episodes:
            if (type(episode["success"]) is not bool or episode["checkpoint"] != job["checkpoint"] or
                    episode["selection_sha256"] != selection_sha or episode["condition"] != condition or
                    not 0 < episode["primitive_steps"] <= 280 or not Path(episode["video"]).is_file()):
                raise ValueError("Missing or inconsistent intervention outcome/selection/evidence")


def validate_concurrency_summary(smoke):
    """Validate the rich per-condition results of the mixed-profile smoke."""
    if (not isinstance(smoke, dict) or smoke.get("complete") is not True
            or smoke.get("passed") is not True or smoke.get("workers") != 4
            or smoke.get("memory_fraction_per_worker") != 0.2):
        raise ValueError("Four-worker research requires the passed mixed-profile native concurrency proof")
    rows = smoke.get("conditions")
    if (not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows)
            or [row.get("condition") for row in rows] != list(CONDITIONS)):
        raise ValueError("Four-worker proof must cover all nine ordered conditions")
    reference_conditions = {"none", "no_image", "target:mosaic"}
    for row in rows:
        checks = row.get("native_no_op")
        if (row.get("four_workers_finite") is not True or not isinstance(checks, list)
                or len(checks) != 4 or any(not isinstance(check, dict)
                    or check.get("passed") is not True or check.get("max_abs_difference") != 0.0
                    or check.get("persisted_noise_recurrence_max_abs_difference") != 0.0
                    for check in checks)):
            raise ValueError("Four-worker proof requires finite, exact native and explicit-noise checks")
        if (row["condition"] in reference_conditions
                and row.get("baseline_bit_exact_to_previous_three_worker_reference") is not True):
            raise ValueError("Four-worker baseline differs from its original native reference")


def run_interventions(args):
    startup_sources = queue_source_hashes("interventions")
    concurrency_paths = []
    if args.workers == 4:
        directory = args.run/"intervention_trained_concurrency_profile_v3"
        smoke = read(directory/"summary.json")
        validate_concurrency_summary(smoke)
        concurrency_paths = [directory/"summary.json", directory/"plan.json", directory/"release_verification.json"]
        release = read(concurrency_paths[-1])
        if not release or release.get("physical_gpu_lock_acquired_after_smoke") is not True or release.get("live_worker_pids") != []:
            raise ValueError("Four-worker technical processes have not been certified released")
        smoke_sources = read(directory/"plan.json")["source_input_sha256"]
        for path in (ROOT/"scripts/run_vla_trace_interventions.py", ROOT/"src/rl_vla/diagnostics/pi05_attention.py",
                     ROOT/"src/rl_vla/diagnostics/interventions.py", ROOT/"src/rl_vla/diagnostics/compiler_dispatch.py"):
            if smoke_sources.get(str(path)) != file_hash(path):
                raise ValueError(f"Four-worker numerical source changed: {path}")
        case_plans = sorted(directory.glob("worker_*/case_plan.json"))
        if len(case_plans) != 4:
            raise ValueError("Missing four-worker child source contracts")
        for path in case_plans:
            if any(file_hash(source) != expected for source, expected in read(path)["source_sha256"].items()):
                raise ValueError("Four-worker child source or input changed")
        concurrency_paths += case_plans
    task = args.task
    method = args.run/"methods/rldt"/task
    selection = method/"selection.json"
    wait_file(selection, failure_path=method/"status.json")
    audit = wait_file(method/"checkpoint_audit.json", failure_path=method/"status.json")
    checkpoint_files = checkpoint_contract(method, audit)
    moving_mask_gate = wait_file(args.run/"moving_mask_alignment_gate.json")
    if not moving_mask_gate.get("passed"):
        raise ValueError("Moving-frame native observation/mask fidelity gate failed")
    profiles = profiles_for(args)
    if args.workers == 4:
        for kind, profile in profiles.items():
            expected = smoke["compiler_profiles"][kind]
            if any(expected.get(key) != profile[key] for key in ("manifest_path", "manifest_sha256", "profile_sha256", "helper_sha256")):
                raise ValueError("Research compiler profile differs from verified four-worker launch")
    fidelity = {}
    while len(fidelity) != 2:
        verify_queue_sources(startup_sources)
        for kind, profile in profiles.items():
            proof = policy_fidelity_contract(args.run, task, profile)
            if proof is not None: fidelity[kind] = proof
        if len(fidelity) != 2:
            failure = read(method/"status.json")
            if failure and failure.get("state") == "failed":
                raise RuntimeError("Upstream task failed while awaiting numerical fidelity")
            time.sleep(30)
    output = args.run/"interventions"/task
    output.mkdir(parents=True, exist_ok=True)
    selection_sha = file_hash(selection)
    plan = read(args.run/"tasks"/task/"protocol.json")
    expected_ids = plan["tasks"][task]["test_state_ids"][:10]
    expected_seeds = list(range(100000, 100010))
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", args.gpu,
                    "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    lock_path = ROOT/"runs/.vla_trace_gpu_locks"/f"{gpu_uuid}.lock"
    lock_path.parent.mkdir(exist_ok=True)
    policies = {"pi05_libero": None, **{f"rldt_{i}": method/"training"/f"iteration_{i:06d}" for i in (25, 50, 100)}}
    jobs = policy_jobs(args, policies, selection, output, compiler_profile=profiles["trained"], baseline_profile=profiles["baseline"])
    lanes = cpu_lanes(args.cpuset, args.workers)
    overflow = GridReleasedOverflow(args.run, task)
    if args.gpu == overflow.gpu:
        raise ValueError("Primary causal GPU must be distinct from reserved GPU3 overflow")
    input_paths = [selection, method/"checkpoint_audit.json", args.run/"tasks"/task/"protocol.json", args.run/"tasks"/task/"test.toml",
                   args.run/"gate_b_actual/metadata.json", args.run/"moving_mask_alignment_gate.json",
                   Path(__file__), ROOT/"scripts/run_vla_trace_interventions.py",
                   ROOT/"src/rl_vla/diagnostics/pi05_attention.py", ROOT/"src/rl_vla/diagnostics/interventions.py",
                   ROOT/"src/rl_vla/diagnostics/compiler_profile.py", ROOT/"src/rl_vla/diagnostics/trained_fidelity.py",
                   ROOT/"src/rl_vla/diagnostics/compiler_profile_baseline.py", ROOT/"src/rl_vla/diagnostics/compiler_dispatch.py",
                   *[Path(profile[key]) for profile in profiles.values() for key in ("manifest_path", "profile_path")],
                   *concurrency_paths,
                   *sorted((output/"compiler_contracts").glob("*.json"))]
    hashes = {str(path): file_hash(path) for path in input_paths}
    def immutable_check():
        verify_queue_sources(startup_sources)
        if any(file_hash(path) != expected for path, expected in hashes.items()):
            raise ValueError("Intervention source/input/selection changed after queue launch")
        verify_checkpoint_contract(checkpoint_files)
        for proof in fidelity.values(): verify_task_fidelity(proof)
    write(output/"queue_plan.json", {"jobs": jobs, "cpu_lanes": lanes, "max_workers": 4,
          "max_primary_workers": args.workers,
          "overflow": {"gpu": overflow.gpu, "cpuset": overflow.cpuset, "max_policies_per_task": 1,
                       "max_models_on_shared_gpu": 3, "memory_fraction": "0.25",
                       "requires": str(overflow.gate), "mode": "optional nonblocking shared GPU lock plus exclusive task slot"} if args.workers == 3 else None,
          "input_sha256": {**hashes, **startup_sources}, "checkpoint_files": checkpoint_files,
          "compiler_profiles": {kind: profile["manifest_path"] for kind, profile in profiles.items()}, "policy_fidelity": fidelity,
          "states": expected_ids, "seeds": expected_seeds,
          "policy_scheduling": "Four primary lanes, one fresh process per policy" if args.workers == 4 else "Three primary lanes; optional fourth on released grid GPU"})
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_jobs(jobs, lanes, output, immutable_check=immutable_check,
                 validate=lambda job: validate_policy(job, expected_ids, expected_seeds, selection_sha),
                 overflow=overflow if args.workers == 3 else None)
    verify_checkpoint_contract(checkpoint_files, full_hash=True)
    write(output/"queue_complete.json", {"complete": True, "passed": True, "technical_only": False,
          "task": task, "policies": list(policies), "conditions": list(CONDITIONS),
          "episodes_per_condition": 10, "selection_sha256": selection_sha,
          "max_concurrent_policies": 4 if overflow.used else args.workers, "max_primary_policies": args.workers,
          "overflow_policy": overflow.record if overflow.used else None,
          "cpu_lanes": lanes, "input_sha256": {**hashes, **startup_sources},
          "checkpoint_files": checkpoint_files, "checkpoint_hashes_verified_start_and_final": True,
          "compiler_profiles": {kind: profile["manifest_path"] for kind, profile in profiles.items()}, "policy_fidelity": fidelity,
          "scope": "Predeclared first10 frozen test cases; exploratory causal study"})
    write(output/"queue_status.json", {"state": "completed"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["grid", "interventions", "rollout"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=["object_0", "object_1", "object_2"])
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpuset", required=True)
    parser.add_argument("--compiler-profile", type=Path, required=True)
    parser.add_argument("--baseline-compiler-profile", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3, choices=[3, 4])
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.mode == "interventions" and not args.task:
        parser.error("interventions requires a task")
    def terminate(signum, frame):
        raise SystemExit(f"Received signal {signum}; stopping owned diagnostic children")
    signal.signal(signal.SIGTERM, terminate)
    try:
        {"grid": run_grid, "interventions": run_interventions, "rollout": run_rollout}[args.mode](args)
    except BaseException as exc:
        destination = args.run/"interventions"/args.task if args.task else args.run
        write(destination/"queue_status.json", {"state": "failed", "error": repr(exc), "mode": args.mode})
        raise
