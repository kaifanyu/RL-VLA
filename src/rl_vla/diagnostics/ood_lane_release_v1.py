"""Read-only resource release and provenance gates for isolated OOD scheduling."""

import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

from rl_vla.diagnostics.ood_evaluation import VARIANTS, read_json

VERSION = "per_lane_release_v1"
REPO = Path(__file__).resolve().parents[3]
GPU6_ORACLE_SHA256 = "d217c613a7d6361348e97b6fa7362daa2401fdf64de338537603c85429da497e"
RESOURCES = {
    "object_0": (("main", "0", "24-35"), ("causal", "4", "0-7"), ("trace", "5", "68-71,82-85")),
    "object_1": (("main", "1", "36-47"), ("causal", "6", "8-15")),
    "object_2": (("main", "2", "56-67"), ("causal", "7", "16-23"), ("trace", "3", "48-55")),
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def cpus(text):
    result = []
    for token in text.split(","):
        parts = [int(x) for x in token.split("-")]
        result.extend(range(parts[0], parts[-1] + 1))
    return result


def scientific_readiness(campaign, task):
    campaign = Path(campaign)
    method = campaign / "methods/rldt" / task
    paths = {"selection": method / "selection.json", "main": method / "status.json",
             "checkpoint_audit": method / "checkpoint_audit.json",
             "kinematics": campaign / "diagnostic_layouts" / task / "kinematic_audit.json",
             "layouts": campaign / "diagnostic_layouts" / task / "summary.json"}
    data = {name: read_json(path) for name, path in paths.items()}
    missing = [name for name, value in data.items() if value is None]
    main = data["main"]
    if main:
        if main.get("state") == "failed":
            raise RuntimeError("Main experiment failed")
        if (main.get("stage"), main.get("state")) != ("action_evaluations_complete", "completed"):
            missing.append("main_action_evaluations_complete")
    selection = data["selection"]
    if selection and (selection.get("iteration") not in (25, 50, 100) or not selection.get("locked")
                      or selection.get("test_results_seen") is not False):
        raise ValueError("Expected locked pre-test checkpoint selection")
    for name in ("checkpoint_audit", "kinematics"):
        if data[name] is not None and data[name].get("passed") is not True:
            missing.append(name + "_passed")
    layout = data["layouts"]
    if layout is not None:
        if layout.get("complete") is not True:
            missing.append("layouts_complete")
        elif (layout.get("task") != task or layout.get("variants") != list(VARIANTS)
              or layout.get("states_per_variant") != 50):
            raise ValueError("Expected exactly four frozen fifty-state OOD variants")
    return {"ready": not missing, "waiting_for": sorted(set(missing)),
            "gate_paths": {name: str(path.resolve()) for name, path in paths.items()}}


def process_table():
    """Snapshot process identities, retaining zombie identity but treating it as exited."""
    result = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
            command = (directory / "cmdline").read_text().split("\0")[:-1]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        pid = int(directory.name)
        result[pid] = {"pid": pid, "proc_start_ticks": int(fields[19]), "ppid": int(fields[1]),
                       "state": fields[0], "command": command}
    return result


def same_process(identity, table):
    current = table.get(identity["pid"])
    return bool(current and current["proc_start_ticks"] == identity["proc_start_ticks"]
                and current["state"] != "Z")


class ProcessTracker:
    def __init__(self, parent, observed, reader=process_table):
        self.reader = reader
        self.parent = parent
        self.observed = {(item["pid"], item["proc_start_ticks"]): item for item in [parent, *observed]}

    def observe(self):
        table = self.reader()
        frontier = {item["pid"] for item in self.observed.values() if same_process(item, table)}
        while frontier:
            children = {pid for pid, item in table.items() if item["ppid"] in frontier
                        and (pid, item["proc_start_ticks"]) not in self.observed}
            for pid in children:
                item = table[pid]
                self.observed[(pid, item["proc_start_ticks"])] = item
            frontier = children
        return table

    def released(self):
        table = self.observe()
        return not any(same_process(item, table) for item in self.observed.values())


def marker_for(campaign, task, role):
    if role == "main":
        return campaign / "methods/rldt" / task / "status.json"
    if role == "causal":
        return campaign / "interventions" / task / "queue_complete.json"
    return campaign / ("rollout_trace_queue_complete.json" if task == "object_0"
                       else "attention_grid_queue_complete.json")


def validate_marker(resource, table_reader=process_table):
    data = read_json(resource["release_gate"])
    if data is None:
        return False
    role = resource["role"]
    if role == "main":
        if data.get("state") == "failed":
            raise RuntimeError("Main evaluation owner failed")
        return (data.get("stage"), data.get("state")) == ("action_evaluations_complete", "completed")
    if data.get("complete") is not True:
        return False
    if data.get("passed") is False:
        raise RuntimeError("Resource owner published failed completion")
    if role == "causal":
        if (data.get("passed") is not True or data.get("technical_only") is not False
                or data.get("task") != resource["task"] or data.get("episodes_per_condition") != 10
                or data.get("policies") != ["pi05_libero", "rldt_25", "rldt_50", "rldt_100"]
                or data.get("conditions") != ["none", "no_image", "no_text", "target:background_fill",
                   "distractor:background_fill", "random:background_fill", "target:mosaic",
                   "distractor:mosaic", "random:mosaic"]):
            raise ValueError("Expected the complete research causal queue for this task")
        return True
    if data.get("tasks") != ["object_0", "object_1", "object_2"]:
        raise ValueError("Trace release must cover all three tasks")
    jobs = data.get("jobs", [])
    children = data.get("child_exits", [])
    if not jobs or children != [job.get("child_exit") for job in jobs]:
        raise ValueError("Trace release children differ from completed jobs")
    if resource["task"] == "object_0":
        if (data.get("passed") is not True or data.get("policies") !=
                ["pi05_libero", "rldt_000025", "rldt_000050", "rldt_000100"]
                or [job.get("stage") for job in jobs] != ["baseline", "trained"]
                or data.get("child_exit") != children[-1]):
            raise ValueError("Rollout release lacks the complete two-stage four-policy trace")
    else:
        expected = {(task, stage) for task in ("object_0", "object_1", "object_2")
                    for stage in ("baseline", "trained", "baseline_projection", "aggregate")}
        if (data.get("count_per_task") != 50 or len(jobs) != 12
                or {(job.get("task"), job.get("stage")) for job in jobs} != expected
                or data.get("child_exit") != next(job["child_exit"] for job in reversed(jobs)
                                                   if job["stage"] != "aggregate")):
            raise ValueError("Grid release lacks the complete twelve-stage fifty-case trace")
    for child in children:
        if (not isinstance(child, dict) or type(child.get("pid")) is not int or child["pid"] <= 0
                or type(child.get("proc_start_ticks")) is not int or child["proc_start_ticks"] <= 0
                or type(child.get("returncode")) is not int or child["returncode"] != 0):
            raise ValueError("Trace release lacks exact successful child identity")
    table = table_reader()
    return not any(same_process(child, table) for child in children)


def capture_release_plan(campaign, task, main_pid, causal_pid, trace_pid=None):
    table = process_table()
    resources = []
    for role, gpu, cpuset in RESOURCES[task]:
        pid = {"main": main_pid, "causal": causal_pid, "trace": trace_pid}[role]
        if type(pid) is not int or pid not in table or table[pid]["state"] == "Z":
            raise ValueError(f"Capture {role} process identity while its owner is still running")
        parent = table[pid]
        command = " ".join(parent["command"])
        expected = {"main": "run_vla_trace_objects.py", "causal": "interventions", "trace": "queue_vla_trace_diagnostics.py"}[role]
        if expected not in command or str(campaign) not in command or (role != "trace" and task not in command):
            raise ValueError("Resource parent command does not identify the expected campaign/task")
        if role != "main" and f"--gpu {gpu}" not in command:
            raise ValueError("Resource parent command has a different GPU")
        tracker = ProcessTracker(parent, [])
        tracker.observe()
        resources.append({"role": role, "task": task, "gpu": gpu, "cpuset": cpuset, "cpus": cpus(cpuset),
            "release_gate": str(marker_for(campaign, task, role).resolve()), "parent": parent,
            "observed_processes": list(tracker.observed.values())})
    return {"version": VERSION, "campaign": str(campaign), "task": task,
            "helper_sha256": digest(__file__), "resources": resources,
            "scope": "Resource-only parent identities captured before replacement; no result-driven cell selection"}


def validate_release_plan(plan, campaign, task):
    if (plan.get("version") != VERSION or plan.get("campaign") != str(campaign)
            or plan.get("task") != task or plan.get("helper_sha256") != digest(__file__)):
        raise ValueError("Release plan identity/source differs")
    capture = plan.get("process_capture")
    if capture:
        if digest(capture["path"]) != capture["sha256"]:
            raise ValueError("Original pre-exit process capture changed")
        original = json.loads(Path(capture["path"]).read_text())
        if any(plan.get(key) != original.get(key) for key in ("campaign", "task", "resources", "version")):
            raise ValueError("Final release plan changed the captured process/resource identities")
        if capture.get("helper_sha256") != original["helper_sha256"]:
            raise ValueError("Original process capture helper identity differs")
    resources = plan.get("resources", [])
    if [(r["role"], r["gpu"], r["cpuset"]) for r in resources] != list(RESOURCES[task]):
        raise ValueError("Release plan must preserve the predefined physical GPUs and CPU pools")
    for resource in resources:
        if (resource["cpus"] != cpus(resource["cpuset"]) or resource["task"] != task
                or resource["release_gate"] != str(marker_for(campaign, task, resource["role"]).resolve())):
            raise ValueError("Release resource differs from the frozen role contract")
        for item in [resource["parent"], *resource["observed_processes"]]:
            if (type(item.get("pid")) is not int or item["pid"] <= 0
                    or type(item.get("proc_start_ticks")) is not int or item["proc_start_ticks"] <= 0):
                raise ValueError("Release process identity is malformed")
    return resources


def gpu_uuid(gpu):
    result = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not result.startswith("GPU-") or "\n" in result:
        raise ValueError("Expected exactly one physical GPU")
    return result


def gpu_processes(gpu):
    result = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    return [int(line.strip()) for line in result.splitlines() if line.strip()]


class Lane:
    def __init__(self, resource, *, lock_directory=None, table_reader=process_table,
                 uuid_reader=gpu_uuid, gpu_process_reader=gpu_processes):
        self.resource = resource
        self.optional = resource["role"] != "main"
        self.lock_directory = Path(lock_directory) if lock_directory else REPO / "runs/.vla_trace_gpu_locks"
        self.tracker = ProcessTracker(resource["parent"], resource["observed_processes"], reader=table_reader)
        self.table_reader, self.uuid_reader, self.gpu_process_reader = table_reader, uuid_reader, gpu_process_reader
        self.stream = self.uuid = self.release_record = None

    def observe(self):
        if self.stream is None:
            self.tracker.observe()

    def acquire(self):
        if self.stream is not None:
            return True
        self.observe()
        if not validate_marker(self.resource, self.table_reader) or not self.tracker.released():
            return False
        marker = Path(self.resource["release_gate"])
        before = marker.read_bytes()
        self.uuid = self.uuid_reader(self.resource["gpu"])
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        stream = (self.lock_directory / f"{self.uuid}.lock").open("a+")
        try:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            if (marker.read_bytes() != before or not validate_marker(self.resource, self.table_reader)
                    or not self.tracker.released()):
                raise ValueError("Lane release changed while acquiring its GPU lock")
            if self.gpu_process_reader(self.resource["gpu"]):
                return False
            self.release_record = {"path": str(marker.resolve()), "sha256": hashlib.sha256(before).hexdigest(),
                "role": self.resource["role"], "gpu_uuid": self.uuid, "parent": self.resource["parent"],
                "observed_processes": list(self.tracker.observed.values()), "owned_processes_exited": True,
                "gpu_compute_processes_at_acquisition": []}
            self.stream = stream
            return True
        finally:
            if self.stream is not stream:
                stream.close()

    def release(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def verify_cache_environment(contract):
    if (os.environ.get("JAX_COMPILATION_CACHE_DIR") != contract["cache_root"]
            or os.environ.get("XLA_FLAGS") not in (None, "")
            or os.environ.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES") not in (None, "")):
        raise ValueError("Native GPU6 OOD must inherit the proved original cache without diagnostic profile flags")


def gpu6_cache_contract(campaign):
    oracle = Path(campaign) / "compiler_profile_integration/gpu6_native_cache_oracle_v1/summary.json"
    if digest(oracle) != GPU6_ORACLE_SHA256:
        raise ValueError("Native GPU6 cache oracle differs from independently accepted evidence")
    proof = json.loads(oracle.read_text())
    if (proof.get("complete") is not True or proof.get("passed") is not True or proof.get("gpu") != 6
            or proof.get("original_native_cache_compatibility_verified") is not True):
        raise ValueError("Native GPU6 cache oracle did not pass")
    cache = REPO / ".cache/jax-rl-vla"
    files, binaries, identities = {str(oracle): GPU6_ORACLE_SHA256}, {}, []
    for case in proof["cases"]:
        identities.append((case["role"], case["iteration"], case["state_id"], case["decision_index"]))
        if case.get("context_exact") is not True or any(value is not True for value in case["same_as_original_proved_oracle"].values()):
            raise ValueError("GPU6 natural native cache proof includes non-exact evidence")
        files[case["summary"]] = case["summary_sha256"]
        files[case["original_gpu5_oracle_summary"]] = case["original_gpu5_oracle_summary_sha256"]
        for row in case["cached_executables"]:
            path = Path(row["path"]).resolve()
            if path.parent != cache or path.name != row["cache_key"] + "-cache":
                raise ValueError("GPU6 cache binary is not under its original natural key")
            if str(path) in binaries and binaries[str(path)] != row["sha256"]:
                raise ValueError("Conflicting native cache binary identity")
            binaries[str(path)] = row["sha256"]
    expected = {(role, iteration, state, decision) for role, iteration in
                (("baseline", None), ("trained", 25), ("trained", 50), ("trained", 100))
                for state in (100, 101) for decision in (0, 1)}
    if len(identities) != 16 or set(identities) != expected or len(binaries) != 3:
        raise ValueError("GPU6 natural cache proof lacks exact four-policy coverage")
    files.update(binaries)
    for path, sha256 in files.items():
        if digest(path) != sha256:
            raise ValueError(f"Native GPU6 cache provenance changed: {path}")
    return {"cache_root": str(cache), "inputs": files, "binary_sha256": binaries,
            "oracle": str(oracle), "oracle_sha256": GPU6_ORACLE_SHA256,
            "scope": "Original native cache; all three binary hashes checked at freeze, each cell and final completion"}
