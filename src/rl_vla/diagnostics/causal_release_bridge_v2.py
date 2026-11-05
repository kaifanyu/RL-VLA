"""Release a repaired causal resource without changing its frozen OOD plan.

Version2 compares immutable path/SHA bindings independently of optional file-stat
metadata, and verifies that metadata whenever present.

This certificate proves process/resource release. Structural unavailability is
never represented as a native Boolean outcome or a successful old execution.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path

from rl_vla.diagnostics.ood_lane_release_v1 import gpu_processes, gpu_uuid, process_table

POLICIES = ["pi05_libero", "rldt_25", "rldt_50", "rldt_100"]
CONDITIONS = ["none", "no_image", "no_text", "target:background_fill",
              "distractor:background_fill", "random:background_fill", "target:mosaic",
              "distractor:mosaic", "random:mosaic"]
GPU_UUID = "GPU-ffd52ae9-e4f5-4012-abc6-9dd9f5200e8d"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def check_record(value, expected_path=None):
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise TypeError("Missing immutable file record")
    path = Path(value["path"])
    if expected_path is not None and path.resolve() != Path(expected_path).resolve():
        raise ValueError("Certificate references an unexpected path")
    if sha(path) != value.get("sha256"):
        raise ValueError(f"Changed certificate input: {path}")
    if "identity" in value:
        stat = path.stat()
        if value["identity"] != [stat.st_size, stat.st_mtime_ns, stat.st_ino]:
            raise ValueError(f"Changed immutable file identity: {path}")
    return path


def same_binding(left, right):
    return (Path(left["path"]).resolve() == Path(right["path"]).resolve()
            and left["sha256"] == right["sha256"])


def identity(value):
    pid = value.get("pid")
    ticks = value.get("start_ticks", value.get("proc_start_ticks"))
    if type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0:
        raise ValueError("Process identity lacks positive PID/start ticks")
    return {"pid": pid, "proc_start_ticks": ticks}


def live(identity_record, table):
    item = identity(identity_record)
    current = table.get(item["pid"])
    return bool(current and current["proc_start_ticks"] == item["proc_start_ticks"]
                and current["state"] != "Z")


def process_groups():
    result = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if fields[0] != "Z":
            result.setdefault(int(fields[2]), []).append(int(path.name))
    return result


def validate_inputs(run, bridge_acceptance_path):
    """Validate immutable execution coverage; do not infer it from a marker."""
    run = Path(run).resolve()
    approval = read(bridge_acceptance_path)
    if approval.get("approved") is not True or approval.get("passed") is not True:
        raise ValueError("Resource bridge lacks concrete independent acceptance")
    for path, expected in approval["bridge_sources"].items():
        if sha(path) != expected:
            raise ValueError("Reviewed bridge source changed")
    if approval["bridge_sources"].get(str(Path(__file__).resolve())) != sha(__file__):
        raise ValueError("Current bridge helper is not accepted")
    root_path = check_record(approval["root_acceptance"],
                            run / "compiler_profile_integration/causal_structural_na_v1/root_acceptance.json")
    root = read(root_path)
    if root.get("approved") is not True or root.get("passed") is not True:
        raise ValueError("Resume execution is not accepted")
    check_record(root["reuse_manifest"])
    proof = read(check_record(root["resume_proof"]))
    if proof.get("passed") is not True or proof.get("complete") is not True:
        raise ValueError("Accepted resume proof is incomplete")
    if proof.get("device", {}).get("uuid") != GPU_UUID or proof["device"].get("index") != 6:
        raise ValueError("Accepted replay belongs to another physical device")
    amendment_path = check_record(root["amendment"])
    amendment = read(amendment_path)
    if amendment.get("approved_for_execution") is not True:
        raise ValueError("Scientific support amendment is not accepted")
    for path, expected in root["producer_sources"].items():
        if sha(path) != expected:
            raise ValueError("Accepted resume producer changed")
    completion_path = run / "interventions_structural_na_v1/object_1/queue_complete.json"
    coverage_path = run / "compiler_profile_integration/causal_structural_na_v1/execution_coverage.json"
    if not completion_path.exists() or not coverage_path.exists():
        return None
    completion, coverage = read(completion_path), read(coverage_path)
    if (completion.get("complete") is not True or completion.get("passed") is not True
            or completion.get("execution_coverage_complete") is not True
            or completion.get("canonical_completion_published") is not False):
        raise ValueError("Resume queue did not resolve its complete execution plan")
    if (coverage.get("schema") != "causal-structural-execution-coverage-v1"
            or coverage.get("complete") is not True or coverage.get("passed") is not True
            or coverage.get("task") != "object_1"):
        raise ValueError("Independent execution coverage is incomplete")
    check_record(coverage["queue_completion"], completion_path)
    check_record(coverage["root_acceptance"], root_path)
    check_record(coverage["amendment"], amendment_path)
    check_record(coverage["validator_source"])
    if coverage["validator_source"] != root["execution_validator_source"]:
        raise ValueError("Execution coverage came from an unaccepted validator")
    case_path = check_record(coverage["case_records"],
        run / "compiler_profile_integration/causal_structural_na_v1/execution_cases.json")
    cases = read(case_path)
    if (cases.get("schema") != "causal-structural-execution-cases-v1"
            or cases.get("complete") is not True or cases.get("task") != "object_1"
            or cases.get("validator_source") != coverage["validator_source"]
            or len(cases.get("cases", [])) != 360):
        raise ValueError("Coverage lacks its complete independently validated case records")
    key_path = check_record(coverage["native_key_recurrence"])
    if coverage["native_key_recurrence"] != root["native_key_recurrence"]:
        raise ValueError("Coverage uses a different native key recurrence audit")
    counts = coverage["coverage"]
    if (counts != completion["coverage"] or counts.get("planned_cases") != 360
            or counts.get("total_resolved") != 360
            or type(counts.get("native_completed")) is not int
            or type(counts.get("structural_NA")) is not int
            or not 256 <= counts["native_completed"] <= 360
            or counts["structural_NA"] < 0
            or counts["native_completed"] + counts["structural_NA"] != 360):
        raise ValueError("Native and structural counts do not exhaust the planned360 cases")
    expected_cases = {(policy, condition, index, 150 + index, 100000 + index)
                      for policy in POLICIES for condition in CONDITIONS for index in range(10)}
    if {(item["policy"], item["condition"], item["episode"], item["state_id"], item["seed"])
            for item in cases["cases"]} != expected_cases:
        raise ValueError("Independent case identities differ from the original plan")
    if (sum(item.get("status") == "completed" for item in cases["cases"]) != counts["native_completed"]
            or sum(item.get("status") == "structural_NA" for item in cases["cases"]) != counts["structural_NA"]):
        raise ValueError("Independent case statuses differ from resolved counts")
    if coverage["policy_summaries"] != completion["policy_summaries"]:
        raise ValueError("Coverage and producer bind different policy summaries")
    policies = completion["policy_summaries"]
    if len(policies) != 4 or {item["policy"] for item in policies} != set(POLICIES):
        raise ValueError("Completion lacks the four predeclared policies")
    for item in policies:
        check_record(item["summary"], run / "interventions_structural_na_v1/object_1" / item["policy"] / "summary.json")
        if (type(item.get("native_completed")) is not int
                or type(item.get("structural_NA")) is not int
                or item["native_completed"] < 0 or item["structural_NA"] < 0
                or item.get("resolved") != 90
                or item["native_completed"] + item["structural_NA"] != 90):
            raise ValueError("Policy completion does not resolve its original90 cases")
    if (sum(item["native_completed"] for item in policies) != counts["native_completed"]
            or sum(item["structural_NA"] for item in policies) != counts["structural_NA"]):
        raise ValueError("Policy counts differ from independent coverage")
    for name in ("queue_plan", "reuse_manifest", "amendment"):
        check_record(completion[name])
    if not same_binding(completion["amendment"], root["amendment"]):
        raise ValueError("Queue executed a different amendment")
    if not same_binding(completion["reuse_manifest"], root["reuse_manifest"]):
        raise ValueError("Queue reused a different predecessor manifest")
    for path, expected in completion["source_input_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Resumed source/input changed after execution coverage")
    plan = read(completion["queue_plan"]["path"])
    if identity(plan["parent"]) != identity(completion["parent"]):
        raise ValueError("Resume parent differs from its immutable launch plan")
    exits = completion["child_exits"]
    if len(exits) != 4 or {item["policy"] for item in exits} != set(POLICIES):
        raise ValueError("Missing policy child exits")
    for item in exits:
        identity(item)
        if (item.get("returncode") != 0 or item.get("reaped") is not True
                or item.get("process_group_gone") is not True):
            raise ValueError("A resumed worker was not successfully reaped")
    owners = [completion["parent"], *exits, *completion["observed_descendants"]]
    for item in owners:
        identity(item)
    groups = completion["owned_process_groups"]
    if (not isinstance(groups, list) or not groups
            or any(type(group) is not int or group <= 0 for group in groups)
            or not {item["pgid"] for item in exits} <= set(groups)):
        raise ValueError("Separate owned process groups are not fully recorded")
    journals = completion["child_process_journals"]
    expected_journals = {run / "interventions_structural_na_v1/object_1" / policy / "owned_processes.jsonl"
                         for policy in POLICIES}
    if len(journals) != 4 or {Path(item["path"]).resolve() for item in journals} != expected_journals:
        raise ValueError("Completion lacks the four exact child process journals")
    observed_ids = {(identity(item)["pid"], identity(item)["proc_start_ticks"], item["pgid"])
                    for item in completion["observed_descendants"]}
    for item in journals:
        journal = check_record(item)
        for line in journal.read_text().splitlines():
            process = json.loads(line)
            normalized = identity(process)
            pgid = process.get("pgid")
            if (type(pgid) is not int or pgid <= 0 or pgid not in groups
                    or (normalized["pid"], normalized["proc_start_ticks"], pgid) not in observed_ids):
                raise ValueError("A journaled descendant/process group is not covered by release checks")
    plan_path = run / "ood_per_lane_release_review_v1/release_plans/object_1.json"
    check_record(approval["frozen_ood_release_plan"], plan_path)
    live_ood_manifest = run / "ood/object_1/input_manifest.json"
    check_record(approval["live_ood_input_manifest"], live_ood_manifest)
    if read(live_ood_manifest)["inputs"].get(str(plan_path)) != sha(plan_path):
        raise ValueError("Release plan differs from the live OOD input contract")
    old_plan = read(plan_path)
    old = [item for item in old_plan["resources"] if item["role"] == "causal"]
    if len(old) != 1 or old[0]["gpu"] != "6" or old[0]["cpuset"] != "8-15":
        raise ValueError("Frozen OOD plan has a different causal resource")
    if old[0]["parent"]["pid"] != 363111:
        raise ValueError("Bridge does not supersede the reviewed failed owner")
    owners.extend([old[0]["parent"], *old[0]["observed_processes"]])
    return {"completion": completion, "coverage": coverage, "owners": owners,
            "groups": groups, "old_owner": old[0], "bindings": {
                "bridge_acceptance": record(bridge_acceptance_path),
                "root_acceptance": record(root_path), "amendment": record(amendment_path),
                "queue_completion": record(completion_path), "execution_coverage": record(coverage_path),
                "case_records": record(case_path), "native_key_recurrence": record(key_path),
                **{f"child_process_journal_{index}": record(item["path"])
                   for index, item in enumerate(journals)},
                "frozen_ood_release_plan": record(plan_path),
                "live_ood_input_manifest": record(live_ood_manifest)}}


def publish(run, approval, *, table_reader=process_table, group_reader=process_groups,
            uuid_reader=gpu_uuid, gpu_reader=gpu_processes):
    """Return False while owners are alive; publish only while holding GPU6's lock."""
    run = Path(run).resolve()
    data = validate_inputs(run, approval)
    if data is None:
        return False
    table = table_reader()
    if any(live(item, table) for item in data["owners"]):
        return False
    groups = group_reader()
    if any(groups.get(group) for group in data["groups"]):
        return False
    uuid = uuid_reader("6")
    if uuid != GPU_UUID:
        raise ValueError("Physical GPU6 differs from the accepted replay device")
    lock_dir = run.parents[1] / "runs/.vla_trace_gpu_locks"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f"{uuid}.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        # Recheck after acquiring the same physical-device lease used by OOD.
        table, groups = table_reader(), group_reader()
        if (any(live(item, table) for item in data["owners"])
                or any(groups.get(group) for group in data["groups"]) or gpu_reader("6")):
            return False
        for value in data["bindings"].values():
            check_record(value)
        marker = run / "interventions/object_1/queue_complete.json"
        certificate = {"version": "strict-support-resource-release-v2", "complete": True,
            "passed": True, "technical_only": False, "resource_release_only": True,
            "task": "object_1", "policies": POLICIES, "conditions": CONDITIONS,
            "episodes_per_condition": 10, "episode_count_semantics": "Predeclared planned cases; native and structural-NA counts are separate",
            "execution_coverage": data["coverage"]["coverage"],
            "scientific_state": "Resolved strict controls; any condition with structural_NA has unavailable primary outcome/comparison",
            "superseded_failed_parent": data["old_owner"]["parent"],
            "resume_parent": data["completion"]["parent"],
            "child_exits": data["completion"]["child_exits"],
            "observed_descendants": data["completion"]["observed_descendants"],
            "owned_process_groups": data["groups"], "all_recorded_owners_exited": True,
            "all_owned_process_groups_gone": True, "gpu_uuid": uuid,
            "gpu_compute_processes_at_release": [], "exclusive_gpu_lease_held_at_publication": True,
            "bindings": data["bindings"],
            "scope": "Resource-release certificate for an explicitly superseded failed queue; not a claim that the old queue succeeded or that unavailable cases had native outcomes"}
        temporary = marker.with_name(f".resource_release_{os.getpid()}.json")
        with temporary.open("x") as output:
            output.write(json.dumps(certificate, indent=2, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            # Atomic creation which cannot overwrite any existing marker.
            os.link(temporary, marker)
        finally:
            temporary.unlink()
        return True
