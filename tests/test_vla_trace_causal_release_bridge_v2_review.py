"""Version2 actual stat-bearing bindings plus independent release regressions."""

import copy
import fcntl
import json
from pathlib import Path

import pytest

from rl_vla.diagnostics import causal_release_bridge_v2 as bridge

UUID = "GPU-ffd52ae9-e4f5-4012-abc6-9dd9f5200e8d"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return bridge.record(path)


@pytest.fixture
def case(tmp_path):
    run = tmp_path / "repo/runs/campaign"
    canonical = run / "interventions/object_1/queue_complete.json"
    canonical.parent.mkdir(parents=True)
    base = run / "compiler_profile_integration/causal_structural_na_v1"
    source = tmp_path / "producer.py"
    source.write_text("# frozen test producer\n")
    validator = tmp_path / "validator.py"
    validator.write_text("# frozen test validator\n")
    amendment = save(base / "amendment.json", {"approved_for_execution": True})
    reuse = save(base / "reuse.json", {"completed_episodes": 256, "whole_conditions_complete": 23})
    proof = save(
        base / "proof.json",
        {"passed": True, "complete": True, "gpu_uuid": UUID, "device": {"uuid": UUID, "index": 6}},
    )
    native_key = save(base / "native_key_recurrence.json", {"passed": True})
    root_value = {
        "approved": True,
        "passed": True,
        "amendment": amendment,
        "reuse_manifest": reuse,
        "resume_proof": proof,
        "producer_sources": {str(source): bridge.sha(source)},
        "source_input_sha256": {str(source): bridge.sha(source)},
        "execution_validator_source": bridge.record(validator),
        "native_key_recurrence": native_key,
    }
    root = save(base / "root_acceptance.json", root_value)
    raw = run / "interventions_structural_na_v1/object_1"
    parent = {"pid": 5000, "start_ticks": 50, "pgid": 5000, "ppid": 1, "state": "S"}
    exits, descendants, journals, policies, groups, cases, conditions = [], [], [], [], [], [], []
    for number, policy in enumerate(bridge.POLICIES):
        child = {
            "pid": 5001 + number,
            "start_ticks": 51 + number,
            "pgid": 5001 + number,
            "ppid": 5000,
            "state": "S",
        }
        exits.append(
            {**child, "policy": policy, "returncode": 0, "reaped": True, "process_group_gone": True}
        )
        descendant = {
            "pid": 6001 + number,
            "start_ticks": 61 + number,
            "pgid": 6001 + number,
            "ppid": child["pid"],
            "state": "S",
        }
        descendants.append(descendant)
        groups.extend((child["pgid"], descendant["pgid"]))
        journal = raw / policy / "owned_processes.jsonl"
        journal.parent.mkdir(parents=True)
        journal.write_text(json.dumps(descendant) + "\n")
        journals.append(bridge.record(journal))
        na = int(policy == "rldt_100")
        policies.append(
            {
                "policy": policy,
                "summary": save(raw / policy / "summary.json", {"complete": True}),
                "native_completed": 90 - na,
                "structural_NA": na,
                "resolved": 90,
            }
        )
        for condition in bridge.CONDITIONS:
            cell_na = int(na and condition == "random:mosaic")
            conditions.append(
                {
                    "policy": policy,
                    "condition": condition,
                    "planned_cases": 10,
                    "native_completed": 10 - cell_na,
                    "structural_NA": cell_na,
                    "total_resolved": 10,
                    "primary_outcome_available": not cell_na,
                }
            )
            for i in range(10):
                unavailable = bool(cell_na and i == 9)
                cases.append(
                    {
                        "task": "object_1",
                        "policy": policy,
                        "condition": condition,
                        "episode": i,
                        "state_id": 150 + i,
                        "seed": 100000 + i,
                        "status": "structural_NA" if unavailable else "completed",
                        "row": {"success": None if unavailable else True},
                    }
                )
    counts = {
        "planned_cases": 360,
        "native_completed": 359,
        "structural_NA": 1,
        "total_resolved": 360,
    }
    plan = save(
        raw / "queue_plan.json",
        {
            "parent": parent,
            "task": "object_1",
            "original_parent": 363111,
            "states": list(range(150, 160)),
            "seeds": list(range(100000, 100010)),
            "conditions": bridge.CONDITIONS,
            "planned_cases": 360,
            "amendment": amendment,
            "reuse_manifest": reuse,
            "resume_proof": proof,
            "source_input_sha256": {str(source): bridge.sha(source)},
        },
    )
    completion = {
        "complete": True,
        "passed": True,
        "execution_coverage_complete": True,
        "canonical_completion_published": False,
        "parent": parent,
        "child_exits": exits,
        "owned_process_groups": groups,
        "child_process_journals": journals,
        "observed_descendants": descendants,
        "coverage": counts,
        "policy_summaries": policies,
        "queue_plan": plan,
        "reuse_manifest": reuse,
        "amendment": amendment,
        "source_input_sha256": {str(source): bridge.sha(source)},
    }

    def with_identity(item):
        stat = Path(item["path"]).stat()
        return {**item, "identity": [stat.st_size, stat.st_mtime_ns, stat.st_ino]}

    actual_plan = json.loads(Path(plan["path"]).read_text())
    for name in ("amendment", "reuse_manifest"):
        completion[name] = with_identity(completion[name])
        actual_plan[name] = completion[name]
    completion["queue_plan"] = with_identity(save(Path(plan["path"]), actual_plan))
    for item in policies:
        item["summary"] = with_identity(item["summary"])
    completion["child_process_journals"] = [with_identity(item) for item in journals]
    completion_record = save(raw / "queue_complete.json", completion)
    details = save(
        base / "execution_cases.json",
        {
            "schema": "causal-structural-execution-cases-v1",
            "complete": True,
            "task": "object_1",
            "cases": cases,
            "native_artifacts": [],
            "validator_source": bridge.record(validator),
        },
    )
    coverage = {
        "schema": "causal-structural-execution-coverage-v1",
        "complete": True,
        "passed": True,
        "task": "object_1",
        "queue_completion": completion_record,
        "root_acceptance": root,
        "amendment": amendment,
        "validator_source": bridge.record(validator),
        "native_key_recurrence": native_key,
        "coverage": counts,
        "policy_summaries": policies,
        "case_records": details,
        "conditions": conditions,
        "preserved_native_cases": 256,
        "preserved_whole_conditions": 23,
    }
    coverage_path = base / "execution_coverage.json"
    save(coverage_path, coverage)
    old_owner = {"pid": 363111, "proc_start_ticks": 30}
    release_plan = save(
        run / "ood_per_lane_release_review_v1/release_plans/object_1.json",
        {
            "resources": [
                {
                    "role": "causal",
                    "gpu": "6",
                    "cpuset": "8-15",
                    "parent": old_owner,
                    "observed_processes": [{"pid": 363112, "proc_start_ticks": 31}],
                }
            ]
        },
    )
    manifest = save(
        run / "ood/object_1/input_manifest.json",
        {"inputs": {release_plan["path"]: release_plan["sha256"]}},
    )
    approval = save(
        base / "bridge_acceptance.json",
        {
            "approved": True,
            "passed": True,
            "bridge_sources": {str(Path(bridge.__file__).resolve()): bridge.sha(bridge.__file__)},
            "root_acceptance": root,
            "frozen_ood_release_plan": release_plan,
            "live_ood_input_manifest": manifest,
            "gpu_uuid": UUID,
        },
    )
    return {
        "run": run,
        "approval": Path(approval["path"]),
        "completion": completion,
        "completion_path": Path(completion_record["path"]),
        "coverage": coverage,
        "coverage_path": coverage_path,
        "canonical": canonical,
        "root": root_value,
    }


def publish(case, **overrides):
    defaults = {
        "table_reader": dict,
        "group_reader": dict,
        "uuid_reader": lambda _: UUID,
        "gpu_reader": lambda _: [],
    }
    return bridge.publish(case["run"], case["approval"], **(defaults | overrides))


def test_actual_nested_schema_publishes_resource_only(case):
    assert bridge.validate_inputs(case["run"], case["approval"])
    assert publish(case)
    marker = json.loads(case["canonical"].read_text())
    assert marker["resource_release_only"] is True
    assert marker["execution_coverage"] == {
        "planned_cases": 360,
        "native_completed": 359,
        "structural_NA": 1,
        "total_resolved": 360,
    }
    assert marker["gpu_uuid"] == UUID
    assert marker["superseded_failed_parent"]["pid"] == 363111
    assert marker["exclusive_gpu_lease_held_at_publication"] is True


def test_absent_independent_coverage_remains_pending(case):
    case["coverage_path"].unlink()
    assert not publish(case)
    assert not case["canonical"].exists()


@pytest.mark.parametrize("pid", [5000, 5001, 6001, 363111, 363112])
def test_any_current_or_previous_owner_blocks_release(case, pid):
    ticks = {5000: 50, 5001: 51, 6001: 61, 363111: 30, 363112: 31}[pid]
    assert not publish(case, table_reader=lambda: {pid: {"proc_start_ticks": ticks, "state": "S"}})
    assert not case["canonical"].exists()


def test_pid_reuse_does_not_deadlock_resource_release(case):
    assert publish(case, table_reader=lambda: {5001: {"proc_start_ticks": 999, "state": "S"}})


def test_owned_group_gpu_process_and_gpu_lock_each_block(case):
    assert not publish(case, group_reader=lambda: {6001: [7001]})
    assert not publish(case, gpu_reader=lambda _: [7001])
    directory = case["run"].parents[1] / "runs/.vla_trace_gpu_locks"
    directory.mkdir(exist_ok=True)
    with (directory / f"{UUID}.lock").open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not publish(case)
    assert not case["canonical"].exists()


def test_existing_canonical_marker_never_overwritten(case):
    original = b'{"prior":"preserve"}\n'
    case["canonical"].write_bytes(original)
    with pytest.raises(FileExistsError):
        publish(case)
    assert case["canonical"].read_bytes() == original


def test_post_lock_owner_reappearance_blocks_publication(case):
    calls = []

    def table():
        calls.append(1)
        return {} if len(calls) == 1 else {6001: {"proc_start_ticks": 61, "state": "S"}}

    assert not publish(case, table_reader=table)
    assert not case["canonical"].exists()


def test_post_lock_bound_artifact_mutation_rejected(case):
    calls = []

    def table():
        calls.append(1)
        if len(calls) == 2:
            case["coverage_path"].write_text("{}")
        return {}

    with pytest.raises(ValueError, match="Changed"):
        publish(case, table_reader=table)
    assert not case["canonical"].exists()


@pytest.mark.parametrize(
    "field",
    [
        "policy_summary",
        "case_records",
        "native_key",
        "journal",
        "validator",
        "source",
        "release_plan",
        "manifest",
    ],
)
def test_bound_input_mutations_fail_closed(case, field):
    coverage = case["coverage"]
    completion = case["completion"]
    approval = json.loads(case["approval"].read_text())
    target = {
        "policy_summary": completion["policy_summaries"][0]["summary"]["path"],
        "case_records": coverage["case_records"]["path"],
        "native_key": coverage["native_key_recurrence"]["path"],
        "journal": completion["child_process_journals"][0]["path"],
        "validator": coverage["validator_source"]["path"],
        "source": next(iter(completion["source_input_sha256"])),
        "release_plan": approval["frozen_ood_release_plan"]["path"],
        "manifest": approval["live_ood_input_manifest"]["path"],
    }[field]
    Path(target).write_text("{}")
    with pytest.raises((ValueError, TypeError)):
        publish(case)
    assert not case["canonical"].exists()


@pytest.mark.parametrize(
    "mutation",
    ["counts", "duplicate_policy", "child_failure", "missing_descendant", "missing_group"],
)
def test_rehashed_completed_marker_cannot_weaken_coverage_or_exit_chain(case, mutation):
    completion = copy.deepcopy(case["completion"])
    coverage = copy.deepcopy(case["coverage"])
    if mutation == "counts":
        completion["coverage"]["native_completed"] = 360
        coverage["coverage"] = copy.deepcopy(completion["coverage"])
    elif mutation == "duplicate_policy":
        completion["policy_summaries"][1] = copy.deepcopy(completion["policy_summaries"][0])
        coverage["policy_summaries"] = copy.deepcopy(completion["policy_summaries"])
    elif mutation == "child_failure":
        completion["child_exits"][0]["returncode"] = 1
    elif mutation == "missing_descendant":
        completion["observed_descendants"] = []
    else:
        completion["owned_process_groups"].remove(6001)
    coverage["queue_completion"] = save(case["completion_path"], completion)
    save(case["coverage_path"], coverage)
    with pytest.raises(ValueError):
        publish(case)
    assert not case["canonical"].exists()


def test_wrong_physical_gpu_uuid_rejected(case):
    with pytest.raises(ValueError, match="Physical GPU6"):
        publish(case, uuid_reader=lambda _: "GPU-another-device")
    assert not case["canonical"].exists()


@pytest.mark.parametrize("mutation", ["state", "duplicate", "status"])
def test_rehashed_case_records_cannot_change_plan_or_status_counts(case, mutation):
    coverage = copy.deepcopy(case["coverage"])
    path = Path(coverage["case_records"]["path"])
    details = json.loads(path.read_text())
    if mutation == "state":
        details["cases"][0]["state_id"] = 159
    elif mutation == "duplicate":
        details["cases"][0] = copy.deepcopy(details["cases"][1])
    else:
        details["cases"][0]["status"] = "structural_NA"
    coverage["case_records"] = save(path, details)
    save(case["coverage_path"], coverage)
    with pytest.raises(ValueError):
        publish(case)
    assert not case["canonical"].exists()


def test_actual_identity_bearing_records_match_compact_root_acceptance(case):
    completion = case["completion"]
    root = json.loads(Path(case["coverage"]["root_acceptance"]["path"]).read_text())
    for name in ("amendment", "reuse_manifest"):
        assert "identity" in completion[name] and "identity" not in root[name]
        assert bridge.same_binding(completion[name], root[name])
    assert bridge.validate_inputs(case["run"], case["approval"])
    assert publish(case)
    assert (
        json.loads(case["canonical"].read_text())["version"] == "strict-support-resource-release-v2"
    )


@pytest.mark.parametrize("name", ["amendment", "reuse_manifest"])
@pytest.mark.parametrize("mutation", ["size", "mtime", "inode", "alternate_path", "wrong_sha"])
def test_actual_optional_stats_and_binding_mutations_fail_closed(case, name, mutation):
    completion = copy.deepcopy(case["completion"])
    coverage = copy.deepcopy(case["coverage"])
    item = completion[name]
    if mutation in ("size", "mtime", "inode"):
        item["identity"][("size", "mtime", "inode").index(mutation)] += 1
    elif mutation == "alternate_path":
        original = Path(item["path"])
        alternate = original.with_name("substituted_" + original.name)
        alternate.write_bytes(original.read_bytes())
        stat = alternate.stat()
        item.update(
            path=str(alternate),
            sha256=bridge.sha(alternate),
            identity=[stat.st_size, stat.st_mtime_ns, stat.st_ino],
        )
    else:
        item["sha256"] = "0" * 64
    coverage["queue_completion"] = save(case["completion_path"], completion)
    save(case["coverage_path"], coverage)
    with pytest.raises(ValueError):
        publish(case)
    assert not case["canonical"].exists()
