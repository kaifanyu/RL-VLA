"""Independent consumer tests using the actual v2 bridge publication schema."""

import json

import pytest
import test_vla_trace_causal_release_bridge_v2_review as fixtures

from rl_vla.diagnostics import causal_release_bridge_v2 as bridge
from rl_vla.diagnostics import ood_completion_support_v2 as wrapper

case = fixtures.case
publish = fixtures.publish


def test_actual_published_bridge_accepted_without_current_resource_probe(case, monkeypatch):
    assert publish(case)

    def unavailable(*_):
        raise AssertionError("A historical release consumer must not probe current GPU ownership")

    monkeypatch.setattr(bridge, "gpu_processes", unavailable)
    monkeypatch.setattr(bridge, "gpu_uuid", unavailable)
    monkeypatch.setattr(bridge, "process_table", unavailable)
    result = wrapper.bridge_contract(case["run"])
    assert result["coverage"] == case["coverage"]["coverage"]
    assert result["coverage"]["structural_NA"] == 1
    assert result["marker"] == bridge.record(case["canonical"])
    assert result["bindings"] == bridge.validate_inputs(case["run"], case["approval"])["bindings"]


@pytest.mark.parametrize("field,value", [
    ("resource_release_only", False),
    ("execution_coverage", {"native_completed": 360, "structural_NA": 0}),
    ("conditions", ["none"]),
    ("gpu_uuid", "another-physical-device"),
    ("owned_process_groups", []),
    ("all_recorded_owners_exited", False),
    ("exclusive_gpu_lease_held_at_publication", False),
    ("child_exits", []),
    ("resume_parent", {"pid": 363111, "start_ticks": 30}),
])
def test_mutated_marker_cannot_preserve_accepted_release(case, field, value):
    assert publish(case)
    marker = json.loads(case["canonical"].read_text())
    marker[field] = value
    case["canonical"].write_text(json.dumps(marker))
    with pytest.raises(ValueError):
        wrapper.bridge_contract(case["run"])


def test_changed_underlying_execution_certificate_rejected(case):
    assert publish(case)
    case["coverage_path"].write_text(case["coverage_path"].read_text() + "\n")
    with pytest.raises(ValueError):
        wrapper.bridge_contract(case["run"])


def test_native_ood_completion_stays_pending_until_native_reports_complete(case, monkeypatch):
    assert publish(case)
    native = {"task": "object_1", "complete": False, "passed": None, "sources": {}}
    monkeypatch.setattr(wrapper.original, "completion", lambda *_: native)
    result = wrapper.completion(case["run"], "object_1")
    assert result["complete"] is False and result["passed"] is None
    assert result["causal_resource_bridge"]["coverage"]["structural_NA"] == 1
    assert result["sources"][str(case["canonical"])] == bridge.sha(case["canonical"])


def test_invalid_bridge_demotes_apparently_complete_native_ood(case, monkeypatch):
    assert publish(case)
    marker = json.loads(case["canonical"].read_text())
    marker["gpu_uuid"] = "unaccepted-device"
    case["canonical"].write_text(json.dumps(marker))
    native = {"task": "object_1", "complete": True, "passed": True, "sources": {}}
    monkeypatch.setattr(wrapper.original, "completion", lambda *_: native)
    result = wrapper.completion(case["run"], "object_1")
    assert result["complete"] is False and result["passed"] is False
    assert "resource bridge" in result["reason"]
