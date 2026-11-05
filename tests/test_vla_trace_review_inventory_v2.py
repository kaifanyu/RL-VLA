"""Compatibility checks for immutable markers; synthetic evidence only."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def helper(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "review_inventory_v2", scripts / "inspect_vla_trace_review_inventory_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def expected():
    return {"complete": True, "task": "object_0", "report_sha256": "report-hash",
            "review_records": [{"path": "/labels.json", "sha256": "labels-hash"}],
            "report": "/baseline_test.json", "failed_episodes": 33,
            "native_successes": 17, "no_failure_review_required": False}


def test_original_and_richer_marker_contracts_are_accepted(helper, expected):
    original = {key: expected[key] for key in ("complete", "task", "report_sha256", "review_records")}
    helper.validate_baseline_marker(original, expected)
    helper.validate_baseline_marker(expected.copy(), expected)
    helper.validate_baseline_marker({**original, "failed_episodes": 33}, expected)


@pytest.mark.parametrize("field,value", [
    ("complete", False), ("task", "object_1"), ("report_sha256", "changed"),
    ("review_records", []), ("report", "/different.json"),
    ("failed_episodes", 32), ("native_successes", 18), ("no_failure_review_required", True),
])
def test_changed_core_or_present_optional_metadata_is_rejected(helper, expected, field, value):
    with pytest.raises(ValueError, match=field):
        helper.validate_baseline_marker({**expected, field: value}, expected)


def test_missing_core_and_unknown_claim_are_rejected(helper, expected):
    missing = expected.copy()
    del missing["review_records"]
    with pytest.raises(ValueError, match="required core"):
        helper.validate_baseline_marker(missing, expected)
    with pytest.raises(ValueError, match="invented_claim"):
        helper.validate_baseline_marker({**expected, "invented_claim": True}, expected)


def test_inventory_preserves_existing_original_marker_bytes(helper, tmp_path):
    root = tmp_path
    definitions = {"validation_state_ids": list(range(100, 150)),
                   "test_state_ids": list(range(150, 200)), "source_state_ids": list(range(50))}
    helper.write(root / "tasks/object_1/protocol.json", {"tasks": {"object_1": definitions}})
    for iteration in (25, 50, 100):
        helper.write(root / f"methods/rldt/object_1/validation_{iteration:06d}.json", {
            "episodes": [{"seed": 100000 + i, "reset": {"initial_state_id": 100 + i},
                          "success": True} for i in range(50)]})
    helper.write(root / "methods/rldt/object_1/selection.json", {"iteration": 25, "locked": True})
    report = root / "methods/rldt/object_1/baseline_test.json"
    helper.write(report, {"episodes": [{"seed": 100000 + i, "reset": {"initial_state_id": 150 + i},
                                       "success": True} for i in range(50)]})
    marker = root / "video_reviews/object_1/baseline_test_review_complete.json"
    helper.write(marker, {"complete": True, "task": "object_1", "report_sha256": helper.Hashes().get(report),
                          "review_records": []})
    before = marker.read_bytes(), marker.stat().st_mtime_ns
    result = helper.inspect(root, "object_1", helper.Hashes())
    assert result["reports"][0]["status"] == "complete_review"
    assert (marker.read_bytes(), marker.stat().st_mtime_ns) == before
    assert json.loads(marker.read_text()).keys() == {"complete", "task", "report_sha256", "review_records"}
