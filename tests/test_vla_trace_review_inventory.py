"""Synthetic review inventory checks; no campaign outcomes are simulated."""

import importlib.util
import json
from pathlib import Path


def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("review_inventory", scripts / "inspect_vla_trace_review_inventory.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def setup(root, helper, *, locked=False):
    definitions = {"validation_state_ids": list(range(100, 150)), "test_state_ids": list(range(150, 200)),
                   "source_state_ids": list(range(50))}
    helper.write(root / "tasks/object_1/protocol.json", {"tasks": {"object_1": definitions}})
    if locked:
        for iteration in (25, 50, 100):
            helper.write(root / "methods/rldt/object_1" / f"validation_{iteration:06d}.json", {
                "episodes": [{"seed": 100000 + i, "reset": {"initial_state_id": 100 + i}, "success": True} for i in range(50)]})
        helper.write(root / "methods/rldt/object_1/selection.json", {"iteration": 25, "locked": True})


def test_preselection_inventory_does_not_read_test_outcomes(tmp_path, monkeypatch):
    helper = module(monkeypatch)
    setup(tmp_path, helper)
    path = tmp_path / "methods/rldt/object_1/baseline_test.json"
    path.parent.mkdir(parents=True)
    path.write_text("invalid content deliberately unread before selection")
    result = helper.inspect(tmp_path, "object_1", helper.Hashes())
    assert len(result["reports"]) == 28
    assert result["reports"][0]["status"] == "pending_locked_selection"
    assert not (tmp_path / "video_reviews/object_1/baseline_test_review_complete.json").exists()


def test_zero_failure_baseline_marker_is_bound_to_full_report(tmp_path, monkeypatch):
    helper = module(monkeypatch)
    setup(tmp_path, helper, locked=True)
    path = tmp_path / "methods/rldt/object_1/baseline_test.json"
    helper.write(path, {"episodes": [{"seed": 100000 + i, "reset": {"initial_state_id": 150 + i}, "success": True} for i in range(50)]})
    result = helper.inspect(tmp_path, "object_1", helper.Hashes())
    marker = json.loads((tmp_path / "video_reviews/object_1/baseline_test_review_complete.json").read_text())
    assert marker["complete"] and marker["failed_episodes"] == 0 and marker["review_records"] == []
    assert marker["report_sha256"] == helper.Hashes().get(path)
    assert result["reports"][0]["status"] == "complete_review"
    assert any(row["report"].endswith("selected_test.json") and row["policy"] == "rldt_25" for row in result["reports"])
