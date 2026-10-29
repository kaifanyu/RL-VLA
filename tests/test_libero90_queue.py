"""The extra suite uses a distinct factory and waits without owning the GPU."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
from benchmark_bridge import screen90


def test_original_suite_selects_only_its_new_factory(monkeypatch):
    monkeypatch.setattr(screen90, "base_make_config", lambda *args: {"env": {"factory": "old"}})
    case = {"benchmark": "libero90", "suite": "libero_90", "max_steps": 400}
    config = screen90.make_config(None, None, case, None)
    assert config["env"]["factory"] == "benchmark_bridge.libero90_environment:make_env"
    with pytest.raises(ValueError, match="400-step"):
        screen90.make_config(None, None, case | {"max_steps": 520}, None)


def test_dependency_wait_finishes_after_all_predecessor_cells_are_processed(tmp_path, monkeypatch):
    previous, current = tmp_path / "previous", tmp_path / "current"
    previous.mkdir()
    current.mkdir()
    path = previous / "screen_status.json"
    path.write_text(json.dumps({"state": "evaluating"}))
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        status = json.loads((current / "screen_status.json").read_text())
        assert status["state"] == "waiting_for_predecessor"
        path.write_text(json.dumps({"state": "complete_with_errors"}))

    monkeypatch.setattr(screen90.time, "sleep", sleep)
    screen90.wait_for_predecessor(current, previous, 1470624, 30)
    assert waits == [30]


def test_failed_predecessor_is_not_bypassed(tmp_path):
    previous, current = tmp_path / "previous", tmp_path / "current"
    previous.mkdir()
    current.mkdir()
    (previous / "screen_status.json").write_text(json.dumps({"state": "failed"}))
    with pytest.raises(RuntimeError, match="Predecessor"):
        screen90.wait_for_predecessor(current, previous, 1470624, 30)
    assert json.loads((current / "screen_status.json").read_text())["state"] == "failed"
