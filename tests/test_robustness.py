"""Verify state compatibility and queue accounting without running a simulator."""
import importlib.util
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_vla.robustness import load_states


@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_published_numpy_archives_are_read_without_arbitrary_classes(tmp_path, protocol):
    path = tmp_path / "states.pruned_init"
    states = np.arange(24, dtype=np.float64).reshape(3, 8)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(states, protocol=protocol))
        archive.writestr("archive/version", b"1")
    np.testing.assert_array_equal(load_states(path), states)


def test_state_archive_rejects_non_numpy_objects(tmp_path):
    path = tmp_path / "states.pruned_init"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(Path("unexpected")))
        archive.writestr("archive/version", b"1")
    with pytest.raises(pickle.UnpicklingError, match="Unexpected class"):
        load_states(path)


def test_plus_single_layout_state_and_nonfinite_rejection(tmp_path):
    path = tmp_path / "layout.pruned_init"
    torch.save(np.arange(16, dtype=np.float64), path)
    assert load_states(path).shape == (1, 16)
    torch.save(np.array([np.nan]), path)
    with pytest.raises(ValueError, match="Invalid initial states"):
        load_states(path)


def queue_module():
    spec = importlib.util.spec_from_file_location("robustness_queue",
        Path(__file__).resolve().parents[1] / "scripts/robustness_queue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_queue_covers_each_episode_once_and_starts_both_benchmarks():
    queue = queue_module()
    cases = [{"case_id": "a", "benchmark": "plus", "category": "camera", "episodes": 1},
             {"case_id": "b", "benchmark": "plus", "category": "camera", "episodes": 1},
             {"case_id": "c", "benchmark": "pro", "category": "Task", "episodes": 3}]
    jobs = list(queue.schedule(cases))
    assert len(jobs) == len({queue.episode_id(c, i) for c, i in jobs}) == 5
    assert {c["benchmark"] for c, _ in jobs[:2]} == {"plus", "pro"}
    assert {queue.episode_id(c, i) for c, i in jobs} == {
        "a_init000", "b_init000", "c_init000", "c_init001", "c_init002"}


def test_partial_benchmark_denominators_are_kept_separate():
    queue = queue_module()
    cases = [{"benchmark": "plus", "category": "Camera", "suite": "spatial", "episodes": 1},
             {"benchmark": "pro", "category": "Task", "suite": "spatial", "episodes": 50}]
    rows = [{**cases[0], "success": True}]
    result = queue.summary(rows, cases, "running")
    assert result["episodes"] == 1 and result["planned_episodes"] == 51
    assert result["metrics"]["plus"]["success_rate"] == 1
    assert result["metrics"]["pro"]["success_rate"] is None
    assert result["metrics"]["pro"]["status"] == "partial"


def test_capacity_wait_restarts_stability_timer_after_another_job_arrives(tmp_path, monkeypatch):
    import json

    spec = importlib.util.spec_from_file_location("resource_queue",
        Path(__file__).resolve().parents[1] / "scripts/resource_queue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    readings = iter(["0, 40", "0, 40", "50000, 45", "0, 40", "0, 40", "0, 40", "0, 40"])
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: next(readings))
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw:
        "MemAvailable: 67108864 kB\n" if str(path) == "/proc/meminfo" else original_read(path, *a, **kw))
    module.wait_for_capacity(tmp_path)
    record = json.loads((tmp_path / "resource_wait.json").read_text())
    assert clock[0] == 60
    assert record["status"] == "capacity_available" and record["stable_seconds"] == 30
