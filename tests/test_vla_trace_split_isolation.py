"""Official Object 0/1/2 preparation must never leak retention/test states."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def prep(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("vla_objects", scripts / "prepare_vla_trace_objects.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exclusion_union_rejects_source_and_screen_even_with_different_clock(prep, tmp_path):
    excluded = np.asarray([[0., 1., 2., 0.], [4., 3., 4., 1.], [9., 1., 2., 0.]])
    attempts = []
    for i, state in enumerate(([13., 1., 2., 8.], [0., 3., 4., 0.], [0., 5., 6., 0.], [8., 5., 6., 1.], [0., 7., 8., 0.])):
        path = tmp_path / f"{i}.npz"
        np.savez(path, seed=100 + i, nq=2, state=np.asarray(state), error="")
        attempts.append(path)
    states, accepted, rejected = prep.collect(attempts, excluded, count=2)
    assert [row["generation_seed"] for row in accepted] == [102, 104]
    assert [row["generation_seed"] for row in rejected] == [100, 101, 103]
    assert len(states) == 2
    assert all(row["nq"] == 2 for row in accepted)


def make_protocol(prep, directory, source_overlap=False):
    pro = np.arange(800, dtype=np.float64).reshape(200, 4)
    source = np.arange(200, dtype=np.float64).reshape(50, 4) + 10000
    if source_overlap:
        source[0] = pro[0] + np.asarray([12., 0., 0., 20.])
    np.save(directory / "pro.npy", pro)
    np.save(directory / "source.npy", source)
    task = {"pro_state_count": 200, "train_state_ids": list(range(100)),
            "validation_state_ids": list(range(100, 150)), "test_state_ids": list(range(150, 200)),
            "source_state_ids": list(range(50)),
            "pro": {"states": prep.record(directory / "pro.npy")},
            "source": {"states": prep.record(directory / "source.npy")}}
    (directory / "protocol.json").write_text(json.dumps({"version": 2, "tasks": {"object_1": task}}))
    (directory / "generation.json").write_text(json.dumps({"accepted": [{"nq": 2}]}))


def test_split_audit_detects_retention_physical_overlap(prep, tmp_path):
    make_protocol(prep, tmp_path, source_overlap=True)
    with pytest.raises(ValueError, match="Physical state overlap"):
        prep.split_audit(tmp_path)


def test_split_audit_records_all_250_states_and_pairwise_zero_overlap(prep, tmp_path):
    make_protocol(prep, tmp_path)
    prep.split_audit(tmp_path)
    result = json.loads((tmp_path / "split_audit.json").read_text())
    assert result["passed"] is True
    assert len(result["qpos_overlap_counts"]) == 6
    assert not any(result["qpos_overlap_counts"].values())
    assert sum(map(len, result["qpos_sha256"].values())) == 250


def test_immutable_artifacts_cannot_be_rewritten(prep, tmp_path):
    destination = tmp_path / "protocol.json"
    prep.immutable_json(destination, {"validation_iterations": [25, 50, 100]})
    prep.immutable_json(destination, {"validation_iterations": [25, 50, 100]})
    with pytest.raises(ValueError, match="Refusing to alter frozen"):
        prep.immutable_json(destination, {"validation_iterations": [25, 50, 75, 100]})


def test_only_three_official_tasks_and_nonoverlapping_seed_ranges(prep):
    assert list(prep.TASKS) == ["object_0", "object_1", "object_2"]
    seeds = [set(range(task["generation_seed"], task["generation_seed"] + 1000)) for task in prep.TASKS.values()]
    assert len(set.union(*seeds)) == 3000
