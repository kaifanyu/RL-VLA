"""Sharding must preserve every paired state/seed and reject incomplete reports."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rl_vla.envs import libero_pro_shard


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_vla_trace_sharded.py"
    spec = importlib.util.spec_from_file_location("evaluation_shards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_reports(helper):
    config = {"env": {"kwargs": {"split": "validation"}}}
    ids = list(range(100, 150))
    specs, reports = [], []
    for index, (start, count) in enumerate(helper.partition(50, 3)):
        shard_config = {"shard": index}
        specs.append({"start": start, "count": count, "config": shard_config})
        rows = [{"seed": 100000 + start + i, "reset": {"initial_state_id": ids[start + i],
                 "experiment_split": "validation"}, "success": index < 2,
                 "video": f"shard_{index}/episode_{i}.mp4", "primitive_steps": 5} for i in range(count)]
        reports.append({"config": shard_config, "checkpoint": "/expert/25", "episodes": rows,
                        "success_rate": float(index < 2)})
    kwargs = {"original_config": config, "expected_ids": ids, "seed": 100000,
              "checkpoint": "/expert/25", "specifications": specs}
    return reports, kwargs


def test_exact_fifty_episode_partition_and_cpu_affinity(helper):
    assert helper.partition(50, 3) == [(0, 17), (17, 17), (34, 16)]
    assert helper.parse_cpuset("24-35") == list(range(24, 36))
    assert helper.parse_cpuset("0-3,8-11,20,22") == [0, 1, 2, 3, 8, 9, 10, 11, 20, 22]
    for invalid in ("3-1", "0-3,2", "4,", "-1"):
        with pytest.raises(ValueError):
            helper.parse_cpuset(invalid)


def test_weighted_merge_preserves_original_config_and_artifact_paths(helper):
    reports, kwargs = fixture_reports(helper)
    before = copy.deepcopy(reports)
    merged = helper.merge_reports(reports, **kwargs)
    assert merged["success_rate"] == 34 / 50
    assert merged["config"] is kwargs["original_config"]
    assert merged["episodes"][17]["seed"] == 100017
    assert merged["episodes"][17]["reset"]["initial_state_id"] == 117
    assert merged["episodes"][17]["video"] == "shard_1/episode_0.mp4"
    assert reports == before


@pytest.mark.parametrize("corruption", ["seed", "state", "checkpoint", "config", "missing_episode", "wrong_split", "success_rate"])
def test_merge_rejects_pairing_and_completion_corruption(helper, corruption):
    reports, kwargs = fixture_reports(helper)
    if corruption == "seed":
        reports[1]["episodes"][0]["seed"] = 100000
    elif corruption == "state":
        reports[1]["episodes"][0]["reset"]["initial_state_id"] = 100
    elif corruption == "checkpoint":
        reports[1]["checkpoint"] = "/expert/50"
    elif corruption == "config":
        reports[1]["config"] = {"wrong": "config"}
    elif corruption == "missing_episode":
        reports[1]["episodes"].pop()
    elif corruption == "wrong_split":
        reports[1]["episodes"][0]["reset"]["experiment_split"] = "train"
    else:
        reports[1]["success_rate"] = 0.5
    with pytest.raises(ValueError):
        helper.merge_reports(reports, **kwargs)


def test_reordered_or_missing_shards_never_publish(helper):
    reports, kwargs = fixture_reports(helper)
    with pytest.raises(ValueError):
        helper.merge_reports(reports[::-1], **kwargs)
    with pytest.raises(ValueError):
        helper.merge_reports(reports[:2], **kwargs)


def test_shard_config_preserves_sampler_and_control_settings(helper):
    import tomllib

    root = Path(__file__).resolve().parents[1]
    text = (root / "runs/vla_trace_objects_20261031/tasks/object_0/validation.toml").read_text() if (
        root / "runs/vla_trace_objects_20261031/tasks/object_0/validation.toml").exists() else '''
[env]
factory = "rl_vla.envs.libero_pro:make_env"
[env.kwargs]
protocol = "/frozen/protocol.json"
task_key = "object_0"
split = "validation"
[rldt]
ode_steps = 10
execute_steps = 5
'''
    before = tomllib.loads(text)
    after = tomllib.loads(helper.shard_config(text, 17, 17))
    assert after["rldt"] == before["rldt"]
    assert after["env"]["kwargs"] == dict(before["env"]["kwargs"], shard_start=17, shard_count=17)
    assert "technical_max_steps" not in after["env"]["kwargs"]
    assert tomllib.loads(text) == before


def test_factory_slices_wrapped_enumeration_without_mutating_state_matrix(monkeypatch):
    initial_ids = tuple(range(100, 150))
    states = np.arange(400).reshape(200, 2)
    base = SimpleNamespace(state_ids=initial_ids, evaluation_index=0, initial_states=states, max_steps=280)
    wrapper = SimpleNamespace(unwrapped=base, close=lambda: None)
    monkeypatch.setattr(libero_pro_shard, "make_frozen_env", lambda **kwargs: wrapper)
    result = libero_pro_shard.make_env(protocol="unchanged.json", task_key="object_0", split="validation", shard_start=17, shard_count=17)
    assert result is wrapper
    assert base.state_ids == tuple(range(117, 134))
    assert initial_ids == tuple(range(100, 150))
    assert base.initial_states is states
    assert base.max_steps == 280


def test_training_shards_are_rejected_before_simulator_construction(monkeypatch):
    def should_not_run(**kwargs):
        raise AssertionError("Simulator should not be constructed")
    monkeypatch.setattr(libero_pro_shard, "make_frozen_env", should_not_run)
    with pytest.raises(ValueError, match="evaluation-only"):
        libero_pro_shard.make_env(protocol="frozen", task_key="object_0", split="train", shard_start=0, shard_count=1)
