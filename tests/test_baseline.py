"""Evaluation protocol checks: repeatable noise, honest partial counts, durable results."""
import json

import pytest

from rl_vla.baseline import atomic_json, noise_seed, summarize


def test_noise_is_independent_of_episode_execution_order():
    expected = [noise_seed("task0:init0:100000", i) for i in range(5)]
    for i in range(100):
        noise_seed("unrelated-episode", i)
    assert [noise_seed("task0:init0:100000", i) for i in range(5)] == expected
    assert len(set(expected)) == 5
    assert noise_seed("task0:init1:100001", 0) != expected[0]


def test_partial_results_do_not_count_missing_episodes_as_failures():
    rows = [{"suite": "libero_spatial", "task_id": 0, "success": True, "return": 1.0,
             "primitive_steps": 25, "elapsed_seconds": 3.0, "inference_latency_ms": [10, 20]},
            {"suite": "libero_spatial", "task_id": 1, "success": False, "return": 0.0,
             "primitive_steps": 220, "elapsed_seconds": 7.0, "inference_latency_ms": [30]}]
    result = summarize(rows, planned=10)
    assert result["status"] == "partial"
    assert result["episodes"] == 2 and result["success_rate"] == 0.5
    assert result["success_95pct_wilson"][0] < 0.5 < result["success_95pct_wilson"][1]
    assert result["mean_steps_on_success"] == 25
    assert result["inference_latency_p50_ms"] == 20
    assert result["per_task"]["libero_spatial/0"]["success_rate"] == 1
    assert summarize(rows, planned=2)["status"] == "complete"


def test_invalid_results_cannot_replace_a_valid_saved_result(tmp_path):
    target = tmp_path / "episode.json"
    atomic_json(target, {"success": True})
    with pytest.raises(ValueError):
        atomic_json(target, {"return": float("nan")})
    assert json.loads(target.read_text()) == {"success": True}
