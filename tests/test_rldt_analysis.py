"""Prevent misleading success aggregation or unmatched before/after comparisons."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "analyze_rldt", Path(__file__).resolve().parents[1] / "scripts/analyze_rldt.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_training_success_weights_episodes_and_preserves_empty_windows():
    rows = [{"episodes": 0, "successes": 0}, {"episodes": 10, "successes": 1},
            {"episodes": 11, "successes": 2}, {"episodes": 11, "successes": 2}]
    cumulative, recent = analysis.success_series(rows, window=2)
    assert np.isnan(cumulative[0]) and np.isnan(recent[0])
    np.testing.assert_allclose(cumulative[1:], [.1, 2 / 11, 2 / 11])
    np.testing.assert_allclose(recent[1:], [.1, 2 / 11, 1.])
    _, recent = analysis.success_series(rows, window=1)
    assert np.isnan(recent[-1])


def test_partial_live_record_is_ignored_but_corrupt_complete_record_is_rejected(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"iterations":1}\n{"iterations":')
    assert analysis.load_records(path) == [{"iterations": 1}]
    path.write_text('{"iterations":1}\n{"iterations":\n')
    with pytest.raises(ValueError):
        analysis.load_records(path)


def evaluation(successes, steps):
    return {"config": {"policy": {}, "env": {}, "rldt": {}}, "episodes": [
        {"seed": i, "reset": {"initial_state_id": i, "backend_seed": i},
         "success": ok, "primitive_steps": count}
        for i, (ok, count) in enumerate(zip(successes, steps, strict=True))]}


def test_paired_evaluation_counts_gains_losses_and_matched_success_speed():
    before = evaluation([True, False, True, False], [20, 220, 30, 220])
    after = evaluation([True, True, False, False], [16, 40, 220, 220])
    result = analysis.compare_evaluations(before, after)
    assert result["change_percentage_points"] == 0
    assert result["new_successes"] == result["lost_successes"] == 1
    assert result["both_succeed"] == result["both_fail"] == 1
    assert result["mean_step_change_on_paired_successes"] == -4
    after["episodes"][0]["reset"]["initial_state_id"] = 99
    with pytest.raises(ValueError, match="seed/reset"):
        analysis.compare_evaluations(before, after)


def test_complete_success_still_has_finite_sample_uncertainty():
    low, high = analysis.wilson(50, 50)
    assert .92 < low < .94
    assert np.isclose(high, 1)
