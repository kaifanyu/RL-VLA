"""Aggregate completed, partial and failed method runs without simulator imports."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compare_pro_methods", ROOT / "scripts/compare_pro_methods.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def metric(baseline=.4, trained=.78, recovered=19, lost=0):
    return {
        "episodes": 50, "frozen_success_rate": baseline, "trained_success_rate": trained,
        "change_percentage_points": round(100 * (trained - baseline), 8),
        "new_successes": recovered, "lost_successes": lost,
        "both_succeed": int(50 * baseline) - lost,
        "both_fail": 50 - int(50 * baseline) - recovered,
        "paired_bootstrap_95_change_percentage_points": [24., 52.],
        "changed_episodes": [{"large": "episode/trajectory data should not be duplicated"}],
    }


def test_paired_metrics_and_measured_selection_preserve_missing_runs(tmp_path):
    run = tmp_path / "methods/rldt/object_0"
    write_json(run / "comparison.json", {
        "selected_iteration": 50, "test": metric(),
        "source": metric(.98, .98, recovered=1, lost=1),
    })
    write_json(run / "training/summary.json", {
        "primitive_steps": 15900, "decisions": 3200, "elapsed_seconds": 3481,
    })
    (run / "training/metrics.jsonl").write_text(
        json.dumps({"iterations": 50, "primitive_steps": 7952}) + "\n"
        + json.dumps({"iterations": 100, "primitive_steps": 15900}) + "\n")
    report = comparison.build_comparison(tmp_path)
    assert len(report["results"]) == report["completed_experiments"] == 1
    assert len(report["experiments"]) == report["planned_experiments"] == 9
    result = report["results"][0]
    assert result["test"]["change_percentage_points"] == 38
    assert result["test"]["paired_bootstrap_95_change_percentage_points"] == [24, 52]
    assert result["test"]["new_successes"] == 19
    assert result["test"]["lost_successes"] == 0
    assert result["source"]["change_percentage_points"] == 0
    assert result["source"]["new_successes"] == result["source"]["lost_successes"] == 1
    assert result["selected_iteration"] == 50
    assert result["selected_training_primitive_steps"] == 7952
    assert result["total_training_primitive_steps"] == 15900
    assert result["evaluation_rng_backend"] == "jax"
    assert "changed_episodes" not in result["test"]
    pending = [row for row in report["experiments"] if row["status"] == "pending"]
    assert len(pending) == 8
    assert all(row["test"] is None and row["source"] is None for row in pending)
    assert all(row["selected_training_primitive_steps"] is None for row in pending)


def test_failed_run_partial_results_and_torch_iteration_accounting(tmp_path):
    run = tmp_path / "methods/flow_sde/spatial_1"
    write_json(run / "comparison.json", {
        "selected_iteration": 25, "test": metric(.2, .18, recovered=3, lost=4),
    })
    write_json(run / "status.json", {"state": "failed", "stage": "selected_source", "error": "test failure"})
    write_json(run / "training/summary.json", {"primitive_steps": 15999})
    (run / "training/iteration_metrics.jsonl").write_text(
        '{"iteration": 25, "primitive_steps": 3997}\n{"iteration":')
    write_json(tmp_path / "pi_rl_queue/progress.json", {"failures": [
        {"task": "spatial_1", "method": "flow_sde", "exit_code": 1},
        {"task": "spatial_2", "method": "flow_noise", "exit_code": 2},
    ]})
    write_json(tmp_path / "methods/flow_noise/spatial_2/selection.json", {"iteration": 75})
    report = comparison.build_comparison(tmp_path)
    assert report["completed_experiments"] == 0
    assert report["failed_experiments"] == 2
    result = report["results"][0]
    assert result["status"] == "failed"
    assert result["test"]["change_percentage_points"] == -2
    assert result["test"]["new_successes"] == 3
    assert result["test"]["lost_successes"] == 4
    assert "source" not in result
    assert result["selected_training_primitive_steps"] == 3997
    assert result["evaluation_rng_backend"] == "torch"
    failed_before_report = next(row for row in report["experiments"]
                               if row["task"] == "spatial_2" and row["method"] == "flow_noise")
    assert failed_before_report["test"] is None
    assert failed_before_report["exit_code"] == 2
    assert failed_before_report["selected_iteration"] == 75
    assert failed_before_report["selected_training_primitive_steps"] is None


def test_successful_zero_is_available_but_missing_retention_is_partial(tmp_path):
    run = tmp_path / "methods/rldt/spatial_2"
    write_json(run / "comparison.json", {
        "selected_iteration": 25, "test": metric(1, 1, recovered=0, lost=0),
    })
    report = comparison.build_comparison(tmp_path)
    assert report["results"][0]["test"]["change_percentage_points"] == 0
    assert report["results"][0]["status"] == "partial"
    assert report["completed_experiments"] == 0
    assert report["results"][0]["selected_training_primitive_steps"] is None


def test_writes_json_and_png_for_all_pending_suite(tmp_path):
    pytest.importorskip("matplotlib")
    report = comparison.write_comparison(tmp_path)
    assert json.loads((tmp_path / "comparison_matrix.json").read_text()) == report
    assert (tmp_path / "comparison_matrix.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert report["results"] == []
