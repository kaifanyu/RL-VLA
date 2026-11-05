"""Qualitative examples retain pairing and distinguish reviewed failure labels."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_vla_trace_qualitative import (
    baseline_review_ready,
    choose_cases,
    sha,
    verify_initial_observation,
)


def episode(state_id, success):
    return {
        "seed": 100000 + state_id - 150,
        "success": success,
        "reset": {"initial_state_id": state_id, "experiment_split": "test"},
    }


def test_case_selection_is_paired_outcome_rule_with_reviewed_wrong_object_priority():
    baseline = [episode(150, False), episode(151, False), episode(152, True), episode(153, False)]
    candidate = [episode(150, True), episode(151, True), episode(152, False), episode(153, False)]
    ordinary = choose_cases(baseline, candidate)
    reviewed = choose_cases(baseline, candidate, [151])
    assert ordinary["recovered"][1] == 150
    assert reviewed["recovered"][1] == 151
    assert reviewed["unsolved"][1] == 153
    assert choose_cases(baseline, baseline)["recovered"] is None


def test_case_selection_refuses_dropped_states_or_changed_noise():
    baseline, candidate = [episode(150, False)], [episode(150, True)]
    candidate[0]["seed"] += 1
    with pytest.raises(ValueError, match="reset/noise"):
        choose_cases(baseline, candidate)
    with pytest.raises(ValueError, match="reset/noise"):
        choose_cases(baseline, [])


def test_recorded_initial_pixels_must_match_fixed_input_panel(tmp_path):
    image = np.zeros((224, 224, 3), np.uint8)
    bank = {
        "image": image.copy(),
        "wrist_image": image.copy(),
        "state": np.zeros(8, np.float32),
        "prompt": np.asarray("pick up alphabet soup"),
    }
    schema = {name: ["openpi", f"observation/{name}"] for name in ("image", "wrist_image", "state")}
    schema["prompt"] = ["openpi", "prompt"]
    np.savez(
        tmp_path / "chunk_000001.npz",
        **{name: value[None] for name, value in bank.items()},
        metadata_json=json.dumps({"observation_schema": schema}),
    )
    recorded = {"trajectory": str(tmp_path)}
    assert verify_initial_observation(recorded, bank)["exact_initial_pixels_state_prompt"]
    bank["image"][12, 23, 0] = 1
    with pytest.raises(ValueError, match="initial image"):
        verify_initial_observation(recorded, bank)


def test_fixed_examples_wait_for_complete_failed_baseline_reviews(tmp_path):
    report = tmp_path / "baseline_test.json"
    report.write_text("{}")
    assert not baseline_review_ready(tmp_path, "object_0", report, [episode(150, False)])
    marker = tmp_path / "video_reviews/object_0/baseline_test_review_complete.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "complete": True,
                "task": "object_0",
                "report_sha256": sha(report),
                "review_records": [],
            }
        )
    )
    with pytest.raises(ValueError, match="omits failed"):
        baseline_review_ready(tmp_path, "object_0", report, [episode(150, False)])
    assert baseline_review_ready(tmp_path, "object_0", report, [episode(150, True)])
    report.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="identity"):
        baseline_review_ready(tmp_path, "object_0", report, [episode(150, True)])
