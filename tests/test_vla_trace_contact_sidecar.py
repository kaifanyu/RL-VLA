"""Sustained-grasp evidence is cached, paired, and separate from human review."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rl_vla.diagnostics.contact_evidence import causal_steps, evidence, native_steps

TARGET, OTHER = "alphabet_soup_1", "butter_1"


def telemetry(index, *, target=False, other=False, target_lift=.04, other_lift=.04):
    return {"objects": {name: {"both_fingerpads_contact": contact, "lift_from_reset_m": lift,
                              "gripper_distance_m": .01} for name, contact, lift in
                             ((TARGET, target, target_lift), (OTHER, other, other_lift))},
            "simulator_state": [.5+index*.05]}


def measure(entries, success=False, primitive_steps=None):
    return evidence(entries, target=TARGET, distractor=OTHER, primitive_steps=primitive_steps or len(entries),
                    success=success, reset_telemetry={"simulator_state": [.5]})


def steps(flags):
    return [{"step": index, "telemetry": telemetry(index, target=a, other=b), "unix_ns": 1000+index}
            for index, (a, b) in enumerate(flags, start=1)]


def test_target_grasp_has_first_confirmation_and_does_not_claim_review():
    result = measure(steps([(True, False)]*3))
    assert result["sustained_grasp_candidates"] == [TARGET]
    assert result["first_contact_confirmed_grasp_id"] == TARGET
    assert result["first_candidate_confirmation_step"] == 3
    assert result["objects"][TARGET]["first_candidate_onset_step"] == 1
    assert result["first_candidate_confirmation_elapsed_sim_seconds"] == pytest.approx(.15)
    assert result["operational_outcome"] == "target_candidate_goal_failed"
    assert result["reviewed"] is False and result["automatic"] is True


def test_non_target_then_target_is_mixed_even_with_unique_first_candidate():
    result = measure(steps([(False, True)]*3+[(True, False)]*3))
    assert result["first_contact_confirmed_grasp_id"] == OTHER
    assert result["mixed_target_non_target_candidates"] is True
    assert result["provisional_failure_type"] == "other/ambiguous"
    assert result["objects"][TARGET]["first_candidate_confirmation_step"] == 6


def test_simultaneous_candidates_are_explicit_tie_not_arbitrary_object():
    result = measure(steps([(True, True)]*3))
    assert result["first_contact_confirmed_grasp_id"] is None
    assert result["first_contact_confirmed_grasp_ids"] == sorted([TARGET, OTHER])
    assert result["first_confirmation_tie"] and result["first_candidate_role"] == "tie"


def test_contact_without_sustained_lift_is_not_wrong_object_and_success_survives():
    entries = [{"step": i, "telemetry": telemetry(i, other=True, other_lift=.02)} for i in range(1, 5)]
    result = measure(entries)
    assert result["no_grasp_candidate"] and result["provisional_failure_type"] == "other/ambiguous"
    successful = measure(steps([(False, True)]*3), success=True)
    assert successful["non_target_grasp_candidate"] and successful["operational_outcome"] == "native_success"


def test_missing_controlled_steps_cannot_join_two_short_streaks():
    entries = [{"step": i, "telemetry": telemetry(i, target=True)} for i in (1, 2, 4, 5)]
    result = measure(entries, primitive_steps=5)
    assert result["sustained_grasp_candidates"] == []
    assert result["missing_telemetry_steps"] == [3] and result["telemetry_complete"] is False
    assert result["operational_outcome"] == "insufficient_telemetry"


def native_fixture(directory, state=100):
    directory.mkdir(parents=True)
    episode = {"seed": 100000+state-100, "primitive_steps": 3, "success": False,
               "reset": {"initial_state_id": state, "task_id": 0, "experiment_split": "validation"}}
    header = {"complete": True, "primitive_steps": 3, "chunks": 1,
              "metadata": {"seed": episode["seed"], "reset": episode["reset"]}, "summary": {"success": False}}
    (directory/"episode.json").write_text(json.dumps(header))
    metadata = {"chunk": 1, "first_primitive_step": 0, "executed_steps": 3,
                "step_infos": [{"object_telemetry": telemetry(i, target=True)} for i in range(1, 4)]}
    np.savez(directory/"chunk_000001.npz", metadata_json=json.dumps(metadata), step_unix_ns=np.arange(1001, 1004))
    return episode


def test_native_and_causal_formats_yield_same_candidate_times(tmp_path):
    episode = native_fixture(tmp_path/"native")
    path = tmp_path/"episode_000_telemetry.json"
    path.write_text(json.dumps([{**telemetry(i, target=True), "primitive_steps": i} for i in range(1, 4)]))
    native_sources, causal_sources = [], []
    a = measure(native_steps(tmp_path/"native", episode, native_sources), primitive_steps=3)
    b = measure(causal_steps(path, episode, causal_sources), primitive_steps=3)
    for field in ("sustained_grasp_candidates", "first_candidate_confirmation_step", "first_candidate_confirmation_sim_seconds", "telemetry_complete"):
        assert a[field] == b[field]
    assert len(native_sources) == 2 and "metadata_json_sha256" in native_sources[1]
    assert len(causal_sources) == 1 and causal_sources[0]["sha256"]


@pytest.fixture
def sidecar():
    path = Path(__file__).resolve().parents[1]/"scripts/analyze_vla_trace_contacts.py"
    spec = importlib.util.spec_from_file_location("contact_sidecar", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def campaign_fixture(root, sidecar):
    task = root/"tasks/object_0"
    task.mkdir(parents=True)
    (task/"protocol.json").write_text(json.dumps({"tasks": {"object_0": {"validation_state_ids": list(range(100, 150))}}}))
    method = root/"methods/rldt/object_0"
    method.mkdir(parents=True)
    episodes = []
    for state in range(100, 150):
        trajectory = method/f"episode_{state}"
        row = native_fixture(trajectory, state)
        video = trajectory/"video.mp4"
        video.write_bytes(b"synthetic video fixture")
        row.update(trajectory=str(trajectory), video=str(video))
        episodes.append(row)
    report_path = method/"baseline_validation.json"
    report_path.write_text(json.dumps({"episodes": episodes, "checkpoint": None, "technical_only": False}))
    manual = root/"video_reviews/reviewed_labels"
    manual.mkdir(parents=True)
    sheet, manifest = manual/"sheet.jpg", manual.parent/"evidence.json"
    sheet.write_bytes(b"synthetic reviewed fixture")
    manifest.write_text("{}")
    label = {"task": "object_0", "report": str(report_path), "state_id": 100, "seed": 100000,
             "success": False, "reviewed": True, "failure_type": "correct grasp but placement failure",
             "grasped_object_id": TARGET, "video": episodes[0]["video"], "sheet": str(sheet),
             "evidence_manifest": str(manifest), "notes": "Synthetic test fixture only"}
    for name in ("report", "video", "sheet", "evidence_manifest"):
        label[name+"_sha256"] = sidecar.checksum(label[name])
    (manual/"labels.json").write_text(json.dumps({"records": [label]}))
    destination = root/"contact_analysis"
    destination.mkdir()
    return report_path, destination


def test_cached_reports_never_reparse_npz_and_manual_join_stays_separate(sidecar, tmp_path, monkeypatch):
    report, destination = campaign_fixture(tmp_path, sidecar)
    cache = {}
    summary = sidecar.refresh(tmp_path, destination, cache, "test-code")
    assert summary["reports_processed"] == 1 and summary["episodes_processed"] == 50
    assert summary["manual_reviewed_episodes"] == 1 and summary["automatic_labels_reviewed"] is False
    artifact = destination/"reports"/sidecar.checksum(report)/"episodes.jsonl"
    rows = [json.loads(line) for line in artifact.read_text().splitlines()]
    assert all(row["reviewed"] is False and "manual_reviewed" not in row for row in rows)
    stat_before = (destination/"summary.json").stat().st_mtime_ns
    def fail(*args, **kwargs):
        raise AssertionError("Cached NPZ must not be reopened")
    monkeypatch.setattr(sidecar, "native_steps", fail)
    again = sidecar.refresh(tmp_path, destination, cache, "test-code")
    assert again == summary
    assert (destination/"summary.json").stat().st_mtime_ns == stat_before


def test_transient_partial_json_is_retried_but_stable_invalid_source_is_reported(sidecar, tmp_path):
    report, destination = campaign_fixture(tmp_path, sidecar)
    (tmp_path/"video_reviews/reviewed_labels/labels.json").unlink()  # producer has not completed, so no valid review can exist yet
    complete = report.read_text()
    report.write_text('{"episodes":')
    cache = {}
    first = sidecar.refresh(tmp_path, destination, cache, "test-code")
    assert not first["persistent_errors"] and first["pending_producers"]
    sidecar.refresh(tmp_path, destination, cache, "test-code")
    third = sidecar.refresh(tmp_path, destination, cache, "test-code")
    assert len(third["persistent_errors"]) == 1 and third["reports_processed"] == 0
    report.write_text(complete)
    recovered = sidecar.refresh(tmp_path, destination, cache, "test-code")
    assert recovered["reports_processed"] == 1 and not recovered["persistent_errors"]


def test_selection_aliases_do_not_create_extra_scientific_cells(sidecar, tmp_path):
    method = tmp_path/"methods/rldt/object_0"
    method.mkdir(parents=True)
    (method/"selection.json").write_text(json.dumps({"iteration": 50}))
    (method/"selected_test.json").touch()
    plan = sidecar.report_plan(tmp_path)
    assert len(plan) == 192 and sum(row["episodes"] for row in plan) == 5280
    chosen = [row for row in plan if row["domain"] == "main" and row["task"] == "object_0" and row["policy"] == "rldt_50" and row["variant"] == "test"]
    assert len(chosen) == 1 and Path(chosen[0]["report"]).name == "selected_test.json"


def test_changed_manual_evidence_never_retains_a_verified_review_flag(sidecar, tmp_path):
    _, destination = campaign_fixture(tmp_path, sidecar)
    cache = {}
    first = sidecar.refresh(tmp_path, destination, cache, "test-code")
    assert first["manual_reviewed_episodes"] == 1
    (tmp_path/"video_reviews/reviewed_labels/sheet.jpg").write_bytes(b"changed sheet")
    changed = sidecar.refresh(tmp_path, destination, cache, "test-code")
    assert changed["manual_reviewed_episodes"] == 0
    assert any(row["domain"] == "manual_reviews" for row in changed["persistent_errors"])
