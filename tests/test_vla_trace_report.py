import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("trace_report",Path(__file__).resolve().parents[1]/"scripts/analyze_vla_trace_objects.py")
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_only_final_narrative_and_explicit_lead_marker_are_adopted(tmp_path):
    (tmp_path / "scientific_interpretation_draft_v1.md").write_text("Unaccepted draft")
    assert report.scientific_interpretation_sections(tmp_path) == ("", "")
    final = tmp_path / "scientific_interpretation.md"
    final.write_text("Existing complete narrative\n")
    assert report.scientific_interpretation_sections(tmp_path) == ("", "Existing complete narrative\n")
    final.write_text("Concise finding\n\n| Selected results |\n<!-- report-lead-end -->\nDetailed interpretation\n")
    assert report.scientific_interpretation_sections(tmp_path) == (
        "Concise finding\n\n| Selected results |", "Detailed interpretation")
    final.write_text("Lead<!-- report-lead-end -->Detail<!-- report-lead-end -->Duplicate")
    with pytest.raises(ValueError, match="multiple"):
        report.scientific_interpretation_sections(tmp_path)


def test_complete_probe_metrics_without_figure_manifest_remain_pending(tmp_path):
    write(tmp_path/"probes/object_0/pi05_base/metrics.json",{"complete":True})
    write(tmp_path/"probes_counterfactual/object_0/pi05_base/metrics.json",{"complete":True})
    row=next(row for row in report.diagnostic_inventory(tmp_path)
             if row["task"]=="object_0" and row["policy"]=="pi05_base" and row["test"]=="localization_probe")
    assert row["status"]=="pending"
    assert "artifact validation" in row["reason"]


def evaluation(successes=0):
    return {"episodes":[{"seed":100000+i,"reset":{"initial_state_id":100+i},
                         "success":i<successes,"primitive_steps":280} for i in range(50)]}


def test_selection_uses_counts_and_earliest_tie(tmp_path):
    for iteration in (25,50,100):
        write(tmp_path/f"validation_{iteration:06d}.json",evaluation(10))
    write(tmp_path/"selection.json",{"iteration":25})
    assert report.audit_selection(tmp_path)["iteration"]==25
    write(tmp_path/"selection.json",{"iteration":100})
    with pytest.raises(ValueError,match="count/earliest"):
        report.audit_selection(tmp_path)


def test_missing_results_remain_pending_not_zero(tmp_path):
    data=report.collect(tmp_path)
    assert len(data["success"])==45
    assert sum(row["status"]=="pending" for row in data["success"])==36
    assert all("successes" not in row for row in data["success"])
    assert sum(row["status"]=="not_applicable" for row in data["success"])==9


@pytest.mark.parametrize("gate,expected", [
    ({"passed": True, "checks": [{"passed": True}], "failed": [], "pending": []}, "passed"),
    ({"passed": False, "checks": [{"passed": None}], "failed": [], "pending": ["training"]}, "pending"),
    ({"passed": False, "checks": [{"passed": False}], "failed": ["fidelity"], "pending": []}, "failed"),
    ({"passed": False, "checks": [{"passed": False}], "failed": [], "pending": ["training"]}, "failed"),
])
def test_acceptance_status_preserves_pending_versus_failed(gate, expected):
    assert report.acceptance_status(gate) == expected


def test_final_test_is_rejected_before_selection_lock(tmp_path):
    path=tmp_path/"methods/rldt/object_0/baseline_test.json"
    write(path,evaluation())
    with pytest.raises(ValueError,match="before locked selection"):
        report.collect(tmp_path)


def test_frozen_episode_ids_and_noise_seeds_are_enforced(tmp_path):
    value=evaluation()
    definition={"validation_state_ids":list(range(100,150))}
    assert len(report.checked_episodes(value,tmp_path,definition,"validation"))==50
    value["episodes"][0]["reset"]["initial_state_id"]=99
    with pytest.raises(ValueError,match="differ from frozen"):
        report.checked_episodes(value,tmp_path,definition,"validation")


def ood_review_fixture(root):
    path = root / "ood/object_1/pi05_libero/both_new_regions/summary.json"
    video, sheet, manifest = [root / name for name in ("fixture_video", "fixture_sheet", "fixture_manifest")]
    for source in (video, sheet, manifest):
        source.write_bytes(b"synthetic test evidence, not campaign data")
    value = {"complete": True, "task": "object_1", "variant": "both_new_regions", "episodes": [
        {"seed": 100000 + i, "reset": {"initial_state_id": i}, "success": i < 47,
         "video": str(video), "primitive_steps": 280, "failure_type": "wrong object"} for i in range(50)]}
    write(path, value)
    rows = []
    for state, kind in ((47, "wrong object"), (48, "other/ambiguous")):
        rows.append({"task": "object_1", "report": str(path), "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "state_id": state, "seed": 100000 + state, "success": False, "video": str(video),
                     "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "sheet": str(sheet),
                     "sheet_sha256": hashlib.sha256(sheet.read_bytes()).hexdigest(), "evidence_manifest": str(manifest),
                     "evidence_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                     "reviewed": True, "failure_type": kind, "reviewer": "synthetic fixture",
                     "notes": "Synthetic test only; mixed/weak-contact evidence remains ambiguous."})
    labels = root / "video_reviews/reviewed_labels/synthetic.json"
    write(labels, {"records": rows})
    return path, value, labels, rows


def test_ood_wrong_failure_fraction_withheld_until_all_failures_reviewed(tmp_path):
    path, value, labels, records = ood_review_fixture(tmp_path)
    rows, coverage, taxonomy = report.ood_review_rows("object_1", "pi05_libero", "both_new_regions", path, value, report.reviewed_labels(tmp_path))
    assert coverage["episodes"] == 50 and coverage["native_successes"] == 47
    assert coverage["confirmed_wrong_object_cases_so_far"] == 1
    assert coverage["status"] == "pending_review" and coverage["confirmed_fraction_all_episodes"] is None
    assert coverage["unreviewed_failures"] == 1 and coverage["reviewed_ambiguous_failures"] == 1
    assert rows[0]["failure_type"] == "success" and rows[0]["reviewed"] is False
    assert rows[49]["failure_type"] == "other/ambiguous — not reviewed"
    assert sum(row["count"] for row in taxonomy) == 50
    records.append({**records[0], "state_id": 49, "seed": 100049,
                    "failure_type": "correct grasp but placement failure"})
    write(labels, {"records": records})
    _, complete, _ = report.ood_review_rows("object_1", "pi05_libero", "both_new_regions", path, value, report.reviewed_labels(tmp_path))
    assert complete["status"] == "complete_review" and complete["confirmed_fraction_all_episodes"] == .02
    assert [complete["confirmed_wilson95_low"], complete["confirmed_wilson95_high"]] == report.wilson(1, 50)


def test_ood_review_requires_exact_identity_and_verified_source_hashes(tmp_path):
    path, value, _, records = ood_review_fixture(tmp_path)
    reviews = report.reviewed_labels(tmp_path)
    value["episodes"][47]["seed"] += 1
    with pytest.raises(ValueError, match="noise seeds"):
        report.ood_review_rows("object_1", "pi05_libero", "both_new_regions", path, value, reviews)
    Path(records[0]["video"]).write_bytes(b"modified fixture")
    with pytest.raises(ValueError, match="source hash changed"):
        report.reviewed_labels(tmp_path)
