"""Native contact evidence and undefined-control accounting remain disjoint."""

import copy
import json
from pathlib import Path

import pytest

from rl_vla.diagnostics import contact_support_overlay_v1 as overlay


@pytest.fixture
def resolved(tmp_path):
    identity = {
        "domain": "causal",
        "task": "object_1",
        "policy": "rldt_100",
        "variant": "random:background_fill",
    }
    rows = []
    for i in range(9):
        rows.append(
            {
                **identity,
                "episode": i,
                "state_id": 150 + i,
                "seed": 100000 + i,
                "report": "report",
                "report_sha256": "hash",
                "video": f"video{i}",
                "success": False,
                "automatic": True,
                "reviewed": False,
                "telemetry_complete": True,
                "target_grasp_candidate": i == 0,
                "non_target_grasp_candidate": i == 1,
                "mixed_target_non_target_candidates": False,
                "no_grasp_candidate": i > 1,
                "first_confirmation_tie": False,
                "first_candidate_role": "target" if i == 0 else "distractor" if i == 1 else "none",
                "sustained_grasp_candidates": ["cream_cheese_1"] if i == 0 else [],
                "provisional_failure_type": "other/ambiguous",
            }
        )
    unavailable = {
        **identity,
        "episode": 9,
        "state_id": 159,
        "seed": 100009,
        "status": "structural_NA",
        "success": None,
        "partial_telemetry_excluded_from_contact_classification": True,
    }
    report = {"identity": identity, "rows": rows, "unavailable": [unavailable]}
    plan = [{**identity, "episodes": 10}]
    return report, plan


def test_one_undefined_case_disables_whole_condition_rates(resolved):
    report, plan = resolved
    native, na, rates, manual = overlay.assemble([report], plan, {})
    assert len(native) == 9 and len(na) == 1 and not manual
    assert rates[0]["native_successes"] is None
    assert rates[0]["observed_native_successes"] == 0
    assert rates[0]["target_grasp_candidate_count"] == 1
    assert rates[0]["non_target_grasp_candidate_count"] == 1
    assert all(value is None for key, value in rates[0].items() if key.endswith("_rate"))
    assert all(row["reviewed"] is False for row in native)


def test_full_native_condition_retains_original_denominator(resolved):
    report, plan = resolved
    last = {**report["rows"][-1], "episode": 9, "state_id": 159, "seed": 100009}
    report["rows"].append(last)
    report["unavailable"] = []
    native, na, rates, _ = overlay.assemble([report], plan, {})
    assert len(native) == 10 and not na
    assert rates[0]["target_grasp_candidate_rate"] == 0.1
    assert rates[0]["native_successes"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_report",
        "duplicate_case",
        "native_null",
        "na_false",
        "automatic_reviewed",
        "wrong_na_domain",
    ],
)
def test_no_duplicate_outcome_imputation_or_review_inference(resolved, mutation):
    report, plan = resolved
    reports = [report]
    if mutation == "duplicate_report":
        reports.append(copy.deepcopy(report))
    elif mutation == "duplicate_case":
        report["unavailable"][0]["episode"] = 8
    elif mutation == "native_null":
        report["rows"][0]["success"] = None
    elif mutation == "na_false":
        report["unavailable"][0]["success"] = False
    elif mutation == "automatic_reviewed":
        report["rows"][0]["reviewed"] = True
    else:
        for row in [report["identity"], *report["rows"], *report["unavailable"], plan[0]]:
            row["task"] = "object_2"
    with pytest.raises(ValueError):
        overlay.assemble(reports, plan, {})


def test_verified_manual_labels_join_separately_and_never_change_automatic_flag(resolved):
    report, plan = resolved
    row = report["rows"][1]
    manual = {
        ("object_1", "hash", 151, 100001): {
            "success": False,
            "video": row["video"],
            "failure_type": "wrong object",
            "grasped_object_id": "alphabet_soup_1",
            "label_source": "review.json",
            "label_source_sha256": "sha",
        }
    }
    native, _, _, calibration = overlay.assemble([report], plan, manual)
    assert len(calibration) == 1 and native[1]["manual_reviewed"]
    assert native[1]["reviewed"] is False and native[1]["automatic"] is True
    manual[("object_1", "hash", 151, 100001)]["video"] = "other"
    with pytest.raises(ValueError):
        overlay.assemble([report], plan, manual)


def test_partial_prefix_never_enters_native_contact_classifier(tmp_path, monkeypatch):
    run = tmp_path / "run"
    destination = run / overlay.DESTINATION
    destination.mkdir(parents=True)
    cases = []
    report = tmp_path / "report.json"
    report.write_text("{}")
    condition = {
        "policy": "rldt_100",
        "condition": "random:background_fill",
        "summary": overlay.validation.binding(report),
    }
    for i in range(10):
        row = {
            "episode": i,
            "seed": 100000 + i,
            "success": None if i == 9 else False,
            "primitive_steps": 3,
            "video": str(tmp_path / f"{i}.mp4"),
            "reset": {
                "object_telemetry": {
                    "objects": {"cream_cheese_1": {}, "alphabet_soup_1": {}},
                    "simulator_state": [0],
                }
            },
        }
        case = {
            "task": "object_1",
            "policy": "rldt_100",
            "condition": "random:background_fill",
            "episode": i,
            "state_id": 150 + i,
            "seed": 100000 + i,
            "status": "structural_NA" if i == 9 else "completed",
            "row": row,
        }
        if i == 9:
            witness = tmp_path / "na.json"
            witness.write_text(
                json.dumps(
                    {"decision": 1, "first_failing_camera": "image", "support": [], "artifacts": []}
                )
            )
            case["structural_evidence"] = overlay.validation.binding(witness)
        else:
            telemetry = tmp_path / f"{i}.json"
            telemetry.write_text("[]")
            case["artifacts"] = {"telemetry": overlay.validation.binding(telemetry)}
        cases.append(case)
    called = []

    def only_native(*args, **kwargs):
        called.append(kwargs["success"])
        assert type(kwargs["success"]) is bool
        return {"automatic": True, "reviewed": False, "telemetry_complete": True}

    monkeypatch.setattr(overlay, "evidence", only_native)
    monkeypatch.setattr(overlay, "causal_steps", lambda *args: iter(()))
    reports = overlay._resumed_reports(
        run,
        destination,
        {"conditions": [condition]},
        {"cases": cases},
        {},
        overlay.legacy().FileCache({}),
    )
    assert called == [False] * 9
    assert len(reports[0]["rows"]) == 9 and len(reports[0]["unavailable"]) == 1
    # Cached immutable per-report evidence does not reparse or classify again.
    overlay._resumed_reports(
        run,
        destination,
        {"conditions": [condition]},
        {"cases": cases},
        {},
        overlay.legacy().FileCache({}),
    )
    assert called == [False] * 9
    assert "provisional_failure_type" not in reports[0]["unavailable"][0]


@pytest.fixture
def publication(tmp_path, resolved):
    report, plan = resolved
    report["rows"].append({**report["rows"][-1], "episode": 9, "state_id": 159, "seed": 100009})
    report["unavailable"] = []
    run = tmp_path / "run"
    root = run / overlay.DESTINATION
    root.mkdir(parents=True)
    report_path = tmp_path / "native_report.json"
    report_path.write_text("{}")
    for row in report["rows"]:
        row["report"] = str(report_path)
        row["report_sha256"] = overlay.sha(report_path)
    native, na, rates, manual = overlay.assemble([report], plan, {})
    tables = root / "tables"
    tables.mkdir()
    legacy = overlay.legacy()
    for filename, rows, fields in (
        ("contact_cases.csv", native, list(native[0])),
        ("contact_candidate_rates.csv", rates, list(rates[0])),
        ("contact_manual_calibration.csv", manual, ["task"]),
        ("support_unavailable_cases.csv", na, ["status", "success"]),
    ):
        legacy.write_csv(tables / filename, rows, fields)
    protocol = run / "tasks/object_1/protocol.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(
        json.dumps({"tasks": {"object_1": {"test_state_ids": list(range(150, 200))}}})
    )
    sources = [
        overlay.Path(overlay.__file__).resolve(),
        overlay.LEGACY_SOURCE,
        overlay.Path(overlay.validation.__file__).resolve(),
        overlay.ROOT / "src/rl_vla/diagnostics/contact_evidence.py",
        overlay.ROOT / "scripts/analyze_vla_trace_contacts_support_v1.py",
        report_path,
        protocol,
    ]
    manifest = root / "inputs.json"
    manifest.write_text(json.dumps({"inputs": {str(p): overlay.sha(p) for p in sources}}))
    summary = {
        "schema": "contact-support-overlay-v1",
        "complete": False,
        "automatic_labels_reviewed": False,
        "rule": overlay.RULE,
        "source_manifest": overlay.validation.binding(manifest),
        "tables": {p.name: overlay.validation.binding(p) for p in tables.iterdir()},
        "completed_native_cases": 10,
        "structural_na_cases": 0,
        "resolved_cases": 10,
        "resolved_reports": 1,
        "completed_main_ood_native_cases": 0,
        "completed_causal_native_cases": 10,
        "manual_reviewed_cases": 0,
        "telemetry_complete_native_cases": 10,
        "support_execution_adopted": False,
        "persistent_errors": [],
    }
    overlay.atomic(root / "summary.json", summary)
    assert overlay.verify_overlay(run)["resolved_cases"] == 10
    return run, root, summary


def test_verified_pending_publication_never_claims_final_coverage(publication):
    run, _, _ = publication
    assert overlay.verify_overlay(run, require_complete=True) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "summary_count",
        "false_complete",
        "source_omission",
        "seed",
        "duplicate",
        "candidate_rate",
        "raw_table_bytes",
    ],
)
def test_consumer_rejects_tampered_or_rehashed_publication(publication, mutation):
    import csv

    run, root, summary = publication
    if mutation == "summary_count":
        summary["completed_native_cases"] = 11
    elif mutation == "false_complete":
        summary["complete"] = True
    elif mutation == "source_omission":
        path = Path(summary["source_manifest"]["path"])
        value = json.loads(path.read_text())
        value["inputs"].pop(str(Path(overlay.__file__).resolve()))
        overlay.atomic(path, value)
        summary["source_manifest"] = overlay.validation.binding(path)
    else:
        name = (
            "contact_candidate_rates.csv" if mutation == "candidate_rate" else "contact_cases.csv"
        )
        path = Path(summary["tables"][name]["path"])
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if mutation == "seed":
            rows[0]["seed"] = "100009"
        elif mutation == "duplicate":
            rows[1] = copy.deepcopy(rows[0])
        elif mutation == "candidate_rate":
            rows[0]["target_grasp_candidate_rate"] = "0.9"
        else:
            rows[0]["success"] = "None"
        overlay.legacy().write_csv(path, rows, list(rows[0]))
        if mutation != "raw_table_bytes":
            summary["tables"][name] = overlay.validation.binding(path)
    overlay.atomic(root / "summary.json", summary)
    with pytest.raises(ValueError):
        overlay.verify_overlay(run)


def test_original_raw_source_identity_is_checked_without_reparsing_arrays(tmp_path):
    path = tmp_path / "native_chunk.npz"
    path.write_bytes(b"cached-metadata-array-file")
    manifest = tmp_path / "source_inputs.jsonl"
    row = {
        "path": str(path),
        **overlay.legacy().stat_record(path),
        "metadata_json_sha256": "original-metadata-hash",
    }
    manifest.write_text(json.dumps(row) + "\n")
    files = overlay.legacy().FileCache({})
    overlay.verify_legacy_raw_sources(manifest, files)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="since contact extraction"):
        overlay.verify_legacy_raw_sources(manifest, files)


def test_original_full_telemetry_hash_is_checked_even_with_preserved_stat(tmp_path):
    import os

    path = tmp_path / "telemetry.json"
    path.write_bytes(b"original")
    stat = path.stat()
    manifest = tmp_path / "source_inputs.jsonl"
    row = {"path": str(path), **overlay.legacy().stat_record(path), "sha256": overlay.sha(path)}
    manifest.write_text(json.dumps(row) + "\n")
    path.write_bytes(b"mutated!")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="bytes changed"):
        overlay.verify_legacy_raw_sources(manifest, overlay.legacy().FileCache({}))
