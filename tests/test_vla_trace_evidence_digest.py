"""Synthetic digest fixtures verify provenance and faithful copying, never results."""

import csv
import hashlib
import json

import pytest

from rl_vla.diagnostics.evidence_digest import Sources, build_digest, decode_cell


def table(root, name, rows):
    path = root / "tables" / name
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    return path


def attention_row(step=5):
    return {"task": "object_0", "policy": "rldt_000025", "camera": "image", "step": step,
            "region": "target", "metric": "region_density", "states": 50, "baseline_mean": .01,
            "checkpoint_mean": .02, "difference": .01, "paired_bootstrap95_low": -.005,
            "paired_bootstrap95_high": .025, "bootstrap_unit": "whole observation", "bootstrap_draws": 20000}


def test_missing_cells_remain_pending_not_zero(tmp_path):
    result = build_digest(tmp_path)
    digest = json.loads((tmp_path / "evidence_digest.json").read_text())
    assert result["pending_cells"] == 120
    assert not result["evidence_complete"]
    assert len(digest["success"]) == 45 and len(digest["diagnostic_inventory"]) == 75
    assert all("successes" not in row for row in digest["success"])
    assert digest["sources"]["tables/success.csv"]["status"] == "missing"
    assert len(digest["failure_review_coverage"]["pending_reports"]) == 84
    assert not digest["failure_review_coverage"]["complete"]


def test_run_readme_is_hashed_without_parsing_and_arrays_remain_excluded(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Frozen artifact index\n\n[Report](report.md)\n")
    build_digest(tmp_path)
    digest = json.loads((tmp_path / "evidence_digest.json").read_text())
    assert digest["sources"]["README.md"] == {
        "status": "present", "sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
        "bytes": len(readme.read_bytes())}
    (tmp_path / "large_attention.npz").write_bytes(b"Never an indexable source")
    with pytest.raises(ValueError, match="Markdown documentation"):
        Sources(tmp_path).index("large_attention.npz")


def test_final_freeze_and_resource_receipts_are_bound_without_changing_pending_evidence(tmp_path):
    names = ["resource_cost_final_v1.json",
             "compiler_profile_integration/final_writer_freeze_v1/receipt.json",
             "contact_analysis_structural_na_v1/final_acceptance_v1.json"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"scope":"immutable final receipt"}\n')
    build_digest(tmp_path)
    digest = json.loads((tmp_path / "evidence_digest.json").read_text())
    assert not digest["evidence_complete"]
    for name in names:
        assert digest["sources"][name]["sha256"] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()


def test_partial_failure_review_counts_do_not_establish_complete_coverage(tmp_path):
    table(tmp_path, "wrong_object.csv", [{"task": "object_1", "split": "validation", "policy": "pi05_libero",
          "status": "complete_review", "episodes": 50, "unreviewed_failures": 0, "review_coverage_complete": True}])
    table(tmp_path, "ood_wrong_object.csv", [{"task": "object_1", "policy": "pi05_libero", "variant": "both_new_regions",
          "status": "pending_review", "episodes": 50, "unreviewed_failures": 1, "review_coverage_complete": False,
          "confirmed_wrong_object_cases_so_far": 3, "confirmed_fraction_all_episodes": ""}])
    build_digest(tmp_path)
    digest = json.loads((tmp_path / "evidence_digest.json").read_text())
    assert len(digest["failure_review_coverage"]["pending_reports"]) == 83
    assert digest["unseen_layouts"]["wrong_object_failures"][0]["confirmed_fraction_all_episodes"] is None
    assert not digest["evidence_complete"]


def test_existing_means_and_intervals_are_copied_with_exact_source_hash(tmp_path):
    path = table(tmp_path, "paired_attention.csv", [attention_row()])
    build_digest(tmp_path)
    digest = json.loads((tmp_path / "evidence_digest.json").read_text())
    row = digest["attention"]["all_predeclared_step_comparisons"][0]
    assert row["policy"] == "rldt_25"
    assert (row["baseline_mean"], row["checkpoint_mean"], row["difference"]) == (.01, .02, .01)
    assert (row["paired_bootstrap95_low"], row["paired_bootstrap95_high"]) == (-.005, .025)
    assert row["source"] == {"path": "tables/paired_attention.csv", "csv_line": 2}
    assert digest["sources"]["tables/paired_attention.csv"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert decode_cell("False") is False and decode_cell("") is None


def test_digest_is_stable_when_only_live_monitor_files_change(tmp_path):
    table(tmp_path, "paired_attention.csv", [attention_row()])
    build_digest(tmp_path)
    before = [(tmp_path / name).read_bytes() for name in ("evidence_digest.json", "evidence_digest.md")]
    (tmp_path / "acceptance_gates.json").write_text('{"updated_utc":"changed","complete":false}')
    (tmp_path / "gpu_telemetry.csv").write_text("new telemetry\n")
    (tmp_path / "report.md").write_text("new timestamp\n")
    (tmp_path / "large_attention.npz").write_bytes(b"Not a valid NPZ, deliberately never read")
    build_digest(tmp_path)
    assert before == [(tmp_path / name).read_bytes() for name in ("evidence_digest.json", "evidence_digest.md")]


def test_undeclared_flow_step_is_rejected(tmp_path):
    table(tmp_path, "paired_attention.csv", [attention_row(step=2)])
    with pytest.raises(ValueError, match="predeclared flow"):
        build_digest(tmp_path)
