"""Presentation completeness cannot hide missing research/review evidence."""
import copy
from pathlib import Path

import pytest

from rl_vla.diagnostics import ood_figures as plots


def fixture(root, *, pending_review=False):
    scores, pairs, reviews, categories, episodes = [], [], [], [], []
    for variant in plots.VARIANTS:
        for policy in plots.POLICIES:
            cell = {"task": "object_1", "policy": policy, "variant": variant}
            path = root / "sources" / f"{policy}_{variant}.json"
            plots.write(path, {"synthetic_fixture": True, **cell})
            scores.append({**cell, "episodes": 50, "successes": 45, "success_rate": .9,
                           "wilson95_low": .78, "wilson95_high": .96})
            if policy != "pi05_libero":
                pairs.append({**cell, "difference": 0., "paired_bootstrap95": [-.08, .08],
                              "recovered": 2, "regressed": 2})
            reviews.append({**cell, "status": "pending_review" if pending_review else "complete_review",
                            "review_coverage_complete": not pending_review, "episodes": 50,
                            "unreviewed_failures": int(pending_review), "report": str(path),
                            "report_sha256": plots.sha(path)})
            counts = {"success": 45, "wrong object": 2, "correct grasp but placement failure": 2,
                      "other/ambiguous": 0 if pending_review else 1, "not reviewed": int(pending_review)}
            categories.extend({**cell, "failure_type": kind, "count": count} for kind, count in counts.items())
            episodes.append({**cell, "report": str(path)})
    return scores, pairs, reviews, categories, episodes


def test_absent_cells_are_explicit_pending_not_zero(tmp_path):
    data = plots.inputs(tmp_path, "object_1", [], [], [], [], [])
    assert len(data["success"]) == 16 and len(data["paired"]) == 12
    assert not plots.complete_data(data)
    assert all("successes" not in row and row["status"] == "pending" for row in data["success"])
    assert all(row["count"] is None for row in data["categories"])
    assert plots.verify(tmp_path, "object_1")["passed"] is None


def test_complete_native_results_need_all_failed_cases_reviewed(tmp_path):
    rows = fixture(tmp_path, pending_review=True)
    assert not plots.complete_data(plots.inputs(tmp_path, "object_1", *rows))
    rows = fixture(tmp_path)
    assert plots.complete_data(plots.inputs(tmp_path, "object_1", *rows))
    rows[2][0]["report_sha256"] = "0" * 64
    assert not plots.complete_data(plots.inputs(tmp_path, "object_1", *rows))


def test_duplicate_cells_unknown_categories_and_inconsistent_counts_fail(tmp_path):
    rows = fixture(tmp_path)
    duplicate = copy.deepcopy(rows)
    duplicate[0].append(duplicate[0][0])
    with pytest.raises(ValueError, match="duplicate"):
        plots.inputs(tmp_path, "object_1", *duplicate)
    for kind, count in (("automatic grasp proxy", 1), ("wrong object", -1), ("wrong object", 1)):
        broken = copy.deepcopy(rows)
        broken[3][1].update(failure_type=kind, count=count)
        with pytest.raises(ValueError, match="taxonomy"):
            plots.inputs(tmp_path, "object_1", *broken)


def test_actual_png_pdf_manifest_byte_and_review_guards(tmp_path):
    value = plots.render(tmp_path, "object_1", *fixture(tmp_path))
    assert value["complete"] and len(value["artifacts"]) == 9
    assert plots.verify(tmp_path, "object_1", full_hash=True)["passed"]
    assert plots.render(tmp_path, "object_1", *fixture(tmp_path)) == value
    assert plots.verify(tmp_path, "object_1", full_hash=True)["passed"]
    summary = tmp_path / "figures/ood/object_1/summary.json"
    original = summary.read_bytes()
    value["complete_review_cells"] = 15
    plots.write(summary, value)
    assert plots.verify(tmp_path, "object_1")["passed"] is False
    summary.write_bytes(original)
    png = next(Path(row["path"]) for row in value["artifacts"] if row["path"].endswith(".png"))
    png.write_bytes(b"changed")
    assert plots.verify(tmp_path, "object_1", full_hash=True)["passed"] is False


def test_missing_artifacts_and_changed_sources_cannot_pass(tmp_path):
    value = plots.render(tmp_path, "object_1", *fixture(tmp_path))
    summary = tmp_path / "figures/ood/object_1/summary.json"
    original = summary.read_bytes()
    value["artifacts"].pop()
    plots.write(summary, value)
    assert plots.verify(tmp_path, "object_1")["passed"] is False
    summary.write_bytes(original)
    source = next(iter(value["sources"]))
    Path(source).write_text('{"changed":true}')
    assert plots.verify(tmp_path, "object_1")["passed"] is False


def test_invalid_summary_is_failure_not_unhandled_or_passed(tmp_path):
    path = tmp_path / "figures/ood/object_1/summary.json"
    path.parent.mkdir(parents=True)
    for value in ("{", "[]"):
        path.write_text(value)
        assert plots.verify(tmp_path, "object_1")["passed"] is False
    plots.write(path, {"complete": False})
    assert plots.verify(tmp_path, "object_1")["passed"] is None
