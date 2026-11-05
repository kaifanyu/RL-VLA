import json
from pathlib import Path

import pytest

import prepare_vla_trace_review_queue as queue


def fixture(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    episodes = [{"reset": {"initial_state_id": 150}, "seed": 100000, "success": False},
                {"reset": {"initial_state_id": 151}, "seed": 100001, "success": True}]
    report.write_text(json.dumps({"episodes": episodes}))
    item = {"task": "object_2", "domain": "main", "policy": "rldt_25",
            "variant": "test", "report": str(report)}
    monkeypatch.setattr(queue, "report_episodes", lambda *args: episodes)
    return report, item


def existing(tmp_path, report, *, reviewed=False, state=150):
    directory = tmp_path / "video_reviews/object_2/review"
    directory.mkdir(parents=True)
    (directory / "sheet_000.jpg").write_bytes(b"existing image")
    manifest = {"report": str(report), "report_sha256": queue.sha(report), "reviewed": reviewed,
                "entries": [{"state_id": state, "seed": 100000, "sheet": "sheet_000.jpg",
                             "evidence": {"sustained_grasp_candidates": ["tomato_sauce_1"]}}]}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_existing_exact_evidence_reused_without_marking_reviewed(tmp_path, monkeypatch):
    report, item = fixture(tmp_path, monkeypatch)
    path = existing(tmp_path, report)
    before = path.read_bytes()
    monkeypatch.setattr(queue, "build_primary", lambda *args: pytest.fail("must reuse"))
    result = queue.prepare(tmp_path, item)
    assert result["episodes"] == 2 and result["native_failures"] == 1
    assert result["reviewed"] is False and result["requires_explicit_image_review"] is True
    assert result["manifest"]["path"] == str(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.rglob("*review_complete*"))


@pytest.mark.parametrize("reviewed,state,error", [(True, 150, "must not claim"),
                                                  (False, 152, "exactly the native failures")])
def test_invalid_existing_review_evidence_rejected(tmp_path, monkeypatch, reviewed, state, error):
    report, item = fixture(tmp_path, monkeypatch)
    existing(tmp_path, report, reviewed=reviewed, state=state)
    with pytest.raises(ValueError, match=error):
        queue.prepare(tmp_path, item)


def test_report_mutation_during_build_is_rejected(tmp_path, monkeypatch):
    report, item = fixture(tmp_path, monkeypatch)

    def build(path, task, output):
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(json.dumps({"report_sha256": queue.sha(report)}))
        report.write_text(report.read_text() + "\n")

    monkeypatch.setattr(queue, "build_primary", build)
    with pytest.raises(ValueError, match="changed during evidence"):
        queue.prepare(tmp_path, item)


def test_partial_output_never_overwritten(tmp_path, monkeypatch):
    report, item = fixture(tmp_path, monkeypatch)
    output = tmp_path / "video_reviews/object_2" / f"main_rldt_25_test_ai_prepare_objects_{queue.sha(report)[:12]}"
    output.mkdir(parents=True)
    saved = output / "partial"
    saved.write_bytes(b"retain evidence")
    with pytest.raises(FileExistsError):
        queue.prepare(tmp_path, item)
    assert saved.read_bytes() == b"retain evidence"
