import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rl_vla.diagnostics import figure_artifacts as audit


def probe_fixture(root):
    folder = root / "probes/object_0/pi05_base"
    folder.mkdir(parents=True)
    spec = {"fit_split": "train", "fit_states": 100}
    summary = {"complete": True, "task": "object_0", "model": "pi05_base", "specification": spec,
               "measurements": {kind: {split: {"status": "complete"} for split in ("validation", "test", "source")}
                                for kind in ("image_features", "prefix_features")}}
    audit.write(folder / "metrics.json", summary)
    audit.write(folder / "specification.json", spec)
    np.save(folder / "fixed_projection.npy", np.ones((4, 2), np.float32))
    for kind in ("image_features", "prefix_features"):
        np.savez(folder / f"{kind}_probe.npz", center=np.zeros(2), scale=np.ones(2),
                 coefficients=np.zeros((3, 3)), classes=np.arange(3))
        for split in ("validation", "test", "source"):
            np.savez(folder / f"{kind}_{split}_predictions.npz", scores=np.zeros((2, 3)), labels=np.zeros(2),
                     state_ids=np.array([150, 151]), cameras=np.array(["image", "wrist_image"]))
            (folder / f"{kind}_{split}_localization.csv").write_text("state_id,camera,region,iou\n150,image,target,0.5\n")
    return folder


def figures(folder):
    image = Image.new("RGB", (20, 20), "white")
    image.save(folder / "confusion.png")
    image.save(folder / "confusion.pdf")


def test_early_complete_metrics_do_not_certify_missing_figures(tmp_path):
    folder = probe_fixture(tmp_path)
    result = audit.validate_cell(tmp_path, "probes", "object_0", "pi05_base")
    assert result["status"] == "pending"
    assert not audit.cell_ready(tmp_path, "probes", "object_0", "pi05_base")
    figures(folder)
    result = audit.validate_cell(tmp_path, "probes", "object_0", "pi05_base")
    assert result["status"] == "complete"
    assert audit.cell_ready(tmp_path, "probes", "object_0", "pi05_base")
    audit.verify_manifest(result["manifest"]["path"], full_hash=True)


def test_invalid_pdf_waits_and_changed_completed_png_fails(tmp_path):
    folder = probe_fixture(tmp_path)
    figures(folder)
    (folder / "confusion.pdf").write_bytes(b"%PDF-1.4 truncated")
    assert audit.validate_cell(tmp_path, "probes", "object_0", "pi05_base")["status"] == "pending"
    figures(folder)
    ready = audit.validate_cell(tmp_path, "probes", "object_0", "pi05_base")
    Image.new("RGB", (20, 20), "black").save(folder / "confusion.png")
    with pytest.raises(ValueError, match="changed"):
        audit.verify_manifest(ready["manifest"]["path"])
    assert not audit.cell_ready(tmp_path, "probes", "object_0", "pi05_base")


def test_csv_npz_content_and_declared_hash_guards(tmp_path):
    path = tmp_path / "truncated.csv"
    path.write_text("a,b\n1\n")
    with pytest.raises(ValueError, match="truncated"):
        audit.artifact(path)
    path = tmp_path / "nonfinite.npz"
    np.savez(path, values=np.array([np.nan]))
    with pytest.raises(ValueError, match="nonfinite"):
        audit.artifact(path)
    path = tmp_path / "good.json"
    audit.write(path, {"complete": True})
    with pytest.raises(ValueError, match="declared hash"):
        audit.artifact(path, expected_sha="0" * 64)


def test_explicit_na_metrics_require_exact_boolean_masks_and_never_allow_infinity(tmp_path):
    path=tmp_path/"metrics.npz"
    np.savez(path,values=np.array([1.,np.nan]),valid=np.array([True,False]))
    record=audit.artifact(path)
    assert record["validation"]["arrays"]["values"]["explicit_na"]["explicit_na_cells"]==1
    np.savez(path,first_command_cosine=np.array([np.nan]),first_command_cosine_valid=np.array([False]))
    audit.artifact(path)
    for values,mask in (([1.,np.nan],[False,False]),([1.,np.inf],[True,False])):
        np.savez(path,values=np.asarray(values),valid=np.asarray(mask))
        with pytest.raises(ValueError,match="validity mask"):
            audit.artifact(path)
    np.savez(path,decoded_commands=np.array([np.nan]),valid=np.array([False]))
    with pytest.raises(ValueError,match="validity mask"):
        audit.artifact(path)


def test_missing_supplemental_groups_and_false_global_completion_fail(tmp_path):
    folder = tmp_path / "supplemental_analysis/object_0"
    audit.write(folder / "summary.json", {"complete": True, "task": "object_0", "states": 50,
                                          "coverage": {"states": 50}, "required_artifact_groups": {}})
    assert audit.validate_cell(tmp_path, "supplemental_analysis", "object_0", "all_policies")["status"] == "pending"
    summary = audit.refresh(tmp_path)
    assert summary["expected_cells"] == 27
    assert len(summary["cells"]) == 27
    assert summary["complete"] is False and summary["passed"] is None
    summary["complete"] = summary["passed"] = True
    audit.write(tmp_path / "figure_artifacts/summary.json", summary)
    with pytest.raises(ValueError, match="complete manifest"):
        audit.verify_coverage(tmp_path)
