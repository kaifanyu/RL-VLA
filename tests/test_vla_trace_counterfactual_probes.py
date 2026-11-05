"""Prompt transfer preserves physical identity classes and frozen probe weights."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def scorer():
    path = Path(__file__).resolve().parents[1]/"scripts/score_vla_trace_counterfactual_probes.py"
    spec = importlib.util.spec_from_file_location("counterfactual_probes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def probe():
    return {"classes": np.arange(3), "center": np.zeros(3), "scale": np.ones(3),
            "coefficients": np.r_[np.eye(3), np.zeros((1, 3))]}


def test_requested_class_switch_does_not_relabel_identity(scorer, tmp_path):
    labels = np.array([1, 2, 0, 0])
    x = np.eye(3)[labels]
    row = {"state_id": 150, "camera": "image", "labels": labels, "valid": np.ones(4, bool),
           "target_occupancy": np.array([[1., 0.], [0., 0.]]),
           "distractor_occupancy": np.array([[0., 1.], [0., 0.]]),
           "original": x, "counterfactual": x}
    result = scorer.score_kind(probe(), [row], tmp_path, "image_features")
    assert result["original"]["confusion"] == result["counterfactual"]["confusion"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert result["original"]["requested_object"]["physical_identity_class"] == 1
    assert result["counterfactual"]["requested_object"]["physical_identity_class"] == 2
    assert result["counterfactual"]["requested_object"]["mean_iou_visible"] == 1
    assert result["paired_prompt_change"]["max_abs_score_change"] == 0


def test_changed_language_free_control_rejected(scorer, tmp_path):
    x = np.eye(3)[[1, 2, 0, 0]]
    row = {"state_id": 150, "camera": "image", "labels": np.array([1, 2, 0, 0]), "valid": np.ones(4, bool),
           "target_occupancy": np.array([[1., 0.], [0., 0.]]),
           "distractor_occupancy": np.array([[0., 1.], [0., 0.]]),
           "original": x, "counterfactual": x[:, [0, 2, 1]]}
    with pytest.raises(ValueError, match="Language-free"):
        scorer.score_kind(probe(), [row], tmp_path, "image_features")


def test_roc_ties_and_missing_class_are_well_defined(scorer):
    curve = scorer.binary_roc([False, True, False, True], [0., 0., 1., 1.])
    assert curve["false_positive_rate"] == [0., .5, 1.]
    assert curve["true_positive_rate"] == [0., .5, 1.]
    assert scorer.binary_roc([False, False], [0., 1.])["thresholds"] == []


def fixture_campaign(root, model, scorer):
    task = "object_0"
    selection = root/"methods/rldt"/task/"selection.json"
    save(selection, {"validation_counts": {"25": 20, "50": 20, "100": 10}, "iteration": 25,
                     "locked": "before-test", "test_results_seen": False,
                     "checkpoint": str(selection.parent/"training/iteration_000025")})
    selection_hash = scorer.sha(selection)
    probes = root/"probes"/task/model
    spec = {"fit_split": "train", "fit_states": 100, "feature_kinds": list(scorer.KINDS),
            "selection_sha256": selection_hash}
    save(probes/"specification.json", spec)
    save(probes/"metrics.json", {"complete": True, "task": task, "model": model, "specification": spec})
    np.save(probes/"fixed_projection.npy", np.eye(3, dtype=np.float32))
    for kind in scorer.KINDS:
        np.savez(probes/f"{kind}_probe.npz", **probe())
    save(root/"traces"/task/"summary.json", {"complete": True, "task": task, "states": 50,
                                             "selection_sha256": selection_hash})
    protocol = root/"tasks"/task/"protocol.json"
    save(protocol, {"tasks": {task: {"test_state_ids": list(range(150, 200))}}})
    bank = root/"observation_bank"/task/"test"
    records = []
    spans = [{"name": "base_0_rgb", "start": 0, "end": 4, "valid": True, "grid": [2, 2]},
             {"name": "left_wrist_0_rgb", "start": 4, "end": 8, "valid": True, "grid": [2, 2]},
             {"name": "right_wrist_0_rgb", "start": 8, "end": 12, "valid": False, "grid": [2, 2]}]
    x = np.tile(np.eye(3)[[1, 2, 0, 0]], (3, 1))[None].astype(np.float32)
    for kind in scorer.KINDS:
        np.savez(probes/f"{kind}_test_predictions.npz", scores=np.tile(x[0, :4].astype(float), (100, 1)),
                 labels=np.tile([1, 2, 0, 0], 100), state_ids=np.repeat(np.arange(150, 200), 8),
                 cameras=np.tile(np.repeat(["image", "wrist_image"], 4), 50))
    for state in range(150, 200):
        folder = bank/f"state_{state:06d}"
        folder.mkdir(parents=True)
        np.savez(folder/"observation.npz", state=np.array([state]))
        a, b = np.array([[1, 0], [0, 0]], bool), np.array([[0, 1], [0, 0]], bool)
        masks = {f"{camera}__{name}": values for camera in ("image", "wrist_image")
                 for name, values in (("alphabet_soup_1", a), ("butter_1", b))}
        np.savez(folder/"masks.npz", **masks)
        record = {"initial_state_id": state, "prompt": "pick the alphabet soup and place it in the basket",
                  "protocol_sha256": scorer.sha(protocol), "masks_path": str(folder/"masks.npz"),
                  "observation_path": str(folder/"observation.npz"),
                  "masks_sha256": scorer.sha(folder/"masks.npz"),
                  "observation_sha256": scorer.sha(folder/"observation.npz")}
        records.append(record)
        trace = root/"traces"/task/model/f"state_{state:06d}"
        trace.mkdir(parents=True)
        for name in ("prefix", "counterfactual_prefix"):
            np.savez(trace/f"{name}.npz", image_features=x, prefix_features=x)
        save(trace/"metadata.json", {"bank_record": record, "camera_spans": spans, "prefix_features_sha256": "same-frozen-feature"})
        save(root/"features"/model/task/"test"/f"state_{state:06d}"/"metadata.json", {"prefix_features_sha256": "same-frozen-feature"})
        cf_metadata = "counterfactual_prefix_metadata.json" if model == "pi05_base" else "counterfactual_metadata.json"
        save(trace/cf_metadata, {"instruction": "pick the butter and place it in the basket",
                                "camera_spans": spans})
    manifest = bank/"manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in records)+"\n")
    save(bank/"summary.json", {"complete": True, "manifest_sha256": scorer.sha(manifest)})
    return probes


@pytest.mark.parametrize("model", ["pi05_base", "pi05_libero"])
def test_complete_fifty_state_scoring_without_refit_or_input_mutation(scorer, tmp_path, model):
    probes = fixture_campaign(tmp_path, model, scorer)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in probes.iterdir()}
    assert scorer.ready(tmp_path, "object_0", model)
    scorer.run(tmp_path, "object_0", model)
    result = json.loads((tmp_path/"probes_counterfactual/object_0"/model/"metrics.json").read_text())
    assert result["complete"] and result["states"] == 50 and result["refit"] is False
    assert result["measurements"]["prefix_features"]["counterfactual"]["accuracy"] == 1
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == expected for p, expected in before.items())
    for kind in scorer.KINDS:
        with np.load(tmp_path/"probes_counterfactual/object_0"/model/f"{kind}_prediction_maps.npz") as data:
            assert data["original"].shape == (100, 4, 3)
    selection = tmp_path/"methods/rldt/object_0/selection.json"
    bad = json.loads(selection.read_text())
    bad["iteration"] = 50
    save(selection, bad)
    with pytest.raises(ValueError, match="earliest tie"):
        scorer.ready(tmp_path, "object_0", model)
