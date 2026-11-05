import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("contact_review", Path(__file__).resolve().parents[1]/"scripts/review_vla_trace_episodes.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_incidental_contact_does_not_establish_wrong_object_grasp(tmp_path):
    infos = [{"object_telemetry": {"objects": {"butter_1": {
        "both_fingerpads_contact": True, "lift_from_reset_m": lift, "gripper_distance_m": .01}}}}
        for lift in [0., .01, .02, .04]]
    np.savez(tmp_path/"chunk_000001.npz", metadata_json=json.dumps({"step_infos": infos}))
    evidence = module.episode_evidence(tmp_path, "alphabet_soup_1")
    assert not evidence["sustained_grasp_candidates"]
    assert evidence["provisional_failure_type"] == "other/ambiguous"
    assert evidence["reviewed"] is False


def test_sustained_contact_and_lift_are_evidence_pending_review(tmp_path):
    info = {"object_telemetry": {"objects": {"butter_1": {
        "both_fingerpads_contact": True, "lift_from_reset_m": .05, "gripper_distance_m": .01}}}}
    np.savez(tmp_path/"chunk_000001.npz", metadata_json=json.dumps({"step_infos": [info]*3}))
    evidence = module.episode_evidence(tmp_path, "alphabet_soup_1")
    assert evidence["sustained_grasp_candidates"] == ["butter_1"]
    assert evidence["provisional_failure_type"] == "wrong object"
    assert evidence["reviewed"] is False
