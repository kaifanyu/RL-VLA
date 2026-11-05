"""Causal audit contracts; synthetic fixtures are not experiment results."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("causal_analysis",Path(__file__).resolve().parents[1]/"scripts/analyze_vla_trace_causal.py")
causal=importlib.util.module_from_spec(spec)
spec.loader.exec_module(causal)


def test_padded_trace_policy_resolves_canonical_closed_loop_folder(tmp_path):
    directory=tmp_path/"interventions/object_0/rldt_25"
    directory.mkdir(parents=True)
    (directory/"summary.json").write_text(json.dumps({"complete":False}))
    with pytest.raises(ValueError,match="campaign is incomplete"):
        causal.analyze(tmp_path,"object_0","rldt_000025")


def test_incidental_contact_is_not_a_sustained_grasp_choice():
    def frame(contact,lift):
        return {"objects":{"target":{"both_fingerpads_contact":False,"lift_from_reset_m":0.,"gripper_distance_m":.2},
                           "distractor":{"both_fingerpads_contact":contact,"lift_from_reset_m":lift,"gripper_distance_m":.01}}}
    result=causal.contact_choice([frame(True,.05),frame(True,.05),frame(False,.05)],"target","distractor")
    assert result["first_grasp_candidate_role"]=="none"
    result=causal.contact_choice([frame(True,.05)]*3,"target","distractor")
    assert result["first_grasp_candidate_role"]=="distractor" and result["first_candidate_step"]==3
    assert result["reviewed"] is False


def test_nominal_physical_requests_clip_each_command_before_comparison():
    contract={"use_delta":True,"control_dim":6,"impedance_mode":"fixed","input_min":[-1]*6,"input_max":[1]*6,
              "output_min":[-.05]*3+[-.5]*3,"output_max":[.05]*3+[.5]*3}
    reference=np.zeros((10,7));candidate=np.zeros_like(reference)
    reference[0,0]=2;candidate[0,0]=1
    assert causal.nominal_request_delta(reference,candidate,contract)["translation_request_delta_l2_m"]==0
    candidate[0,0]=-1
    result=causal.nominal_request_delta(reference,candidate,contract)
    assert result["translation_request_delta_l2_m"]==.1
    assert result["per_step_translation_request_delta_xyz_m"][0]==[-.1,0.,0.]
