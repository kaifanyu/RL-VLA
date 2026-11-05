"""Regression checks for frozen-input CPU supplements; no model or GPU imports."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("supplement", REPO/"scripts/analyze_vla_trace_supplement.py")
supplement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supplement)


def metadata_cases():
    return json.loads((REPO/"tests/fixtures/vla_trace_noun_metadata.json").read_text())["cases"]


def test_real_saved_metadata_controls_and_random_exclusions():
    originals = []
    changed = []
    for case in metadata_cases():
        metadata = case["metadata"]
        original = "alphabet soup" in case["expected"]
        condition = "prefix" if original else "counterfactual_prefix"
        result = supplement.select_controls(metadata,"object_0",150,condition)
        expected = {"pick":[1],"and":[5 if original else 4],"in":[8 if original else 7],"basket":[10 if original else 9]}
        for name, indices in expected.items():
            assert result["controls"][name]["prompt_indices"] == indices
        excluded = {0,11 if original else 10,3,10 if original else 9}
        if original: excluded.add(4)
        assert not excluded.intersection(result["eligible_random_prompt_indices"])
        selected = result["controls"]["random_valid_non_noun"]["prompt_indices"][0]
        assert selected in result["eligible_random_prompt_indices"]
        (originals if original else changed).append(result)
    assert originals[0] == originals[1]
    assert changed[0] == changed[1]


def test_random_seed_is_stable_and_does_not_depend_on_policy():
    metadata=metadata_cases()[0]["metadata"]
    before=json.dumps(metadata,sort_keys=True)
    a=supplement.select_controls(metadata,"object_0",150,"prefix")
    b=supplement.select_controls(metadata,"object_0",150,"prefix")
    assert a==b
    assert a["random_seed_material"]=="20261031|object_0|150|prefix"
    assert json.dumps(metadata,sort_keys=True)==before


def test_missing_control_rejected_instead_of_substituting_token():
    metadata=json.loads(json.dumps(metadata_cases()[0]["metadata"]))
    metadata["prompt_token_pieces"][1]="move"
    with pytest.raises(ValueError,match="required named control"):
        supplement.select_controls(metadata,"object_0",150,"prefix")


def contract():
    return {"control_dim":6,"use_delta":True,"impedance_mode":"fixed",
            "input_min":[-1]*6,"input_max":[1]*6,
            "output_min":[-.05]*3+[-.5]*3,"output_max":[.05]*3+[.5]*3}


def test_native_affine_scaling_and_clipping_keep_channels_separate():
    actions=np.tile([2,-2,.5,2,-2,.5,2.],(10,1))
    expected=np.tile([.05,-.05,.025,.5,-.5,.25,1.],(10,1))
    np.testing.assert_array_equal(supplement.nominal_requests(actions,contract()),expected)
    np.testing.assert_array_equal(actions[0],[2,-2,.5,2,-2,.5,2.])


@pytest.mark.parametrize("bad",[np.zeros((7,)),np.zeros((1,10,7)),np.full((10,7),np.nan)])
def test_nonfinite_or_wrong_decoded_contract_rejected(bad):
    with pytest.raises(ValueError,match="ten finite"):
        supplement.nominal_requests(bad,contract())


def test_non_delta_controller_is_not_given_physical_vector_units():
    changed=contract();changed["use_delta"]=False
    with pytest.raises(ValueError,match="Unsupported"):
        supplement.nominal_requests(np.zeros((10,7)),changed)


def test_nominal_direction_is_cosine_not_endpoint_and_zero_is_na():
    assert supplement.cosine([1,0,0],[0,0,0],[2,0,0])==1
    assert supplement.cosine([1,0,0],[0,0,0],[-2,0,0])==-1
    assert supplement.cosine([1,0,0],[0,0,0],[0,2,0])==0
    assert supplement.cosine([0,0,0],[0,0,0],[2,0,0]) is None
    assert supplement.cosine([1,0,0],[0,0,0],[0,0,0]) is None


def test_changed_frozen_input_rejected_before_completion(tmp_path):
    path=tmp_path/"source";path.write_text("first")
    evidence=supplement.Evidence();evidence.bind(path)
    path.write_text("changed")
    with pytest.raises(ValueError,match="Input changed"):
        evidence.verify()


def test_summary_writer_does_not_publish_invalid_json(tmp_path):
    path=tmp_path/"summary.json"
    with pytest.raises(ValueError):supplement.write_json(path,{"complete":True,"bad":float("nan")})
    assert not path.exists()
