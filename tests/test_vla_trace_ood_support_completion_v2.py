import json

import pytest

from rl_vla.diagnostics import ood_completion_support_v2 as wrapper


def test_no_bridge_changes_no_native_completion_contract(tmp_path,monkeypatch):
    value={'complete':False,'passed':None,'sources':{},'task':'object_1'}
    monkeypatch.setattr(wrapper.original,'completion',lambda *_:value)
    assert wrapper.completion(tmp_path,'object_1')==value


def test_accepted_successor_rejects_plain_native_completion_marker(tmp_path):
    marker=tmp_path/'interventions/object_1/queue_complete.json';marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({'complete':True,'passed':True}))
    approval=tmp_path/'compiler_profile_integration/causal_structural_na_v1/root_acceptance.json';approval.parent.mkdir(parents=True);approval.write_text('{}')
    with pytest.raises(ValueError,match='resource-only'):
        wrapper.bridge_contract(tmp_path)


def test_resource_release_claim_cannot_skip_bound_structural_coverage(tmp_path,monkeypatch):
    from rl_vla.diagnostics import causal_release_bridge_v2 as bridge
    acceptance=tmp_path/'approval.json';acceptance.write_text('{}')
    marker=tmp_path/'interventions/object_1/queue_complete.json';marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({'resource_release_only':True,'bindings':{'bridge_acceptance':wrapper.binding(acceptance)}}))
    monkeypatch.setattr(bridge,'validate_inputs',lambda *_:None)
    with pytest.raises(ValueError,match='precedes'):
        wrapper.bridge_contract(tmp_path)
