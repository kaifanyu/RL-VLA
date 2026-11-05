"""Synthetic export-format tests; these fixtures are never experiment evidence."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("trace_grid_analysis",Path(__file__).resolve().parents[1]/"scripts/analyze_vla_trace_grid.py")
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    traces=root/"traces/object_0"
    put(traces/"summary.json",{"completed":True,"states":1})
    bank=root/"observation_bank/object_0/test/state_000150"
    bank.mkdir(parents=True)
    image=np.zeros((224,224,3),np.uint8)
    np.savez(bank/"observation.npz",image=image,wrist_image=image)
    masks={}
    for camera in ("image","wrist_image"):
        target=np.zeros((224,224),bool);target[20:60,20:60]=True
        distractor=np.zeros_like(target);distractor[150:200,150:200]=True
        masks[f"{camera}__alphabet_soup_1"]=target
        masks[f"{camera}__butter_1"]=distractor
    np.savez(bank/"masks.npz",**masks)
    auxiliary=root/"observation_bank_aux/object_0/test/state_000150"
    auxiliary.mkdir(parents=True)
    np.savez(auxiliary/"old_target_region.npz",**{f"{camera}__old_target_prism":np.zeros((224,224),bool) for camera in ("image","wrist_image")})
    metadata={"camera_spans":[{"name":"base_0_rgb","start":0,"end":16,"grid":[4,4],"valid":True},
                              {"name":"left_wrist_0_rgb","start":16,"end":32,"grid":[4,4],"valid":True}],
              "text_span":[32,35],"prefix_tokens":35,"prefill_query_indices":[32,33,34],
              "prompt_token_pieces":["▁alphabet","▁soup","▁basket"],"prompt_ids":[11,12,13],"prompt_mask":[True]*3,
              "object_noun_tokens":{"alphabet soup":{"prompt_indices":[0,1],"prefix_positions":[32,33]},
                                    "basket":{"prompt_indices":[2],"prefix_positions":[34]}}}
    counter=dict(metadata,prompt_token_pieces=["▁butter","▁basket","<pad>"],prompt_ids=[14,13,0],
                 prompt_mask=[True,True,False],prefill_query_indices=[32,33],
                 object_noun_tokens={"butter":{"prompt_indices":[0],"prefix_positions":[32]},
                                     "basket":{"prompt_indices":[1],"prefix_positions":[33]}})
    attention=np.ones((2,1,2,3,37),np.float32)/37
    for policy in analysis.POLICIES:
        directory=traces/policy/"state_000150"
        put(directory/("prefix_metadata.json" if policy=="pi05_base" else "metadata.json"),metadata)
        put(directory/("counterfactual_prefix_metadata.json" if policy=="pi05_base" else "counterfactual_metadata.json"),counter)
        np.savez(directory/"prefix.npz",prefill_attention=attention)
        np.savez(directory/"counterfactual_prefix.npz",prefill_attention=attention[:,:,:,:2])
        if policy!="pi05_base":
            np.savez(directory/"expert_step05_none.npz",attention=attention,velocity=np.zeros((1,2,3)),x_t=np.zeros((1,2,3)),t=np.array([.5],np.float32))
        if policy.startswith("rldt"):
            put(directory.parent/"prefix_equality.json",{"passed":True})


def test_grid_aggregation_retains_denominator_and_marks_raw_base_action_na(tmp_path,monkeypatch):
    fixture(tmp_path)
    monkeypatch.setattr(analysis,"plot",lambda *args:None)
    monkeypatch.setattr(analysis,"plot_routing",lambda *args:None)
    monkeypatch.setattr(analysis,"plot_text_grounding",lambda *args:None)
    result=analysis.analyze(tmp_path,"object_0")
    assert result["complete"]
    rows=json.loads((tmp_path/"trace_analysis/object_0/layer_summary.json").read_text())
    assert rows and all(row["camera_mass_all_keys"]<.5 for row in rows)
    assert not (tmp_path/"traces/object_0/pi05_base/expert_summary.json").exists()
    assert (tmp_path/"traces/object_0/pi05_base/text_summary.json").exists()


def test_grid_rejects_latent_mismatch_instead_of_plotting_unmatched_inputs(tmp_path,monkeypatch):
    fixture(tmp_path)
    path=tmp_path/"traces/object_0/rldt_000025/state_000150/expert_step05_none.npz"
    with np.load(path) as data:arrays={key:data[key] for key in data.files}
    arrays["x_t"]=np.ones((1,2,3))
    np.savez(path,**arrays)
    monkeypatch.setattr(analysis,"plot",lambda *args:None)
    monkeypatch.setattr(analysis,"plot_routing",lambda *args:None)
    monkeypatch.setattr(analysis,"plot_text_grounding",lambda *args:None)
    with pytest.raises(ValueError,match="exact baseline flow latent"):
        analysis.analyze(tmp_path,"object_0")


def test_actual_frozen_empty_noun_metadata_resolves_exact_captured_rows():
    fixture=json.loads((Path(__file__).parent/"fixtures/vla_trace_noun_metadata.json").read_text())
    for case in fixture["cases"]:
        metadata=case["metadata"]
        assert metadata["object_noun_tokens"]=={}
        spans=analysis.resolve_noun_spans(metadata,["alphabet soup","butter","basket"])
        assert set(spans)==set(case["expected"])
        for noun,indices in case["expected"].items():
            assert spans[noun]["prompt_indices"]==indices
            assert spans[noun]["prefix_positions"]==[768+index for index in indices]
            assert [metadata["prefill_query_indices"].index(position) for position in spans[noun]["prefix_positions"]]==indices
        assert metadata["object_noun_tokens"]=={}  # Original export is never modified.


def prompt_metadata(pieces,mask=None):
    mask=mask or [True]*len(pieces)
    return {"prompt_token_pieces":pieces,"prompt_ids":list(range(len(pieces))),"prompt_mask":mask,
            "text_span":[768,768+len(pieces)],"prefill_query_indices":[768+i for i,valid in enumerate(mask) if valid],
            "object_noun_tokens":{}}


def test_saved_phrase_subwords_boundaries_and_hard_barriers():
    metadata=prompt_metadata(["<bos>","pick","▁cream","▁che","ese","▁basket","<pad>"])
    assert analysis.resolve_noun_spans(metadata,["cream cheese"])["cream cheese"]["prompt_indices"]==[2,3,4]
    assert not analysis.resolve_noun_spans(prompt_metadata(["▁buttery"]),["butter"])
    assert not analysis.resolve_noun_spans(prompt_metadata(["▁alphabet","<pad>","▁soup"]),["alphabet soup"])
    assert not analysis.resolve_noun_spans(prompt_metadata(["▁alphabet","▁x","▁soup"],[True,False,True]),["alphabet soup"])


def test_noun_resolution_rejects_ambiguity_uncaptured_queries_and_conflicting_export():
    with pytest.raises(ValueError,match="Ambiguous"):
        analysis.resolve_noun_spans(prompt_metadata(["▁butter","▁and","▁butter"]),["butter"])
    metadata=prompt_metadata(["▁butter"])
    metadata["prefill_query_indices"]=[]
    with pytest.raises(ValueError,match="not captured"):
        analysis.resolve_noun_spans(metadata,["butter"])
    metadata=prompt_metadata(["▁butter"])
    metadata["object_noun_tokens"]={"butter":{"prompt_indices":[1],"prefix_positions":[769]}}
    with pytest.raises(ValueError,match="disagrees"):
        analysis.resolve_noun_spans(metadata,["butter"])
