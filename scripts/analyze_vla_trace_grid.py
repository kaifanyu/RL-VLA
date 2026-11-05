"""Aggregate matched native attention traces without dropping the softmax denominator."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.metrics import grounding_scores, jensen_shannon, patch_occupancy

OBJECTS={"object_0":("alphabet_soup_1","butter_1"),"object_1":("cream_cheese_1","alphabet_soup_1"),"object_2":("salad_dressing_1","tomato_sauce_1")}
CAMERAS={"base_0_rgb":"image","left_wrist_0_rgb":"wrist_image"}
POLICIES=("pi05_base","pi05_libero","rldt_000025","rldt_000050","rldt_000100")


def resolve_noun_spans(metadata,nouns):
    """Map exact saved SentencePiece phrases to captured queries, without retokenizing.

    Some frozen exports contain an empty standalone-token-ID lookup. Their
    original per-token pieces, valid mask and query indices still unambiguously
    locate each requested phrase. Controls and masked gaps are hard barriers.
    """
    pieces=metadata["prompt_token_pieces"]
    mask=metadata["prompt_mask"]
    ids=metadata["prompt_ids"]
    start,end=metadata["text_span"]
    queries=metadata["prefill_query_indices"]
    if not (len(pieces)==len(mask)==len(ids)==end-start) or len(set(queries))!=len(queries):
        raise ValueError("Inconsistent saved prompt/query metadata")
    characters=[];owners=[]
    for index,(piece,valid) in enumerate(zip(pieces,mask,strict=True)):
        if not valid or re.fullmatch(r"<[^>]+>",piece):
            characters.append("\0");owners.append(None);continue
        for character in piece.replace("▁"," "):
            for normalized in character.casefold():
                characters.append(normalized);owners.append(index)
    decoded="".join(characters)
    result={}
    for noun in nouns:
        phrase=r"\s+".join(re.escape(word) for word in noun.casefold().split())
        matches=list(re.finditer(r"(?<!\w)"+phrase+r"(?!\w)",decoded))
        if len(matches)>1:
            raise ValueError(f"Ambiguous repeated noun phrase in saved prompt: {noun}")
        if not matches:
            if noun in metadata.get("object_noun_tokens",{}):
                raise ValueError(f"Exported noun span disagrees with saved token pieces: {noun}")
            continue
        match=matches[0]
        matched_owners=set(owners[match.start():match.end()])
        if None in matched_owners:
            raise ValueError(f"Noun phrase crosses a control token: {noun}")
        indices=sorted(matched_owners)
        if (not indices or indices!=list(range(indices[0],indices[-1]+1))
                or not all(mask[index] for index in indices)):
            raise ValueError(f"Noun phrase crosses an invalid prompt span: {noun}")
        positions=[start+index for index in indices]
        if any(position not in queries for position in positions):
            raise ValueError(f"Noun phrase query was not captured: {noun}")
        record={"prompt_indices":indices,"prefix_positions":positions,
                "aggregation":"report individual subwords and arithmetic mean of their full-denominator attention rows"}
        exported=metadata.get("object_noun_tokens",{}).get(noun)
        if exported is not None and any(exported.get(key)!=record[key] for key in ("prompt_indices","prefix_positions")):
            raise ValueError(f"Exported noun span disagrees with saved token pieces: {noun}")
        result[noun]=record
    return result


def write_json(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def projection_rows(traces,task):
    """Final projection factorial decomposition; no claim that attention is causal."""
    rows=[]
    for path in sorted(traces.glob("rldt_*/state_*/projection_ablation_step*.npz")):
        metadata=json.loads(path.with_suffix(".json").read_text())
        if not metadata.get("complete") or metadata["native_reconstruction_max_abs_difference"]!=0:
            raise ValueError("Projection swap lacks exact native reconstruction")
        with np.load(path,allow_pickle=False) as data:
            baseline=data["baseH_baseW"]
            projection=data["baseH_checkpointW"]-baseline
            hidden=data["checkpointH_baseW"]-baseline
            total=data["checkpointH_checkpointW"]-baseline
            interaction=total-projection-hidden
            hidden_drift=np.linalg.norm(data["hidden_checkpoint"]-data["hidden_base"])
        denominator=float(np.sum(total.astype(np.float64)**2))
        row={"task":task,"policy":path.parent.parent.name,"state_id":int(path.parent.name.split("_")[1]),
             "step":int(path.stem.rsplit("step",1)[1]),"hidden_drift_l2":float(hidden_drift),
             "units":"normalized flow velocity"}
        if metadata.get("profile_ownership"):
            row["projection_profile_ownership"] = json.dumps(metadata["profile_ownership"], sort_keys=True)
            row["projection_interpretation"] = metadata["interpretation"]
        for name, effects in metadata.get("same_hidden_baseline_weights_compiler_effect", {}).items():
            for metric, value in effects.items():
                row[f"compiler_{name}_{metric}"] = value
        for name,value in (("projection_only",projection),("hidden_only",hidden),("interaction",interaction),("total",total)):
            row[f"{name}_l2"]=float(np.linalg.norm(value))
            row[f"{name}_signed_alignment_fraction"]=float(np.sum(value.astype(np.float64)*total)/denominator) if denominator else None
        rows.append(row)
    return rows


def paired_attention_rows(states,task):
    """Resample whole matched observations after averaging layers and heads."""
    groups=defaultdict(dict)
    for row in states:
        groups[row["policy"],row["camera"],row["step"],row["region"]][row["state_id"]]=row
    rows=[]
    for (policy,camera,step,region),values in sorted(groups.items()):
        if policy=="pi05_libero":continue
        baseline=groups["pi05_libero",camera,step,region]
        if values.keys()!=baseline.keys():raise ValueError("Attention comparison dropped matched states")
        for metric in ("region_mass_all_keys","region_density","region_fraction_camera"):
            ids=[state for state in sorted(values) if values[state][metric] is not None and baseline[state][metric] is not None]
            if not ids:continue
            a=np.array([baseline[state][metric] for state in ids]);b=np.array([values[state][metric] for state in ids])
            differences=b-a
            boot=np.random.default_rng(20261031).choice(differences,size=(20000,len(ids))).mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append({"task":task,"policy":policy,"camera":camera,"step":step,"region":region,"metric":metric,
                         "states":len(ids),"baseline_mean":float(a.mean()),"checkpoint_mean":float(b.mean()),
                         "difference":float(differences.mean()),"paired_bootstrap95_low":float(lo),"paired_bootstrap95_high":float(hi),
                         "bootstrap_unit":"whole observation after averaging all18layers/8heads/10queries","bootstrap_draws":20000})
    return rows


def layer_knockout_rows(traces,task):
    rows=[];velocities=[]
    expected={(mode,window) for mode in ("no_image","no_text") for window in
              ["early","middle","late"]+[f"layer_{index:02d}" for index in range(18)]}
    for path in sorted(traces.glob("*/state_*/layer_knockout_summary.json")):
        data=json.loads(path.read_text())
        if not data.get("complete") or data["no_op_action_max_abs_difference"]!=0:
            raise ValueError("Layer knockout no-op fidelity is not exact")
        if {(row["mode"],row["window"]) for row in data["conditions"]}!=expected:
            raise ValueError("Incomplete predeclared layer-window/single-layer intervention set")
        for value in data["conditions"]:
            if value["blocked_attention_max"]!=0:raise ValueError("Layer knockout leaked blocked attention")
            row={"task":task,"policy":path.parent.parent.name,"state_id":int(path.parent.name.split("_")[1]),
                 "mode":value["mode"],"window":value["window"],"layers_zero_based":value["layers_zero_based"],
                 "normalized_action_l2":value["normalized_action_delta"]["l2"],
                 "decoded_controller_l2":value["decoded_controller_delta"]["l2"],
                 "decoded_units":"LIBERO OSC/gripper commands; not metres"}
            rows.append(row)
            for velocity in value["fixed_common_latent_velocity_effects"]:
                velocities.append({"task":task,"policy":row["policy"],"state_id":row["state_id"],
                                   "mode":row["mode"],"window":row["window"],"step":velocity["step"],
                                   "normalized_velocity_l2":velocity["normalized_velocity_delta"]["l2"]})
    return rows,velocities


def analyze(root,task):
    root=Path(root).resolve()
    traces=root/"traces"/task
    complete=json.loads((traces/"summary.json").read_text())
    if not complete.get("completed",complete.get("complete")):
        raise ValueError("Trace generation has not completed")
    output=root/"trace_analysis"/task
    if (output/"summary.json").exists():
        previous=json.loads((output/"summary.json").read_text())
        if previous.get("complete"):return previous
    output.mkdir(parents=True,exist_ok=True)
    target,distractor=OBJECTS[task]
    noun_names=[name.removesuffix("_1").replace("_"," ") for name in (target,distractor)]
    accumulator=defaultdict(list)
    text_accumulator=defaultdict(list)
    state_accumulator=defaultdict(list)
    action_rows=[]
    routing=[]
    noun_span_audit=[]
    grounding_path=output/"grounding.csv"
    text_path=output/"text_grounding.csv"
    with grounding_path.open("w",newline="") as stream, text_path.open("w",newline="") as text_stream:
        writer=None;text_writer=None
        for policy in POLICIES:
            policy_dir=traces/policy
            state_dirs=sorted(policy_dir.glob("state_*"))
            if len(state_dirs)!=complete["states"]:
                raise ValueError(f"Trace policy has missing predetermined observations: {policy}")
            protocol=root/"tasks"/task/"protocol.json"
            if protocol.exists():
                ids=json.loads(protocol.read_text())["tasks"][task]["test_state_ids"]
                if [int(path.name.split("_")[1]) for path in state_dirs]!=ids:
                    raise ValueError("Scientific trace summary must cover all frozen test states")
            for state_dir in state_dirs:
                state_id=int(state_dir.name.split("_")[1])
                bank=root/"observation_bank"/task/"test"/state_dir.name
                with np.load(bank/"masks.npz",allow_pickle=False) as archive:
                    masks={key:archive[key] for key in archive.files}
                auxiliary=root/"observation_bank_aux"/task/"test"/state_dir.name
                with np.load(auxiliary/"old_target_region.npz",allow_pickle=False) as archive:
                    masks.update({key:archive[key] for key in archive.files})
                meta_path=state_dir/("prefix_metadata.json" if policy=="pi05_base" else "metadata.json")
                metadata=json.loads(meta_path.read_text())
                attention_files=sorted(state_dir.glob("expert_step*_none.npz"))
                if protocol.exists() and policy!="pi05_base" and [path.name for path in attention_files]!=[f"expert_step{step:02d}_none.npz" for step in (0,5,9)]:
                    raise ValueError("Scientific expert traces require actual flow steps0,5,9")
                for attention_file in attention_files:
                    step=int(attention_file.stem.split("_")[1][4:])
                    with np.load(attention_file,allow_pickle=False) as archive:
                        attention=archive["attention"]
                        velocity=archive["velocity"]
                        latent=archive["x_t"]
                        flow_time=archive["t"]
                    baseline=traces/"pi05_libero"/state_dir.name/attention_file.name
                    with np.load(baseline,allow_pickle=False) as archive:
                        if not np.array_equal(latent,archive["x_t"]):
                            raise ValueError("Checkpoint traces do not share the exact baseline flow latent")
                        if not np.array_equal(flow_time,archive["t"]):
                            raise ValueError("Checkpoint traces do not share the exact baseline flow time")
                        baseline_velocity=archive["velocity"]
                        baseline_attention=archive["attention"]
                    action_rows.append({"task":task,"policy":policy,"state_id":state_id,"step":step,"time":float(flow_time[0]),
                                        "velocity_l2_vs_libero":float(np.linalg.norm(velocity-baseline_velocity))})
                    for span in metadata["camera_spans"]:
                        if not span["valid"]:continue
                        camera=CAMERAS[span["name"]];grid=tuple(span["grid"])
                        values=attention[:,0,:,:,span["start"]:span["end"]].mean(axis=2)
                        occupancies={role:patch_occupancy(masks[f"{camera}__{obj}"],grid) for role,obj in (("target",target),("distractor",distractor),("old_target","old_target_prism"))}
                        for layer in range(values.shape[0]):
                            for head in range(values.shape[1]):
                                heatmap=values[layer,head].reshape(grid)
                                reference=baseline_attention[layer,0,head,:,span["start"]:span["end"]].mean(axis=0).reshape(grid)
                                full=attention[layer,0,head].mean(axis=0)
                                start,end=metadata["text_span"]
                                masses={role:float((heatmap*occupancy).sum()) for role,occupancy in occupancies.items()}
                                areas={role:float(occupancy.sum()) for role,occupancy in occupancies.items()}
                                positive=full[full>0]
                                routing.append({"task":task,"policy":policy,"state_id":state_id,"step":step,"time":float(flow_time[0]),
                                    "camera":camera,"layer":layer,"head":head,
                                    "target_distractor_mass_ratio":(masses["target"]+1e-12)/(masses["distractor"]+1e-12),
                                    "target_distractor_density_ratio":((masses["target"]/areas["target"]+1e-12)/(masses["distractor"]/areas["distractor"]+1e-12)) if min(areas["target"],areas["distractor"])>0 else None,
                                    "camera_conditional_js_vs_libero_nats":jensen_shannon(heatmap,reference) if min(heatmap.sum(),reference.sum())>0 else None,
                                    "camera_attention_l1_all_keys_vs_libero":float(np.abs(heatmap-reference).sum()),
                                    "action_to_text_mass_all_keys":float(full[start:end].sum()),
                                    "action_to_action_mass_all_keys":float(full[metadata["prefix_tokens"]:].sum()),
                                    "camera_mass_all_keys":float(heatmap.sum()),
                                    "full_attention_entropy_nats":float(-(positive*np.log(positive)).sum()),"ratio_epsilon":1e-12})
                                for role,occupancy in occupancies.items():
                                    scores=grounding_scores(heatmap,occupancy)
                                    row={"task":task,"policy":policy,"state_id":state_id,"step":step,"time":float(flow_time[0]),
                                         "camera":camera,"layer":layer,"head":head,"query_aggregation":"mean_all10_action_queries","region":role,**scores}
                                    if writer is None:
                                        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
                                    writer.writerow(row)
                                    accumulator[policy,camera,step,layer,role].append(scores)
                                    state_accumulator[policy,camera,step,state_id,role].append(scores)
                # Frozen native prefix maps are stored once, and shared with
                # RLDT only after explicit per-checkpoint equality verification.
                if policy not in ("pi05_base","pi05_libero"):continue
                for condition in ("prefix","counterfactual_prefix"):
                    if condition=="prefix":
                        prompt_metadata=metadata
                        prompt_metadata_path=meta_path
                    else:
                        filename="counterfactual_prefix_metadata.json" if policy=="pi05_base" else "counterfactual_metadata.json"
                        prompt_metadata_path=state_dir/filename
                        prompt_metadata=json.loads(prompt_metadata_path.read_text())
                    noun_tokens=resolve_noun_spans(prompt_metadata,noun_names+["basket"])
                    noun_span_audit.append({"task":task,"policy":policy,"state_id":state_id,"condition":condition,
                        "source_metadata":str(prompt_metadata_path),
                        "source_metadata_sha256":hashlib.sha256(prompt_metadata_path.read_bytes()).hexdigest(),
                        "exported_mapping_was_empty":not bool(prompt_metadata.get("object_noun_tokens")),
                        "resolved_mapping":noun_tokens})
                    with np.load(state_dir/f"{condition}.npz",allow_pickle=False) as archive:
                        attention=archive["prefill_attention"]
                    requested=noun_names[0 if condition=="prefix" else 1]
                    if requested not in noun_tokens:
                        raise ValueError(f"Requested noun token span missing: {policy}/{condition}/{requested}")
                    absolute=prompt_metadata["prefill_query_indices"]
                    if attention.shape[-2]!=len(absolute):
                        raise ValueError("Saved prefill query dimension differs from metadata")
                    for noun in noun_names+["basket"]:
                        token_record=noun_tokens.get(noun)
                        if token_record is None:continue
                        queries=[absolute.index(position) for position in token_record["prefix_positions"]]
                        for token_i,query in zip(token_record["prompt_indices"],queries,strict=True):
                            for span in prompt_metadata["camera_spans"]:
                                if not span["valid"]:continue
                                camera=CAMERAS[span["name"]];grid=tuple(span["grid"])
                                values=attention[:,0,:,query,span["start"]:span["end"]].mean(axis=1)
                                for layer,values_layer in enumerate(values):
                                    for role,obj in (("target",target),("distractor",distractor),("old_target","old_target_prism")):
                                        scores=grounding_scores(values_layer.reshape(grid),patch_occupancy(masks[f"{camera}__{obj}"],grid))
                                        row={"task":task,"policy":policy,"state_id":state_id,"prompt_condition":condition,"noun":noun,
                                             "token_index":token_i,"token_piece":prompt_metadata["prompt_token_pieces"][token_i],
                                             "camera":camera,"layer":layer,"head_aggregation":"mean_all_heads","region":role,**scores}
                                        if text_writer is None:
                                            text_writer=csv.DictWriter(text_stream,fieldnames=list(row));text_writer.writeheader()
                                        text_writer.writerow(row)
                                        text_accumulator[policy,camera,condition,noun,layer,role].append(scores)
            if policy!="pi05_base":
                write_json(policy_dir/"expert_summary.json",{"complete":bool(state_dirs),"states":len(state_dirs),"task":task,"policy":policy,
                    "artifact":str(grounding_path),"query_aggregation":"Mean10action queries; all18layers and8heads retained; no region renormalization"})
            if policy.startswith("rldt"):
                equality=json.loads((policy_dir/"prefix_equality.json").read_text())
                if not equality.get("passed"):
                    raise ValueError("RLDT text attention cannot be shared without frozen-prefix equality")
            write_json(policy_dir/"text_summary.json",{"complete":bool(state_dirs),"states":len(state_dirs),"task":task,"policy":policy,
                "artifact":str(text_path),"shared_with":"pi05_libero" if policy.startswith("rldt") else None,
                "equality_audit":str(policy_dir/"prefix_equality.json") if policy.startswith("rldt") else None,
                "counterfactual":"Offline prompt edit only; no closed-loop success scored with mismatched goal"})
    aggregates=[]
    for key,values in accumulator.items():
        policy,camera,step,layer,role=key
        row={"task":task,"policy":policy,"camera":camera,"step":step,"layer":layer,"region":role,"state_head_rows":len(values)}
        for metric in values[0]:
            finite=[value[metric] for value in values if value[metric] is not None]
            row[metric]=float(np.mean(finite)) if finite else None
        aggregates.append(row)
    projection=projection_rows(traces,task)
    state_rows=[]
    for (policy,camera,step,state_id,role),values in state_accumulator.items():
        row={"task":task,"policy":policy,"camera":camera,"step":step,"state_id":state_id,"region":role}
        for metric in values[0]:
            finite=[value[metric] for value in values if value[metric] is not None]
            row[metric]=float(np.mean(finite)) if finite else None
        state_rows.append(row)
    paired_attention=paired_attention_rows(state_rows,task)
    layer_knockouts,layer_velocities=layer_knockout_rows(traces,task)
    for name,rows in (("layer_summary",aggregates),("action_changes",action_rows),("projection_ablation",projection),
                      ("state_attention",state_rows),("paired_attention",paired_attention),
                      ("layer_knockout_actions",layer_knockouts),("layer_knockout_velocities",layer_velocities),("routing",routing)):
        if rows:
            with (output/f"{name}.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output/"layer_summary.json",aggregates)
    write_json(output/"projection_ablation.json",projection)
    write_json(output/"paired_attention.json",paired_attention)
    write_json(output/"layer_knockout_actions.json",layer_knockouts)
    write_json(output/"noun_span_resolution.json",{"complete":True,
        "method":"Exact bounded phrases in saved SentencePiece token pieces; controls and masked tokens are hard barriers; contiguous valid indices must have captured query rows. Existing exported mappings are checked for agreement. No tokenizer/model rerun and no trace mutation.",
        "records":noun_span_audit})
    plot(root,task,traces,output,aggregates)
    plot_routing(task,output,routing,action_rows,traces)
    plot_text_grounding(root,task,traces,output)
    if (root/"tasks"/task/"protocol.json").exists():
        from summarize_vla_trace_frame import summarize
        first=min((traces/"pi05_libero").glob("state_*")).name
        toolkit=Path(__file__).resolve().parents[1]/"third_party/vla_trace_tools/.venv/bin/vla-trace"
        for policy in POLICIES[1:]:
            destination=output/"vla_trace"/policy
            if not (destination/"summary.json").exists():
                summarize(traces/policy/first,root/"observation_bank"/task/"test"/first,task,destination,toolkit,scope="fixed-test")
    summary={"complete":True,"task":task,"states":complete["states"],"policies":list(POLICIES),
             "layer_summary":str(output/"layer_summary.json"),"grounding":str(grounding_path),"text_grounding":str(text_path),
             "noun_span_resolution":str(output/"noun_span_resolution.json"),
             "interpretation":"Matched fixed-input attention and velocities; neither actual divergent rollout attention nor causal action attribution by itself"}
    write_json(output/"summary.json",summary)
    return summary


def plot_text_grounding(root,task,traces,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first=min((traces/"pi05_libero").glob("state_*")).name
    bank=root/"observation_bank"/task/"test"/first
    with np.load(bank/"observation.npz",allow_pickle=False) as data:images={camera:data[camera] for camera in CAMERAS.values()}
    with np.load(bank/"masks.npz",allow_pickle=False) as data:masks={key:data[key] for key in data.files}
    cache={}
    for model in ("pi05_base","pi05_libero"):
        directory=traces/model/first
        for condition in ("prefix","counterfactual_prefix"):
            filename=("prefix_metadata.json" if condition=="prefix" else "counterfactual_prefix_metadata.json") if model=="pi05_base" else ("metadata.json" if condition=="prefix" else "counterfactual_metadata.json")
            metadata=json.loads((directory/filename).read_text())
            requested=OBJECTS[task][0 if condition=="prefix" else 1].removesuffix("_1").replace("_"," ")
            positions=resolve_noun_spans(metadata,[requested])[requested]["prefix_positions"]
            indices=[metadata["prefill_query_indices"].index(position) for position in positions]
            with np.load(directory/f"{condition}.npz",allow_pickle=False) as data:
                attention=data["prefill_attention"]
            cache[model,condition]=(metadata,attention[:,:,:,indices,:].mean(axis=(1,2,3)))
    for layer in (0,8,17):
        fig,axes=plt.subplots(4,5,figsize=(15,12),layout="constrained")
        for camera_index,camera in enumerate(CAMERAS.values()):
            values={}
            for (model,condition),(metadata,attention) in cache.items():
                span=next(span for span in metadata["camera_spans"] if CAMERAS.get(span["name"])==camera)
                values[model,condition]=attention[layer,span["start"]:span["end"]].reshape(span["grid"])
            vmax=max(float(value.max()) for value in values.values())
            for condition_index,condition in enumerate(("prefix","counterfactual_prefix")):
                requested=OBJECTS[task][condition_index].removesuffix("_1").replace("_"," ")
                for column,policy in enumerate(POLICIES):
                    model="pi05_libero" if policy.startswith("rldt") else policy
                    axis=axes[2*camera_index+condition_index,column]
                    axis.imshow(images[camera]);axis.imshow(values[model,condition],extent=(-.5,223.5,223.5,-.5),cmap="inferno",alpha=.55,vmin=0,vmax=vmax,interpolation="nearest")
                    for obj,color in zip(OBJECTS[task],("lime","cyan"),strict=True):axis.contour(masks[f"{camera}__{obj}"],levels=[.5],colors=[color],linewidths=.7)
                    axis.set_axis_off();axis.set_title(f"{policy}\n{camera} · request {requested}",fontsize=8)
        fig.suptitle(f"{task} {first} · VLM layer {layer} · requested noun subwords and heads averaged\nTrue task target green; valid distractor cyan. Same per-camera absolute scale across policies/prompts.\nRLDT maps share the verified frozen LIBERO prefix; counterfactual prompt is offline, without goal scoring.",fontsize=11)
        for extension in ("png","pdf"):fig.savefig(output/f"text_grounding_layer{layer:02d}.{extension}",dpi=140)
        plt.close(fig)


def plot_routing(task,output,rows,actions,traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:return
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for camera,row_axes in zip(CAMERAS.values(),axes,strict=True):
        for policy in POLICIES[1:]:
            for metric,axis in zip(("target_distractor_mass_ratio","target_distractor_density_ratio","camera_mass_all_keys"),row_axes,strict=True):
                values=[]
                steps=sorted({row["step"] for row in rows})
                for step in steps:
                    subset=[r[metric] for r in rows if r["policy"]==policy and r["camera"]==camera and r["step"]==step and r[metric] is not None]
                    values.append(float(np.mean(np.log2(np.maximum(subset,1e-12)))) if "ratio" in metric and subset else float(np.mean(subset)) if subset else np.nan)
                axis.plot([1-step/10 for step in steps],values,marker="o",label=policy)
                axis.set(title=f"{camera}\n{metric}",xlabel="Actual flow time t",ylabel="Mean log2 ratio" if "ratio" in metric else "Full-denominator mass")
                axis.invert_xaxis() if not axis.xaxis_inverted() else None
                axis.grid(alpha=.2)
        row_axes[-1].legend(fontsize=7)
    fig.suptitle(f"{task} · fixed baseline latents; mean across predetermined states/layers/heads\nRatio curves average log2 per-head ratios (epsilon1e-12); raw ratios and areas remain in CSV.")
    for extension in ("png","pdf"):fig.savefig(output/f"flow_time_evolution.{extension}",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout="constrained")
    for camera,row_axes in zip(CAMERAS.values(),axes,strict=True):
        for policy,axis in zip(POLICIES[2:],row_axes,strict=True):
            matrix=np.full((18,8),np.nan)
            for layer in range(18):
                for head in range(8):
                    values=[r["camera_conditional_js_vs_libero_nats"] for r in rows if r["camera"]==camera and r["policy"]==policy and r["step"]==5 and r["layer"]==layer and r["head"]==head]
                    values=[value for value in values if value is not None]
                    if values:matrix[layer,head]=np.mean(values)
            mesh=axis.imshow(matrix,aspect="auto",vmin=0,vmax=np.log(2),cmap="magma")
            axis.set(title=f"{camera} · {policy}",xlabel="Head",ylabel="Expert layer")
    fig.colorbar(mesh,ax=axes.ravel().tolist(),label="Camera-conditional Jensen–Shannon divergence (nats)",shrink=.7)
    fig.suptitle(f"{task} · expert attention drift vs LIBERO · common latent t=.5")
    fig.savefig(output/"head_attention_drift.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for policy in POLICIES[2:]:
        path=traces/policy/"weight_drift.json"
        if path.exists():
            drift=json.loads(path.read_text())
            layers=[(name,value["relative_l2"]) for name,value in drift.items() if "layer" in name]
            if layers:axes[0].plot(range(len(layers)),[value for _,value in layers],marker=".",label=policy)
        for step in sorted({row["step"] for row in actions}):
            values=[r["velocity_l2_vs_libero"] for r in actions if r["policy"]==policy and r["step"]==step]
            if values:axes[1].plot([1-step/10],[np.mean(values)],marker="o",label=f"{policy}, t={1-step/10:.1f}")
    axes[0].set(title="Per-expert-layer parameter drift",xlabel="Expert layer",ylabel="Relative parameter L2")
    axes[1].set(title="Output velocity change at common latent",xlabel="Actual flow time t",ylabel="Mean normalized velocity L2 change")
    for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=6)
    axes[1].invert_xaxis();fig.suptitle(task)
    fig.savefig(output/"weight_and_velocity_drift.png",dpi=160);plt.close(fig)
    sources=sorted((traces/"pi05_libero").glob("state_*/masked_inputs.npz"))
    if sources:
        with np.load(sources[0],allow_pickle=False) as inputs:
            fig,axes=plt.subplots(4,7,figsize=(17,10),layout="constrained")
            for camera_index,camera in enumerate(CAMERAS.values()):
                for mode_index,mode in enumerate(("background_fill","mosaic")):
                    for column,region in enumerate(("none","target","distractor","random","robot","background","old_target")):
                        key=f"{region}_{mode}" if region!="none" else "none"
                        axis=axes[2*camera_index+mode_index,column]
                        axis.imshow(inputs[f"{key}__{camera}"]);axis.set_axis_off()
                        axis.set_title(f"{region}\n{camera} · {mode}",fontsize=8)
            fig.suptitle(f"{task} {sources[0].parent.name} · exact intervention inputs exported by the native grid\nPixel inputs are checkpoint-independent. Prior-location prism is distinct from distractor instance; random control matches target area.",fontsize=11)
            for extension in ("png","pdf"):fig.savefig(output/f"mask_demonstration.{extension}",dpi=140)
            plt.close(fig)


def plot(root,task,traces,output,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target,distractor=OBJECTS[task]
    first=min((traces/"pi05_libero").glob("state_*")).name
    bank=root/"observation_bank"/task/"test"/first
    with np.load(bank/"observation.npz",allow_pickle=False) as data:
        images={camera:data[camera] for camera in CAMERAS.values()}
    with np.load(bank/"masks.npz",allow_pickle=False) as data:
        masks={key:data[key] for key in data.files}
    old_region=json.loads((root/"observation_bank_aux"/task/"test"/first/"metadata.json").read_text())
    fig,axes=plt.subplots(2,5,figsize=(15,6),layout="constrained")
    for i,camera in enumerate(CAMERAS.values()):
        heatmaps={}
        for policy in POLICIES[1:]:
            state=traces/policy/first
            metadata=json.loads((state/"metadata.json").read_text())
            span=next(s for s in metadata["camera_spans"] if CAMERAS.get(s["name"])==camera)
            with np.load(state/"expert_step05_none.npz",allow_pickle=False) as data:
                heatmaps[policy]=data["attention"][...,span["start"]:span["end"]].mean(axis=(0,1,2,3)).reshape(span["grid"])
        vmax=max(float(value.max()) for value in heatmaps.values())
        for j,policy in enumerate(POLICIES):
            ax=axes[i,j];ax.imshow(images[camera]);ax.set_axis_off();ax.set_title(f"{policy}\n{camera}",fontsize=9)
            if policy=="pi05_base":
                ax.text(.5,.5,"Action attention N/A\nincompatible action contract",transform=ax.transAxes,ha="center",va="center",bbox={"facecolor":"white","alpha":.9},fontsize=8);continue
            ax.imshow(heatmaps[policy],extent=(-.5,223.5,223.5,-.5),cmap="inferno",alpha=.55,
                      interpolation="nearest",vmin=0,vmax=vmax)
            for obj,color in ((target,"lime"),(distractor,"cyan")):
                ax.contour(masks[f"{camera}__{obj}"],levels=[.5],colors=[color],linewidths=.8)
            polygon=np.asarray(old_region["cameras"][camera]["floor_polygon_xy"])
            polygon=np.concatenate([polygon,polygon[:1]])
            ax.plot(polygon[:,0],polygon[:,1],color="yellow",ls="--",lw=1)
            ax.set_xlim(-.5,223.5);ax.set_ylim(223.5,-.5)
    fig.suptitle(f"{task} {first} · fixed baseline latent t=0.5 · all layers/heads/queries mean\nTarget green; distractor cyan; prior floor-region yellow. Shared absolute color scale across policies within each camera.",fontsize=10)
    fig.savefig(output/"fixed_observation_attention.png",dpi=160);fig.savefig(output/"fixed_observation_attention.pdf");plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout="constrained")
    for camera,ax in zip(CAMERAS.values(),axes,strict=True):
        matrix=np.full((18,4),np.nan)
        for j,policy in enumerate(POLICIES[1:]):
            for layer in range(18):
                target_rows=[r for r in rows if r["policy"]==policy and r["camera"]==camera and r["step"]==5 and r["layer"]==layer and r["region"]=="target"]
                distractor_rows=[r for r in rows if r["policy"]==policy and r["camera"]==camera and r["step"]==5 and r["layer"]==layer and r["region"]=="distractor"]
                if target_rows and distractor_rows:
                    a,b=target_rows[0]["region_density"],distractor_rows[0]["region_density"]
                    if a is not None and b is not None:matrix[layer,j]=np.log2((a+1e-12)/(b+1e-12))
        mesh=ax.imshow(matrix,aspect="auto",cmap="coolwarm",vmin=-3,vmax=3)
        ax.set(title=camera,xlabel="Policy",ylabel="Expert layer",xticks=range(4),xticklabels=["LIBERO","25","50","100"])
        fig.colorbar(mesh,ax=ax,label="log2(mean target density / mean distractor density)")
    fig.suptitle(f"{task} · fixed latent t=.5 · mean over predetermined states and all heads")
    fig.savefig(output/"layer_density_ratio.png",dpi=160);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,type=Path);p.add_argument("--task",required=True,choices=list(OBJECTS))
    args=p.parse_args();print(json.dumps(analyze(args.run,args.task),indent=2))
