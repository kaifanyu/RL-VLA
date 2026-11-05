"""Audit and summarize paired open-loop and closed-loop intervention evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.metrics import paired_success

POLICIES=("pi05_libero","rldt_000025","rldt_000050","rldt_000100")
CLOSED=("none","no_image","no_text","target:background_fill","distractor:background_fill","random:background_fill",
        "target:mosaic","distractor:mosaic","random:mosaic")
OPEN=("none","no_image","no_text",*[f"{region}:{mode}" for region in ("target","distractor","random","robot","background","old_target") for mode in ("background_fill","mosaic")])


def intervention_policy(policy):
    return f"rldt_{int(policy.split('_')[1])}" if policy.startswith("rldt_") else policy


def contact_choice(telemetry,target,distractor):
    """Automated grasp-choice proxy; never a reviewed semantic-failure label."""
    streak={};first={};nearest={target:None,distractor:None}
    for index,step in enumerate(telemetry):
        for name,object_state in step["objects"].items():
            if name in nearest:
                value=object_state["gripper_distance_m"]
                nearest[name]=value if nearest[name] is None else min(nearest[name],value)
            lifted=name!="basket_1" and object_state["both_fingerpads_contact"] and object_state["lift_from_reset_m"]>=.03
            streak[name]=streak.get(name,0)+1 if lifted else 0
            if streak[name]>=3:first.setdefault(name,index+1)
    earliest=min(first.values()) if first else None
    candidates=[name for name,value in first.items() if value==earliest]
    choice=candidates[0] if len(candidates)==1 else None
    role="ambiguous_multiple" if len(candidates)>1 else "none" if choice is None else "target" if choice==target else "distractor" if choice==distractor else "other_object"
    return {"first_grasp_candidate":choice,"first_grasp_candidate_role":role,"first_candidate_step":earliest,
            "all_sustained_grasp_candidates":sorted(first),"minimum_target_gripper_distance_m":nearest[target],
            "minimum_distractor_gripper_distance_m":nearest[distractor],"reviewed":False,
            "interpretation":"Native contact/lift grasp-choice proxy only; requires video review for semantic failure classification"}


def nominal_request_delta(reference,candidate,contract):
    """Native affine OSC scaling after clipping; never integrate into a trajectory."""
    a,b=np.asarray(reference,float),np.asarray(candidate,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=7 or not np.isfinite([a,b]).all():
        raise ValueError("Expected matched finite decoded seven-dimensional controller commands")
    if not contract["use_delta"] or contract["control_dim"]!=6 or contract["impedance_mode"]!="fixed":
        raise ValueError("Physical request conversion requires the audited fixed-impedance delta OSC contract")
    low,high=np.asarray(contract["input_min"]),np.asarray(contract["input_max"])
    output_low,output_high=np.asarray(contract["output_min"]),np.asarray(contract["output_max"])
    def scale(values):
        return (np.clip(values[:,:6],low,high)-(low+high)/2)*((output_high-output_low)/(high-low))+(output_low+output_high)/2
    difference=scale(b)-scale(a)
    return {"translation_request_delta_l2_m":float(np.linalg.norm(difference[:,:3])),
            "first5_translation_request_delta_l2_m":float(np.linalg.norm(difference[:5,:3])),
            "per_step_translation_request_delta_xyz_m":difference[:,:3].tolist(),
            "rotation_request_delta_l2_rad":float(np.linalg.norm(difference[:,3:6])),
            "gripper_clipped_command_delta_l2":float(np.linalg.norm(np.clip(b[:,6],-1,1)-np.clip(a[:,6],-1,1))),
            "physical_request_scope":"Nominal per-command OSC delta targets after native clipping/scaling, not achieved displacement or a predicted trajectory"}


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({key:json.dumps(value) if isinstance(value,(list,dict)) else value for key,value in row.items()})


def analyze(root,task,policy):
    root=Path(root).resolve();directory=root/"interventions"/task/intervention_policy(policy)
    top=json.loads((directory/"summary.json").read_text())
    if not top.get("complete",top.get("completed")):raise ValueError("Closed-loop intervention campaign is incomplete")
    plan=top["case_plan"]
    if plan.get("technical_smoke") or set(plan["conditions"])!=set(CLOSED):
        raise ValueError("Expected all nine predeclared scientific intervention conditions")
    protocol=json.loads((root/"tasks"/task/"protocol.json").read_text())
    expected_ids=protocol["tasks"][task]["test_state_ids"][:10]
    if plan["state_ids"]!=expected_ids or plan["seeds"]!=list(range(100000,100010)):
        raise ValueError("Closed-loop intervention cases differ from the predeclared first10 paired test cases")
    selection=root/"methods/rldt"/task/"selection.json"
    selection_sha=hashlib.sha256(selection.read_bytes()).hexdigest()
    if plan["selection_sha256"]!=selection_sha:
        raise ValueError("Intervention checkpoint selection identity changed")
    reports={};choices=[];choice_reports={}
    for condition in CLOSED:
        report=json.loads((directory/condition.replace(":","_")/"summary.json").read_text())
        if not report.get("complete",report.get("completed")):raise ValueError("Closed-loop condition is incomplete")
        episodes=report["episodes"]
        if report.get("technical_smoke") or len(episodes)!=10:
            raise ValueError("Scientific closed-loop intervention needs all10 episodes")
        if [row["reset"]["initial_state_id"] for row in episodes]!=expected_ids:
            raise ValueError("Intervention report reset IDs differ")
        if [row["seed"] for row in episodes]!=plan["seeds"]:
            raise ValueError("Intervention noise seeds differ")
        reports[condition]=episodes
        proxies=[]
        for episode in episodes:
            telemetry_path=directory/condition.replace(":","_")/f"episode_{episode['episode']:03d}_telemetry.json"
            telemetry=json.loads(telemetry_path.read_text())
            if len(telemetry)!=episode["primitive_steps"]:raise ValueError("Incomplete native intervention telemetry")
            choice=contact_choice(telemetry,episode["target"],episode["distractor"])
            choices.append({"task":task,"policy":policy,"condition":condition,"state_id":episode["reset"]["initial_state_id"],
                            "seed":episode["seed"],"native_success":episode["success"],"telemetry":str(telemetry_path),**choice})
            proxies.append(dict(episode,success=choice["first_grasp_candidate_role"]=="target"))
        choice_reports[condition]=proxies
    paired=[]
    for condition in CLOSED:
        paired.append(dict(task=task,policy=policy,condition=condition,**paired_success(reports["none"],reports[condition])))
    proxy_pairs=[dict(task=task,policy=policy,condition=condition,**paired_success(choice_reports["none"],choice_reports[condition])) for condition in CLOSED]
    trace=root/"traces"/task/policy
    states=sorted(trace.glob("state_*"))
    if [int(state.name.split("_")[1]) for state in states]!=protocol["tasks"][task]["test_state_ids"]:
        raise ValueError("Open-loop intervention analysis needs all50 predetermined test states")
    open_rows=[]
    controller_path=root/"native_controller_contract.json"
    controller=json.loads(controller_path.read_text())
    for state in states:
        record=json.loads((state/"open_loop_summary.json").read_text())
        if not record.get("complete") or {row["condition"] for row in record["conditions"]}!=set(OPEN):
            raise ValueError("Expected all15 open-loop conditions")
        with np.load(state/"open_loop_actions.npz",allow_pickle=False) as archive:
            commands={name:archive[name] for name in archive.files if name.endswith("__decoded")}
        for row in record["conditions"]:
            decoded=row.get("decoded_controller_delta",row.get("physical_delta"))
            if row["condition"]=="none" and (row["normalized_delta"]["l2"]!=0 or decoded["l2"]!=0):
                raise ValueError("No-op intervention changed actions")
            open_rows.append({"task":task,"policy":policy,"state_id":int(state.name.split("_")[1]),"condition":row["condition"],
                              "normalized_l2":row["normalized_delta"]["l2"],"decoded_controller_l2":decoded["l2"],
                              "decoded_controller_xyz_l2":decoded["xyz_l2"],"severity":row["severity"],
                              "units":"LIBERO OSC/gripper controller command coordinates; not metres",
                              **nominal_request_delta(commands["none__decoded"],commands[row["condition"].replace(":","_")+"__decoded"],controller)})
    output=root/"causal_analysis"/task/policy
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"closed_loop_pairs.csv",paired);write_csv(output/"open_loop_actions.csv",open_rows)
    write_csv(output/"grasp_choice_proxies.csv",choices);write_csv(output/"paired_target_grasp_proxies.csv",proxy_pairs)
    ratios=[]
    indexed={(row["state_id"],row["condition"]):row for row in open_rows}
    for state in protocol["tasks"][task]["test_state_ids"]:
        for mode in ("background_fill","mosaic"):
            a=indexed[state,f"target:{mode}"];b=indexed[state,f"distractor:{mode}"];control=indexed[state,f"random:{mode}"]
            ratios.append({"task":task,"policy":policy,"state_id":state,"replacement":mode,
                           "target_distractor_controller_sensitivity_ratio":(a["decoded_controller_l2"]+1e-12)/(b["decoded_controller_l2"]+1e-12),
                           "target_random_controller_sensitivity_ratio":(a["decoded_controller_l2"]+1e-12)/(control["decoded_controller_l2"]+1e-12),
                           "target_action_delta":a["decoded_controller_l2"],"distractor_action_delta":b["decoded_controller_l2"],
                           "matched_random_action_delta":control["decoded_controller_l2"],"epsilon":1e-12,
                           "target_pixel_severity":a["severity"],"distractor_pixel_severity":b["severity"],"random_pixel_severity":control["severity"],
                           "caveat":"Target/distractor areas and pixel severities differ; ratio alone is not object importance. Random control matches target mask area."})
    write_csv(output/"causal_sensitivity_ratios.csv",ratios)
    aggregates=[]
    for condition in OPEN:
        values=np.asarray([row["decoded_controller_l2"] for row in open_rows if row["condition"]==condition])
        rng=np.random.default_rng(20261031)
        boot=rng.choice(values,size=(20000,len(values))).mean(1)
        aggregates.append({"condition":condition,"states":len(values),"mean_decoded_controller_l2":float(values.mean()),
                           "paired_state_bootstrap95_mean":np.quantile(boot,[.025,.975]).tolist()})
    summary={"complete":True,"task":task,"policy":policy,"closed_loop_conditions":9,"paired_episodes_per_condition":10,
             "open_loop_conditions":15,"open_loop_states":50,"paired_success":paired,"open_loop_aggregates":aggregates,
             "paired_target_first_grasp_proxy":proxy_pairs,"grasp_choice_semantics":"Unreviewed native contact/lift proxy, not confirmed wrong-object failure classification",
             "selection_sha256":selection_sha,"units":"Decoded deltas are LIBERO OSC/gripper controller commands, not metres",
             "controller_contract_sha256":hashlib.sha256(controller_path.read_bytes()).hexdigest(),
             "nominal_physical_request_units":"Additional translation request deltas are metres after native clipping/scaling; rotation requests radians; not achieved displacement",
             "limitations":["Single RL seed; first10cases exploratory closed-loop; no checkpoint reselection",
                            "Direct image-key knockout can retain visual information in contextualized text keys; analogous caveat for text-key knockout",
                            "Old-target mask is a source-BDDL vertical-prism projection, not an instance mask; secondary layer-window interventions are reported separately in trace_analysis"]}
    temporary=output/"summary.tmp";temporary.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n");temporary.replace(output/"summary.json")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    means=np.asarray([row["mean_decoded_controller_l2"] for row in aggregates])
    ci=np.asarray([row["paired_state_bootstrap95_mean"] for row in aggregates]).T
    ax.errorbar(range(len(OPEN)),means,yerr=np.abs(ci-means),fmt="o",capsize=3)
    ax.set(xticks=range(len(OPEN)),xticklabels=OPEN,ylabel="Action-chunk L2 change (controller-command units)",title=f"{task} · {policy} · identical inputs/noise,50cases")
    ax.tick_params(axis="x",rotation=60);ax.grid(alpha=.2)
    fig.savefig(output/"open_loop_sensitivity.png",dpi=160);fig.savefig(output/"open_loop_sensitivity.pdf");plt.close(fig)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True,type=Path)
    p.add_argument("--task",required=True);p.add_argument("--policy",required=True,choices=POLICIES)
    a=p.parse_args();print(json.dumps(analyze(a.run,a.task,a.policy),indent=2))
