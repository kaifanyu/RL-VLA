"""Replay recorded native actions and export predetermined rollout decision phases."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.old_target import project_old_target_region
from rl_vla.diagnostics.sensor_masks import SensorMaskCapture
from rl_vla.envs.libero import LiberoEnv
from rl_vla.envs.libero_pro import make_env

TARGETS={"object_0":"alphabet_soup_1","object_1":"cream_cheese_1","object_2":"salad_dressing_1"}


def read_chunk(path):
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data["metadata_json"].item()))
        schema=metadata["observation_schema"]
        observation={}
        for name,parts in schema.items():
            if parts[0]=="openpi":observation[parts[1]]=data[name][0]
        observation["prompt"]=str(observation["prompt"].item())
        return metadata,observation,data["executed_actions"],data["normalized_actions"],data["decoded_actions"]


def phase_indices(chunks,initial_telemetry,target):
    """Phases are chosen by a fixed contact/distance rule, never model attention."""
    current=initial_telemetry;decisions=[];streaks={};grasp_step=None;grasp_object=None
    for index,(metadata,_,_,_,_) in enumerate(chunks):
        objects=current.get("objects",{})
        decisions.append({"index":index,"step":metadata["first_primitive_step"],"objects":objects})
        for local,info in enumerate(metadata["step_infos"]):
            current=info.get("object_telemetry",{})
            for name,state in current.get("objects",{}).items():
                valid=name!="basket_1" and state["both_fingerpads_contact"] and state["lift_from_reset_m"]>=.03
                streaks[name]=streaks.get(name,0)+1 if valid else 0
                if streaks[name]>=3 and grasp_step is None:
                    grasp_step=metadata["first_primitive_step"]+local+1;grasp_object=name
    phases={"initial_decision":0,"last_decision":len(chunks)-1}
    approach=[row for row in decisions if target in row["objects"] and (grasp_step is None or row["step"]<grasp_step)]
    if approach:phases["closest_requested_target_approach"]=min(approach,key=lambda r:(r["objects"][target]["gripper_distance_m"],r["index"]))["index"]
    after=[row for row in decisions if grasp_step is not None and row["step"]>=grasp_step]
    if after:
        phases["first_decision_after_sustained_grasp"]=after[0]["index"]
        for row in after:
            objects=row["objects"]
            if grasp_object in objects and "basket_1" in objects:
                distance=np.linalg.norm(np.asarray(objects[grasp_object]["position_xyz"])[:2]-np.asarray(objects["basket_1"]["position_xyz"])[:2])
                if distance<=.12:
                    phases["first_grasped_object_approach_to_basket"]=row["index"];break
    return phases,{"grasp_event_step":grasp_step,"grasped_object_candidate":grasp_object,
                   "grasp_rule":"At least3 consecutive controlled steps of both-fingerpad contact and lift>=3cm; candidate only, not visual semantic review"}


def extract(protocol,task,split,episode,output,policy,checkpoint=None):
    protocol,episode,output=map(Path,(protocol,episode,output))
    output.mkdir(parents=True,exist_ok=False)
    episode_meta=json.loads((episode/"episode.json").read_text())
    if not episode_meta.get("complete"):raise ValueError("Recorded episode is not complete")
    reset=episode_meta["metadata"]["reset"];seed=episode_meta["metadata"]["seed"]
    paths=sorted(episode.glob("chunk_*.npz"));chunks=[read_chunk(path) for path in paths]
    initial=reset.get("object_telemetry",{})
    if "simulator_state" not in initial:raise ValueError("Rollout lacks native simulator-state telemetry")
    phases,events=phase_indices(chunks,initial,TARGETS[task])
    plan={"task":task,"split":split,"policy":policy,"checkpoint":str(checkpoint) if checkpoint else None,
          "episode":str(episode.resolve()),"initial_state_id":reset["initial_state_id"],"seed":seed,"phases":phases,
          "events":events,"frame_semantics":"Actual rollout-conditioned decision observations; policies may have different pixels. Not a fixed-input cross-policy comparison.",
          "noise":"Recorded native JAX sampling key and exact normalized action chunk retained for replay fidelity"}
    (output/"phase_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    definition=json.loads(protocol.read_text())["tasks"][task]
    env=make_env(protocol=str(protocol),task_key=task,split=split)
    capture=SensorMaskCapture(env)
    entries=[]
    try:
        capture.enabled=True
        raw,_=LiberoEnv.reset(env,seed=seed,options={"initial_state_id":reset["initial_state_id"]})
        capture.enabled=False
        for index,(metadata,recorded,actions,normalized,decoded) in enumerate(chunks):
            names=[name for name,value in phases.items() if value==index]
            if names:
                for key in ("observation/image","observation/wrist_image","observation/state"):
                    if not np.array_equal(raw["openpi"][key],recorded[key]):
                        raise ValueError(f"Recorded action replay changed decision pixels/state: {index}/{key}")
                if raw["openpi"]["prompt"]!=recorded["prompt"]:raise ValueError("Recorded instruction changed")
                masks,mask_meta=capture.masks(raw)
                old_masks,old_meta=project_old_target_region(env,source_bddl=definition["source"]["bddl"]["path"],target=TARGETS[task],allow_invisible=True,camera_transforms=capture.transforms)
                masks.update(old_masks)
                destination=output/f"decision_{index:06d}";destination.mkdir()
                np.savez_compressed(destination/"observation.npz",image=recorded["observation/image"],wrist_image=recorded["observation/wrist_image"],state=recorded["observation/state"],prompt=recorded["prompt"],
                                    sim_state=env.backend.get_sim_state(),sampling_key=np.asarray(metadata["decision"]["sampling_key"],np.uint32),
                                    recorded_normalized_actions=normalized,recorded_decoded_actions=decoded)
                np.savez_compressed(destination/"masks.npz",**masks)
                record={**plan,"decision_index":index,"first_primitive_step":metadata["first_primitive_step"],"phase_names":names,
                        "source_chunk":str(paths[index].resolve()),"source_chunk_sha256":hashlib.sha256(paths[index].read_bytes()).hexdigest(),
                        "observation_path":str((destination/"observation.npz").resolve()),"mask_metadata":mask_meta,"old_target_region":old_meta,
                        "exact_recorded_pixels_state_verified":True}
                (destination/"metadata.json").write_text(json.dumps(record,indent=2)+"\n");entries.append(record)
                print(json.dumps({"episode":str(episode),"decision":index,"phases":names}),flush=True)
            for action_index,action in enumerate(actions):
                capture.enabled=index+1 in phases.values() and action_index==len(actions)-1
                raw,_,_,_,_=env.step(action)
                expected_sim=metadata["step_infos"][action_index].get("object_telemetry",{}).get("simulator_state")
                if expected_sim is None or not np.array_equal(env.backend.get_sim_state(),np.asarray(expected_sim)):
                    raise ValueError(f"Recorded action replay changed full simulator state: {index}/{action_index}")
            capture.enabled=False
    finally:env.close()
    (output/"manifest.jsonl").write_text("".join(json.dumps(row)+"\n" for row in entries))
    (output/"summary.json").write_text(json.dumps({"complete":True,"phase_decisions":len(entries),"plan":plan,
        "primitive_steps_with_exact_recorded_simulator_state":sum(len(chunk[2]) for chunk in chunks),
        "sensor_timed_masks_exact_policy_rgb":True},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--protocol",required=True,type=Path)
    p.add_argument("--task",required=True,choices=list(TARGETS));p.add_argument("--split",required=True)
    p.add_argument("--episode",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--policy",required=True);p.add_argument("--checkpoint",type=Path)
    a=p.parse_args();extract(a.protocol,a.task,a.split,a.episode,a.output,a.policy,a.checkpoint)
