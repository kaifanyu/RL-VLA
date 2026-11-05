"""Replay a native recording through sensor-timed mask wrappers without a policy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.interventions import make_env


def audit(protocol,task,split,episode,output):
    from extract_vla_trace_rollout_bank import read_chunk

    episode=Path(episode);output=Path(output)
    record=json.loads((episode/"episode.json").read_text())
    if not record.get("complete"):raise ValueError("Incomplete episode")
    paths=sorted(episode.glob("chunk_*.npz"))
    chunks=[read_chunk(path) for path in paths]
    env=make_env(protocol=str(protocol),task_key=task,split=split,mask_every=5,limit=1)
    checks=[];primitives=0
    try:
        raw,reset=env.reset(seed=record["metadata"]["seed"])
        if reset["initial_state_id"]!=record["metadata"]["reset"]["initial_state_id"]:
            raise ValueError("Technical replay case must be the first predetermined split ID")
        for index,(metadata,expected,actions,_,_) in enumerate(chunks):
            for name in ("observation/image","observation/wrist_image","observation/state","prompt"):
                if not np.array_equal(raw["openpi"][name],expected[name]):
                    raise ValueError(f"Mask wrapper changed recorded observation: {index}/{name}")
            alignment=raw["diagnostic_mask_metadata"]["camera_alignment"]
            if not all(value["exact_sensor_rgb_match"] and value["simulator_state_unchanged"] for value in alignment.values()):
                raise ValueError("Mask camera synchronization failed")
            checks.append({"decision":index,"first_primitive_step":metadata["first_primitive_step"],
                           "all_policy_pixels_state_prompt_exact":True,"camera_alignment":alignment})
            for local,action in enumerate(actions):
                raw,_,_,_,_=env.step(action)
                expected_sim=np.asarray(metadata["step_infos"][local]["object_telemetry"]["simulator_state"])
                if not np.array_equal(env.backend.get_sim_state(),expected_sim):
                    raise ValueError(f"Mask wrapper changed full recorded simulator state: {index}/{local}")
                primitives+=1
    finally:env.close()
    result={"passed":True,"technical_only":True,"task":task,"split":split,"episode":str(episode.resolve()),
            "episode_sha256":hashlib.sha256((episode/"episode.json").read_bytes()).hexdigest(),
            "decisions":len(checks),"primitive_steps":primitives,"exact_recorded_full_state_every_primitive":True,
            "exact_recorded_policy_images_state_prompt_every_decision":True,"native_actions_executed_unchanged":True,
            "policy_model_action_recomputation":"Separate native-model sampling-key audit required",
            "mask_timing":"Segmentation immediately after original RGB sensor; camera transforms retained from same callback",
            "checks":checks}
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key!="checks"}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("protocol","episode","output"):parser.add_argument(f"--{name}",type=Path,required=True)
    for name in ("task","split"):parser.add_argument(f"--{name}",required=True)
    args=parser.parse_args();audit(args.protocol,args.task,args.split,args.episode,args.output)
