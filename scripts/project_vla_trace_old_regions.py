"""Write immutable prior-target-location auxiliary masks for saved observations."""

import argparse
import json
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.observation_bank import digest, simulator_masks
from rl_vla.diagnostics.old_target import project_old_target_region
from rl_vla.envs.libero_pro import make_env


def run(root,task,split="test"):
    root=Path(root).resolve();bank=root/"observation_bank"/task/split
    protocol=root/"tasks"/task/"protocol.json";plan=json.loads(protocol.read_text())
    definition=plan["tasks"][task];output=root/"observation_bank_aux"/task/split
    if (output/"summary.json").exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    env=make_env(protocol=str(protocol),task_key=task,split=split)
    rows=[]
    try:
        records=[json.loads(line) for line in (bank/"manifest.jsonl").read_text().splitlines()]
        for record in records:
            source=Path(record["observation_path"])
            with np.load(source,allow_pickle=False) as data:
                sim_state=data["sim_state"]
                raw={"state":data["state"],"openpi":{"observation/image":data["image"],"observation/wrist_image":data["wrist_image"],"observation/state":data["state"],"prompt":str(data["prompt"])}}
            # Replay the exact frozen reset, then verify the saved physical
            # state. MuJoCo render camera caches can correspond to the final
            # integration substep; blindly restoring qpos and calling forward
            # shifts those caches relative to the recorded policy image.
            replayed,_=env.reset(seed=record["evaluation_seed"])
            actual=env.backend.get_sim_state()
            state_error=float(np.max(np.abs(actual-sim_state)))
            if state_error>1e-10:raise ValueError(f"Frozen reset does not reconstruct saved simulator state: {state_error}")
            for key in ("observation/image","observation/wrist_image","observation/state"):
                if not np.array_equal(replayed["openpi"][key],raw["openpi"][key]):
                    raise ValueError(f"Frozen reset does not reconstruct exact saved observation: {key}")
            # Verify direct RGB and instance orientation against immutable pixels.
            _,alignment=simulator_masks(env,raw,audit_rgb=True)
            masks,metadata=project_old_target_region(env,source_bddl=definition["source"]["bddl"]["path"],target=record["target"]+"_1")
            destination=output/f"state_{record['initial_state_id']:06d}";destination.mkdir(exist_ok=True)
            if (destination/"metadata.json").exists():
                existing=json.loads((destination/"metadata.json").read_text())
                if existing["observation_sha256"]!=record["observation_sha256"] or existing["artifact_sha256"]!=digest(destination/"old_target_region.npz"):
                    raise ValueError("Existing immutable auxiliary artifact changed")
                rows.append({"state_id":record["initial_state_id"],"mask_path":str(destination/"old_target_region.npz"),"metadata_path":str(destination/"metadata.json"),"observation_sha256":record["observation_sha256"]})
                continue
            np.savez_compressed(destination/"old_target_region.npz",**masks)
            metadata.update(observation_path=str(source),observation_sha256=record["observation_sha256"],
                            simulator_state_source="exact frozen reset replay; physical state verified against saved sim_state and pixels/state exactly matched",
                            simulator_state_max_abs_difference=state_error,
                            camera_alignment=alignment["camera_alignment"],artifact_sha256=digest(destination/"old_target_region.npz"))
            (destination/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
            rows.append({"state_id":record["initial_state_id"],"mask_path":str(destination/"old_target_region.npz"),"metadata_path":str(destination/"metadata.json"),"observation_sha256":record["observation_sha256"]})
            print(json.dumps({"task":task,"completed":len(rows),"total":len(records)}),flush=True)
    finally:env.close()
    (output/"manifest.json").write_text(json.dumps(rows,indent=2)+"\n")
    (output/"summary.json").write_text(json.dumps({"complete":True,"task":task,"split":split,"states":len(rows),"primary_mask":"old_target_prism"},indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",required=True,type=Path)
    p.add_argument("--task",required=True);p.add_argument("--split",default="test")
    args=p.parse_args();run(args.run,args.task,args.split)
