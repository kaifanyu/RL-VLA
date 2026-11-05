"""Measure native expert attention and render VLA-Trace overlays for one matched frame."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.metrics import grounding_scores, patch_occupancy

CAMERA_NAMES = {"base_0_rgb": "image", "left_wrist_0_rgb": "wrist_image"}
OBJECTS = {"object_0": ("alphabet_soup_1", "butter_1"),
           "object_1": ("cream_cheese_1", "alphabet_soup_1"),
           "object_2": ("salad_dressing_1", "tomato_sauce_1")}


def summarize(trace, bank, task, output, toolkit, scope="technical-fixed-input"):
    trace, bank, output = Path(trace), Path(bank), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((trace / "metadata.json").read_text())
    bank_meta = json.loads((bank / "metadata.json").read_text())
    masks = np.load(bank / "masks.npz", allow_pickle=False)
    observation = np.load(bank / "observation.npz", allow_pickle=False)
    if (trace/"observation.npz").exists():
        with np.load(trace/"observation.npz",allow_pickle=False) as trace_observation:
            for key in ("image", "wrist_image", "state", "prompt"):
                if not np.array_equal(observation[key], trace_observation[key]):
                    raise ValueError(f"Trace/bank input mismatch: {key}")
    else:
        expected=metadata["bank_record"]["observation_sha256"]
        if expected!=bank_meta["observation_sha256"] or hashlib.sha256((bank/"observation.npz").read_bytes()).hexdigest()!=expected:
            raise ValueError("Trace/bank observation hash mismatch")
    if scope=="fixed-test":
        selection=bank.resolve().parents[3]/"methods/rldt"/task/"selection.json"
        if bank_meta["split"]!="test" or not json.loads(selection.read_text()).get("locked"):
            raise ValueError("Scientific toolkit export requires locked selection and frozen test inputs")
    target, distractor = OBJECTS[task]
    rows, modality = [], []
    toolkit_commands = []
    toolkit_attention={camera:{} for camera in CAMERA_NAMES.values()}
    toolkit_masks={camera:{} for camera in CAMERA_NAMES.values()}
    for attention_path in sorted(trace.glob("expert_step*_none.npz")):
        step = int(attention_path.stem.split("_")[1][4:])
        with np.load(attention_path, allow_pickle=False) as data:
            attention = data["attention"]
            time = float(data["t"][0])
        if attention.ndim != 5 or attention.shape[1] != 1:
            raise ValueError("Expected [layer,one_batch,head,query,key] native attention")
        for layer in range(attention.shape[0]):
            for head in range(attention.shape[2]):
                queries = attention[layer, 0, head]
                text_lo, text_hi = metadata["text_span"]
                image_mass = sum(queries[:, s["start"]:s["end"]].sum(-1) for s in metadata["camera_spans"])
                modality.append({"task": task, "step":step, "time":time, "layer":layer, "head":head,
                    "image_mass":float(image_mass.mean()), "text_mass":float(queries[:,text_lo:text_hi].sum(-1).mean()),
                    "action_mass":float(queries[:,metadata["action_key_span"][0]:metadata["action_key_span"][1]].sum(-1).mean()),
                    "row_sum":float(queries.sum(-1).mean()),
                    "entropy_nats":float(np.mean(-np.sum(np.where(queries>0, queries*np.log(np.maximum(queries,1e-30)),0),axis=-1)))})
        for span in metadata["camera_spans"]:
            if not span["valid"]:
                continue
            camera = CAMERA_NAMES.get(span["name"])
            if camera is None:
                raise ValueError(f"Unknown valid camera mapping: {span['name']}")
            grid = tuple(span["grid"])
            values = attention[...,span["start"]:span["end"]]
            occupancy = {role:patch_occupancy(masks[f"{camera}__{name}"],grid)
                         for role,name in (("target",target),("distractor",distractor))}
            for layer in range(values.shape[0]):
                for head in range(values.shape[2]):
                    for query in range(values.shape[3]):
                        heatmap = values[layer,0,head,query].reshape(grid)
                        for role,mask in occupancy.items():
                            rows.append({"task":task,"state_id":bank_meta.get("initial_state_id"),
                                         "step":step,"time":time,"camera":camera,"layer":layer,
                                         "head":head,"action_query":query,"region":role,
                                         **grounding_scores(heatmap,mask)})
            # Toolkit overlays use mean over ALL layers/heads/queries, declared here.
            average = values.mean(axis=(0,1,2,3)).reshape(grid)
            np.save(output/f"{camera}.npy",observation[camera],allow_pickle=False)
            np.savez_compressed(output/f"step{step:02d}_{camera}_attention.npz",action_to_image=average)
            toolkit_attention[camera][f"step_{step:03d}"]=average
            for role,name in (("target",target),("distractor",distractor)):
                toolkit_masks[camera][f"step_{step:03d}_{role}"]=masks[f"{camera}__{name}"]
                command=[str(toolkit),"attention-overlay","--image",str(output/f"{camera}.npy"),
                    "--attention",str(output/f"step{step:02d}_{camera}_attention.npz"),"--attention-key","action_to_image",
                    "--masks",str(bank/"masks.npz"),"--mask-key",f"{camera}__{name}",
                    "--grid-size",str(grid[0]),"--output",str(output/f"step{step:02d}_{camera}_{role}.png")]
                subprocess.run(command,check=True,capture_output=True,text=True)
                toolkit_commands.append(command)
    for camera in CAMERA_NAMES.values():
        np.savez(output/f"toolkit_{camera}_attention.npz",**toolkit_attention[camera])
        np.savez(output/f"toolkit_{camera}_masks.npz",**toolkit_masks[camera])
        command=[str(toolkit),"attention-metrics","--attention",str(output/f"toolkit_{camera}_attention.npz"),
                 "--masks",str(output/f"toolkit_{camera}_masks.npz"),"--output",str(output/f"toolkit_{camera}_metrics.csv"),
                 "--summary",str(output/f"toolkit_{camera}_summary.json"),"--grid-size","16"]
        subprocess.run(command,check=True,capture_output=True,text=True);toolkit_commands.append(command)
    for name,values in (("grounding",rows),("modality",modality)):
        if values:
            with (output/f"{name}.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    (output/"summary.json").write_text(json.dumps({"complete":bool(rows),"task":task,"trace":str(trace),
        "bank":str(bank),"rows":len(rows),"overlay_aggregation":"mean over all layers, heads, action queries; separately each camera and ODE time",
        "metrics":"Full-denominator native region mass; fractional patch occupancy; per-layer/head/query rows",
        "toolkit_commands":toolkit_commands,"scope":scope,
        "toolkit_metric_convention":"Official VLA-Trace attention-metrics uses camera-conditional mass, nearest-sampled binary masks, entropy in bits. These supplemental outputs are distinct from our primary full-denominator/fractional-occupancy metrics.",
        "caveat":"Fixed-input attention is not a causal or actual closed-loop behavior result"},indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",required=True,type=Path)
    parser.add_argument("--bank",required=True,type=Path)
    parser.add_argument("--task",required=True,choices=list(OBJECTS))
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--toolkit",type=Path,default=Path("third_party/vla_trace_tools/.venv/bin/vla-trace"))
    parser.add_argument("--scope",choices=["technical-fixed-input","fixed-test"],default="technical-fixed-input")
    args=parser.parse_args()
    summarize(args.trace,args.bank,args.task,args.output,args.toolkit,args.scope)
