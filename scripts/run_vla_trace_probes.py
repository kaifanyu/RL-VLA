"""Fit diagnostic-only linear patch probes on frozen train-state features."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.metrics import patch_occupancy
from rl_vla.diagnostics.probes import fit_linear_probe, probe_scores, score_probe

OBJECTS={"object_0":("alphabet_soup_1","butter_1"),"object_1":("cream_cheese_1","alphabet_soup_1"),"object_2":("salad_dressing_1","tomato_sauce_1")}
CAMERAS={"base_0_rgb":"image","left_wrist_0_rgb":"wrist_image"}


def examples(features_root, bank_root, task, split, feature_kind, projection, source_features=None):
    target,distractor=OBJECTS[task]
    rows=[]
    manifests=[json.loads(line) for line in (bank_root/split/"manifest.jsonl").read_text().splitlines()]
    for record in manifests:
        state=record["initial_state_id"]
        split_features=source_features if split=="source" and source_features is not None else features_root/split
        folder=split_features/f"state_{state:06d}"
        metadata=json.loads((folder/"metadata.json").read_text())
        if metadata.get("source_observation") and Path(metadata["source_observation"]).resolve()!=Path(record["observation_path"]).resolve():
            raise ValueError("Feature and mask banks refer to different observations")
        with np.load(folder/"prefix.npz",allow_pickle=False) as data:
            features=data[feature_kind][0]
        with np.load(record["masks_path"],allow_pickle=False) as masks:
            for span in metadata["camera_spans"]:
                if not span["valid"]:
                    continue
                camera=CAMERAS[span["name"]]
                grid=tuple(span["grid"])
                a=patch_occupancy(masks[f"{camera}__{target}"],grid)
                b=patch_occupancy(masks[f"{camera}__{distractor}"],grid)
                # Ambiguous overlapping target/distractor patches are excluded
                # from classification, while full occupancy remains for localization.
                labels=np.zeros(a.size,dtype=np.int32)
                labels[a.ravel()>=.1]=1
                labels[b.ravel()>=.1]=2
                valid=~((a.ravel()>=.1)&(b.ravel()>=.1))
                x=features[span["start"]:span["end"]].astype(np.float32)@projection
                rows.append({"state_id":state,"camera":camera,"features":x,"labels":labels,
                             "valid":valid,"target_occupancy":a,"distractor_occupancy":b})
    return rows


def run(task,model,features_root,bank_root,output,source_features=None):
    features_root,bank_root,output=map(Path,(features_root,bank_root,output))
    source_features=Path(source_features) if source_features is not None else None
    selection_path=features_root.resolve().parents[2]/"methods/rldt"/task/"selection.json"
    if not selection_path.exists():
        raise ValueError("Locked checkpoint selection is required before probe test/source scoring")
    import hashlib
    selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest()
    output.mkdir(parents=True,exist_ok=False)
    train_summary=json.loads((bank_root/"train/summary.json").read_text())
    if train_summary["entries"]!=100:
        raise ValueError("Full probe fitting requires all100 frozen training states")
    first=next((features_root/"train").glob("state_*/prefix.npz"))
    with np.load(first,allow_pickle=False) as data:
        dimension=data["image_features"].shape[-1]
    projection=(np.random.default_rng(20261031).standard_normal((dimension,256))/16).astype(np.float32)
    np.save(output/"fixed_projection.npy",projection,allow_pickle=False)
    specification={"task":task,"model":model,"fit_split":"train","fit_states":100,
                   "feature_kinds":["image_features","prefix_features"],"classes":{"0":"background","1":"target","2":"distractor"},
                   "label_rule":"Patch occupancy>=0.1; target/distractor overlaps excluded from classification",
                   "ridge":10.,"projection":"Fixed Gaussian2048to256 (actual input dimensions recorded), linear overall; no fitted PCA or test preprocessing",
                   "input_features":dimension,"projected_features":256,"projection_seed":20261031,
                   "selection":"No hyperparameter selection; predeclared fixed settings",
                   "policy_unchanged":True,"base_action_contract":"not used by probe"}
    specification.update(selection_path=str(selection_path),selection_sha256=selection_sha256)
    (output/"specification.json").write_text(json.dumps(specification,indent=2)+"\n")
    measurements={}
    for kind in specification["feature_kinds"]:
        print(f"Loading training features: {task}/{model}/{kind}",flush=True)
        rows=examples(features_root,bank_root,task,"train",kind,projection)
        x=np.concatenate([r["features"][r["valid"]] for r in rows])
        y=np.concatenate([r["labels"][r["valid"]] for r in rows])
        probe=fit_linear_probe(x,y,ridge=10.,fit_split="train")
        np.savez_compressed(output/f"{kind}_probe.npz",**probe)
        del x,y,rows
        by_split={}
        for split in ("validation","test","source"):
            split_features=source_features if split=="source" and source_features is not None else features_root/split
            if not (split_features/"summary.json").exists():
                by_split[split]={"status":"pending","reason":"Completed feature batch missing"}
                continue
            rows=examples(features_root,bank_root,task,split,kind,projection,source_features)
            x=np.concatenate([r["features"][r["valid"]] for r in rows])
            y=np.concatenate([r["labels"][r["valid"]] for r in rows])
            metrics=score_probe(probe,x,y)
            del x,y
            predictions=[]
            raw_scores=[]
            raw_labels=[]
            raw_state_ids=[]
            raw_cameras=[]
            for row in rows:
                scores=probe_scores(probe,row["features"])
                raw_scores.append(scores)
                raw_labels.append(row["labels"])
                raw_state_ids.extend([row["state_id"]]*len(scores))
                raw_cameras.extend([row["camera"]]*len(scores))
                prediction=scores.argmax(axis=1).reshape(row["target_occupancy"].shape)
                for role,label in (("target",1),("distractor",2)):
                    actual=row[f"{role}_occupancy"]>=.1
                    predicted=prediction==label
                    union=actual|predicted
                    actual_yx=np.argwhere(actual)
                    predicted_yx=np.argwhere(predicted)
                    distance=float(np.linalg.norm(actual_yx.mean(0)-predicted_yx.mean(0))) if len(actual_yx) and len(predicted_yx) else None
                    predictions.append({"state_id":row["state_id"],"camera":row["camera"],"region":role,
                        "visible_target_patches":int(actual.sum()),"predicted_patches":int(predicted.sum()),
                        "iou":float((actual&predicted).sum()/union.sum()) if union.any() else None,
                        "centroid_error_patches":distance})
            np.savez_compressed(output/f"{kind}_{split}_predictions.npz",scores=np.concatenate(raw_scores),
                                labels=np.concatenate(raw_labels),state_ids=np.asarray(raw_state_ids),
                                cameras=np.asarray(raw_cameras))
            with (output/f"{kind}_{split}_localization.csv").open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(predictions[0]))
                writer.writeheader();writer.writerows(predictions)
            for role in ("target","distractor"):
                subset=[p for p in predictions if p["region"]==role and p["visible_target_patches"]>0]
                metrics[f"{role}_mean_iou_visible"]=float(np.mean([p["iou"] for p in subset])) if subset else None
                distances=[p["centroid_error_patches"] for p in subset if p["centroid_error_patches"] is not None]
                metrics[f"{role}_mean_centroid_error_when_predicted"]=float(np.mean(distances)) if distances else None
                metrics[f"{role}_visible_camera_states"]=len(subset)
                metrics[f"{role}_visible_but_no_prediction"]=sum(p["predicted_patches"]==0 for p in subset)
            by_split[split]=dict(metrics,status="complete")
            print(json.dumps({"kind":kind,"split":split,"accuracy":metrics["accuracy"],"balanced_accuracy":metrics["balanced_accuracy"]}),flush=True)
        measurements[kind]=by_split
    summary={"complete":all(s.get("status")=="complete" for v in measurements.values() for s in v.values()),
             "task":task,"model":model,"specification":specification,"measurements":measurements,
             "interpretation":"A positive result establishes decodability under this distribution, not causal policy use. Negative result does not establish absence. Image-only features control for contextual language integration."}
    (output/"metrics.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout="constrained")
    for kind,row_axes in zip(specification["feature_kinds"],axes,strict=True):
        for split,axis in zip(("validation","test","source"),row_axes,strict=True):
            result=measurements[kind][split]
            if result.get("status")!="complete":
                axis.set_axis_off();continue
            matrix=np.asarray(result["confusion"],float)
            matrix=np.divide(matrix,matrix.sum(1,keepdims=True),out=np.zeros_like(matrix),where=matrix.sum(1,keepdims=True)>0)
            axis.imshow(matrix,vmin=0,vmax=1,cmap="Blues")
            for i in range(3):
                for j in range(3):
                    axis.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",color="white" if matrix[i,j]>.6 else "black")
            axis.set(title=f"{kind} · {split}",xticks=range(3),yticks=range(3),
                     xticklabels=["background","target","distractor"],yticklabels=["background","target","distractor"],
                     xlabel="Predicted",ylabel="Simulator truth")
    fig.suptitle(f"{task} · {model} · diagnostic linear probe, train states only")
    fig.savefig(output/"confusion.png",dpi=160)
    fig.savefig(output/"confusion.pdf")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",required=True,choices=list(OBJECTS))
    p.add_argument("--model",required=True,choices=["pi05_base","pi05_libero"])
    p.add_argument("--features",required=True,type=Path)
    p.add_argument("--bank",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--source-features",type=Path)
    a=p.parse_args()
    run(a.task,a.model,a.features,a.bank,a.output,a.source_features)
