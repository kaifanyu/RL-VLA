"""Visualize native sensor-timed instance alignment over a real recorded episode."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(bank,output):
    bank=Path(bank);output=Path(output)
    records=[json.loads(line) for line in (bank/"manifest.jsonl").read_text().splitlines()]
    fig,axes=plt.subplots(2,len(records),figsize=(3*len(records),6),layout="constrained")
    for column,record in enumerate(records):
        directory=Path(record["observation_path"]).parent
        with np.load(directory/"observation.npz",allow_pickle=False) as images,np.load(directory/"masks.npz",allow_pickle=False) as masks:
            for row,camera in enumerate(("image","wrist_image")):
                axis=axes[row,column];axis.imshow(images[camera]);axis.set_axis_off()
                for obj,color in (("alphabet_soup_1","lime"),("butter_1","cyan"),("old_target_prism","yellow")):
                    mask=masks[f"{camera}__{obj}"]
                    if mask.any() and not mask.all():axis.contour(mask,levels=[.5],colors=[color],linewidths=.8)
                axis.set_title(f"Decision {record['decision_index']} · {camera}\n"+"\n".join(name.replace("_"," ") for name in record["phase_names"]),fontsize=8)
    fig.suptitle("Technical sensor-alignment audit · actual Object0 validation rollout\nRequested soup green; butter cyan; prior-location prism yellow. RGB/state/action replay is unchanged.\nFive predetermined phases span approach, grasp, transport and final failure; these are divergent rollout observations, not fixed-input comparisons.",fontsize=11)
    for extension in ("png","pdf"):fig.savefig(output.with_suffix("."+extension),dpi=160)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--bank",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();plot(args.bank,args.output)
