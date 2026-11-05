"""Plot the predetermined first test state's physical prior-location regions."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root):
    root=Path(root)
    fig,axes=plt.subplots(3,2,figsize=(9,12),layout="constrained")
    for index,task in enumerate(("object_0","object_1","object_2")):
        bank=root/"observation_bank"/task/"test/state_000150"
        aux=root/"observation_bank_aux"/task/"test/state_000150"
        metadata=json.loads((aux/"metadata.json").read_text())
        with np.load(bank/"observation.npz",allow_pickle=False) as data:
            for camera,ax in zip(("image","wrist_image"),axes[index],strict=True):
                ax.imshow(data[camera]);ax.set_axis_off()
                for key,color in (("floor_polygon_xy","yellow"),("prism_silhouette_polygon_xy","cyan")):
                    vertices=np.asarray(metadata["cameras"][camera][key]);vertices=np.concatenate([vertices,vertices[:1]])
                    ax.plot(vertices[:,0],vertices[:,1],color=color,lw=1.2)
                ax.set_xlim(-.5,223.5);ax.set_ylim(223.5,-.5)
                ax.set_title(f"{task} · {camera}\nFloor z={metadata['floor_plane_z']:.2f} m; prism height={metadata['prism_height_m']:.3f} m",fontsize=10)
    fig.suptitle("Original target location projected onto immutable policy images\nYellow: physical floor region. Cyan: target-height volume silhouette.\nGeometric regions include occluded volume; they are not visible object masks.",fontsize=12)
    for extension in ("png","pdf"):fig.savefig(root/f"figures/old_target_physical_floor_audit.{extension}",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",type=Path,required=True)
    plot(parser.parse_args().run)
