"""Exercise each frozen PRO/source split through the real simulator, without a GPU."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rl_vla.envs.libero_pro import make_env


def check(experiment):
    output = experiment / "setup_check"
    output.mkdir(exist_ok=True)
    records = []
    for task in ("spatial_0", "object_0"):
        for split in ("train", "heldout", "source"):
            print(f"Checking {task}/{split}", flush=True)
            env = make_env(protocol=str(experiment / "protocol.json"), task_key=task, split=split)
            try:
                raw, info = env.reset(seed=100000)
                image = raw["openpi"]["observation/image"]
                assert image.shape == (224, 224, 3) and float(image.std()) > 1
                Image.fromarray(image).save(output / f"{task}_{split}.png")
                _, reward, terminated, truncated, _ = env.step(np.array([0.] * 6 + [-1.]))
                records.append({"task": task, "split": split, "reset": info,
                                "image_std": float(image.std()), "reward": reward,
                                "terminated": terminated, "truncated": truncated})
                print(json.dumps(records[-1]), flush=True)
            finally:
                env.close()
    (output / "report.json").write_text(json.dumps({"passed": True, "checks": records},
                                                  indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    check(parser.parse_args().experiment.resolve())
