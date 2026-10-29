"""Real renderer/reset/controller/success-predicate checks; requires Slurm GPU."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .adapter import make_env, UPSTREAM_COMMIT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=110000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    env = None
    report = {"upstream_commit": UPSTREAM_COMMIT, "seed": args.seed,
              "job_id": os.getenv("SLURM_JOB_ID"), "step_id": os.getenv("SLURM_STEP_ID"),
              "versions": {x: importlib.metadata.version(x) for x in ("sapien", "mplib", "numpy", "torch")}}
    try:
        env = make_env(max_steps=3)
        obs, info = env.reset(seed=args.seed)
        assert not info["is_success"]
        assert env.observation_space.contains(obs)
        report["reset"] = info
        report["state"] = obs["state"].tolist()
        for name, key in [("head", "observation/image"), ("left", "observation/wrist_image"),
                          ("right", "observation/right_wrist_image")]:
            image = obs["openpi"][key]
            assert image.max() > image.min(), "Camera is blank"
            Image.fromarray(image).save(args.output / f"{name}.png")
        for _ in range(3):
            obs, reward, term, trunc, info = env.step(obs["state"].copy())
            assert reward == 0 and not term
        assert trunc and info["controller_steps"] == 3
        report["step"] = info | {"terminated": term, "truncated": trunc}
        # This is a predicate check, not a successful rollout: place the block's
        # functional point exactly on the pad and test the unmodified predicate.
        task = env.task
        pose = task.box.get_pose()
        delta = task.target_box.get_functional_point(1, "pose").p - task.box.get_functional_point(0, "pose").p
        pose.p = pose.p + delta
        task.box.actor.set_pose(pose)
        assert task.check_success(), "Upstream goal predicate failed at goal geometry"
        report["goal_predicate_fixture"] = True
        report["ready"] = True
    except BaseException as exc:
        report["ready"] = False
        report["error"] = repr(exc)
        raise
    finally:
        if env is not None:
            env.close()
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
