"""Real CPU-rendered reset/step/camera/success checks; no policy evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from PIL import Image

from integrations.multibench.robocasa.environment import CAMERA_KEYS, RoboCasaEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "evidence/smoke")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 1])
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "simulator smoke, not policy success measurement", "checks": [],
              "python": platform.python_version(),
              "packages": {n: importlib.metadata.version(n) for n in ("robocasa", "robosuite", "mujoco", "numpy", "gymnasium")}}
    for seed in args.seeds:
        start = time.perf_counter()
        env = RoboCasaEnv(seed=seed)
        try:
            obs, info = env.reset(seed=seed)
            assert env.observation_space.contains(obs), repr(obs["openpi"]["prompt"])
            assert not info["is_success"]
            images = {}
            for key in CAMERA_KEYS:
                pixels = obs["openpi"][key]
                assert pixels.std() > 1, f"Camera {key} is blank"
                name = key.rsplit("/", 1)[1]
                path = args.output / f"seed_{seed}_{name}.png"
                Image.fromarray(pixels).save(path)
                images[key] = {"path": str(path.resolve()), "shape": list(pixels.shape),
                               "std": float(pixels.std()), "sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}
            action = np.zeros(12, dtype=np.float32)
            action[6] = -1.0  # open gripper
            action[11] = -1.0  # arm mode
            state = obs["state"].copy()
            np.savez(args.output / f"seed_{seed}_observation.npz", **obs["openpi"])
            for step in range(args.steps):
                obs, reward, terminated, truncated, step_info = env.step(action)
                assert env.observation_space.contains(obs)
                assert float(reward) == float(step_info["is_success"])
                if terminated or truncated:
                    break
            record = {"seed": seed, "reset_info": info, "images": images,
                      "initial_state": state.tolist(), "final_state": obs["state"].tolist(),
                      "action_dim": 12, "state_dim": 16, "steps": step + 1,
                      "success": step_info["is_success"], "reward": reward,
                      "terminated": terminated, "truncated": truncated,
                      "elapsed_seconds": time.perf_counter()-start}
            previous = next((r for r in report["checks"] if r["seed"] == seed), None)
            if previous:
                assert np.allclose(state, previous["initial_state"], atol=1e-7, rtol=0)
                assert info["task_prompt"] == previous["reset_info"]["task_prompt"]
                for key in images:
                    assert images[key]["sha256"] == previous["images"][key]["sha256"]
                record["repeated_seed_identical"] = True
            # Verify the explicit published horizon is a truncation, not a fake
            # success, without spending a full 750-step rollout on a smoke test.
            if not terminated and not truncated:
                env.steps = env.max_steps - 1
                _, cutoff_reward, cutoff_terminated, cutoff_truncated, cutoff_info = env.step(action)
                assert cutoff_terminated == bool(cutoff_info["is_success"])
                assert cutoff_truncated == (not cutoff_terminated)
                record["horizon_truncation_checked"] = True
            report["checks"].append(record)
            print(json.dumps({k: record[k] for k in ("seed", "steps", "success", "elapsed_seconds")}), flush=True)
        finally:
            env.close()
    report["passed"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
