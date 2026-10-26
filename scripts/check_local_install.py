"""Verify local pi05_libero inference and real LIBERO rollouts without training.

Run with third_party/openpi/.venv/bin/python on an allocated GPU, after sourcing
scripts/env.sh and setting PYTHONPATH=src. The simulator uses the root .venv.
"""

import argparse
import json
import socket
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config
from PIL import Image

from rl_vla.rldt_env import ProcessEnv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Run this check on an allocated GPU")
    start = time.monotonic()
    policy = create_trained_policy(get_config("pi05_libero"), ROOT / "checkpoints/openpi/pi05_libero")
    report = {
        "host": socket.gethostname(), "devices": [str(device) for device in jax.devices()],
        "packages": {name: version(name) for name in ("jax", "jaxlib", "flax", "torch", "transformers")},
        "openpi_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party/openpi"), "rev-parse", "HEAD"], text=True).strip(),
        "load_seconds": time.monotonic() - start, "optimizer_updates": 0,
        "episodes": [], "purpose": "installation smoke test; not a benchmark estimate",
    }
    rng = np.random.default_rng(7)
    for suite in args.suites:
        horizon = {"libero_spatial": 220, "libero_object": 280}[suite]
        env = ProcessEnv({
            "python": str(ROOT / ".venv/bin/python"), "kind": "external",
            "factory": "rl_vla.envs.libero:make_env",
            "kwargs": {"task_suite": suite, "task_id": 0, "initial_state_id": 0, "max_steps": horizon},
        })
        try:
            raw, reset = env.reset(seed=7)
            for key, label in [("observation/image", "front"), ("observation/wrist_image", "wrist")]:
                frame = raw["openpi"][key]
                assert frame.shape == (224, 224, 3) and frame.dtype == np.uint8 and frame.std() > 0
                Image.fromarray(frame).save(args.output / f"{suite}_{label}.png")
            steps, calls, total_reward = 0, 0, 0.0
            latencies = []
            while steps < horizon:
                before = time.monotonic()
                noise = rng.standard_normal((10, 32)).astype(np.float32)
                actions = np.asarray(policy.infer(raw["openpi"], noise=noise)["actions"])
                latencies.append(time.monotonic() - before)
                assert actions.shape == (10, 7) and np.isfinite(actions).all()
                if calls == 0:
                    np.save(args.output / f"{suite}_first_actions.npy", actions, allow_pickle=False)
                calls += 1
                for action in actions[:5]:
                    raw, reward, terminated, truncated, info = env.step(np.clip(action, env.low, env.high))
                    steps += 1
                    total_reward += reward
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
            row = {"suite": suite, "task_id": 0, "reset": reset, "primitive_steps": steps,
                   "inference_calls": calls, "success": bool(info["is_success"]),
                   "reward": total_reward, "action_shape": list(actions.shape),
                   "inference_seconds": latencies}
            assert terminated or truncated
            assert total_reward == float(row["success"])
            report["episodes"].append(row)
            print(json.dumps(row), flush=True)
        finally:
            env.close()
    report["passed"] = True
    report["elapsed_seconds"] = time.monotonic() - start
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Installation smoke test passed", flush=True)


if __name__ == "__main__":
    main()
