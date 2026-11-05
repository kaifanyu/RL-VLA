#!/usr/bin/env python
"""Evaluate a DICE-RL checkpoint or its frozen prior under the shared K=1 contract.

One proposal per decision, so no critic candidate selection. Reset seed and JAX noise key are
seed + i for the i-th state of the ordered split, the stream of rl_vla.rldt.evaluate.
DICE source is imported from --dice-source and is not modified.
"""
import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/workspace/smrithi/dice-pilot-output/train.toml")
    parser.add_argument("--dice-source", default="/workspace/smrithi/dice-pilot/src")
    parser.add_argument("--checkpoint", help="DICE checkpoint; omit to evaluate the frozen prior")
    parser.add_argument("--split", required=True, choices=["validation", "test", "source"],
                        help="source is the original LIBERO retention split, as RLDT's --env-split source")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, help="First N states (diagnostic; never selection eligible)")
    parser.add_argument("--seed", type=int, default=100000)
    args = parser.parse_args()
    for key, value in {"LIBERO_CONFIG_PATH": str(ROOT / ".cache/libero-pro"), "MUJOCO_GL": "osmesa",
                       "PYOPENGL_PLATFORM": "osmesa", "NUMBA_DISABLE_JIT": "1", "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LP_NUM_THREADS": "1",
                       "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.35",
                       "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}.items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, args.dice_source)
    os.environ["PYTHONPATH"] = args.dice_source + os.pathsep + os.environ.get("PYTHONPATH", "")

    import jax
    import numpy as np
    import torch

    from rl_vla.dice import tensor
    from rl_vla.dice_core import DICE
    from rl_vla.dice_pi05 import FrozenPi05
    from rl_vla.rldt_env import make_primitive_env

    config = tomllib.loads(Path(args.config).read_text())
    config["env"]["kwargs"]["split"] = args.split
    protocol = json.loads(Path(config["env"]["kwargs"]["protocol"]).read_text())
    task = protocol["tasks"][config["env"]["kwargs"]["task_key"]]
    state_ids = task[f"{args.split}_state_ids"]
    count = args.episodes or len(state_ids)
    if not 1 <= count <= len(state_ids):
        raise SystemExit(f"--episodes must be in [1, {len(state_ids)}]")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    prior = FrozenPi05(config["policy"], {**config["dice"], "seed": args.seed})
    learner = None
    if args.checkpoint:
        value = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if value.get("format") != "dice-pi05-v1" or value["prior"] != prior.metadata:
            raise SystemExit("Checkpoint format or prior fingerprint mismatch")
        for section in ("policy", "dice"):
            if config[section] != value["config"][section]:
                raise SystemExit(f"Config [{section}] differs from the checkpoint")
        learner = DICE(prior.feature_dim, prior.noise_dim, prior.action_dim, config["dice"])
        learner.load_state_dict(value["learner"])
        learner.actor.eval()
        learner.critics.eval()
    env = make_primitive_env(config["env"])
    rows = []
    try:
        for episode in range(count):
            seed = args.seed + episode
            raw, info = env.reset(seed=seed)
            if info.get("initial_state_id") != state_ids[episode]:
                raise RuntimeError(f"Evaluated state {info.get('initial_state_id')}, expected {state_ids[episode]}")
            prior.key = jax.random.PRNGKey(seed)
            terminated = truncated = False
            steps, total, decisions, started = 0, 0.0, 0, time.monotonic()
            while not (terminated or truncated):
                features, noise, proposal, full = prior.propose(raw, 1)
                if learner:
                    action, index = learner.select(tensor(features), tensor(noise), tensor(proposal))
                    action = action.numpy()
                else:
                    action, index = proposal[0], 0
                for primitive in prior.decode(action, full[index], raw):
                    raw, reward, terminated, truncated, final_info = env.step(np.clip(primitive, env.low, env.high))
                    steps += 1
                    total += float(reward)
                    if terminated or truncated:
                        break
                decisions += 1
            row = {"episode": episode, "seed": seed, "initial_state_id": info["initial_state_id"],
                   "backend_seed": info.get("backend_seed"), "split": args.split,
                   "success": bool(final_info.get("is_success", terminated)), "truncated": bool(truncated),
                   "return": total, "primitive_steps": steps, "decisions": decisions,
                   "seconds": time.monotonic() - started}
            rows.append(row)
            with (args.output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    summary = {"method": "dice" if learner else "prior", "checkpoint": args.checkpoint, "split": args.split,
               "episodes": len(rows), "successes": sum(r["success"] for r in rows), "inference_k": 1,
               "first_seed": args.seed, "noise_protocol": "jax-prngkey-split-normal-v1",
               "selection_eligible": args.split == "validation" and count == len(state_ids)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
