"""Matched-seed evaluation of zero-residual and learned controllers."""

import math
import time
from pathlib import Path

import numpy as np
import torch

from rl_vla.runtime import (
    build_env,
    load_checkpoint,
    make_learner,
    seed_everything,
    tensor,
    write_json,
)


def wilson_interval(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    p = successes / count
    denom = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def evaluate(config: dict, output: str, checkpoint: str | None = None,
             episodes: int = 20, seed: int = 100_000, deterministic: bool = False,
             max_decisions: int = 10_000) -> dict:
    if episodes < 1 or max_decisions < 1:
        raise ValueError("episodes and max_decisions must be positive")
    path = Path(output)
    if path.exists():
        raise FileExistsError(f"Evaluation output already exists: {path}")
    seed_everything(seed, int(config["train"].get("torch_threads", 1)))
    device = config["train"].get("device", "cpu")
    env = build_env(config)
    rows, learner = [], None
    started = time.perf_counter()
    try:
        for episode in range(episodes):
            # Identical local sampling seed sequence for repeated evaluations.
            seed_everything(seed + episode, int(config["train"].get("torch_threads", 1)))
            obs, _ = env.reset(seed=seed + episode)
            if checkpoint and learner is None:
                learner = make_learner(config, env)
                saved = load_checkpoint(checkpoint, device)
                for key in ("algorithm", "base", "chunk"):
                    if saved["config"][key] != config[key]:
                        raise ValueError(f"Evaluation {key} differs from trained checkpoint")
                learner.load_state_dict(saved["learner"])
                learner.actor.eval()
            total_return, steps, success, projections = 0.0, 0, False, 0.0
            for decision in range(1, max_decisions + 1):
                if learner is None:
                    action = np.zeros(env.action_dim, dtype=np.float32)
                else:
                    with torch.no_grad():
                        action = learner.actor.sample(tensor(obs["obs"], device).unsqueeze(0),
                                                      deterministic=deterministic)[0][0].cpu().numpy()
                obs, _, terminated, truncated, info = env.step(action)
                total_return += float(info["raw_return"])
                steps += int(info["executed_steps"])
                success |= bool(info.get("is_success", False))
                projections += float(info["projection_fraction"]) * int(info["executed_steps"])
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Evaluation hit max_decisions; implement task termination/time limit")
            rows.append({"seed": seed + episode, "success": success, "return": total_return,
                         "primitive_steps": steps, "decisions": decision,
                         "projection_fraction": projections / steps})
        successes = sum(row["success"] for row in rows)
        result = {"controller": "residual" if checkpoint else "frozen_base",
                  "residual_sampling": ("zero" if checkpoint is None else
                                        "mean" if deterministic else "stochastic"),
                  "checkpoint": checkpoint, "episodes": episodes, "successes": successes,
                  "success_rate": successes / episodes,
                  "success_95pct_wilson": wilson_interval(successes, episodes),
                  "mean_return": float(np.mean([row["return"] for row in rows])),
                  "elapsed_seconds": time.perf_counter() - started, "results": rows,
                  "config": config,
                  "base_rng_note": "Remote OpenPI RNG is server-owned; environment seeds do not reseed it."}
        write_json(path, result)
        return result
    finally:
        env.close()
