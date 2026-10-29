"""Evaluate a frozen prior or DICE checkpoint without optimizer updates."""

import json
import time
from pathlib import Path

import numpy as np
import torch


def evaluate(config, output, *, checkpoint=None, episodes=50, seed=100000):
    from rl_vla.dice import check, resource_check, tensor
    from rl_vla.dice_core import DICE
    from rl_vla.dice_pi05 import FrozenPi05
    from rl_vla.rldt_env import make_primitive_env

    if episodes < 1 or seed < 0:
        raise ValueError("Evaluation requires positive episodes and nonnegative seed")
    value = None
    if checkpoint:
        value = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if value.get("format") != "dice-pi05-v1":
            raise ValueError("Expected a DICE inference checkpoint (latest.pt or final.pt)")
        # Preserve the trained action/feature convention. Allow new environment
        # conditions for explicitly labeled generalization tests.
        for section in ("policy", "dice"):
            if config[section] != value["config"][section]:
                raise ValueError(f"Evaluation [{section}] differs from checkpoint")
    setup = check(config)
    if setup is not None and not setup["setup_ok"]:
        raise RuntimeError("DICE setup is incomplete")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    resource_check()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    prior = FrozenPi05(config["policy"], {**config["dice"], "seed": seed})
    learner = None
    if value:
        if value["prior"] != prior.metadata:
            raise ValueError("Checkpoint normalization/prior fingerprint mismatch")
        learner = DICE(prior.feature_dim, prior.noise_dim, prior.action_dim, config["dice"])
        learner.load_state_dict(value["learner"])
        learner.actor.eval()
        learner.critics.eval()
    env = make_primitive_env(config["env"])
    results = []
    try:
        for episode in range(episodes):
            raw, info = env.reset(seed=seed + episode)
            terminated = truncated = False
            total, steps, decisions, clipped = 0.0, 0, 0, 0
            started = time.monotonic()
            while not (terminated or truncated):
                resource_check()
                count = config["dice"]["selection_samples"] if learner else 1
                f, z, p, full = prior.propose(raw, count)
                if learner:
                    # Always use the learned selection rule; no training warmup.
                    action, index = learner.select(
                        tensor(f),
                        tensor(z),
                        tensor(p),
                        strategy=config["dice"].get("selection_strategy", "max_q_min"),
                    )
                    action = action.numpy()
                else:
                    action, index = p[0], 0
                for primitive in prior.decode(action, full[index], raw):
                    command = np.clip(primitive, env.low, env.high)
                    clipped += int(np.any(command != primitive))
                    raw, reward, terminated, truncated, final_info = env.step(command)
                    steps += 1
                    total += float(reward)
                    if terminated or truncated:
                        break
                decisions += 1
            row = {
                "episode": episode,
                "seed": seed + episode,
                "initial_state_id": info.get("initial_state_id"),
                "success": bool(final_info.get("is_success", terminated)),
                "truncated": bool(truncated),
                "return": total,
                "primitive_steps": steps,
                "decisions": decisions,
                "clipped_commands": clipped,
                "seconds": time.monotonic() - started,
            }
            results.append(row)
            with (output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        successes = sum(r["success"] for r in results)
        # Wilson 95% interval, useful for sparse binary success rates.
        n, z = len(results), 1.959963984540054
        rate = successes / n
        center = (rate + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        summary = {
            "method": "dice" if learner else "prior",
            "checkpoint": checkpoint,
            "episodes": n,
            "successes": successes,
            "success_rate": rate,
            "wilson_95": [center - radius, center + radius],
            "first_seed": seed,
            "optimizer_updates": 0,
            "config": config,
            "prior": prior.metadata,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        env.close()
