"""Evaluate a frozen prior or DICE checkpoint without optimizer updates."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def evaluation_contract(config):
    kwargs = config["env"].get("kwargs", {})
    if config["env"].get("factory") != "rl_vla.envs.libero_pro:make_env":
        return kwargs, None, None, None
    from rl_vla.dice import validate_replay_provenance
    from rl_vla.envs.libero_pro import validate_splits

    if not kwargs.get("protocol") or not kwargs.get("task_key"):
        raise ValueError("LIBERO-PRO evaluation requires explicit protocol and task_key")
    protocol_bytes = Path(kwargs["protocol"]).read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    plan = json.loads(protocol_bytes)
    task = plan["tasks"].get(kwargs["task_key"])
    if task is None:
        raise ValueError(f"Task key is absent from protocol: {kwargs['task_key']}")
    if plan.get("version") != 2:
        raise ValueError("DICE LIBERO-PRO evaluation requires protocol version 2")
    validate_splits(task, 2)
    provenance = config.get("data", {}).get("provenance")
    validate_replay_provenance(provenance, provenance)
    if provenance["task"]["task_key"] != kwargs["task_key"]:
        raise ValueError("Evaluation task key differs from configured task provenance")
    if (provenance["task"]["task_suite"], provenance["task"]["task_id"]) != (
        task["suite"], task["task_id"]
    ):
        raise ValueError("Evaluation task identity differs from protocol")
    if provenance["environment"]["protocol_sha256"] != protocol_sha256:
        raise ValueError("Evaluation protocol fingerprint differs from configured provenance")
    if provenance["environment"]["protocol_version"] != plan["version"]:
        raise ValueError("Evaluation protocol version differs from configured provenance")
    if provenance["environment"]["split"] != "train":
        raise ValueError("Demonstration provenance must identify the train split")
    split = kwargs.get("split")
    if split not in ("validation", "test"):
        raise ValueError("DICE policy evaluation on LIBERO-PRO requires validation or test split")
    return kwargs, task[f"{split}_state_ids"], protocol_sha256, provenance


def evaluate(config, output, *, checkpoint=None, episodes=50, seed=100000):
    from rl_vla.dice import check, resource_check, tensor
    from rl_vla.dice_core import DICE
    from rl_vla.dice_pi05 import FrozenPi05
    from rl_vla.rldt_env import make_primitive_env

    if episodes < 1 or seed < 0:
        raise ValueError("Evaluation requires positive episodes and nonnegative seed")
    kwargs, expected_state_ids, protocol_sha256, provenance = evaluation_contract(config)
    if expected_state_ids is not None and episodes != len(expected_state_ids):
        raise ValueError(
            f"Requested {episodes} episodes, but {kwargs['split']} protocol has "
            f"{len(expected_state_ids)} states"
        )
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
            if provenance is not None:
                if raw["openpi"].get("prompt") != provenance["task"]["instruction"]:
                    raise ValueError("Evaluation environment instruction differs from task provenance")
                if (info.get("task_suite"), info.get("task_id")) != (
                    provenance["task"]["task_suite"], provenance["task"]["task_id"]
                ):
                    raise ValueError("Evaluation environment task identity differs from provenance")
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
                "task_key": kwargs.get("task_key"),
                "task_suite": info.get("task_suite"),
                "task_id": info.get("task_id"),
                "split": kwargs.get("split"),
                "success": bool(final_info.get("is_success", terminated)),
                "truncated": bool(truncated),
                "return": total,
                "primitive_steps": steps,
                "controller_steps": steps,
                "decisions": decisions,
                "clipped_commands": clipped,
                "seconds": time.monotonic() - started,
            }
            if expected_state_ids is not None and row["initial_state_id"] != expected_state_ids[episode]:
                raise ValueError(
                    f"DICE evaluated state {row['initial_state_id']!r} at episode {episode}; "
                    f"protocol requires {expected_state_ids[episode]}"
                )
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
            "state_ids": [row["initial_state_id"] for row in results],
            "protocol_sha256": protocol_sha256,
            "selection_eligible": kwargs.get("split") == "validation",
            "test_results_used_for_selection": False,
            "optimizer_updates": 0,
            "config": config,
            "prior": prior.metadata,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        env.close()
