"""Single-environment collection with explicit macro-step accounting."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from rl_vla.buffers import ReplayBuffer
from rl_vla.runtime import (
    build_env,
    load_checkpoint,
    make_learner,
    metadata,
    save_checkpoint,
    seed_everything,
    tensor,
    write_json,
)


def train(config: dict, output: str, init_from: str | None = None) -> dict:
    settings = config["train"]
    seed = int(settings.get("seed", 0))
    seed_everything(seed, int(settings.get("torch_threads", 1)))
    device = settings.get("device", "cpu")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)  # Never silently replace an experiment.
    env = build_env(config)
    started = time.perf_counter()
    counters = {"decisions": 0, "primitive_steps": 0, "episodes": 0, "updates": 0}
    try:
        current, _ = env.reset(seed=seed)
        learner = make_learner(config, env)
        if init_from:
            saved = load_checkpoint(init_from, device)
            for key in ("algorithm", "base", "chunk", "env"):
                if saved["config"][key] != config[key]:
                    raise ValueError(f"Warm-start {key} must match checkpoint")
            if learner.config != saved["learner"]["config"]:
                raise ValueError("Warm-start learner settings must match checkpoint; optimizer settings are restored")
            learner.load_state_dict(saved["learner"])
        write_json(out / "manifest.json", metadata(config, env) | {"init_from": init_from})
        replay = ReplayBuffer(int(settings.get("replay_capacity", 100_000)), seed=seed)
        rollout = []
        episode_return, episode_steps = 0.0, 0
        batch_size = int(settings.get("batch_size", 256))
        total = int(settings.get("total_decisions", 10_000))
        metrics = {}
        with (out / "metrics.jsonl").open("w", encoding="utf-8") as log:
            for decision in range(1, total + 1):
                with torch.no_grad():
                    action, logp, pre_tanh = learner.actor.sample(
                        tensor(current["obs"], device).unsqueeze(0))
                    value = (learner.value(tensor(current["critic_obs"], device).unsqueeze(0)).item()
                             if config["algorithm"] == "ppo" else 0.0)
                action_np = action.squeeze(0).cpu().numpy()
                following, reward, terminated, truncated, info = env.step(action_np)
                # Keep requested and executed variables distinct, including a
                # validity length for an early-terminated chunk's padded tail.
                executed = np.zeros((env.execute_steps, env.motor_dim), dtype=np.float32)
                primitive_rewards = np.zeros(env.execute_steps, dtype=np.float32)
                executed[:info["executed_steps"]] = info["executed_actions"]
                primitive_rewards[:info["executed_steps"]] = info["primitive_rewards"]
                transition = {
                    "obs": current["obs"], "critic_obs": current["critic_obs"],
                    "action": action_np, "reward": reward, "discount": info["discount"],
                    "next_obs": following["obs"], "next_critic_obs": following["critic_obs"],
                    "terminated": terminated, "truncated": truncated,
                    "pre_tanh": pre_tanh.squeeze(0).cpu().numpy(),
                    "executed_actions": executed, "primitive_rewards": primitive_rewards,
                    "executed_steps": info["executed_steps"], "episode_id": counters["episodes"],
                }
                counters["decisions"] = decision
                counters["primitive_steps"] += int(info["executed_steps"])
                episode_return += float(info["raw_return"])
                episode_steps += int(info["executed_steps"])
                if config["algorithm"] == "sac":
                    replay.add(**transition)
                    if decision >= int(settings.get("learning_starts", 256)) and len(replay) >= batch_size:
                        for _ in range(int(settings.get("updates_per_decision", 1))):
                            metrics = learner.update(replay.sample(batch_size, device=device))
                            counters["updates"] += 1
                else:
                    with torch.no_grad():
                        next_value = (0.0 if terminated else learner.value(
                            tensor(following["critic_obs"], device).unsqueeze(0)).item())
                    rollout.append(transition | {
                        "pre_tanh": pre_tanh.squeeze(0).cpu().numpy(),
                        "logp": logp.item(), "value": value, "next_value": next_value,
                    })
                    if len(rollout) >= int(settings.get("rollout_steps", 256)) or decision == total:
                        batch = {key: tensor(np.stack([row[key] for row in rollout]), device)
                                 for key in rollout[0]}
                        metrics = learner.update(batch)
                        counters["updates"] += 1
                        rollout.clear()
                event = {}
                if terminated or truncated:
                    counters["episodes"] += 1
                    event = {"episode_return": episode_return, "episode_steps": episode_steps,
                             "success": bool(info.get("is_success", False)),
                             "terminated": terminated, "truncated": truncated}
                    episode_return, episode_steps = 0.0, 0
                    if decision < total:
                        current, _ = env.reset(seed=seed + counters["episodes"])
                else:
                    current = following
                if event or decision % int(settings.get("log_every", 100)) == 0 or decision == total:
                    record = counters | {"elapsed_seconds": time.perf_counter() - started,
                        "projection_fraction": float(info["projection_fraction"]),
                        "residual_rms": float(np.sqrt(np.mean(action_np ** 2))),
                        "base_calls": env.base_calls,
                        "base_inference_seconds": env.base_inference_seconds,
                        "gradient_steps_per_decision": (counters["updates"] / decision
                                                        if config["algorithm"] == "sac" else None),
                        "metrics": metrics} | event
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                if decision % int(settings.get("save_every", 1000)) == 0:
                    save_checkpoint(out / f"step_{decision}.pt", learner, config, counters)
                if decision % int(settings.get("log_every", 100)) == 0 or decision == total:
                    print(f"{config['algorithm'].upper()} decisions={decision}/{total} "
                          f"motor_steps={counters['primitive_steps']} updates={counters['updates']}",
                          flush=True)
        save_checkpoint(out / "final.pt", learner, config, counters)
        summary = counters | {"elapsed_seconds": time.perf_counter() - started,
                              "checkpoint": str(out / "final.pt")}
        write_json(out / "summary.json", summary)
        return summary
    finally:
        env.close()
