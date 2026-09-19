"""Shared environment construction, checkpoints and reproducibility metadata."""

import json
import platform
import random
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from rl_vla.base_policy import make_base_policy
from rl_vla.env_adapter import ResidualChunkEnv
from rl_vla.envs import make_env


def seed_everything(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def build_env(config: dict) -> ResidualChunkEnv:
    env_config = dict(config["env"])
    env_config["kwargs"] = dict(env_config.get("kwargs", {}))
    if env_config["kind"] == "toy":
        env_config["kwargs"].setdefault("gamma", config["chunk"].get("gamma", 0.99))
        if env_config["kwargs"]["gamma"] != config["chunk"].get("gamma", 0.99):
            raise ValueError("Toy shaping gamma and learner gamma must match")
    env = make_env(env_config)
    try:
        base = make_base_policy(config["base"], int(env.action_space.shape[0]),
                                int(config["chunk"]["horizon"]))
        return ResidualChunkEnv(env, base, config["chunk"])
    except Exception:
        env.close()
        raise


def make_learner(config: dict, env: ResidualChunkEnv):
    # Imports here keep configuration/environment tools independent of learner choice.
    from rl_vla.ppo import PPO
    from rl_vla.sac import SAC

    cls = SAC if config["algorithm"] == "sac" else PPO
    settings = dict(config.get(config["algorithm"], {}))
    settings.setdefault("batch_size", config["train"].get("batch_size", 256))
    return cls(env.obs_dim, env.critic_dim, env.action_dim, settings,
               device=config["train"].get("device", "cpu"))


def tensor(value, device: str) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metadata(config: dict, env: ResidualChunkEnv) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", "third_party/openpi", "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    base = getattr(env, "base_policy", None)
    return {
        "config": config, "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "torch", "gymnasium", "rl-vla")},
        "openpi_source_commit": commit,
        "base_metadata": getattr(base, "metadata", {}),
        "dimensions": {"obs": env.obs_dim, "critic": env.critic_dim, "action": env.action_dim},
        "device": config["train"].get("device", "cpu"),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def save_checkpoint(path: Path, learner, config: dict, counters: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"format_version": 1, "config": config, "counters": counters,
                "learner": learner.state_dict()}, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported RL-VLA checkpoint format")
    return payload

