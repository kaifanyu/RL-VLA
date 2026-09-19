"""TOML configuration; paths are interpreted from the invoking working directory."""

import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    if config.get("algorithm") not in {"sac", "ppo"}:
        raise ValueError("algorithm must be 'sac' or 'ppo'")
    for section in ("env", "base", "chunk", "train"):
        if section not in config:
            raise ValueError(f"Missing [{section}] in {path}")
    train = config["train"]
    for name in ("total_decisions", "batch_size", "log_every", "save_every"):
        if int(train.get(name, 1)) < 1:
            raise ValueError(f"train.{name} must be positive")
    if not 0 < float(config["chunk"].get("gamma", 0.99)) <= 1:
        raise ValueError("chunk.gamma must be in (0, 1]")
    return config

