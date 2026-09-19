"""TOML configuration; paths are interpreted from the invoking working directory."""

import os
import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Load TOML, applying OpenPI endpoint environment overrides when configured.

    For ``base.kind = "openpi"``, ``RL_VLA_OPENPI_HOST`` replaces ``base.host``
    after stripping whitespace and must be nonempty. ``RL_VLA_OPENPI_PORT``
    replaces ``base.port`` and must be an integer from 1 through 65535.
    Unset variables preserve the TOML values; other base kinds ignore both.
    """
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
    base = config["base"]
    if base.get("kind") == "openpi":
        host = os.environ.get("RL_VLA_OPENPI_HOST")
        if host is not None:
            host = host.strip()
            if not host:
                raise ValueError("RL_VLA_OPENPI_HOST must be a nonempty hostname or IP address")
            base["host"] = host
        port = os.environ.get("RL_VLA_OPENPI_PORT")
        if port is not None:
            try:
                port_number = int(port)
            except ValueError:
                raise ValueError("RL_VLA_OPENPI_PORT must be an integer from 1 through 65535") from None
            if not 1 <= port_number <= 65535:
                raise ValueError("RL_VLA_OPENPI_PORT must be an integer from 1 through 65535")
            base["port"] = port_number
    return config
