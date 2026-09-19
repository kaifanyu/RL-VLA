"""Start the pinned upstream policy server with explicit deployment settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse


def validate_checkpoint(checkpoint: str) -> bool:
    """Validate a mounted checkpoint and return whether it uses PyTorch weights."""
    root = Path(checkpoint)
    if urlparse(checkpoint).scheme and not root.is_absolute():
        raise ValueError(
            "OPENPI_CHECKPOINT must be a mounted local directory. Download the complete "
            "checkpoint, including assets and params, before starting this service."
        )
    if not root.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {root}")
    pytorch_weights = root / "model.safetensors"
    is_pytorch = pytorch_weights.is_file() and pytorch_weights.stat().st_size > 0
    params = root / "params"
    if not is_pytorch and (not params.is_dir() or not any(params.iterdir())):
        raise ValueError(f"Checkpoint {root} needs a nonempty params/ or model.safetensors")
    assets = root / "assets"
    if not assets.is_dir() or not any(assets.rglob("norm_stats.json")):
        raise ValueError(f"Checkpoint {root} is missing assets/**/norm_stats.json")
    return is_pytorch


def require_gpu(is_pytorch: bool) -> None:
    """Fail early instead of silently falling back to a prohibitively slow CPU."""
    if is_pytorch:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access a CUDA GPU")
        print(f"OpenPI GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        import jax

        devices = jax.devices("gpu")
        if not devices:
            raise RuntimeError("JAX cannot access a CUDA GPU")
        print(f"OpenPI GPU: {devices}", flush=True)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] != "serve":
        # Allow diagnostic commands without loading the model or allocating GPU memory.
        os.execvp(args[0], args)
    try:
        port = int(os.environ.get("OPENPI_PORT", "8000"))
        if not 1 <= port <= 65535:
            raise ValueError("OPENPI_PORT must be between 1 and 65535")
        checkpoint = os.environ.get("OPENPI_CHECKPOINT", "/checkpoints/pi05_libero")
        config = os.environ.get("OPENPI_CONFIG", "pi05_libero")
        is_pytorch = validate_checkpoint(checkpoint)
        require_gpu(is_pytorch)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"OpenPI startup failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    command = [
        sys.executable,
        "/opt/openpi/scripts/serve_policy.py",
        f"--port={port}",
        *args[1:],
        "policy:checkpoint",
        f"--policy.config={config}",
        f"--policy.dir={checkpoint}",
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
