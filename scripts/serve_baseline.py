"""Serve the pinned OpenPI checkpoint with provenance and explicit evaluation noise."""
import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
from openpi.policies.policy_config import create_trained_policy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training.config import get_config


def checkpoint_fingerprint(root):
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        records.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                        "sha256": digest.hexdigest()})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return digest, records


class EvaluationPolicy:
    def __init__(self, policy, model_config):
        self.policy = policy
        self.shape = (model_config.action_horizon, model_config.action_dim)

    def infer(self, observation):
        observation = dict(observation)
        seed = observation.pop("evaluation_noise_seed", None)
        if seed is None:
            return self.policy.infer(observation)
        noise = np.random.default_rng(int(seed)).standard_normal(self.shape).astype(np.float32)
        result = self.policy.infer(observation, noise=noise)
        if not np.isfinite(result["actions"]).all():
            raise ValueError("Nonfinite model actions")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-config", default="pi05_libero")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "params").is_dir():
        raise ValueError("Expected a local OpenPI JAX checkpoint with params/")
    if not jax.devices("gpu"):
        raise RuntimeError("No CUDA device is available")
    print("Hashing local checkpoint for reproducible evaluation", flush=True)
    fingerprint, records = checkpoint_fingerprint(checkpoint)
    config = get_config(args.model_config)
    source = Path(__file__).resolve().parents[1] / "third_party/openpi"
    metadata = {
        "model_config": args.model_config, "checkpoint": str(checkpoint),
        "checkpoint_sha256": fingerprint, "checkpoint_files": records,
        "openpi_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "noise_protocol": "numpy-pcg64-float32-standard-normal-v1",
        "action_horizon": config.model.action_horizon,
        "internal_action_dim": config.model.action_dim,
        "server_pid": os.getpid(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "jax_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "jax_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "server_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "packages": {name: version(name) for name in
                     ("jax", "jaxlib", "flax", "orbax-checkpoint", "numpy", "transformers")},
        "gpu": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader"], text=True).strip(),
    }
    started = time.monotonic()
    policy = create_trained_policy(config, checkpoint)
    metadata["load_seconds"] = time.monotonic() - started
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Loaded {args.model_config}; serving 127.0.0.1:{args.port}", flush=True)
    WebsocketPolicyServer(EvaluationPolicy(policy, config.model), host="127.0.0.1",
                          port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
