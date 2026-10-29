"""Portable resource checks and episode-boundary restart state for DICE."""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch


def expert_fraction(data, step):
    if data.get("mode") == "online_only_ablation":
        return 0.0
    start = data.get("expert_fraction", 0.5)
    end = data.get("expert_fraction_end", start)
    duration = data.get("expert_fraction_decay_decisions", 1)
    return float(start + (end - start) * min(step / duration, 1.0))


def check_resources():
    # Query assigned devices individually. Unassigned busy GPUs do not block this job.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible is not None else [None]
    if visible == "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty: no GPU assigned")
    for device in devices:
        command = ["nvidia-smi"]
        if device is not None:
            command += ["--id", device]
        command += ["--query-gpu=memory.free,temperature.gpu", "--format=csv,noheader,nounits"]
        for line in subprocess.check_output(command, text=True).strip().splitlines():
            free, temperature = map(int, line.split(","))
            if free < int(os.environ.get("DICE_MIN_GPU_FREE_MB", "2048")):
                raise RuntimeError("Assigned GPU has too little free memory")
            if temperature >= int(os.environ.get("DICE_MAX_GPU_TEMP", "85")):
                raise RuntimeError("Assigned GPU temperature limit reached")
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < int(os.environ.get("DICE_MIN_RAM_MB", "8192")) * 1024:
        raise RuntimeError("Too little available host RAM")
    if shutil.disk_usage(Path.cwd()).free < int(os.environ.get("DICE_MIN_DISK_GB", "10")) * 10**9:
        raise RuntimeError("Too little free space on repository filesystem")


def save_training(path, learner, config, prior, counters, replay, rng):
    """A trusted local restart file; includes image replay and may be large."""
    prior_rng = None
    if hasattr(prior, "key"):
        import jax

        prior_rng = np.asarray(jax.random.key_data(prior.key))
    payload = {
        "format": "dice-training-v1",
        "learner": learner.state_dict(),
        "config": config,
        "prior": prior.metadata,
        "counters": counters,
        "replay": list(replay),
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "prior_rng": prior_rng,
    }
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_training(path, learner, config, prior, replay, rng):
    # This format contains NumPy replay. Only load files produced by this project
    # on a trusted host; unlike inference checkpoints it requires pickle objects.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("format") != "dice-training-v1" or value["prior"] != prior.metadata:
        raise ValueError("Resume format/prior mismatch")
    previous, current = value["config"], config
    for section in ("policy", "env", "data", "dice"):
        if previous[section] != current[section]:
            raise ValueError(f"Cannot resume with changed [{section}]")
    for key in previous["train"]:
        if (
            key not in ("episodes", "save_every")
            and previous["train"][key] != current["train"][key]
        ):
            raise ValueError(f"Cannot resume with changed train.{key}")
    completed = value["counters"]["episodes"]
    if completed >= config["train"]["episodes"]:
        raise ValueError("Requested episode budget has already been completed")
    learner.load_state_dict(value["learner"])
    replay.extend(value["replay"])
    rng.bit_generator.state = value["numpy_rng"]
    torch.set_rng_state(value["torch_rng"])
    if value["prior_rng"] is not None:
        import jax

        prior.key = jax.random.wrap_key_data(value["prior_rng"])
    return completed, value["counters"]["decisions"]
