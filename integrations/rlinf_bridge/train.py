"""Official RLinf entry point with local interaction/diagnostic logging."""

import json
import math
import os
import runpy
import time
from pathlib import Path

import numpy as np
import ray
import torch
from rlinf.runners.embodied_runner import EmbodiedRunner
from rl_vla.checkpoint_validation import run_checkpoint_validation

ROOT = Path(__file__).resolve().parents[2]

# This launcher owns one local Ray cluster inside one Slurm allocation. Explicit
# resources prevent Ray from reserving the whole shared node's CPUs and RAM.
original_ray_init = ray.init


def local_ray_init(*args, **kwargs):
    if kwargs.get("address") in (None, "auto", "local"):
        kwargs.update(address="local", num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
                      num_gpus=1, object_store_memory=2 * 1024**3,
                      include_dashboard=False)
    return original_ray_init(*args, **kwargs)


ray.init = local_ray_init


def serializable(value, nonfinite=None, path=""):
    if nonfinite is None:
        nonfinite = []
    if isinstance(value, dict):
        return {str(k): serializable(v, nonfinite, f"{path}.{k}".lstrip(".")) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v, nonfinite, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().tolist(), nonfinite, path)
    if isinstance(value, np.ndarray):
        return serializable(value.tolist(), nonfinite, path)
    if isinstance(value, np.generic):
        return serializable(value.item(), nonfinite, path)
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite.append(path)
        return None
    return value


original_log = EmbodiedRunner._log_step_metrics
original_save = EmbodiedRunner._save_checkpoint
original_run = EmbodiedRunner.run


def interaction_record(runner):
    path = Path(runner.cfg.env.train.recording_directory) / "progress.json"
    return json.loads(path.read_text()) if path.exists() else {}


def log_step(self, *args, **kwargs):
    result = original_log(self, *args, **kwargs)
    output = Path(self.cfg.runner.logger.log_path)
    nonfinite = []
    record = serializable({"iteration": self.global_step, **interaction_record(self),
                           "actor_rollout_metrics": kwargs.get("actor_rollout_metrics", {}),
                           "actor_training_metrics": kwargs.get("actor_training_metrics", {})}, nonfinite)
    record["nonfinite_diagnostics"] = nonfinite
    record["elapsed_seconds"] = time.monotonic() - self._controlled_started - self._checkpoint_validation_seconds
    record["checkpoint_validation_seconds"] = self._checkpoint_validation_seconds
    with (output / "iteration_metrics.jsonl").open("a") as log:
        log.write(json.dumps(record, allow_nan=False) + "\n")
    invalid_training = [path for path in nonfinite if "loss" in path.lower() or "grad" in path.lower()]
    if invalid_training:
        raise FloatingPointError(f"Nonfinite PPO loss/gradient diagnostics: {invalid_training}")
    return result


def save_checkpoint(self):
    result = original_save(self)
    path = Path(self.cfg.runner.logger.log_path) / "checkpoint_interactions.jsonl"
    with path.open("a") as log:
        log.write(json.dumps({"iteration": self.global_step, **interaction_record(self)}) + "\n")
    training = Path(self.cfg.runner.logger.log_path)
    checkpoint = training / self.cfg.runner.logger.experiment_name / "checkpoints" / f"global_step_{self.global_step}"
    self._checkpoint_validation_seconds += run_checkpoint_validation(
        training, checkpoint, self.global_step, interaction_record(self) | {
            "elapsed_seconds": time.monotonic() - self._controlled_started - self._checkpoint_validation_seconds}, "flow_sde")
    return result


def run(self):
    started = time.monotonic()
    self._controlled_started = started
    self._checkpoint_validation_seconds = 0.0
    result = original_run(self)
    elapsed = time.monotonic() - started
    if self.cfg.env.train.env_type == "pro_controlled":
        self.env.finalize_recordings().wait()
    path = Path(self.cfg.runner.logger.log_path) / "summary.json"
    path.write_text(json.dumps({"iterations": self.global_step, **interaction_record(self),
                               "elapsed_seconds": elapsed - self._checkpoint_validation_seconds,
                               "wall_elapsed_seconds": elapsed,
                               "checkpoint_validation_seconds": self._checkpoint_validation_seconds,
                               "completed": True}, indent=2) + "\n")
    return result


EmbodiedRunner._log_step_metrics = log_step
EmbodiedRunner._save_checkpoint = save_checkpoint
EmbodiedRunner.run = run
if __name__ == "__main__":
    os.environ.setdefault("EMBODIED_PATH", str(ROOT / "third_party/rlinf/examples/embodiment"))
    try:
        runpy.run_path(str(ROOT / "third_party/rlinf/examples/embodiment/train_embodied_agent.py"),
                      run_name="__main__")
    finally:
        ray.shutdown(wait_for_processes=True)
