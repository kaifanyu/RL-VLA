"""Shared observation/action transforms for the unchanged pi05_base weights."""

import hashlib
import importlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = {"robotwin": {"action_dim": 14, "state_dim": 14, "images": 3, "quantiles": True},
              "robocasa": {"action_dim": 12, "state_dim": 16, "images": 3, "quantiles": False},
              "maniskill": {"action_dim": 7, "state_dim": 8, "images": 1, "quantiles": True}}


def model_config():
    from openpi.models.pi0_config import Pi0Config

    # Preserve the published base model's horizon and discrete proprioception.
    return Pi0Config(pi05=True, action_horizon=50, action_dim=32, discrete_state_input=True)


def build_transforms(benchmark, model=None):
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training.config import ModelTransformFactory

    spec = BENCHMARKS[benchmark]
    module = importlib.import_module(f"multibench.{benchmark}.policy")
    path = Path(module.normalization_path()).resolve()
    stats = normalize.load(path.parent)
    for name, required in (("state", spec["state_dim"]), ("actions", spec["action_dim"])):
        item = stats[name]
        if len(item.mean) < required or not np.isfinite(item.mean).all() or not np.isfinite(item.std).all():
            raise ValueError(f"Invalid {benchmark} {name} normalization")
        if spec["quantiles"] and (item.q01 is None or item.q99 is None):
            raise ValueError("Missing required quantile statistics")
    data = module.data_transforms()
    model_transforms = ModelTransformFactory()(model or model_config())
    inputs = [*data.inputs, transforms.Normalize(stats, use_quantiles=spec["quantiles"]), *model_transforms.inputs]
    outputs = [*model_transforms.outputs, transforms.Unnormalize(stats, use_quantiles=spec["quantiles"]), *data.outputs]
    return transforms.compose(inputs), transforms.compose(outputs), {
        "benchmark": benchmark, "normalization_path": str(path),
        "normalization_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "quantile_normalization": spec["quantiles"], "motor_dim": spec["action_dim"],
        "state_dim": spec["state_dim"], "image_count": spec["images"],
        "weight_origin": "Unchanged original pi05_base; benchmark calibration statistics only; no task SFT weights",
    }
