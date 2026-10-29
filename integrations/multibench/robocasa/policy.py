"""Exact official RoboCasa Pi0.5 transforms for the shared policy interpreter.

The simulator and dataset loaders are deliberately not imported here. The caller
owns normalization (official mean/std), tokenizer, and pretrained model loading.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys

from openpi import transforms
from openpi.models import model

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "third_party/robocasa365/dependencies/openpi_reference/src/openpi/policies/robocasa_policy.py"


def normalization_path() -> Path:
    return Path(__file__).resolve().parent / "assets/norm_stats.json"


def _official_module():
    name = "_multibench_official_robocasa_policy"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REFERENCE)
        if spec is None or spec.loader is None:
            raise ImportError(f"Official pinned RoboCasa adapter missing: {REFERENCE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclasses.dataclass(frozen=True)
class CommonCameraNames(transforms.DataTransformFn):
    """Only rename the common third camera key to the official upstream name."""

    def __call__(self, data: dict) -> dict:
        copied = dict(data)
        copied["observation/right_image"] = copied.pop("observation/right_wrist_image")
        return copied


def data_transforms() -> transforms.Group:
    official = _official_module()
    return transforms.Group(
        inputs=[CommonCameraNames(), official.RobocasaInputs(action_dim=32, model_type=model.ModelType.PI05)],
        outputs=[official.RobocasaOutputs()],
    )
