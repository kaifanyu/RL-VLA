"""Official ManiSkill data transforms, independent of simulator imports.

These transforms do not construct a model or select weights/architecture.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def normalization_path() -> Path:
    path = ROOT / "datasets/benchmark_assets/maniskill3/normalization/physical-intelligence/maniskill/norm_stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"Download the pinned ManiSkill calibration assets first: {path}")
    return path


def data_transforms():
    from openpi import transforms
    from openpi.models import model

    module_name = "multibench_official_maniskill_policy"
    if module_name not in sys.modules:
        source = ROOT / "third_party/rlinf/rlinf/models/embodiment/openpi/policies/maniskill_policy.py"
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    module = sys.modules[module_name]
    return transforms.Group(
        inputs=[module.ManiSkillInputs(model_type=model.ModelType.PI05)],
        outputs=[module.ManiSkillOutputs()],
    )
