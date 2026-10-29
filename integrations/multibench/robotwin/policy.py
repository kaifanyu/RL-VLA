"""Exact pinned RLinf Aloha transforms, shared by unchanged-base and RL runs."""
import dataclasses
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
from openpi import transforms

ROOT = Path(__file__).resolve().parents[3]
POLICY_SOURCE = ROOT / "third_party/rlinf/rlinf/models/embodiment/openpi/policies/aloha_policy.py"
NORM_STATS = Path(__file__).with_name("norm_stats.json")
DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)


def upstream_policy():
    name = "multibench_pinned_rlinf_aloha_policy"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, POLICY_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@dataclasses.dataclass(frozen=True)
class CameraInputs(transforms.DataTransformFn):
    def __call__(self, data):
        data = dict(data.get("openpi", data))
        if "observation/state" in data:
            result = {
                "state": np.asarray(data["observation/state"]).copy(),
                "images": {
                    "cam_high": data["observation/image"],
                    "cam_left_wrist": data["observation/wrist_image"],
                    "cam_right_wrist": data["observation/right_wrist_image"],
                },
                "prompt": data["prompt"],
            }
            if "actions" in data:
                result["actions"] = data["actions"]
            return result
        return data


def data_transforms():
    aloha = upstream_policy()
    return transforms.Group(
        inputs=[CameraInputs(), aloha.AlohaInputs(adapt_to_pi=True),
                transforms.DeltaActions(DELTA_MASK)],
        outputs=[transforms.AbsoluteActions(DELTA_MASK),
                 aloha.AlohaOutputs(adapt_to_pi=True)],
    )


def normalization_path():
    if not NORM_STATS.is_file():
        raise FileNotFoundError("Compute the task's normalization statistics first")
    return NORM_STATS


def source_sha256():
    return hashlib.sha256(POLICY_SOURCE.read_bytes()).hexdigest()
