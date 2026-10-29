"""Regression tests for embodiment mappings that could invalidate comparisons."""
import copy
import importlib
import json

import numpy as np
import pytest

from multibench.policy_spec import BENCHMARKS, model_config


def raw_observation(benchmark):
    state = np.linspace(-.2, .2, BENCHMARKS[benchmark]["state_dim"], dtype=np.float32)
    if benchmark == "robotwin":
        state[[6, 13]] = .5
    return {"observation/state": state,
            "observation/image": np.full((224, 224, 3), 20, dtype=np.uint8),
            "observation/wrist_image": np.full((224, 224, 3), 80, dtype=np.uint8),
            "observation/right_wrist_image": np.full((224, 224, 3), 140, dtype=np.uint8),
            "prompt": "Complete the task."}


def test_original_base_model_geometry_is_preserved():
    cfg = model_config()
    assert cfg.pi05 is True
    assert cfg.discrete_state_input is True
    assert cfg.action_horizon == 50
    assert cfg.action_dim == 32
    assert cfg.max_token_len == 200


@pytest.mark.parametrize("benchmark", ["robotwin", "robocasa", "maniskill"])
def test_camera_identity_and_visibility_follow_the_embodiment(benchmark):
    from openpi import transforms

    module = importlib.import_module(f"multibench.{benchmark}.policy")
    output = transforms.compose(module.data_transforms().inputs)(copy.deepcopy(raw_observation(benchmark)))
    expected = [20, 0, 0] if benchmark == "maniskill" else [20, 80, 140]
    keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    assert [int(output["image"][key][0, 0, 0]) for key in keys] == expected
    assert [bool(output["image_mask"][key]) for key in keys] == ([True, False, False] if benchmark == "maniskill" else [True, True, True])


def test_robocasa_zero_normalized_action_means_official_controller_mean():
    from openpi import transforms
    from openpi.shared import normalize
    from multibench.robocasa import policy

    stats = normalize.load(policy.normalization_path().parent)
    assert stats["actions"].q01 is None and stats["actions"].q99 is None
    output = transforms.compose([
        transforms.Unnormalize(stats, use_quantiles=False),
        *policy.data_transforms().outputs,
    ])({"actions": np.zeros((50, 32), dtype=np.float32), "state": np.zeros(32, dtype=np.float32)})
    expected = np.broadcast_to(stats["actions"].mean[:12], (50, 12))
    np.testing.assert_allclose(output["actions"], expected, atol=1e-7, rtol=0)


def test_robotwin_zero_arm_delta_holds_native_joint_angles():
    from openpi import transforms
    from multibench.robotwin import policy

    raw = raw_observation("robotwin")
    data = policy.data_transforms()
    encoded = transforms.compose(data.inputs)(copy.deepcopy(raw))
    decoded = transforms.compose(data.outputs)({"state": encoded["state"].copy(),
                                               "actions": np.zeros((50, 32), dtype=np.float32)})
    arm_axes = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    np.testing.assert_allclose(decoded["actions"][:, arm_axes],
                               np.broadcast_to(raw["observation/state"][arm_axes], (50, 12)),
                               atol=1e-7, rtol=0)
