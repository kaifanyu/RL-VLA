"""RoboCasa365's native PandaOmron task with the official OpenPI observation contract.

Run this module in .venv-robocasa365; policy inference/training lives in another
interpreter. No LIBERO action conversion or artificial success signal is used.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import string

TASK_ID = "robocasa/PickPlaceCounterToCabinet"
TASK_NAME = "PickPlaceCounterToCabinet"
STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.base_position",
    "state.base_rotation",
    "state.gripper_qpos",
)
CAMERA_KEYS = {
    "observation/image": "video.robot0_agentview_left",
    "observation/wrist_image": "video.robot0_eye_in_hand",
    "observation/right_wrist_image": "video.robot0_agentview_right",
}


class RoboCasaEnv(gym.Env):
    """Flat native 12D controller commands, 16D proprioception, three RGB views.

    The official Gym wrapper flips MuJoCo images vertically, converts the two
    gripper/control-mode commands at threshold 0.5, and handles mobile-base
    action ordering. It is intentionally the sole owner of those conversions.
    """

    def __init__(self, *, task_id=TASK_ID, split="target", max_steps=None,
                 image_size=224, seed=0, randomize_cameras=False):
        super().__init__()
        if task_id != TASK_ID:
            raise ValueError(f"Only the requested task {TASK_ID} is configured")
        if split not in {"target", "pretrain"}:
            raise ValueError("Use RoboCasa's official target or pretrain split")
        import robocasa  # noqa: F401; registers the native Gym environments
        from robocasa.utils.dataset_registry_utils import get_task_horizon

        official_horizon = int(get_task_horizon(TASK_NAME))
        self.max_steps = official_horizon if max_steps is None else int(max_steps)
        if self.max_steps != official_horizon:
            raise ValueError(f"This task uses the published v1.0.1 horizon {official_horizon}")
        self.task_id, self.split, self.image_size = task_id, split, int(image_size)
        self.native_camera_size = 256  # Official RoboCasa Gym/OpenPI evaluation pipeline.
        self.initial_seed = int(seed)
        self.randomize_cameras = bool(randomize_cameras)
        self.native = None
        self.steps = 0
        self._needs_reset = True
        self._closed = False
        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            "openpi": spaces.Dict({
                **{k: spaces.Box(0, 255, shape=(self.image_size, self.image_size, 3),
                                dtype=np.uint8) for k in CAMERA_KEYS},
                "observation/state": spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
                "prompt": spaces.Text(512, charset=string.printable),
            }),
        })

    def _observation(self, raw):
        from openpi_client.image_tools import resize_with_pad

        state = np.concatenate([np.asarray(raw[k], dtype=np.float32) for k in STATE_KEYS])
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("RoboCasa policy requires finite native 16D state")
        images = {}
        for output_key, native_key in CAMERA_KEYS.items():
            value = np.asarray(raw[native_key])
            if value.shape != (self.native_camera_size, self.native_camera_size, 3) or value.dtype != np.uint8:
                raise ValueError(f"Invalid native RGB image {native_key}: {value.shape}, {value.dtype}")
            images[output_key] = np.ascontiguousarray(resize_with_pad(value, self.image_size, self.image_size))
        return {"state": state.copy(), "openpi": {
            **images, "observation/state": state.copy(),
            "prompt": str(raw["annotation.human.task_description"]),
        }}

    def _info(self, native_info, success):
        meta = self.native.unwrapped.env.get_ep_meta()
        return dict(native_info) | {
            "is_success": bool(success), "task_suite": "robocasa365",
            "task_id": self.task_id, "benchmark_split": self.split,
            "backend_seed": self._episode_seed,
            "initial_state_id": self._episode_seed,
            "control_frequency_hz": 20, "max_episode_steps": self.max_steps,
            "layout_id": meta.get("layout_id"), "style_id": meta.get("style_id"),
            "task_prompt": meta.get("lang"),
        }

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("Environment has been closed")
        if options:
            raise ValueError("Use explicit frozen episode seeds; no reset options are supported")
        super().reset(seed=seed)
        self._needs_reset = True
        self._episode_seed = self.initial_seed if seed is None else int(seed)
        # A fresh seeded instance prevents hidden scene/object sampler state from
        # leaking between train and validation resets or between policy methods.
        if self.native is not None:
            self.native.close()
        self.native = gym.make(
            self.task_id, split=self.split, seed=self._episode_seed,
            camera_widths=self.native_camera_size, camera_heights=self.native_camera_size,
            randomize_cameras=self.randomize_cameras,
            disable_env_checker=True,
        )
        raw, info = self.native.reset(seed=self._episode_seed)
        success = bool(self.native.unwrapped.env._check_success())
        if success:
            raise RuntimeError("Task is already successful at reset")
        self.steps, self._needs_reset = 0, False
        return self._observation(raw), self._info(info, success)

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset before step and after episode termination")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Expected 12 finite normalized native controller values in [-1,1]")
        from robocasa.utils.env_utils import convert_action

        self._needs_reset = True
        raw, reward, native_done, native_truncated, info = self.native.step(convert_action(action))
        self.steps += 1
        success = bool(self.native.unwrapped.env._check_success())
        if bool(info["success"]) != success or float(reward) != float(success):
            raise RuntimeError("Native sparse reward disagrees with the task success predicate")
        terminated = success
        truncated = bool((self.steps >= self.max_steps or native_done or native_truncated) and not success)
        self._needs_reset = terminated or truncated
        return self._observation(raw), float(success), terminated, truncated, self._info(info, success)

    def render(self):
        if self.native is None:
            raise RuntimeError("Call reset before rendering")
        return self.native.render()

    def close(self):
        if self.native is not None:
            self.native.close()
            self.native = None
        self._closed = self._needs_reset = True


def make_env(**kwargs):
    return RoboCasaEnv(**kwargs)
