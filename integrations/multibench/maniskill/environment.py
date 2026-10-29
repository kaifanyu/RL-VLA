"""Single-environment plate-25 adapter using the pinned official RLinf task.

Actions are decoded ManiSkill dataset actions (metres/radians, gripper -1..1).
The only conversion is RLinf's published WidowX gripper threshold. Simulation
and policy processes may use separate Python environments.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import string
import subprocess
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "PutOnPlateInScene25Main-v3"
CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
ASSETS = ROOT / "datasets/benchmark_assets/maniskill3"


def register_task():
    # These must be configured before importing ManiSkill (which freezes ASSET_DIR).
    os.environ.setdefault("MS_ASSET_DIR", str(ASSETS / "upstream"))
    os.environ.setdefault("MANISKILL_ASSET_DIR", str(ASSETS / "rlinf"))
    import mani_skill.envs  # noqa: F401

    if TASK_ID not in gym.registry:
        path = ROOT / "third_party/rlinf/rlinf/envs/maniskill/tasks/put_on_in_scene_multi.py"
        spec = importlib.util.spec_from_file_location("multibench_plate25_official", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class Plate25Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, *, episode_ids=None, obj_set="train", sim_backend="cpu",
                 render_backend="gpu", max_episode_steps=80):
        register_task()
        self.native = gym.make(
            TASK_ID, num_envs=1, obs_mode="rgb+segmentation",
            control_mode=CONTROL_MODE, reward_mode="none", sim_backend=sim_backend,
            render_backend=render_backend, max_episode_steps=max_episode_steps,
            sim_config={"sim_freq": 500, "control_freq": 5},
            sensor_configs={"shader_pack": "default"}, render_mode="rgb_array",
            obj_set=obj_set, use_multiple_plates=False,
        )
        device = self.native.unwrapped.backend.render_device
        self.render_device = {"name": device.name, "pci": device.pci_string,
                              "cuda_id": device.cuda_id if device.is_cuda() else None,
                              "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                              "slurm_step_gpus": os.getenv("SLURM_STEP_GPUS")}
        if render_backend in {"gpu", "cuda", "sapien_cuda"}:
            import torch
            if not device.is_cuda() or device.cuda_id != 0:
                raise RuntimeError("ManiSkill renderer must use the assigned CUDA device 0")
            uuid = str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
            rows = subprocess.check_output([
                "nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"], text=True)
            lookup = {key.removeprefix("GPU-"): value for key, value in
                      (line.strip().split(", ", 1) for line in rows.splitlines() if line.strip())}
            if uuid not in lookup:
                raise RuntimeError("Cannot verify assigned GPU UUID against the native renderer")
            canonical_pci = lambda value: ":".join(value.lower().split(":")[-2:])
            if canonical_pci(device.pci_string) != canonical_pci(lookup[uuid]):
                raise RuntimeError("SAPIEN renderer selected a different GPU from assigned CUDA device 0")
            self.render_device.update(cuda_uuid=uuid, cuda_pci=lookup[uuid], affinity_verified=True)
        low = np.array(self.native.unwrapped.single_action_space.low, dtype=np.float32)
        high = np.array(self.native.unwrapped.single_action_space.high, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        policy_space = spaces.Dict({
            "observation/state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            "observation/image": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
            "observation/wrist_image": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
            "observation/right_wrist_image": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
            "prompt": spaces.Text(max_length=512, charset=string.printable),
        })
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            "openpi": policy_space,
        })
        self.episode_ids = None if episode_ids is None else [int(i) for i in episode_ids]
        if self.episode_ids is not None:
            if not self.episode_ids or len(set(self.episode_ids)) != len(self.episode_ids):
                raise ValueError("episode_ids must be nonempty and unique")
            if min(self.episode_ids) < 0 or max(self.episode_ids) >= self.native.unwrapped.total_num_trials:
                raise ValueError("episode_ids outside the selected object split")
        self._cursor = 0
        self._episode_id = None
        self._steps = 0
        self._last_image = None
        self._needs_reset = True

    def _obs(self, raw):
        state = to_numpy(self.native.unwrapped.agent.robot.get_qpos())[0].astype(np.float32)
        image = to_numpy(raw["sensor_data"]["3rd_view_camera"]["rgb"])[0].astype(np.uint8)
        prompt = self.native.unwrapped.get_language_instruction()[0]
        if state.shape != (8,) or image.shape != (480, 640, 3) or not np.isfinite(state).all():
            raise RuntimeError("Unexpected ManiSkill embodiment or camera output")
        self._last_image = image.copy()
        return {"state": state, "openpi": {"observation/state": state.copy(),
                "observation/image": image, "observation/wrist_image": np.zeros_like(image),
                "observation/right_wrist_image": np.zeros_like(image), "prompt": prompt}}

    def _info(self, raw):
        info = {}
        for key, value in raw.items():
            if hasattr(value, "shape"):
                value = to_numpy(value)
                info[key] = value.reshape(-1)[0].item() if value.size == 1 else value
        info.update({"is_success": bool(info.get("success", False)), "task_id": TASK_ID,
                     "episode_id": self._episode_id, "controller_steps": self._steps,
                     "initial_state_id": self._episode_id,
                     "render_device": self.render_device,
                     "control_frequency": 5, "reward_definition": "binary_task_success"})
        return info

    def reset(self, *, seed=None, options=None):
        import torch
        if seed is None:
            raise ValueError("An explicit seed is required for reproducibility")
        options = dict(options or {})
        if "episode_id" in options:
            if self.episode_ids is not None:
                raise ValueError("Cannot override a frozen episode_ids schedule")
            episode_id = int(options.pop("episode_id"))
        elif self.episode_ids is not None:
            episode_id = self.episode_ids[self._cursor % len(self.episode_ids)]
            self._cursor += 1
        else:
            # Hashing seeds avoids accidentally restricting low-numbered seeds to one object.
            digest = hashlib.sha256(str(seed).encode()).digest()
            episode_id = int.from_bytes(digest[:8], "little") % self.native.unwrapped.total_num_trials
        if options:
            raise ValueError(f"Unsupported reset options: {list(options)}")
        self._episode_id, self._steps = episode_id, 0
        raw, info = self.native.reset(seed=int(seed), options={
            "episode_id": torch.tensor([episode_id], device=self.native.unwrapped.device)})
        self._needs_reset = False
        return self._obs(raw), self._info(info)

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("reset() is required before stepping")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite, decoded seven-dimensional ManiSkill action")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        controller_action = action.copy()
        controller_action[-1] = 1. if action[-1] > .5 else -1.
        raw, _, terminated, truncated, info = self.native.step(controller_action[None])
        self._steps += 1
        info = self._info(info)
        terminated = bool(to_numpy(terminated).reshape(-1)[0])
        truncated = bool(to_numpy(truncated).reshape(-1)[0])
        self._needs_reset = terminated or truncated
        info["executed_action"] = controller_action
        return self._obs(raw), float(info["is_success"]), terminated, truncated, info

    def render(self):
        return None if self._last_image is None else self._last_image.copy()

    def close(self):
        self.native.close()


def make_env(**kwargs):
    return Plate25Env(**kwargs)
