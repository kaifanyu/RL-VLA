"""Single-environment Gym adapter over the official RoboTwin task.

No expert seed filtering or automatic reset is applied. Invalid scene seeds
raise rather than silently changing the paired evaluation condition.
"""

from copy import deepcopy
import ctypes
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "third_party/robotwin"
UPSTREAM_COMMIT = "dca9ec682688821c42944eacd7f5dd9bfb15396f"
PROMPT = "Use the left arm to grasp the red block on the table, hand it to the right arm, and place it on the blue pad."


def verify_render_device(scene):
    """Check Vulkan rendering against CUDA's assigned device without loading Torch."""
    device = scene.render_system.device
    if not device.is_cuda() or device.cuda_id != 0:
        raise RuntimeError("RoboTwin renderer must use assigned CUDA device 0")
    driver = ctypes.CDLL("libcuda.so.1")
    cuda_device = ctypes.c_int()
    pci = ctypes.create_string_buffer(32)
    for code in (driver.cuInit(0), driver.cuDeviceGet(ctypes.byref(cuda_device), 0),
                 driver.cuDeviceGetPCIBusId(pci, len(pci), cuda_device)):
        if code:
            raise RuntimeError(f"CUDA driver failed while verifying render GPU: {code}")
    cuda_pci = pci.value.decode()
    canonical = lambda value: ":".join(value.lower().split(":")[-2:])
    if canonical(device.pci_string) != canonical(cuda_pci):
        raise RuntimeError("RoboTwin Vulkan renderer differs from assigned CUDA device 0")
    return {"name": device.name, "pci": device.pci_string, "cuda_id": device.cuda_id,
            "cuda_pci": cuda_pci, "affinity_verified": True,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "slurm_step_gpus": os.getenv("SLURM_STEP_GPUS")}


def configuration(upstream=UPSTREAM):
    upstream = Path(upstream).resolve()
    args = yaml.safe_load((upstream / "task_config/demo_clean.yml").read_text())
    robot = upstream / "assets/embodiments/aloha-agilex"
    embodiment = yaml.safe_load((robot / "config.yml").read_text())
    # The upstream RLinf branch supports mplib without CUDA motion planning.
    # Native take_action(qpos) retains its TOPP trajectory interpolation.
    embodiment["planner"] = "mplib_RRT"
    args.update(
        task_name="handover_block", task_config="demo_clean",
        left_robot_file=str(robot), right_robot_file=str(robot),
        left_embodiment_config=deepcopy(embodiment),
        right_embodiment_config=deepcopy(embodiment),
        dual_arm_embodied=True, planner_backend="mplib", eval_mode=True,
        render_freq=0, save_data=False, collect_data=False,
        eval_video_log=False, instruction=PROMPT,
    )
    camera = yaml.safe_load((upstream / "task_config/_camera_config.yml").read_text())["D435"]
    args["head_camera_h"], args["head_camera_w"] = camera["h"], camera["w"]
    return args


def action_bounds(args):
    robot = Path(args["left_robot_file"])
    emb = args["left_embodiment_config"]
    urdf = ET.parse(robot / emb["urdf_path"])
    joints = {joint.attrib["name"]: joint for joint in urdf.findall("joint")}
    low, high = [], []
    for names in emb["arm_joints_name"]:
        for name in names:
            limits = joints[name].find("limit")
            low.append(float(limits.attrib["lower"]))
            high.append(float(limits.attrib["upper"]))
        low.append(0.0)
        high.append(1.0)
    return np.asarray(low, np.float32), np.asarray(high, np.float32)


class HandoverBlockEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, *, max_steps=800, upstream=None, render_mode="rgb_array"):
        self.upstream = Path(upstream or UPSTREAM).resolve()
        self.args = configuration(self.upstream)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if render_mode != "rgb_array":
            raise ValueError("Only headless RGB rendering is supported")
        self.render_mode = render_mode
        self.task_description = PROMPT
        low, high = action_bounds(self.args)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        rgb = lambda: gym.spaces.Box(0, 255, (240, 320, 3), dtype=np.uint8)
        state = lambda: gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "state": state(),
            "openpi": gym.spaces.Dict({
                "observation/state": state(),
                "observation/image": rgb(),
                "observation/wrist_image": rgb(),
                "observation/right_wrist_image": rgb(),
                "prompt": gym.spaces.Text(max_length=512, min_length=1,
                                           charset="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ,.-0123456789"),
            }),
        })
        os.environ["ASSETS_PATH"] = str(self.upstream)
        sys.path.insert(0, str(self.upstream))
        from envs.handover_block import handover_block
        class HeadlessHandoverBlock(handover_block):
            def setup_scene(self, **kwargs):
                super().setup_scene(**kwargs)
                # SAPIEN 3.0.0b1 bundles an OIDN CUDA denoiser predating the
                # Blackwell GPU. Preserve RT, 32 spp and path depth 8, with
                # denoising disabled so rendered pixels remain valid.
                import sapien
                sapien.render.set_ray_tracing_denoiser("none")
        self._task_class = HeadlessHandoverBlock
        self.task = None
        self._needs_reset = True
        self._last = None
        self._seed = None

    def _observation(self):
        raw = self.task.get_obs()
        state = np.asarray(raw["joint_action"]["vector"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise RuntimeError("Invalid RoboTwin native proprioception")
        cameras = raw["observation"]
        images = {name: np.asarray(cameras[name]["rgb"]) for name in
                  ("head_camera", "left_camera", "right_camera")}
        if any(img.shape != (240, 320, 3) or img.dtype != np.uint8 for img in images.values()):
            raise RuntimeError("RoboTwin camera shape or dtype changed")
        self._last = {
            "state": state.copy(),
            "openpi": {
                "observation/state": state.copy(),
                "observation/image": images["head_camera"].copy(),
                "observation/wrist_image": images["left_camera"].copy(),
                "observation/right_wrist_image": images["right_camera"].copy(),
                "prompt": self.task_description,
            },
        }
        return self._last

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Task reset is controlled only by its explicit seed")
        self.close()
        self._seed = int(seed if seed is not None else self.np_random.integers(2**31 - 1))
        self.task = self._task_class()
        self.task.setup_demo(seed=self._seed, now_ep_num=0, **deepcopy(self.args))
        self.render_device = verify_render_device(self.task.scene)
        self.task.step_lim = self.max_steps
        if self.task.check_success():
            raise RuntimeError("RoboTwin initial condition already satisfies the goal")
        self._needs_reset = False
        raw = self._observation()
        return raw, {"is_success": False, "backend_seed": self._seed,
                     "initial_state_id": self._seed, "task_id": "handover_block",
                     "render_device": self.render_device,
                     "controller_steps": 0}

    def step(self, action):
        if self._needs_reset or self.task is None:
            raise RuntimeError("reset() is required before each episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected finite fourteen-dimensional absolute qpos action")
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            raise ValueError("Action outside URDF joint limits / normalized gripper limits")
        self._needs_reset = True
        self.task.take_action(action, action_type="qpos")
        success = bool(self.task.eval_success or self.task.check_success())
        truncated = self.task.take_action_cnt >= self.max_steps and not success
        raw = self._observation()
        self._needs_reset = success or truncated
        return raw, float(success), success, bool(truncated), {
            "is_success": success, "backend_seed": self._seed,
            "initial_state_id": self._seed, "controller_steps": self.task.take_action_cnt,
        }

    def render(self):
        if self._last is None:
            raise RuntimeError("reset() is required before render()")
        return self._last["openpi"]["observation/image"].copy()

    def close(self):
        if self.task is not None:
            self.task.close_env(clear_cache=True)
            self.task = None
        self._needs_reset = True


def make_env(**kwargs):
    return HandoverBlockEnv(**kwargs)
