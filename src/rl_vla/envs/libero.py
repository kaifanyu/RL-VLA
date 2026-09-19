"""Optional LIBERO reference adapter for the upstream pi05_libero checkpoint.

The observation/reset protocol follows Physical Intelligence's Apache-2.0
``openpi/examples/libero/main.py``; see ``third_party/openpi/LICENSE``.
This is a new Gymnasium adapter: success flags, reset ownership, validation and
collection cutoffs are explicit. The quaternion conversion uses the same xyzw
convention as the example/robosuite, without mutating simulator observations.
No LIBERO, renderer, client, or checkpoint is imported/downloaded at module load.
"""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

SETTLING_ACTION = np.array([0.0] * 6 + [-1.0], dtype=np.float32)


def quaternion_to_axis_angle(quaternion) -> np.ndarray:
    """Convert simulator xyzw quaternion to the checkpoint's 3D axis-angle."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("robot0_eef_quat must be a finite xyzw quaternion of length 4")
    scalar = float(np.clip(quaternion[3], -1.0, 1.0))
    sine_half_angle = np.sqrt(max(0.0, 1.0 - scalar * scalar))
    if sine_half_angle <= 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quaternion[:3] * (2.0 * np.arccos(scalar) / sine_half_angle)).astype(np.float32)


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    try:
        from openpi_client import image_tools
    except ImportError as exc:
        raise ImportError("Install the pinned openpi-client; see docs/openpi.md") from exc
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, size, size))


def _load_initial_states(path: Path):
    """Read official NumPy/tensor state files using PyTorch's restricted loader.

    LIBERO's old bare torch.load predates the modern weights_only default.
    Permit NumPy float-array reconstruction locally; never fall back to an
    unrestricted pickle loader or change the process-wide torch defaults.
    """
    import torch

    allowed = [
        np.core.multiarray._reconstruct,
        np.core.multiarray.scalar,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.float32)),
    ]
    with torch.serialization.safe_globals(allowed):
        values = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(values, torch.Tensor):
        values = values.numpy()
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError(
            "LIBERO initial states must be a nonempty finite [episodes,state_size] array"
        )
    return values


def _create_backend(task_suite: str, task_id: int, camera_resolution: int, horizon: int):
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise ImportError(
            "LIBERO simulator is optional and is not installed in the baseline. "
            "Use a compatible Linux simulator environment; see docs/libero.md."
        ) from exc
    suites = benchmark.get_benchmark_dict()
    if task_suite not in suites:
        raise ValueError(f"Unknown LIBERO task suite {task_suite!r}; available: {sorted(suites)}")
    suite = suites[task_suite]()
    if not 0 <= task_id < suite.n_tasks:
        raise ValueError(f"task_id must be in [0, {suite.n_tasks}) for {task_suite}")
    task = suite.get_task(task_id)
    state_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    initial_states = _load_initial_states(state_path)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    backend = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=camera_resolution,
        camera_widths=camera_resolution,
        horizon=horizon,
    )
    return backend, initial_states, str(task.language)


class LiberoEnv(gym.Env):
    """One LIBERO task with sparse verified success and explicit time truncation.

    Backend injection supports contract tests without installing a simulator.
    Production construction uses ``make_env`` and the official LIBERO package.
    Settling steps belong to reset; ``max_steps`` counts subsequent commands.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        *,
        task_suite="libero_spatial",
        task_id=0,
        max_steps=220,
        camera_resolution=256,
        resize_size=224,
        settle_steps=10,
        initial_state_id=None,
        backend=None,
        initial_states=None,
        task_description=None,
        image_processor: Callable | None = None,
    ):
        super().__init__()
        self.task_suite = str(task_suite)
        self.task_id = int(task_id)
        self.max_steps = int(max_steps)
        self.resize_size = int(resize_size)
        self.settle_steps = int(settle_steps)
        if self.task_id < 0 or min(self.max_steps, self.resize_size, int(camera_resolution)) < 1:
            raise ValueError(
                "task_id must be nonnegative; max_steps and image sizes must be positive"
            )
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be nonnegative")
        if backend is None:
            backend, initial_states, task_description = _create_backend(
                self.task_suite,
                self.task_id,
                int(camera_resolution),
                max(1000, self.max_steps + self.settle_steps + 1),
            )
        if initial_states is None or task_description is None:
            raise ValueError(
                "An injected backend also requires initial_states and task_description"
            )
        self.backend = backend
        self.initial_states = np.asarray(initial_states)
        if self.initial_states.ndim != 2 or len(self.initial_states) < 1:
            raise ValueError("initial_states must be a nonempty 2D array")
        if not np.isfinite(self.initial_states).all():
            raise ValueError("initial_states must be finite")
        self.initial_state_id = self._validate_state_id(initial_state_id)
        self.task_description = str(task_description)
        if not self.task_description:
            raise ValueError("LIBERO task_description must not be empty")
        self._image_processor = image_processor or _resize_image
        self.action_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        image_space = lambda: spaces.Box(
            0, 255, (self.resize_size, self.resize_size, 3), dtype=np.uint8
        )
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
                "openpi": spaces.Dict(
                    {
                        "observation/image": image_space(),
                        "observation/wrist_image": image_space(),
                        "observation/state": spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32),
                        "prompt": spaces.Text(
                            max_length=len(self.task_description),
                            charset=set(self.task_description),
                        ),
                    }
                ),
            }
        )
        self._needs_reset = True
        self._closed = False

    def _validate_state_id(self, value):
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or int(value) != value
            or not 0 <= int(value) < len(self.initial_states)
        ):
            raise ValueError(
                f"initial_state_id must be an integer in [0, {len(self.initial_states)})"
            )
        return int(value)

    def _image(self, raw, key):
        image = np.asarray(raw[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"LIBERO {key} must be an HWC uint8 RGB image")
        # Official OpenPI preprocessing: rotate both cameras 180 degrees.
        image = self._image_processor(np.ascontiguousarray(image[::-1, ::-1]), self.resize_size)
        image = np.asarray(image)
        if image.shape != (self.resize_size, self.resize_size, 3) or image.dtype != np.uint8:
            raise ValueError("Image processor must return a uint8 RGB image at resize_size")
        return np.ascontiguousarray(image)

    def _observation(self, raw):
        position = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
        gripper = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32)
        if position.shape != (3,) or gripper.shape != (2,):
            raise ValueError(
                "LIBERO state requires 3D eef position and two gripper joint positions"
            )
        state = np.concatenate(
            (position, quaternion_to_axis_angle(raw["robot0_eef_quat"]), gripper)
        )
        if not np.isfinite(state).all():
            raise ValueError("LIBERO proprioceptive state must be finite")
        return {
            "state": state.copy(),
            "openpi": {
                "observation/image": self._image(raw, "agentview_image"),
                "observation/wrist_image": self._image(raw, "robot0_eye_in_hand_image"),
                "observation/state": state.copy(),
                "prompt": self.task_description,
            },
        }

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("This LIBERO environment has been closed")
        super().reset(seed=seed)
        self._needs_reset = True
        options = options or {}
        unknown = set(options) - {"initial_state_id"}
        if unknown:
            raise ValueError(f"Unsupported LIBERO reset options: {sorted(unknown)}")
        selected = self._validate_state_id(options.get("initial_state_id", self.initial_state_id))
        if selected is None:
            selected = int(self.np_random.integers(len(self.initial_states)))
        # LIBERO also uses NumPy's global RNG internally. Seed its reset from
        # our independent local generator; selection remains independent of it.
        backend_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.backend.seed(backend_seed)
        self.backend.reset()
        raw = self.backend.set_init_state(self.initial_states[selected].copy())
        for _ in range(self.settle_steps):
            raw, _, done, _ = self.backend.step(SETTLING_ACTION.tolist())
            if done or bool(self.backend.check_success()):
                raise RuntimeError(
                    "LIBERO task ended during reset settling; choose another initial_state_id"
                )
        if bool(self.backend.check_success()):
            raise RuntimeError(
                "LIBERO initial state already satisfies the task; choose another initial_state_id"
            )
        self.steps = 0
        self._selected_state_id = selected
        observation = self._observation(raw)
        self._needs_reset = False
        return observation, {
            "is_success": False,
            "initial_state_id": selected,
            "backend_seed": backend_seed,
            "settling_steps": self.settle_steps,
            "task_suite": self.task_suite,
            "task_id": self.task_id,
        }

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError(
                "Call reset() before stepping LIBERO, including after every episode boundary"
            )
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(
                "LIBERO actions must be seven finite normalized OSC/gripper commands in [-1,1]"
            )
        self._needs_reset = True
        raw, native_reward, done, native_info = self.backend.step(action.tolist())
        self.steps += 1
        success = bool(self.backend.check_success())
        terminated = success
        # The checked backend exposes success separately, so an unexpected
        # native done without success is treated as a cutoff, never a reward.
        truncated = bool((self.steps >= self.max_steps or done) and not success)
        observation = self._observation(raw)
        self._needs_reset = terminated or truncated
        info = dict(native_info)
        info.update(
            {
                "is_success": success,
                "initial_state_id": self._selected_state_id,
                "task_suite": self.task_suite,
                "task_id": self.task_id,
                "native_reward": float(native_reward),
                "native_done": bool(done),
            }
        )
        return observation, float(success), terminated, truncated, info

    def close(self):
        if not self._closed:
            self.backend.close()
            self._closed = True
            self._needs_reset = True


def make_env(**kwargs) -> LiberoEnv:
    """Factory path for TOML: ``rl_vla.envs.libero:make_env``."""
    return LiberoEnv(**kwargs)
