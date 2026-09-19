"""Optional numerical benchmarks for the existing residual SAC/PPO learners.

Use ``env.kind = 'external'`` with factory
``rl_vla.envs.benchmarks:make_env`` and ``env.kwargs.env_id``. These tasks use a
zero mock base: horizon/execute_steps of one and full physical residual scales
exercise ordinary continuous-control learning. Pendulum requires scale 2; the
Reacher and Panda tasks require scale 1. No VLA or model download is involved.
"""

import importlib
from collections.abc import Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

BENCHMARK_IDS = frozenset({
    "Pendulum-v1",
    "Reacher-v5",
    "PandaReachDense-v3",
    "PandaPushDense-v3",
    "PandaPickAndPlaceDense-v3",
})


class NumericalBenchmarkEnv(gym.ObservationWrapper):
    """Expose simulator state and goal without changing rewards or action units.

    Goal dictionaries are ordered as observation, achieved_goal, desired_goal;
    the desired goal is therefore available to both actor and critic. Simulator
    termination, time limits, info (including native success), rendering and
    close remain delegated to the original environment. This wrapper does not
    invent a success threshold for tasks such as Pendulum or MuJoCo Reacher.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        action_space = env.action_space
        if (not isinstance(action_space, spaces.Box)
                or len(action_space.shape) != 1
                or not np.issubdtype(action_space.dtype, np.floating)
                or not np.isfinite(action_space.low).all()
                or not np.isfinite(action_space.high).all()
                or np.any(action_space.low >= action_space.high)):
            raise ValueError("Benchmark actions must be a finite floating vector Box")
        self._zero_action = np.zeros(action_space.shape, dtype=action_space.dtype)
        if not action_space.contains(self._zero_action):
            raise ValueError("Benchmark action bounds must contain the zero base action")

        source = env.observation_space
        if isinstance(source, spaces.Dict):
            self._state_keys = ("observation", "achieved_goal", "desired_goal")
            if set(source.spaces) != set(self._state_keys):
                raise ValueError("Benchmark goal observations require observation, achieved_goal, "
                                 "and desired_goal only")
            components = [source[key] for key in self._state_keys]
        else:
            self._state_keys = None
            components = [source]
        if any(not isinstance(component, spaces.Box)
               or len(component.shape) != 1
               or component.shape[0] == 0
               or not np.issubdtype(component.dtype, np.number)
               for component in components):
            raise ValueError("Benchmark observations must contain numerical vector Boxes")
        self._state_widths = tuple(component.shape[0] for component in components)
        low = np.concatenate([component.low for component in components]).astype(np.float32)
        high = np.concatenate([component.high for component in components]).astype(np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low, high, dtype=np.float32),
            "mock_base_action": spaces.Box(action_space.low.astype(np.float32),
                                           action_space.high.astype(np.float32), dtype=np.float32),
        })

    def observation(self, observation):
        if self._state_keys is not None:
            if not isinstance(observation, Mapping):
                raise ValueError("Benchmark goal observation must be a mapping")
            values = [observation[key] for key in self._state_keys]
        else:
            values = [observation]
        vectors = [np.asarray(value, dtype=np.float32) for value in values]
        if any(vector.shape != (width,) or not np.isfinite(vector).all()
               for vector, width in zip(vectors, self._state_widths, strict=True)):
            raise ValueError("Benchmark observation dimensions changed or state is nonfinite")
        return {"state": np.concatenate(vectors),
                "mock_base_action": self._zero_action.astype(np.float32, copy=True)}


def make_env(env_id: str, render_mode: str | None = None, **kwargs) -> gym.Env:
    """Create a benchmark, importing optional simulator packages only as needed.

    Install ``gymnasium[classic-control]`` for Pendulum rendering,
    ``gymnasium[mujoco]`` for Reacher, and ``panda-gym`` for Panda tasks. Extra
    arguments are passed to Gymnasium, including ``max_episode_steps``. Use
    ``render_mode='rgb_array'`` for headless video capture. Panda requires a
    render mode even during training; its default is headless ``rgb_array``
    with the Tiny renderer. Frames are generated only when ``render()`` is called.
    """
    if env_id not in BENCHMARK_IDS:
        raise ValueError(f"Unsupported benchmark {env_id!r}; choose from {sorted(BENCHMARK_IDS)}")
    if env_id.startswith("Panda"):
        try:
            importlib.import_module("panda_gym")
        except ImportError as exc:
            raise ImportError("Panda benchmarks require the optional 'panda-gym' package") from exc
        if render_mode is None:
            render_mode = "rgb_array"
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    try:
        return NumericalBenchmarkEnv(env)
    except Exception:
        env.close()
        raise
