"""Numerical adapters preserve goals, physical actions and episode semantics."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rl_vla.base_policy import MockBasePolicy
from rl_vla.env_adapter import ResidualChunkEnv
from rl_vla.envs import benchmarks
from rl_vla.envs.benchmarks import NumericalBenchmarkEnv, make_env


class GoalTask(gym.Env):
    """Small goal task to verify the Panda contract without optional simulators."""

    def __init__(self):
        self.action_space = spaces.Box(-2.0, 2.0, (1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "desired_goal": spaces.Box(-10.0, 10.0, (1,), dtype=np.float64),
            "observation": spaces.Box(-10.0, 10.0, (2,), dtype=np.float64),
            "achieved_goal": spaces.Box(-10.0, 10.0, (1,), dtype=np.float64),
        })
        self.closed = False
        self.frame = np.zeros((4, 5, 3), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"observation": np.array([1.0, 2.0]), "achieved_goal": np.array([3.0]),
                "desired_goal": np.array([float((options or {}).get("goal", 4.0))])}, {"seed": seed}

    def step(self, action):
        self.last_action = action.copy()
        observation, _ = self.reset()
        return observation, -0.125, True, False, {"is_success": True, "distance": 0.125}

    def render(self):
        return self.frame

    def close(self):
        self.closed = True


def test_goal_and_action_contract_preserves_semantics():
    task = GoalTask()
    env = NumericalBenchmarkEnv(task)
    observation, info = env.reset(seed=91, options={"goal": 8})
    np.testing.assert_array_equal(observation["state"], [1, 2, 3, 8])
    assert observation["state"].dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info == {"seed": 91}
    assert env.action_space is task.action_space
    np.testing.assert_array_equal(observation["mock_base_action"], [0])
    _, reward, terminated, truncated, info = env.step(np.array([1.75], dtype=np.float32))
    np.testing.assert_array_equal(task.last_action, [1.75])
    assert reward == -0.125
    assert terminated and not truncated
    assert info == {"is_success": True, "distance": 0.125}
    assert env.render() is task.frame
    env.close()
    assert task.closed


def test_actual_pendulum_seed_bounds_and_time_limit():
    env = make_env("Pendulum-v1", max_episode_steps=2)
    try:
        first, info = env.reset(seed=12)
        second, _ = env.reset(seed=12)
        np.testing.assert_array_equal(first["state"], second["state"])
        assert "is_success" not in info
        np.testing.assert_array_equal(env.action_space.low, [-2])
        np.testing.assert_array_equal(env.action_space.high, [2])
        assert env.observation_space.contains(first)
        _, reward, terminated, truncated, info = env.step(np.array([2.0], dtype=np.float32))
        assert np.isfinite(reward)
        assert not terminated and not truncated
        assert "is_success" not in info
        _, _, terminated, truncated, info = env.step(np.array([-2.0], dtype=np.float32))
        assert not terminated and truncated
        assert "is_success" not in info
    finally:
        env.close()


def test_full_scale_residual_reaches_pendulum_physical_action_bound():
    env = ResidualChunkEnv(make_env("Pendulum-v1", max_episode_steps=1),
                           MockBasePolicy(1, 1),
                           {"horizon": 1, "execute_steps": 1, "residual_scale": 2.0})
    try:
        observation, _ = env.reset(seed=0)
        assert observation["obs"].shape == (4,)
        np.testing.assert_array_equal(observation["base_chunk"], [[0]])
        _, _, terminated, truncated, info = env.step(np.ones(1, dtype=np.float32))
        np.testing.assert_array_equal(info["executed_actions"], [[2]])
        assert info["projection_fraction"] == 0
        assert truncated and not terminated
    finally:
        env.close()


@pytest.mark.parametrize("invalid", [np.array([1, np.nan, 2]), np.zeros((3, 1)), np.zeros(4)])
def test_reject_nonfinite_or_changed_state(invalid):
    env = make_env("Pendulum-v1")
    try:
        with pytest.raises(ValueError, match="dimensions changed or state is nonfinite"):
            env.observation(invalid)
    finally:
        env.close()


def test_reject_goal_dict_missing_desired_goal():
    task = GoalTask()
    del task.observation_space.spaces["desired_goal"]
    with pytest.raises(ValueError, match="desired_goal"):
        NumericalBenchmarkEnv(task)


def test_reject_zero_base_outside_physical_action_bounds():
    task = GoalTask()
    task.action_space = spaces.Box(1.0, 2.0, (1,), dtype=np.float32)
    with pytest.raises(ValueError, match="zero base action"):
        NumericalBenchmarkEnv(task)


def test_reject_unknown_benchmark_before_optional_import():
    with pytest.raises(ValueError, match="Unsupported benchmark"):
        make_env("PandaMisspelled-v3")


@pytest.mark.parametrize(
    ("env_id", "requested_mode", "extra", "expected_mode", "expected_extra"),
    [
        ("PandaReachDense-v3", None, {}, "rgb_array", {}),
        ("PandaPushDense-v3", None, {}, "rgb_array", {}),
        ("PandaPickAndPlaceDense-v3", None, {}, "rgb_array", {}),
        ("PandaReachDense-v3", "rgb_array", {}, "rgb_array", {}),
        ("PandaReachDense-v3", "rgb_array", {"renderer": "OpenGL"},
         "rgb_array", {"renderer": "OpenGL"}),
        ("PandaReachDense-v3", "human", {}, "human", {}),
        ("Pendulum-v1", None, {}, None, {}),
        ("Reacher-v5", None, {}, None, {}),
    ],
)
def test_panda_headless_defaults_preserve_explicit_and_other_modes(
        monkeypatch, env_id, requested_mode, extra, expected_mode, expected_extra):
    imports, made = [], []
    task = GoalTask()

    def fake_make(name, **kwargs):
        made.append((name, kwargs))
        return task

    monkeypatch.setattr(benchmarks.importlib, "import_module", imports.append)
    monkeypatch.setattr(benchmarks.gym, "make", fake_make)
    env = make_env(env_id, render_mode=requested_mode, max_episode_steps=12, **extra)
    try:
        assert made == [(env_id, {"render_mode": expected_mode, "max_episode_steps": 12,
                                  **expected_extra})]
        assert imports == (["panda_gym"] if env_id.startswith("Panda") else [])
    finally:
        env.close()
