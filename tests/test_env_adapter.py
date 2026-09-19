"""Regression tests for rollout semantics that affect both SAC and PPO."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from rl_vla.env_adapter import ResidualChunkEnv
from rl_vla.envs import make_env
from rl_vla.envs.toy import ToyReachEnv


class CountingBase:
    def __init__(self, horizon=4, action=(0.2, -0.2)):
        self.horizon = horizon
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0
        self.resets = 0
        self.seen_states = []

    def reset(self):
        self.resets += 1

    def infer(self, observation):
        self.calls += 1
        self.seen_states.append(np.array(observation["state"]).copy())
        return np.tile(self.action, (self.horizon, 1))


class CounterEnv(gym.Env):
    def __init__(self, end_after=10, truncate=False):
        self.end_after, self.truncate = end_after, truncate
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "critic_state": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
        })
        self.actions = []

    def observation(self):
        return {"state": np.array([self.t], dtype=np.float32),
                "critic_state": np.array([123.0], dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.actions = []
        return self.observation(), {}

    def step(self, action):
        self.t += 1
        self.actions.append(action.copy())
        done = self.t >= self.end_after
        return self.observation(), float(self.t), done and not self.truncate, done and self.truncate, {
            "is_success": done and not self.truncate}


def wrapped(*, end_after=10, truncate=False, **overrides):
    env = CounterEnv(end_after, truncate)
    base = CountingBase()
    config = {"horizon": 4, "execute_steps": 3, "gamma": 0.9, "residual_scale": 0.25}
    config.update(overrides)
    return ResidualChunkEnv(env, base, config), base


def test_chunk_reward_actual_length_and_terminal_no_bootstrap_context():
    env, base = wrapped(end_after=2)
    env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(np.zeros(env.action_dim))
    assert reward == pytest.approx(1.0 + 0.9 * 2.0)
    assert info["raw_return"] == 3.0
    assert info["discount"] == pytest.approx(0.9 ** 2)
    assert info["executed_steps"] == 2
    assert info["executed_actions"].shape == (2, 2)
    assert terminated and not truncated and info["is_success"]
    assert base.calls == 1
    np.testing.assert_array_equal(observation["base_chunk"], np.zeros((4, 2)))
    assert observation["obs"][0] == 2
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(env.action_dim))


def test_truncation_keeps_final_observation_and_samples_bootstrap_base_once():
    env, base = wrapped(end_after=2, truncate=True)
    env.reset()
    observation, _, terminated, truncated, info = env.step(np.zeros(env.action_dim))
    assert not terminated and truncated
    assert base.calls == 2
    np.testing.assert_array_equal(base.seen_states[-1], [2])
    assert observation["obs"][0] == 2
    assert info["discount"] == pytest.approx(0.81)
    np.testing.assert_array_equal(observation["base_chunk"][0], base.action)
    saved = observation["obs"].copy()
    env.reset()
    np.testing.assert_array_equal(observation["obs"], saved)
    assert base.resets == 2


def test_base_cached_between_decisions_and_copies_protect_it():
    env, base = wrapped()
    observation, _ = env.reset()
    assert base.calls == 1
    observation["base_chunk"][:] = 999  # Caller mutations must not affect execution.
    _, _, _, _, info = env.step(np.zeros(env.action_dim))
    np.testing.assert_allclose(info["executed_actions"], np.tile(base.action, (3, 1)))
    assert base.calls == 2
    env.step(np.zeros(env.action_dim))
    assert base.calls == 3


def test_projection_preserves_full_requested_residual_and_masks_frozen_axes():
    env, _base = wrapped(residual_scale=[2.0, 0.0])
    env.reset()
    assert env.action_dim == 3
    assert env.active_axes.tolist() == [0]
    _, _, _, _, info = env.step(np.ones(env.action_dim))
    np.testing.assert_allclose(info["executed_actions"], np.tile([1.0, -0.2], (3, 1)))
    np.testing.assert_array_equal(info["requested_residual"], np.ones(3))
    assert info["projection_fraction"] == 0.5


def test_actor_cannot_see_privileged_critic_state():
    env, _ = wrapped(state_mean=[2], state_std=[2])
    observation, _ = env.reset()
    assert observation["obs"][0] == -1
    assert env.obs_dim == 7 and env.critic_dim == 8
    assert 123.0 not in observation["obs"]
    np.testing.assert_array_equal(observation["critic_obs"][:-1], observation["obs"])
    assert observation["critic_obs"][-1] == 123.0


def test_validation_and_reset_gate():
    env, _ = wrapped()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(env.action_dim))
    env.reset()
    for action in (np.zeros((3, 2)), np.full(6, np.nan), np.full(6, 1.5)):
        with pytest.raises(ValueError):
            env.step(action)
    with pytest.raises(ValueError, match="positive"):
        wrapped(residual_scale=0)
    with pytest.raises(ValueError, match="state_std"):
        invalid, _ = wrapped(state_std=0)
        invalid.reset()


def test_toy_sparse_success_and_discount_consistent_potential():
    env = ToyReachEnv(gamma=0.9, shaping_scale=0.5, step_size=0.08, success_radius=0.01)
    env.reset(seed=4, options={"position": [0, 0], "goal": [0.08, 0]})
    _, reward, terminated, truncated, info = env.step(np.array([1, 0], dtype=np.float32))
    # Phi_before=-0.04 and absorbing Phi_after=0.
    assert reward == pytest.approx(1 + 0.5 * 0.04)
    assert terminated and not truncated and info["is_success"]
    assert info["sparse_reward"] == 1
    with pytest.raises(ValueError, match="gamma"):
        ResidualChunkEnv(env, CountingBase(), {"horizon": 4, "execute_steps": 3, "gamma": 0.99})


def test_toy_time_limit_is_truncation_and_retains_potential():
    env = make_env({"kind": "toy", "kwargs": {"max_steps": 1, "gamma": 0.9, "shaping_scale": 0.5}})
    env.reset(seed=7, options={"position": [0, 0], "goal": [1, 0]})
    _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert reward == pytest.approx(0.5 * (0.9 * -0.5 - -0.5))
    assert truncated and not terminated and not info["is_success"]


def test_external_factory_is_explicit_and_returns_gym_env():
    env = make_env({"kind": "external", "factory": "rl_vla.envs.toy:ToyReachEnv", "kwargs": {"max_steps": 2}})
    assert isinstance(env, ToyReachEnv) and env.max_steps == 2
    with pytest.raises(ValueError, match="module:function"):
        make_env({"kind": "external", "factory": "missing_separator"})
