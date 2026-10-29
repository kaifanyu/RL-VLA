"""One RL decision executes a residual correction to a cached frozen VLA chunk."""

import time
from collections.abc import Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_vla.observations import ObservationEncoder


class ResidualChunkEnv(gym.Wrapper):
    """Turn primitive-step rewards into correctly discounted chunk transitions.

    ``v`` is a normalized, flattened [K, active_coordinates] decision in [-1,1].
    Physical corrections are ``residual_scale * v``. A zero scale freezes that
    coordinate and excludes it from the policy's action dimension/density.
    Clipping is an environment-side projection, not a transformed policy density.

    Reset explicitly after termination/truncation. True terminal observations
    contain a zero base chunk because no bootstrap is allowed. Truncations retain
    the final physical observation and a freshly sampled base chunk for bootstrap.
    """

    def __init__(self, env: gym.Env, base_policy, config: dict):
        super().__init__(env)
        self.base_policy = base_policy
        self.base_calls = 0
        self.base_inference_seconds = 0.0
        self.horizon = int(config.get("horizon", 10))
        self.execute_steps = int(config.get("execute_steps", 5))
        self.gamma = float(config.get("gamma", 0.99))
        if not 1 <= self.execute_steps <= self.horizon:
            raise ValueError("Require 1 <= execute_steps <= horizon")
        if not np.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0,1]")
        if not isinstance(env.action_space, spaces.Box) or len(env.action_space.shape) != 1:
            raise ValueError("Primitive action_space must be a one-dimensional Gymnasium Box")
        if not np.issubdtype(env.action_space.dtype, np.floating):
            raise ValueError("Primitive actions must use a floating dtype")
        self.motor_dim = env.action_space.shape[0]
        self._low = np.asarray(env.action_space.low, dtype=np.float32)
        self._high = np.asarray(env.action_space.high, dtype=np.float32)
        if not np.all(np.isfinite(self._low)) or not np.all(np.isfinite(self._high)):
            raise ValueError("Primitive action bounds must be finite")
        if np.any(self._low >= self._high):
            raise ValueError("Each primitive action coordinate must have nonempty bounds")
        scale = np.asarray(config.get("residual_scale", 0.1), dtype=np.float32)
        if scale.ndim == 0:
            scale = np.full(self.motor_dim, float(scale), dtype=np.float32)
        if scale.shape != (self.motor_dim,) or not np.all(np.isfinite(scale)) or np.any(scale < 0):
            raise ValueError("residual_scale must be finite, nonnegative, and scalar or match action width")
        self.residual_scale = scale
        self.active_axes = np.flatnonzero(scale > 0)
        if not self.active_axes.size:
            raise ValueError("At least one residual_scale coordinate must be positive")
        self.action_dim = self.execute_steps * self.active_axes.size
        self.action_space = spaces.Box(-1.0, 1.0, (self.action_dim,), dtype=np.float32)
        self.encoder = ObservationEncoder(config)
        self.obs_dim = None
        self.critic_dim = None
        self._needs_reset = True
        self._base_chunk = None
        current = env
        while isinstance(current, gym.Wrapper):
            if type(current).__name__ in {"Autoreset", "AutoResetWrapper"}:
                raise ValueError("Autoreset wrappers are unsupported: the collector must own resets")
            current = current.env
        if getattr(env.unwrapped, "shaping_scale", 0) != 0:
            task_gamma = getattr(env.unwrapped, "gamma", self.gamma)
            if not np.isclose(task_gamma, self.gamma):
                raise ValueError("Potential shaping gamma must match the chunk primitive-step gamma")

    def _decision_observation(self, raw, *, terminal=False):
        if not isinstance(raw, Mapping):
            raise TypeError("Environment observations must be mappings")
        if terminal:
            base = np.zeros((self.horizon, self.motor_dim), dtype=np.float32)
        else:
            started = time.perf_counter()
            base = np.asarray(self.base_policy.infer(raw), dtype=np.float32)
            self.base_inference_seconds += time.perf_counter() - started
            self.base_calls += 1
            if base.ndim != 2 or base.shape[0] < self.horizon or base.shape[1] != self.motor_dim:
                raise ValueError(f"Base policy must return [>= {self.horizon}, {self.motor_dim}], got {base.shape}")
            if not np.all(np.isfinite(base)):
                raise ValueError("Base policy returned nonfinite actions")
            base = base[:self.horizon].copy()
        self._base_chunk = base
        observation = self.encoder.encode(raw, base[:self.execute_steps])
        observation["base_chunk"] = base.copy()
        if self.obs_dim is None:
            self.obs_dim = observation["obs"].size
            self.critic_dim = observation["critic_obs"].size
            self.observation_space = spaces.Dict({
                "obs": spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32),
                "critic_obs": spaces.Box(-np.inf, np.inf, (self.critic_dim,), dtype=np.float32),
                "base_chunk": spaces.Box(-np.inf, np.inf,
                                         (self.horizon, self.motor_dim), dtype=np.float32),
            })
        return observation

    def reset(self, *, seed=None, options=None):
        self._needs_reset = True
        raw, info = self.env.reset(seed=seed, options=options)
        self.base_policy.reset()
        observation = self._decision_observation(raw)
        self._needs_reset = False
        return observation, info

    def step(self, residual):
        if self._needs_reset:
            raise RuntimeError("Call reset() before step(), including after every episode boundary")
        residual = np.asarray(residual, dtype=np.float32)
        if residual.shape != (self.action_dim,) or not np.all(np.isfinite(residual)):
            raise ValueError(f"Residual must be a finite flat vector of size {self.action_dim}")
        if np.any(np.abs(residual) > 1.0 + 1e-6):
            raise ValueError("Normalized residual must be in [-1,1]")
        residual = np.clip(residual, -1.0, 1.0)
        corrections = np.zeros((self.execute_steps, self.motor_dim), dtype=np.float32)
        corrections[:, self.active_axes] = residual.reshape(self.execute_steps, -1) * self.residual_scale[self.active_axes]
        requested = self._base_chunk[:self.execute_steps] + corrections
        projected = np.clip(requested, self._low, self._high)
        rewards, actions = [], []
        terminated = truncated = False
        # If the underlying task raises after advancing, prevent accidental retry.
        self._needs_reset = True
        for action in projected:
            raw, reward, terminated, truncated, info = self.env.step(action.astype(self.env.action_space.dtype))
            if "final_observation" in info or "final_obs" in info:
                raise ValueError("Environment auto-reset detected; disable it to preserve final observations")
            if not np.isfinite(reward):
                raise ValueError("Environment returned a nonfinite reward")
            rewards.append(float(reward))
            actions.append(action.copy())
            if terminated or truncated:
                break
        k = len(rewards)
        observation = self._decision_observation(raw, terminal=bool(terminated))
        self._needs_reset = bool(terminated or truncated)
        result_info = dict(info)
        result_info.update({
            "discount": self.gamma ** k,
            "executed_steps": k,
            "raw_return": float(sum(rewards)),
            "primitive_rewards": np.asarray(rewards, dtype=np.float32),
            "requested_residual": residual.copy(),
            "executed_actions": np.asarray(actions, dtype=np.float32),
            "projection_fraction": float(np.mean(np.abs(requested[:k] - projected[:k]) > 1e-7)),
            "is_success": bool(info.get("is_success", False)),
        })
        reward = float(sum(self.gamma ** i * r for i, r in enumerate(rewards)))
        return observation, reward, bool(terminated), bool(truncated), result_info
