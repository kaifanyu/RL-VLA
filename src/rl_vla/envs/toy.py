"""Checkpoint-free CPU reaching task for pipeline checks, not a robot benchmark."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ToyReachEnv(gym.Env):
    """Move a 2D point to a goal with bounded velocity commands.

    Success is verified by distance and terminates the episode. The optional
    bounded potential is Phi=-min(distance/2, 1), with Phi=0 at true termination.
    ``gamma`` must match the primitive-step discount in ResidualChunkEnv.
    The time limit is a collection truncation, so its potential is not zeroed.
    """

    metadata: dict = {"render_modes": []}  # noqa: RUF012 - Gymnasium class metadata

    def __init__(self, max_steps=60, gamma=0.99, shaping_scale=0.1,
                 step_size=0.08, success_radius=0.08, base_gain=0.65):
        super().__init__()
        self.max_steps = int(max_steps)
        self.gamma = float(gamma)
        self.shaping_scale = float(shaping_scale)
        self.step_size = float(step_size)
        self.success_radius = float(success_radius)
        self.base_gain = float(base_gain)
        parameters = (self.gamma, self.shaping_scale, self.step_size,
                      self.success_radius, self.base_gain)
        if not all(np.isfinite(x) for x in parameters):
            raise ValueError("Toy parameters must be finite")
        if self.max_steps < 1 or not 0 < self.gamma <= 1 or self.step_size <= 0:
            raise ValueError("Require max_steps>=1, gamma in (0,1], and step_size>0")
        if self.success_radius <= 0 or self.shaping_scale < 0 or self.base_gain < 0:
            raise ValueError("Require success_radius>0 and shaping_scale/base_gain>=0")
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "state": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
            "mock_base_action": spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        })
        self._needs_reset = True

    def _observation(self):
        command = np.clip(self.base_gain * (self.goal - self.position), -1, 1)
        return {"state": np.concatenate((self.position, self.goal)).astype(np.float32),
                "mock_base_action": command.astype(np.float32)}

    def _potential(self):
        return -min(float(np.linalg.norm(self.goal - self.position)) / 2.0, 1.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.position = np.asarray(options.get("position", self.np_random.uniform(-0.7, 0.7, 2)),
                                   dtype=np.float32).copy()
        self.goal = np.asarray(options.get("goal", self.np_random.uniform(-0.7, 0.7, 2)),
                               dtype=np.float32).copy()
        if any(x.shape != (2,) or not np.all(np.isfinite(x)) for x in (self.position, self.goal)):
            raise ValueError("Toy reset position and goal must be finite 2D vectors")
        self.steps = 0
        self._needs_reset = False
        return self._observation(), {"is_success": False}

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping the toy environment")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Toy action must be a finite 2D vector in [-1,1]")
        before = self._potential()
        self.position += self.step_size * action
        self.steps += 1
        distance = float(np.linalg.norm(self.goal - self.position))
        success = distance <= self.success_radius
        terminated = bool(success)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        after = 0.0 if terminated else self._potential()
        reward = float(success) + self.shaping_scale * (self.gamma * after - before)
        self._needs_reset = terminated or truncated
        return self._observation(), reward, terminated, truncated, {
            "is_success": success, "distance": distance, "sparse_reward": float(success),
            "shaping_reward": self.shaping_scale * (self.gamma * after - before),
        }
