"""Pure NumPy residual-action semantics, actor observations and replay storage.

Action coordinates (LIBERO / robosuite OSC_POSE, 7-D, all normalized to [-1, 1]):
  0-2  end-effector translation delta command (x, y, z); controller output_max 0.05 m per step
  3-5  end-effector axis-angle rotation delta command; controller output_max 0.5 rad per step
  6    gripper command: -1 = open, +1 = close (continuous, PandaGripper; not binarized)
pi0.5 outputs are first unnormalized by the checkpoint's own output transforms and
then combined here, in these environment coordinates:

  executed = clip(base + scale * v, action_low, action_high),   v in [-1, 1] (tanh actor)

A zero scale freezes that coordinate: it is excluded from the actor's action
dimension and the base value passes through (still subject to the same clip
as the base-only pipeline). The default gripper scale is 0.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def noise_seed(episode_key: str, call: int) -> int:
    """Identical to ``rl_vla.baseline.noise_seed`` (the robustness queue's protocol)."""
    text = f"rl-vla-evaluation-v1:{episode_key}:{call}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def episode_seed(case_id: str, index: int) -> int:
    """Identical to ``rl_vla.robustness.episode_seed``."""
    return int.from_bytes(hashlib.sha256(f"{case_id}:{index}".encode()).digest()[:4], "little") % (2**31 - 1)


def original_seed(suite_index: int, task_id: int, initial_state_id: int, base: int = 100_000) -> int:
    """Identical to ``rl_vla.baseline`` original-suite seeds (default --seed 100000)."""
    return base + suite_index * 100_000 + task_id * 1000 + initial_state_id


def derived_seed(*parts) -> int:
    text = ":".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % (2**31 - 1)


@dataclass(frozen=True)
class Combination:
    base: np.ndarray          # [7] unnormalized base action (env coordinates)
    residual: np.ndarray      # [7] actor output v in [-1,1]; zeros on inactive dims
    correction: np.ndarray    # [7] scale * v, before clipping
    requested: np.ndarray     # [7] base + correction, before clipping
    executed: np.ndarray      # [7] clipped action sent to the simulator
    clipped_low: np.ndarray   # [7] bool
    clipped_high: np.ndarray  # [7] bool


class ActionSpec:
    def __init__(self, scale, low, high):
        self.scale = np.asarray(scale, np.float32)
        self.low = np.asarray(low, np.float32)
        self.high = np.asarray(high, np.float32)
        if self.scale.shape != (7,) or self.low.shape != (7,) or self.high.shape != (7,):
            raise ValueError("scale/low/high must have seven entries")
        if np.any(self.scale < 0) or np.any(self.low >= self.high):
            raise ValueError("Invalid residual scale or bounds")
        self.active = np.flatnonzero(self.scale > 0)
        if not self.active.size:
            raise ValueError("At least one residual dimension must be active")

    @property
    def action_dim(self) -> int:
        return int(self.active.size)

    def combine(self, base, v_active=None) -> Combination:
        base = np.asarray(base, np.float32)
        if base.shape != (7,) or not np.isfinite(base).all():
            raise ValueError("Base action must be a finite 7-vector")
        residual = np.zeros(7, np.float32)
        if v_active is not None:
            v = np.asarray(v_active, np.float32).reshape(-1)
            if v.shape != (self.action_dim,) or not np.isfinite(v).all() or np.any(np.abs(v) > 1 + 1e-6):
                raise ValueError(f"Residual must be finite in [-1,1] with shape ({self.action_dim},)")
            residual[self.active] = np.clip(v, -1.0, 1.0)
        correction = (self.scale * residual).astype(np.float32)
        requested = (base + correction).astype(np.float32)
        executed = np.clip(requested, self.low, self.high).astype(np.float32)
        return Combination(base.copy(), residual, correction, requested, executed,
                           requested < self.low, requested > self.high)

    def describe(self) -> dict:
        from .spec import ACTION_NAMES

        return {"names": list(ACTION_NAMES), "scale": self.scale.tolist(), "low": self.low.tolist(),
                "high": self.high.tolist(), "active_dims": self.active.tolist(),
                "formula": "executed = clip(base + scale * v, low, high), v = tanh(u) in [-1,1]",
                "units": {"x,y,z": "normalized OSC delta; 1.0 -> 0.05 m per 0.05 s control step",
                          "rx,ry,rz": "normalized OSC axis-angle delta; 1.0 -> 0.5 rad per control step",
                          "gripper": "-1 open, +1 close (continuous command)"},
                "gripper_handling": ("residual scale 0 by default: gripper = clip(base gripper); the "
                                     "base value is never binarized or overridden"
                                     if self.scale[6] == 0 else "gripper residual enabled (scale > 0)")}


class ObservationBuilder:
    """Deployable actor/critic input. No object poses, IDs or other simulator state.

    Layout: [normalized proprio (8) | current base action (7) | chunk position one-hot
    (execute_steps) | frozen pi0.5 features (optional, per-vector standardized)].
    Proprio = eef position (3), eef axis-angle (3), gripper joint positions (2), the
    same 8-D state pi0.5 receives, normalized with the checkpoint's frozen quantile
    statistics. Features are the masked-mean final VLM prefix hidden state computed
    at the most recent base inference (fixed preprocessing, no running statistics).
    """

    def __init__(self, state_q01, state_q99, execute_steps: int, feature_dim: int = 0):
        self.q01 = np.asarray(state_q01, np.float32)[:8]
        self.q99 = np.asarray(state_q99, np.float32)[:8]
        self.execute_steps = int(execute_steps)
        self.feature_dim = int(feature_dim)
        self.dim = 8 + 7 + self.execute_steps + self.feature_dim

    def layout(self) -> dict:
        offset, parts = 0, {}
        for name, size in (("proprio_quantile_normalized", 8), ("base_action", 7),
                           ("chunk_position_onehot", self.execute_steps), ("vlm_features", self.feature_dim)):
            parts[name] = [offset, offset + size]
            offset += size
        return {"dim": self.dim, "slices": parts, "privileged_inputs": "none",
                "critic_inputs": "identical to actor inputs (no privileged critic state)"}

    def build(self, state, base_action, position: int, features=None) -> np.ndarray:
        state = np.asarray(state, np.float32)
        normalized = (state - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0
        onehot = np.zeros(self.execute_steps, np.float32)
        onehot[int(position)] = 1.0
        parts = [normalized, np.asarray(base_action, np.float32), onehot]
        if self.feature_dim:
            f = np.asarray(features, np.float32).reshape(-1)
            if f.size != self.feature_dim:
                raise ValueError(f"Expected {self.feature_dim} features, got {f.size}")
            parts.append((f - f.mean()) / np.sqrt(f.var() + 1e-5))
        vector = np.concatenate(parts).astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("Non-finite actor observation")
        return vector


class ArrayReplay:
    """Preallocated uniform ring buffer of single-environment-step transitions."""

    FIELDS = ("obs", "action", "reward", "discount", "next_obs", "terminated", "truncated")

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0):
        self.capacity, self.obs_dim, self.action_dim = int(capacity), int(obs_dim), int(action_dim)
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((self.capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), np.float32)
        self.action = np.zeros((self.capacity, action_dim), np.float32)
        self.reward = np.zeros(self.capacity, np.float32)
        self.discount = np.zeros(self.capacity, np.float32)
        self.terminated = np.zeros(self.capacity, np.float32)
        self.truncated = np.zeros(self.capacity, np.float32)
        self.size = 0
        self.position = 0

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, discount, next_obs, terminated, truncated):
        i = self.position
        self.obs[i], self.action[i], self.next_obs[i] = obs, action, next_obs
        self.reward[i], self.discount[i] = reward, discount
        self.terminated[i], self.truncated[i] = float(terminated), float(truncated)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device="cpu") -> dict:
        import torch

        if not 0 < batch_size <= self.size:
            raise ValueError("Batch size must be positive and no greater than replay size")
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        batch = {name: torch.as_tensor(getattr(self, name)[idx], device=device) for name in self.FIELDS}
        batch["critic_obs"], batch["next_critic_obs"] = batch["obs"], batch["next_obs"]
        return batch

    def save(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        temporary = path.with_name(path.name + ".tmp.npz")
        n = self.size
        np.savez(temporary, size=n, position=self.position, capacity=self.capacity,
                 rng=np.frombuffer(repr(self.rng.bit_generator.state).encode(), np.uint8),
                 **{name: getattr(self, name)[:n] for name in self.FIELDS})
        temporary.replace(path)

    def load(self, path) -> None:
        import ast

        with np.load(path, allow_pickle=False) as data:
            if int(data["capacity"]) != self.capacity:
                raise ValueError("Replay capacity changed since the checkpoint")
            n = int(data["size"])
            for name in self.FIELDS:
                getattr(self, name)[:n] = data[name]
            self.size, self.position = n, int(data["position"])
            self.rng.bit_generator.state = ast.literal_eval(bytes(data["rng"]).decode())
