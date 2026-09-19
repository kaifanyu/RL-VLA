"""Build small, deployable actor inputs and optional privileged critic inputs."""

from collections.abc import Mapping

import numpy as np


def finite_vector(value, name: str) -> np.ndarray:
    """Read a finite one-dimensional vector without silently flattening images."""
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return vector.copy()


class ObservationEncoder:
    """Fixed preprocessing: no running statistics change PPO likelihood inputs.

    The external environment must provide deployable ``state`` and may provide
    deployable, precomputed ``features``. Images and ``openpi`` inputs are passed
    to the frozen policy separately. Only the critic sees ``critic_state``.
    """

    def __init__(self, config: dict):
        self.state_mean = config.get("state_mean", 0.0)
        self.state_std = config.get("state_std", 1.0)
        self._signature = None

    def encode(self, raw_observation: Mapping, base_prefix: np.ndarray) -> dict:
        if not isinstance(raw_observation, Mapping) or "state" not in raw_observation:
            raise ValueError("Environment observations must be mappings with deployable 'state'")
        state = finite_vector(raw_observation["state"], "state")
        if state.size == 0:
            raise ValueError("state must not be empty")
        mean = np.asarray(self.state_mean, dtype=np.float32)
        std = np.asarray(self.state_std, dtype=np.float32)
        for name, vector in (("state_mean", mean), ("state_std", std)):
            if vector.shape not in ((), state.shape) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be finite and scalar or match state shape {state.shape}")
        if np.any(std <= 0):
            raise ValueError("state_std must be strictly positive")
        features = finite_vector(raw_observation.get("features", []), "features")
        privileged = finite_vector(raw_observation.get("critic_state", []), "critic_state")
        signature = (state.size, features.size, privileged.size)
        if self._signature is not None and signature != self._signature:
            raise ValueError(f"Observation dimensions changed: {self._signature} -> {signature}")
        self._signature = signature
        actor = np.concatenate(((state - mean) / std, features, base_prefix.reshape(-1)))
        critic = np.concatenate((actor, privileged))
        if not np.all(np.isfinite(critic)):
            raise ValueError("Observation normalization produced nonfinite values")
        return {"obs": actor.astype(np.float32), "critic_obs": critic.astype(np.float32)}
