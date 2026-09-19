"""Frozen nominal chunk sources; no learner gradients cross this interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np


class FrozenBasePolicy(Protocol):
    """A single-environment nominal policy in the environment's action units."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Return a finite [H, action_dim] chunk, with H >= configured horizon."""
        ...

    def reset(self) -> None: ...


def _dimensions(action_dim: int, horizon: int) -> None:
    if action_dim < 1 or horizon < 1:
        raise ValueError("action_dim and horizon must both be positive")


def _metadata_value(value: Any) -> Any:
    """Keep upstream NumPy metadata safe for a JSON experiment manifest."""
    if isinstance(value, np.ndarray):
        return _metadata_value(value.tolist())
    if isinstance(value, np.generic):
        return _metadata_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_metadata_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _validate_chunk(chunk: Any, action_dim: int, horizon: int) -> np.ndarray:
    actions = np.asarray(chunk, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < horizon or actions.shape[1] != action_dim:
        raise ValueError(
            f"Base policy must return [H >= {horizon}, {action_dim}], got {actions.shape}. "
            "Check the checkpoint config and action mapping; padded coordinates are not trimmed automatically."
        )
    if not np.isfinite(actions).all():
        raise ValueError("Base policy returned non-finite actions")
    return np.array(actions, dtype=np.float32, copy=True, order="C")


class MockBasePolicy:
    """Repeat the toy environment's nominal action; never loads a VLA."""

    def __init__(self, action_dim: int, horizon: int) -> None:
        _dimensions(action_dim, horizon)
        self.action_dim = action_dim
        self.horizon = horizon

    @property
    def metadata(self) -> dict[str, Any]:
        return {"kind": "mock", "action_dim": self.action_dim, "horizon": self.horizon}

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        action = np.asarray(observation["mock_base_action"], dtype=np.float32)
        if action.shape != (self.action_dim,):
            raise ValueError(f"mock_base_action must have shape ({self.action_dim},), got {action.shape}")
        return _validate_chunk(np.repeat(action[None, :], self.horizon, axis=0), self.action_dim, self.horizon)

    def reset(self) -> None:
        pass


class OpenPIBasePolicy:
    """Use upstream's websocket client in an environment separate from openpi."""

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        *,
        host: str = "localhost",
        port: int = 8000,
        action_key: str = "actions",
        action_indices: Sequence[int] | None = None,
        observation_map: Mapping[str, str] | None = None,
        client: Any = None,
    ) -> None:
        _dimensions(action_dim, horizon)
        self.action_dim = action_dim
        self.horizon = horizon
        self.action_key = action_key
        self.action_indices = tuple(action_indices) if action_indices is not None else None
        self.observation_map = dict(observation_map) if observation_map is not None else None
        if self.action_indices is not None and (
                len(self.action_indices) != action_dim
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.action_indices)
                or len(set(self.action_indices)) != action_dim
        ):
            raise ValueError("action_indices must contain action_dim distinct nonnegative integer indices")
        if self.observation_map is not None and (
            not self.observation_map
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.observation_map.items())
        ):
            raise ValueError("observation_map must map server key strings to environment key strings")
        if client is None:
            try:
                from openpi_client import websocket_client_policy
            except ImportError as exc:
                raise ImportError(
                    "OpenPI client is missing. From the RL-VLA root run "
                    "`uv pip install -e third_party/openpi/packages/openpi-client`; "
                    "see docs/openpi.md for the pinned checkout and separate policy-server setup."
                ) from exc
            client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self._client = client
        self._connection = {"kind": "openpi", "host": host, "port": port}

    @property
    def metadata(self) -> dict[str, Any]:
        return {**self._connection, "server": _metadata_value(self._client.get_server_metadata())}

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        if self.observation_map is not None:
            payload = {server_key: observation[env_key] for server_key, env_key in self.observation_map.items()}
        else:
            payload = observation.get("openpi", observation)
        if not isinstance(payload, Mapping):
            raise TypeError("The OpenPI observation payload must be a dictionary")
        result = self._client.infer(dict(payload))
        if self.action_key not in result:
            raise KeyError(f"OpenPI response has no {self.action_key!r} action key")
        actions = np.asarray(result[self.action_key], dtype=np.float32)
        if self.action_indices is not None:
            if actions.ndim != 2 or max(self.action_indices) >= actions.shape[1]:
                raise ValueError(f"action_indices cannot select coordinates from returned shape {actions.shape}")
            actions = actions[:, self.action_indices]
        return _validate_chunk(actions, self.action_dim, self.horizon)

    def reset(self) -> None:
        # Upstream's websocket reset is a no-op. The collector owns chunk/episode state.
        self._client.reset()


def make_base_policy(config: dict[str, Any], action_dim: int, horizon: int) -> FrozenBasePolicy:
    """Construct a nominal policy. Loading this module never downloads weights."""
    kind = config.get("kind", "mock")
    if kind == "mock":
        return MockBasePolicy(action_dim, horizon)
    if kind == "openpi":
        return OpenPIBasePolicy(
            action_dim,
            horizon,
            host=config.get("host", "localhost"),
            port=config.get("port", 8000),
            action_key=config.get("action_key", "actions"),
            action_indices=config.get("action_indices"),
            observation_map=config.get("observation_map"),
        )
    raise ValueError(f"Unknown base policy kind {kind!r}; choose 'mock' or 'openpi'")
