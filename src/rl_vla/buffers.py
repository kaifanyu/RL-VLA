"""Lazy, bounded replay for compact features and normalized residual actions."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Uniform replay allocated one transition at a time, with owned CPU copies.

    Store compact actor/critic features (including the sampled nominal chunk),
    not full image streams. Scalar transition fields become tensors of [B] on
    sampling. Feature/action fields become [B,D]. Samples are without replacement.
    """

    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._data: list[dict[str, np.ndarray]] = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._data)

    def add(self, **transition) -> None:
        if not transition:
            raise ValueError("A replay transition cannot be empty")
        owned = {}
        for name, value in transition.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.dtype.kind not in "biuf":
                raise TypeError(f"Replay field {name!r} must be numeric")
            owned[name] = array.copy()
        if self._data:
            template = self._data[0]
            if owned.keys() != template.keys():
                raise ValueError("All replay transitions must have the same fields")
            for name in template:
                if owned[name].shape != template[name].shape:
                    raise ValueError(f"Replay field {name!r} changed shape")
        if len(self._data) < self.capacity:
            self._data.append(owned)
        else:
            self._data[self._position] = owned
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        if batch_size <= 0 or batch_size > len(self):
            raise ValueError("Batch size must be positive and no greater than replay size")
        indices = self.rng.choice(len(self), size=batch_size, replace=False)
        batch = {}
        for name in self._data[0]:
            array = np.stack([self._data[int(index)][name] for index in indices])
            tensor = torch.as_tensor(array, device=device)
            batch[name] = tensor.float() if tensor.is_floating_point() else tensor
        return batch
