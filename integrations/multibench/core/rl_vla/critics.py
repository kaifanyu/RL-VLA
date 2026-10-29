"""Small state/chunk critics, optionally conditioned on privileged observations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
    )


class TwinQ(nn.Module):
    """Independent Q functions on critic context and normalized residual v."""

    def __init__(self, critic_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.q1 = _mlp(critic_dim + action_dim, hidden_dim)
        self.q2 = _mlp(critic_dim + action_dim, hidden_dim)

    def forward(self, critic_obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        features = torch.cat((critic_obs, action), dim=-1)
        return self.q1(features).squeeze(-1), self.q2(features).squeeze(-1)


class Value(nn.Module):
    """Value estimates with shape [B]."""

    def __init__(self, critic_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = _mlp(critic_dim, hidden_dim)

    def forward(self, critic_obs: Tensor) -> Tensor:
        return self.net(critic_obs).squeeze(-1)
