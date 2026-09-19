"""Bounded residual policy; densities are always in normalized v=tanh(u) units."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Actor(nn.Module):
    """Gaussian MLP over a flattened, structurally valid residual action chunk.

    Inputs have shape [B, obs_dim]. Outputs are v/pre_tanh [B, action_dim]
    and logp [B], summed over the complete sampled residual decision. Physical
    scales and safety projections belong to the environment action adapter.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        # Start near the frozen base rather than with uniform motor exploration.
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -3.0)

    def distribution(self, obs: Tensor) -> torch.distributions.Normal:
        features = self.net(obs)
        return torch.distributions.Normal(
            self.mean(features), self.log_std(features).clamp(-5.0, 2.0).exp()
        )

    @staticmethod
    def _log_prob(distribution: torch.distributions.Normal, pre_tanh: Tensor) -> Tensor:
        # Stable even when tanh(u) rounds to exactly +/-1; never invert tanh(v).
        log_jacobian = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        return (distribution.log_prob(pre_tanh) - log_jacobian).sum(dim=-1)

    def sample(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(obs)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        return pre_tanh.tanh(), self._log_prob(distribution, pre_tanh), pre_tanh

    def log_prob(self, obs: Tensor, pre_tanh: Tensor) -> Tensor:
        """Score the stored sample against the exact stored policy input."""
        return self._log_prob(self.distribution(obs), pre_tanh)
