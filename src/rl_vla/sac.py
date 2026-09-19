"""Residual SAC with twin critics, automatic temperature, and macro discounts."""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .actors import Actor
from .critics import TwinQ


@torch.no_grad()
def compute_sac_target(
    reward: Tensor, discount: Tensor, terminated: Tensor,
    next_q: Tensor, next_logp: Tensor, alpha: Tensor | float,
) -> Tensor:
    """A true terminal disables bootstrapping; time-limit truncations do not."""
    return reward + discount * (~terminated.bool()).float() * (next_q - alpha * next_logp)


class SAC:
    """Learner for scalar fields [B] and vector fields [B,D].

    critic_obs may contain privileged simulator information; obs must contain
    only information available to the deployed residual actor. Both must include
    their respective context for the exact nominal chunk used for this decision.
    The entropy target defaults to -D in normalized residual coordinates.
    """

    def __init__(self, obs_dim: int, critic_dim: int, action_dim: int, config: dict, device="cpu"):
        self.device = torch.device(device)
        self.config = dict(config)
        hidden = int(config.get("hidden_dim", 128))
        self.actor = Actor(obs_dim, action_dim, hidden).to(self.device)
        self.critic = TwinQ(critic_dim, action_dim, hidden).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.get("actor_lr", 1e-4))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.get("critic_lr", 3e-4))
        initial_alpha = float(config.get("initial_alpha", 0.1))
        if initial_alpha <= 0:
            raise ValueError("initial_alpha must be positive")
        self.log_alpha = torch.tensor(math.log(initial_alpha), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.get("alpha_lr", 3e-4))
        self.target_entropy = float(config.get("target_entropy", -action_dim))
        self.tau = float(config.get("tau", 0.005))
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0,1]")
        self.updates = 0

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.detach().exp()

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        batch = {key: torch.as_tensor(value, device=self.device).detach().float() for key, value in batch.items()}
        reward = batch["reward"].reshape(-1)
        discount = batch["discount"].reshape(-1)
        terminated = batch["terminated"].reshape(-1)
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(batch["next_obs"])
            next_q1, next_q2 = self.target_critic(batch["next_critic_obs"], next_action)
            target = compute_sac_target(reward, discount, terminated,
                                        torch.minimum(next_q1, next_q2), next_logp, self.alpha)
        q1, q2 = self.critic(batch["critic_obs"], batch["action"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Q remains differentiable with respect to the action but receives no
        # parameter gradients during the actor update.
        self.critic.requires_grad_(False)
        action, logp, _ = self.actor.sample(batch["obs"])
        actor_q1, actor_q2 = self.critic(batch["critic_obs"], action)
        actor_loss = (self.alpha * logp - torch.minimum(actor_q1, actor_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        self.updates += 1
        return {
            "critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()), "alpha": float(self.alpha),
            "entropy": float(-logp.detach().mean()), "target_q": float(target.mean()),
            "q": float(torch.minimum(q1, q2).detach().mean()), "updates": float(self.updates),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(), "updates": self.updates,
            "config": self.config, "target_entropy": self.target_entropy, "tau": self.tau,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.updates = int(state["updates"])
        self.config = dict(state["config"])
        self.target_entropy = float(state["target_entropy"])
        self.tau = float(state["tau"])
