"""On-policy residual PPO using stored nominal chunks and pre-tanh samples."""

from __future__ import annotations

import torch
from torch import Tensor

from .actors import Actor
from .critics import Value


@torch.no_grad()
def compute_gae(
    reward: Tensor, discount: Tensor, value: Tensor, next_value: Tensor,
    terminated: Tensor, truncated: Tensor, gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Return advantages/returns for ONE chronological trajectory stream [T].

    Bootstrap truncations from their final pre-reset observation, while stopping
    traces at every reset and at the end of this buffer. Lambda is per decision;
    discount already contains gamma**actual_executed_primitive_steps.
    """
    if reward.ndim != 1 or not all(
        field.shape == reward.shape for field in (discount, value, next_value, terminated, truncated)
    ):
        raise ValueError("GAE expects equally shaped scalar sequences [T], not interleaved environments")
    advantages = torch.zeros_like(reward)
    trace = torch.zeros((), device=reward.device, dtype=reward.dtype)
    for index in reversed(range(len(reward))):
        bootstrap = (~terminated[index].bool()).to(reward.dtype)
        continue_trace = (~(terminated[index].bool() | truncated[index].bool())).to(reward.dtype)
        delta = reward[index] + discount[index] * bootstrap * next_value[index] - value[index]
        trace = delta + discount[index] * gae_lambda * continue_trace * trace
        advantages[index] = trace
    return advantages, advantages + value


class PPO:
    """Single-stream rollout learner. Scalar fields [T], vectors [T,D].

    Store the exact actor input, nominal chunk, and pre_tanh sample at collection.
    Never resample the base chunk or substitute a new visual augmentation when
    computing PPO's likelihood ratio. Normalization uses the complete rollout.
    """

    def __init__(self, obs_dim: int, critic_dim: int, action_dim: int, config: dict, device="cpu"):
        self.device = torch.device(device)
        self.config = dict(config)
        hidden = int(config.get("hidden_dim", 128))
        self.actor = Actor(obs_dim, action_dim, hidden).to(self.device)
        self.value = Value(critic_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam([
            {"params": self.actor.parameters(), "lr": config.get("actor_lr", 1e-4)},
            {"params": self.value.parameters(), "lr": config.get("critic_lr", 3e-4)},
        ])
        self.updates = 0

    def update(self, rollout: dict[str, Tensor]) -> dict[str, float]:
        data = {key: torch.as_tensor(value, device=self.device).detach().float() for key, value in rollout.items()}
        scalar_keys = ("reward", "discount", "value", "next_value", "terminated", "truncated", "logp")
        for key in scalar_keys:
            if data[key].ndim == 2 and data[key].shape[-1] == 1:
                data[key] = data[key].squeeze(-1)
        advantages, returns = compute_gae(
            data["reward"], data["discount"], data["value"], data["next_value"],
            data["terminated"], data["truncated"], float(self.config.get("gae_lambda", 0.95)),
        )
        count = len(advantages)
        if count == 0:
            raise ValueError("PPO rollout cannot be empty")
        if count > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        clip_range = float(self.config.get("clip_range", 0.2))
        entropy_coef = float(self.config.get("entropy_coef", 0.0))
        value_coef = float(self.config.get("value_coef", 0.5))
        target_kl = self.config.get("target_kl", 0.02)
        minibatch_size = int(self.config.get("minibatch_size", self.config.get("batch_size", 64)))
        epochs = int(self.config.get("epochs", 4))
        if minibatch_size < 1 or epochs < 1:
            raise ValueError("PPO epochs and minibatch_size must be positive")
        parameters = list(self.actor.parameters()) + list(self.value.parameters())
        measurements = []
        measured_kls = []
        early_stop = False
        for _ in range(epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, minibatch_size):
                indices = permutation[start:start + minibatch_size]
                logp = self.actor.log_prob(data["obs"][indices], data["pre_tanh"][indices])
                log_ratio = logp - data["logp"][indices]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                measured_kls.append(float(approximate_kl))
                if target_kl is not None and approximate_kl > 1.5 * float(target_kl):
                    early_stop = True
                    break
                policy_loss = -torch.minimum(
                    ratio * advantages[indices],
                    ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages[indices],
                ).mean()
                predicted_value = self.value(data["critic_obs"][indices])
                value_error = (predicted_value - returns[indices]).square()
                value_clip = self.config.get("value_clip_range")
                if value_clip is not None:
                    clipped_value = data["value"][indices] + (
                        predicted_value - data["value"][indices]
                    ).clamp(-float(value_clip), float(value_clip))
                    value_error = torch.maximum(value_error, (clipped_value - returns[indices]).square())
                value_loss = 0.5 * value_error.mean()
                # This entropy is a fresh differentiable sample from the current
                # transformed distribution, not cross-entropy of old actions.
                _, entropy_logp, _ = self.actor.sample(data["obs"][indices])
                entropy = -entropy_logp.mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(self.config.get("max_grad_norm", 0.5))
                )
                self.optimizer.step()
                measurements.append({
                    "policy_loss": float(policy_loss.detach()), "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()), "approx_kl": float(approximate_kl),
                    "clip_fraction": float(((ratio.detach() - 1).abs() > clip_range).float().mean()),
                    "grad_norm": float(grad_norm),
                })
            if early_stop:
                break
        self.updates += 1
        metrics = {
            name: sum(measure[name] for measure in measurements) / max(len(measurements), 1)
            for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")
        }
        with torch.no_grad():
            return_variance = returns.var(unbiased=False)
            residual_variance = (returns - self.value(data["critic_obs"])).var(unbiased=False)
            explained_variance = 1.0 - residual_variance / return_variance if return_variance > 1e-8 else 0.0
        metrics.update({
            "explained_variance": float(explained_variance), "early_stop": float(early_stop),
            "optimizer_steps": float(len(measurements)), "updates": float(self.updates),
            "approx_kl": sum(measured_kls) / len(measured_kls), "max_kl": max(measured_kls),
        })
        return metrics

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(), "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(), "updates": self.updates, "config": self.config,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.value.load_state_dict(state["value"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])
        self.config = dict(state["config"])
