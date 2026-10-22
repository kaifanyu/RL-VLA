"""DICE residual objectives, adapted from upstream 98d936e (see docs/dice.md).

The prior is supplied as detached samples. No policy log density or SAC entropy
term is used. This module never loads OpenPI or opens an environment.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


def mlp(inputs: int, outputs: int, width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, width),
        nn.Mish(),
        nn.Linear(width, width),
        nn.Mish(),
        nn.Linear(width, width),
        nn.Mish(),
        nn.Linear(width, outputs),
    )


def regularization_mask(q, q_prior, calibration, *, step, warmup, threshold, expert):
    """Relax prior penalty only for value improvement AND calibrated underestimation.

    calibration is Q(s,a_replay) - empirical discounted return, not Q(s,a_new).
    q/q_prior are [B,K]; calibration/expert are [B]. Gates are detached.
    """
    if calibration.shape != q.shape[:1] or expert.shape != calibration.shape:
        raise ValueError("Calibration and source flags must be [B]")
    if not torch.isfinite(calibration).all():
        raise ValueError("Finite empirical returns are required for selective regularization")
    with torch.no_grad():
        relax = (q > q_prior) & (calibration[:, None] < threshold)
        if step <= warmup:
            relax = torch.zeros_like(relax)
        relax &= ~expert.bool()[:, None]
        return (~relax).to(q.dtype)


def discounted_returns(rewards, discounts):
    """Finite-episode MC return; discounts are gamma**actual primitive steps."""
    if len(rewards) != len(discounts):
        raise ValueError("Rewards and discounts must have equal length")
    result, running = [], 0.0
    for reward, discount in zip(reversed(rewards), reversed(discounts), strict=True):
        running = float(reward) + float(discount) * running
        result.append(running)
    return list(reversed(result))


class DICE:
    """Noise-conditioned residual actor and independent double-Q ensemble.

    Inputs are frozen visual/language/proprioceptive features, original full
    OpenPI noise, and normalized *executed* action chunks. Torch stays on CPU.
    """

    def __init__(self, features, noise_dim, action_dim, settings):
        self.settings = settings
        width = settings["width"]
        self.actor = mlp(features + noise_dim, action_dim, width)
        # Start exactly at the pretrained policy rather than random corrections.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.critics = nn.ModuleList([mlp(features + action_dim, 1, width) for _ in range(2)])
        self.target = copy.deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=settings["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(), lr=settings["critic_lr"]
        )
        self.updates = 0

    def action(self, features, noise, prior):
        return prior.detach() + self.actor(torch.cat((features.detach(), noise.detach()), -1))

    def q(self, features, actions, *, target=False):
        networks = self.target if target else self.critics
        inputs = torch.cat((features.detach(), actions), -1)
        return torch.stack([net(inputs).squeeze(-1) for net in networks], 0)

    @torch.no_grad()
    def select(self, features, noise, prior, *, strategy="max_q_min"):
        """One state's K proposals; preserve the selected action/noise association."""
        actions = self.action(features, noise, prior)
        values = self.q(features, actions)
        if strategy == "max_q_min":
            index = values.min(0).values.argmax()
        elif strategy == "max_q_std":
            index = values.std(0).argmax()
        else:
            raise ValueError("Unknown selection strategy")
        return actions[index], int(index)

    def update(self, batch, proposals, next_proposals, *, step):
        """Numerical optimizer update; caller owns replay sampling and frozen prior.

        proposals contain [B,K,F/Z/A]; next proposals have one sample per state.
        MC calibration is computed before updating the critic, as in upstream.
        """
        s = self.settings
        f, action = batch["features"], batch["actions"]
        with torch.no_grad():
            calibration = self.q(f, action).min(0).values - batch["returns"]
            nf, nz, np = next_proposals
            next_action = self.action(nf, nz, np)
            target = (
                batch["rewards"]
                + batch["discounts"] * self.q(nf, next_action, target=True).min(0).values
            )
        critic_loss = (self.q(f, action) - target[None]).square().mean(1).sum()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), s["max_grad_norm"])
        self.critic_optimizer.step()
        pf, pz, pp = proposals
        b, k, _ = pp.shape
        pf, pz, pp = (x.reshape(b * k, -1) for x in (pf, pz, pp))
        self.critics.requires_grad_(False)
        try:
            proposed = self.action(pf, pz, pp)
            q = self.q(pf, proposed).min(0).values.reshape(b, k)
            with torch.no_grad():
                qp = self.q(pf, pp).min(0).values.reshape(b, k)
            mask = regularization_mask(
                q,
                qp,
                calibration,
                step=step,
                warmup=s["filter_warmup"],
                threshold=s["underestimation_threshold"],
                expert=batch["expert"],
            )
            penalty = (proposed - pp).square().mean(-1).reshape(b, k)
            actor_loss = -q.mean() + s["bc_weight"] * (mask * penalty).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), s["max_grad_norm"])
            self.actor_optimizer.step()
        finally:
            self.critics.requires_grad_(True)
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target.parameters(), self.critics.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, s["target_tau"])
        self.updates += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "regularization_fraction": float(mask.mean()),
            "q_minus_return": float(calibration.mean()),
            "updates": self.updates,
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target": self.target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, value):
        for name in ("actor", "critics", "target", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(value[name])
        self.updates = int(value["updates"])
