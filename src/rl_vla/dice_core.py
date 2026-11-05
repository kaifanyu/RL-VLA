"""DICE residual objectives, adapted from upstream 98d936e (see docs/dice.md).

The prior is supplied as detached samples. No policy log density or SAC entropy
term is used. This module never loads OpenPI or opens an environment.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


def mlp(inputs: int, outputs: int, width: int, *, activation="Mish", layernorm=False):
    layers = []
    for source in (inputs, width, width):
        layers.append(nn.Linear(source, width))
        if layernorm:
            layers.append(nn.LayerNorm(width))
        layers.append({"Mish": nn.Mish, "GELU": nn.GELU}[activation]())
    return nn.Sequential(*layers, nn.Linear(width, outputs))


def regularization_mask(
    q,
    q_prior,
    calibration,
    *,
    step,
    warmup,
    threshold,
    expert,
    mode="replay",
    enabled=True,
    retain_experts=True,
):
    """Relax prior penalty only for value improvement AND calibrated underestimation.

    Replay mode uses Q(s,a_replay)-G [B], as in pinned code. Candidate mode
    uses Q(s,a_new)-G [B,K], as in paper Eq. 6. Gates are detached.
    """
    expected = q.shape if mode == "candidate" else q.shape[:1]
    if mode not in ("candidate", "replay") or calibration.shape != expected:
        raise ValueError("Invalid calibration mode/shape")
    if expert.shape != q.shape[:1]:
        raise ValueError("Source flags must be [B]")
    if not torch.isfinite(calibration).all():
        raise ValueError("Finite empirical returns are required for selective regularization")
    with torch.no_grad():
        if mode == "candidate":  # Paper v2 Eq. 6, including equality.
            relax = (q >= q_prior) & (calibration <= threshold)
        else:  # Official actor_loss uses strict tests and replay-action calibration.
            relax = (q > q_prior) & (calibration[:, None] < threshold)
        if step <= warmup or not enabled:
            relax = torch.zeros_like(relax)
        if retain_experts:
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
    """Noise-conditioned residual actor and independent critic ensemble.

    Inputs are frozen visual/language/proprioceptive features, original full
    OpenPI noise, and normalized *executed* action chunks. Torch stays on CPU.
    """

    def __init__(self, features, noise_dim, action_dim, settings, device="cpu"):
        self.settings = settings
        self.device = torch.device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        width = settings["width"]
        architecture = dict(
            activation=settings.get("activation", "Mish"),
            layernorm=settings.get("layernorm", False),
        )
        self.actor = mlp(features + noise_dim, action_dim, width, **architecture)
        # Start exactly at the pretrained policy rather than random corrections.
        if settings.get("zero_init", True):
            nn.init.zeros_(self.actor[-1].weight)
            nn.init.zeros_(self.actor[-1].bias)
        self.critics = nn.ModuleList(
            [
                mlp(features + action_dim, 1, width, **architecture)
                for _ in range(settings.get("critic_ensemble_size", 2))
            ]
        )
        self.actor.to(self.device)
        self.critics.to(self.device)
        self.target = copy.deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=settings["actor_lr"],
            weight_decay=settings.get("weight_decay", 0.0),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(),
            lr=settings["critic_lr"],
            weight_decay=settings.get("weight_decay", 0.0),
        )
        self.updates = 0
        self.actor_updates = self.target_updates = 0
        self.actor_scheduler = self.critic_scheduler = None
        if settings.get("lr_scheduler", False):
            from rl_vla.dice_scheduler import CosineAnnealingWarmupRestarts

            for name in ("actor", "critic"):
                scheduler = CosineAnnealingWarmupRestarts(
                    getattr(self, name + "_optimizer"),
                    first_cycle_steps=settings["lr_cycle_steps"],
                    warmup_steps=settings["lr_warmup_steps"],
                    min_lr=settings["min_lr"],
                    max_lr=settings[name + "_lr"],
                )
                setattr(self, name + "_scheduler", scheduler)

    def action(self, features, noise, prior):
        return prior.detach() + self.actor(torch.cat((features.detach(), noise.detach()), -1))

    def q(self, features, actions, *, target=False):
        networks = self.target if target else self.critics
        inputs = torch.cat((features.detach(), actions), -1)
        return torch.stack([net(inputs).squeeze(-1) for net in networks], 0)

    @torch.no_grad()
    def select(self, features, noise, prior, *, strategy="max_q_min"):
        """One state's K proposals; preserve the selected action/noise association."""
        features, noise, prior = (x.to(self.device) for x in (features, noise, prior))
        actions = self.action(features, noise, prior)
        values = self.q(features, actions)
        if strategy == "max_q_min":
            index = values.min(0).values.argmax()
        elif strategy == "max_q_std":
            index = values.std(0).argmax()
        else:
            raise ValueError("Unknown selection strategy")
        return actions[index].cpu(), int(index)

    def update(self, batch, proposals, next_proposals, *, step):
        """Eq. 4–8; min across critics BEFORE the latent expectation.

        `step` is a zero-based collection clock for the new profiles. Official
        delayed updates gate on (step+1), including all gradients in that burst.
        Losses see the same pre-update critics, as in upstream loss().
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        proposals = tuple(x.to(self.device) for x in proposals)
        next_proposals = tuple(x.to(self.device) for x in next_proposals)
        s = self.settings
        f, action = batch["features"], batch["actions"]
        valid = batch.get("action_mask", torch.ones_like(action))
        action = action * valid
        with torch.no_grad():
            calibration = self.q(f, action).min(0).values - batch["returns"]
            nf, nz, np = next_proposals
            multi = nf.ndim == 3
            if multi:
                b, k, _ = nf.shape
                nf, nz, np = (x.reshape(b * k, -1) for x in (nf, nz, np))
            next_action = self.action(nf, nz, np)
            next_value = self.q(nf, next_action, target=True).min(0).values
            if multi:
                next_value = next_value.reshape(b, k).mean(1)
            # where also prevents an undefined terminal bootstrap contaminating y.
            target = batch["rewards"] + torch.where(
                batch["discounts"] > 0, batch["discounts"] * next_value, 0.0
            )
        critic_loss = (self.q(f, action) - target[None]).square().mean(1).sum()
        pf, pz, pp = proposals
        b, k, _ = pp.shape
        pf, pz, pp = (x.reshape(b * k, -1) for x in (pf, pz, pp))
        clock = step if s.get("delay_clock", "optimizer") == "collection" else self.updates
        actor_due = (clock + 1) % s.get("actor_update_freq", 1) == 0
        target_due = (clock + 1) % s.get("target_update_freq", 1) == 0
        self.critics.requires_grad_(False)
        try:
            proposed = self.action(pf, pz, pp)
            proposal_mask = valid[:, None].expand(-1, k, -1).reshape(b * k, -1)
            q = self.q(pf, proposed * proposal_mask).min(0).values.reshape(b, k)
            with torch.no_grad():
                qp = self.q(pf, pp * proposal_mask).min(0).values.reshape(b, k)
            mode = s.get("calibration_mode", "replay")
            gate_calibration = (
                q.detach() - batch["returns"][:, None] if mode == "candidate" else calibration
            )
            mask = regularization_mask(
                q,
                qp,
                gate_calibration,
                step=step,
                warmup=s["filter_warmup"],
                threshold=s["underestimation_threshold"],
                expert=batch["expert"],
                mode=mode,
                enabled=s.get("filter_enabled", True),
                retain_experts=s.get("retain_expert_regularization", True),
            )
            # Official implementation averages over H and motor dimensions.
            penalty = (
                ((proposed - pp).square() * proposal_mask).sum(-1)
                / proposal_mask.sum(-1).clamp_min(1)
            ).reshape(b, k)
            online = (
                (1 - batch["expert"]) if s.get("disable_expert_actor_q", False) else torch.ones(b, device=self.device)
            )
            q_loss = -(q.mean(1) * online).mean()
            if s.get("q_normalization", False):
                count = online.sum() * k
                scale = (
                    (q.detach().abs() * online[:, None]).sum() / count
                    if count > 0
                    else q.detach().abs().mean()
                )
                if scale > 1e-8:
                    q_loss = q_loss / scale
            actor_loss = q_loss + s["bc_weight"] * (mask * penalty).mean()
            if actor_due:
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), s["max_grad_norm"])
                self.actor_optimizer.step()
                self.actor_updates += 1
                if self.actor_scheduler:
                    self.actor_scheduler.step()
        finally:
            self.critics.requires_grad_(True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), s["max_grad_norm"])
        self.critic_optimizer.step()
        if self.critic_scheduler:
            self.critic_scheduler.step()
        if target_due:
            with torch.no_grad():
                for target_parameter, parameter in zip(
                    self.target.parameters(), self.critics.parameters(), strict=True
                ):
                    target_parameter.lerp_(parameter, s["target_tau"])
            self.target_updates += 1
        self.updates += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "regularization_fraction": float(mask.mean()),
            "q_minus_return": float(calibration.mean()),
            "target_mean": float(target.mean()),
            "updates": self.updates,
            "actor_updates": self.actor_updates,
            "target_updates": self.target_updates,
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target": self.target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.updates,
            "actor_updates": self.actor_updates,
            "target_updates": self.target_updates,
            "actor_scheduler": self.actor_scheduler.state_dict() if self.actor_scheduler else None,
            "critic_scheduler": self.critic_scheduler.state_dict()
            if self.critic_scheduler
            else None,
        }

    def load_state_dict(self, value):
        for name in ("actor", "critics", "target", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(value[name])
        self.updates = int(value["updates"])
        self.actor_updates = int(value.get("actor_updates", self.updates))
        self.target_updates = int(value.get("target_updates", self.updates))
        for name in ("actor_scheduler", "critic_scheduler"):
            if getattr(self, name) is not None:
                if value.get(name) is None:
                    raise ValueError("Missing scheduler state")
                getattr(self, name).load_state_dict(value[name])
