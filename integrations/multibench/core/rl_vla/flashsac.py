"""Residual FlashSAC: upstream FlashSAC networks behind RL-VLA's learner interface.

Reference: Kim, Lee et al., "FlashSAC: Fast and Stable Off-Policy Reinforcement
Learning for High-Dimensional Robot Control", arXiv:2604.04539. The critic
backbone, ensembled double critic, categorical value head, BatchNorm/RMSNorm
layers and unit weight normalization are loaded unmodified from the upstream
checkout (see ``flashsac_vendor``). The update rule follows upstream
``flash_rl/agents/flashSAC/update.py`` with the changes RL-VLA's residual
decision semantics require:

* Per-transition discounts. Each replayed decision carries ``discount = gamma**k``
  for the k commands it actually executed; upstream uses a scalar ``gamma**n``.
  True terminals never bootstrap; truncations do, as in ``sac.py``.
* Residual-safe actor head. The trunk is upstream's (embedder, inverted residual
  blocks, post-RMSNorm, unit-norm output projections), but the mean and log-std
  projections are multiplied by per-dimension gains initialised at zero, so the
  policy starts at mean 0 and std exp(-3) exactly like ``actors.Actor``.
  Upstream's NormalTanhPolicy cannot start at zero because its projections are
  unit-norm by construction.
* Entropy target from a target std in normalized residual units
  (``0.5 * D * log(2*pi*e*sigma**2)``). The default sigma=0.05 ~= exp(-3) keeps
  the initial exploration scale; upstream's 0.15 is for full motor commands.
* Return-based reward scaling uses the decision discount for its running return.
* No noise repetition, AMP or torch.compile by default: decisions already span
  K commands, and these speedups only matter with large parallel batches.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .flashsac_vendor import load_flashsac

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # Same clamp as actors.Actor.


@torch.no_grad()
def normalize_weights(module: nn.Module) -> None:
    """Project every upstream unit layer back onto its norm sphere."""
    for child in module.modules():
        if child is not module and hasattr(child, "normalize_parameters"):
            child.normalize_parameters()


class FlashResidualActor(nn.Module):
    """FlashSAC actor trunk with a zero-initialised residual Gaussian head.

    Same contract as ``actors.Actor``: inputs [B, obs_dim]; ``sample`` returns
    v=tanh(u) [B, D], logp [B] in normalized v units, and pre-tanh u [B, D].
    The trunk uses BatchNorm, so acting (batch size 1) uses running statistics
    (``training=False``); learner updates pass ``training=True``.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_blocks: int,
                 initial_log_std: float = -3.0):
        super().__init__()
        layer = load_flashsac().layer
        self.embedder = layer.FlashSACEmbedder(input_dim=obs_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([layer.FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = layer.UnitRMSNorm(hidden_dim)
        self.mean_w = layer.UnitLinear(hidden_dim, action_dim)
        self.std_w = layer.UnitLinear(hidden_dim, action_dim)
        # Unnormalized scale/offset parameters; zero gains make the initial
        # policy independent of the (random, unit-norm) projections.
        self.mean_gain = nn.Parameter(torch.zeros(action_dim))
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std_gain = nn.Parameter(torch.zeros(action_dim))
        self.log_std_bias = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))

    def mean_and_std(self, obs: Tensor, training: bool = False) -> tuple[Tensor, Tensor]:
        x = self.embedder(obs, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        mean = self.mean_gain * self.mean_w(x) + self.mean_bias
        log_std = (self.log_std_bias + self.std_gain * self.std_w(x)).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std.exp()

    @staticmethod
    def _log_prob(mean: Tensor, std: Tensor, pre_tanh: Tensor) -> Tensor:
        log_jacobian = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        normal = torch.distributions.Normal(mean, std)
        return (normal.log_prob(pre_tanh) - log_jacobian).sum(dim=-1)

    def sample(self, obs: Tensor, deterministic: bool = False,
               training: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        mean, std = self.mean_and_std(obs, training)
        pre_tanh = mean if deterministic else mean + std * torch.randn_like(mean)
        return pre_tanh.tanh(), self._log_prob(mean, std, pre_tanh), pre_tanh

    def log_prob(self, obs: Tensor, pre_tanh: Tensor, training: bool = False) -> Tensor:
        mean, std = self.mean_and_std(obs, training)
        return self._log_prob(mean, std, pre_tanh)


@torch.no_grad()
def categorical_td_target(
    target_log_probs: Tensor, reward: Tensor, discount: Tensor, terminated: Tensor,
    entropy_term: Tensor, min_v: float, max_v: float,
) -> Tensor:
    """Project r + discount*(1-terminal)*(Z - alpha*logp) onto the fixed atoms.

    Shapes: target_log_probs [B, N]; reward, discount, terminated, entropy_term [B].
    Returns target probabilities [B, N]. Upstream's version with a per-sample
    discount in place of the scalar gamma**n.
    """
    batch, num_bins = target_log_probs.shape
    width = (max_v - min_v) / (num_bins - 1)
    atoms = torch.linspace(min_v, max_v, num_bins, device=target_log_probs.device,
                           dtype=target_log_probs.dtype).view(1, -1)
    continuation = (discount * (1.0 - terminated.float())).view(-1, 1)
    shifted = reward.view(-1, 1) + continuation * (atoms - entropy_term.view(-1, 1))
    position = (shifted.clamp(min_v, max_v) - min_v) / width
    lower = position.floor().long().clamp(0, num_bins - 1)
    upper = (lower + 1).clamp(max=num_bins - 1)
    upper_weight = position - lower.to(position.dtype)
    probs = target_log_probs.exp()
    target = torch.zeros(batch, num_bins, dtype=probs.dtype, device=probs.device)
    target.scatter_add_(1, lower, probs * (1.0 - upper_weight))
    target.scatter_add_(1, upper, probs * upper_weight)
    return target


class DecisionReturnScaler:
    """Upstream reward scaling: r / max(std(G), max|G| / G_max), G per decision.

    G is the running discounted return of the training stream, carried across
    decisions with each decision's own discount and reset after any episode end.
    """

    def __init__(self, g_max: float, epsilon: float = 1e-8):
        self.g_max = float(g_max)
        self.epsilon = epsilon
        self.running_return = 0.0
        self.carry = 0.0
        self.max_abs_return = 0.0
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4  # Upstream RunningMeanStd init.

    def observe(self, reward: float, discount: float, done: bool) -> None:
        self.running_return = self.carry * self.running_return + float(reward)
        self.max_abs_return = max(self.max_abs_return, abs(self.running_return))
        delta = self.running_return - self.mean
        total = self.count + 1.0
        self.mean += delta / total
        self.var = (self.var * self.count + delta * delta * self.count / total) / total
        self.count = total
        self.carry = 0.0 if done else float(discount)

    def scale(self) -> float:
        return max(math.sqrt(self.var + self.epsilon), self.max_abs_return / self.g_max)

    def state_dict(self) -> dict:
        return {name: getattr(self, name) for name in
                ("running_return", "carry", "max_abs_return", "mean", "var", "count")}

    def load_state_dict(self, state: dict) -> None:
        for name, value in state.items():
            setattr(self, name, float(value))


class FlashSAC:
    """Drop-in alternative to ``sac.SAC`` for ``algorithm = "flashsac"``.

    Batches use the same replay fields as SAC. Call ``observe`` once per collected
    decision so reward scaling sees the training stream (``train.py`` does this).
    """

    def __init__(self, obs_dim: int, critic_dim: int, action_dim: int, config: dict, device="cpu"):
        self.device = torch.device(device)
        self.config = dict(config)
        network = load_flashsac().network
        self.actor = FlashResidualActor(
            obs_dim, action_dim, int(config.get("actor_hidden_dim", 128)),
            int(config.get("actor_num_blocks", 2)),
            float(config.get("initial_log_std", -3.0))).to(self.device)
        self.min_v = -float(config.get("normalized_g_max", 5.0))
        self.max_v = float(config.get("normalized_g_max", 5.0))
        self.critic = network.FlashSACDoubleCritic(
            num_blocks=int(config.get("critic_num_blocks", 2)),
            input_dim=critic_dim + action_dim,
            hidden_dim=int(config.get("critic_hidden_dim", 256)),
            num_bins=int(config.get("critic_num_bins", 101)),
            min_v=self.min_v, max_v=self.max_v).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.temperature = network.FlashSACTemperature(
            float(config.get("initial_temperature", 0.01))).to(self.device)
        for module in (self.actor, self.critic, self.target_critic):
            normalize_weights(module)

        lr = float(config.get("learning_rate", 3e-4))
        lr_end = float(config.get("learning_rate_end", lr))
        decay = int(config.get("lr_decay_updates", 0))

        def schedule(step: int) -> float:  # Upstream: cosine decay from lr to lr_end.
            if decay <= 0:
                return 1.0
            progress = min(step, decay) / decay
            return (lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * progress))) / lr

        self.optimizers, self.schedulers = {}, {}
        for name, module in (("actor", self.actor), ("critic", self.critic),
                             ("temperature", self.temperature)):
            optimizer = torch.optim.Adam(module.parameters(), lr=lr)
            self.optimizers[name] = optimizer
            self.schedulers[name] = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

        if "target_entropy" in config:
            self.target_entropy = float(config["target_entropy"])
        else:
            sigma = float(config.get("target_sigma", 0.05))
            if sigma <= 0:
                raise ValueError("target_sigma must be positive")
            self.target_entropy = 0.5 * action_dim * math.log(2 * math.pi * math.e * sigma ** 2)
        self.tau = float(config.get("tau", 0.01))
        if not 0 < self.tau <= 1:
            raise ValueError("tau must be in (0,1]")
        self.actor_update_period = int(config.get("actor_update_period", 2))
        if self.actor_update_period < 1:
            raise ValueError("actor_update_period must be positive")
        self.normalize_reward = bool(config.get("normalize_reward", True))
        self.reward_scaler = DecisionReturnScaler(self.max_v)
        self.updates = 0

    @property
    def alpha(self) -> Tensor:
        return self.temperature().detach()

    def observe(self, reward: float, discount: float, terminated: bool, truncated: bool) -> None:
        self.reward_scaler.observe(reward, discount, bool(terminated or truncated))

    def _step(self, name: str, loss: Tensor, module: nn.Module | None) -> None:
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.schedulers[name].step()
        if module is not None:
            normalize_weights(module)

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        batch = {key: torch.as_tensor(value, device=self.device).detach().float()
                 for key, value in batch.items()}
        reward = batch["reward"].reshape(-1)
        if self.normalize_reward:
            reward = reward / self.reward_scaler.scale()
        discount = batch["discount"].reshape(-1)
        terminated = batch["terminated"].reshape(-1)
        obs, next_obs = batch["obs"], batch["next_obs"]
        critic_obs, next_critic_obs = batch["critic_obs"], batch["next_critic_obs"]
        info: dict[str, float] = {}

        # Upstream order: (delayed) actor and temperature, then critic, then EMA.
        if self.updates % self.actor_update_period == 0:
            # Current and next observations share one BatchNorm pass upstream.
            actions_all, logp_all, _ = self.actor.sample(torch.cat((obs, next_obs)), training=True)
            action, logp = actions_all.chunk(2)[0], logp_all.chunk(2)[0]
            self.critic.requires_grad_(False)
            qs, _ = self.critic(critic_obs, action, training=False)
            self.critic.requires_grad_(True)
            actor_loss = (self.alpha * logp - torch.minimum(qs[0], qs[1])).mean()
            self._step("actor", actor_loss, self.actor)
            entropy = -logp.detach().mean()
            temperature_loss = self.temperature() * (entropy - self.target_entropy)
            self._step("temperature", temperature_loss.sum(), None)
            info |= {"actor_loss": float(actor_loss.detach()), "entropy": float(entropy),
                     "temperature_loss": float(temperature_loss.detach().sum())}

        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs, training=False)
            entropy_term = self.alpha * next_logp
            obs_all = torch.cat((critic_obs, next_critic_obs))
            actions_all = torch.cat((batch["action"], next_action))
            target_qs, target_info = self.target_critic(obs_all, actions_all, training=True)
            next_qs = target_qs.chunk(2, dim=1)[1]
            next_log_probs = target_info["log_prob"].chunk(2, dim=1)[1]  # [2, B, N]
            index = next_qs.argmin(dim=0)[None, :, None].expand(1, -1, next_log_probs.shape[-1])
            min_log_probs = next_log_probs.gather(0, index)[0]
            target_probs = categorical_td_target(min_log_probs, reward, discount, terminated,
                                                 entropy_term, self.min_v, self.max_v)
        pred_qs, pred_info = self.critic(obs_all, actions_all, training=True)
        pred_log_probs = pred_info["log_prob"].chunk(2, dim=1)[0]
        critic_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1).mean()
        self._step("critic", critic_loss, self.critic)

        with torch.no_grad():
            torch._foreach_lerp_(list(self.target_critic.parameters()),
                                 list(self.critic.parameters()), self.tau)
        self.updates += 1
        q = torch.minimum(pred_qs[0], pred_qs[1]).chunk(2)[0].detach()
        return info | {
            "critic_loss": float(critic_loss.detach()), "alpha": float(self.alpha),
            "q": float(q.mean()), "target_q": float((target_probs * torch.linspace(
                self.min_v, self.max_v, target_probs.shape[-1], device=self.device)).sum(-1).mean()),
            "reward_scale": float(self.reward_scaler.scale()) if self.normalize_reward else 1.0,
            "lr": float(self.schedulers["critic"].get_last_lr()[0]),
            "updates": float(self.updates),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "temperature": self.temperature.state_dict(),
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "schedulers": {k: {"last_epoch": v.last_epoch} for k, v in self.schedulers.items()},
            "reward_scaler": self.reward_scaler.state_dict(), "updates": self.updates,
            "config": self.config, "target_entropy": self.target_entropy, "tau": self.tau,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.temperature.load_state_dict(state["temperature"])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        for name, scheduler in self.schedulers.items():
            scheduler.last_epoch = int(state["schedulers"][name]["last_epoch"])
            for group, factor in zip(scheduler.optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = factor * scheduler.lr_lambdas[0](scheduler.last_epoch)
        self.reward_scaler.load_state_dict(state["reward_scaler"])
        self.updates = int(state["updates"])
        self.config = dict(state["config"])
        self.target_entropy = float(state["target_entropy"])
        self.tau = float(state["tau"])
