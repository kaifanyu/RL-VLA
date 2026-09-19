"""Numerical tests for residual density, macro targets, and learner updates."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch

from rl_vla.actors import Actor
from rl_vla.buffers import ReplayBuffer
from rl_vla.ppo import PPO, compute_gae
from rl_vla.sac import SAC, compute_sac_target


@pytest.fixture(autouse=True)
def reproducible_small_networks():
    torch.manual_seed(17)
    torch.set_num_threads(1)


def test_actor_starts_near_base_and_recomputes_exact_ppo_ratios():
    actor = Actor(5, 3, hidden_dim=16)
    observation = torch.randn(12, 5)
    deterministic, _, _ = actor.sample(observation, deterministic=True)
    assert torch.count_nonzero(deterministic) == 0
    torch.testing.assert_close(actor.distribution(observation).scale,
                               torch.full((12, 3), math.exp(-3)))
    action, old_logp, pre_tanh = actor.sample(observation)
    assert action.shape == pre_tanh.shape == (12, 3)
    assert old_logp.shape == (12,)
    assert action.abs().max() < 1
    ratio = (actor.log_prob(observation, pre_tanh.detach()) - old_logp.detach()).exp()
    torch.testing.assert_close(ratio, torch.ones(12), rtol=0, atol=0)


def test_tanh_density_uses_stable_jacobian_and_full_action_dimensions():
    actor = Actor(2, 3, hidden_dim=8)
    observation = torch.zeros(2, 2)
    moderate_u = torch.tensor([[0.1, 0.2, -0.3], [-0.4, 0.5, 0.6]])
    direct = (actor.distribution(observation).log_prob(moderate_u)
              - torch.log1p(-moderate_u.tanh().square())).sum(-1)
    torch.testing.assert_close(actor.log_prob(observation, moderate_u), direct)
    saturated_u = torch.tensor([[50.0, -50.0, 0.0], [1e6, -1e6, 1.0]], requires_grad=True)
    logp = actor.log_prob(observation, saturated_u)
    assert torch.isfinite(logp).all()
    logp.sum().backward()
    assert torch.isfinite(saturated_u.grad).all()


def test_sac_target_terminal_mask_variable_duration_and_detachment():
    next_q = torch.tensor([10.0, 10.0], requires_grad=True)
    target = compute_sac_target(
        reward=torch.tensor([1.0, 2.0]), discount=torch.tensor([0.9, 0.9**4]),
        terminated=torch.tensor([True, False]), next_q=next_q,
        next_logp=torch.tensor([2.0, 2.0], requires_grad=True), alpha=0.5,
    )
    torch.testing.assert_close(target, torch.tensor([1.0, 2.0 + 0.9**4 * 9.0]))
    assert not target.requires_grad


def test_gae_bootstraps_truncation_but_stops_traces_and_true_terminal_bootstrap():
    reward = torch.tensor([1.0, 2.0, 3.0, 4.0])
    value = torch.tensor([0.4, 0.5, 0.6, 0.7])
    advantage, returns = compute_gae(
        reward=reward, discount=torch.tensor([0.9, 0.81, 0.5, 0.7]), value=value,
        next_value=torch.tensor([0.5, 5.0, 100.0, 8.0]),
        terminated=torch.tensor([False, False, True, False]),
        truncated=torch.tensor([False, True, False, False]), gae_lambda=0.8,
    )
    expected = torch.tensor([1.05 + 0.9 * 0.8 * 5.55, 5.55, 2.4, 8.9])
    torch.testing.assert_close(advantage, expected)
    torch.testing.assert_close(returns, expected + value)


def test_gae_refuses_silently_interleaving_vector_environments():
    fields = [torch.zeros(3, 2) for _ in range(6)]
    with pytest.raises(ValueError, match="chronological|interleaved"):
        compute_gae(*fields)


def test_replay_is_bounded_owns_copies_and_preserves_scalar_batch_shapes():
    buffer = ReplayBuffer(2, seed=5)
    observation = np.array([1.0, 2.0])
    buffer.add(obs=observation, reward=1.0, terminated=False)
    observation[:] = 99
    stored = buffer.sample(1)
    torch.testing.assert_close(stored["obs"], torch.tensor([[1.0, 2.0]]))
    assert stored["reward"].shape == stored["terminated"].shape == (1,)
    assert stored["obs"].dtype == torch.float32
    buffer.add(obs=np.zeros(2), reward=2.0, terminated=False)
    buffer.add(obs=np.ones(2), reward=3.0, terminated=True)
    assert len(buffer) == 2
    assert sorted(buffer.sample(2)["reward"].tolist()) == [2.0, 3.0]
    with pytest.raises(ValueError, match="changed shape"):
        buffer.add(obs=np.zeros(3), reward=0.0, terminated=False)
    with pytest.raises(ValueError, match="Batch size"):
        buffer.sample(3)


def sac_batch(batch_size=16):
    return {
        "obs": torch.randn(batch_size, 5), "critic_obs": torch.randn(batch_size, 7),
        "action": torch.rand(batch_size, 3) * 0.1,
        "reward": torch.linspace(-1, 1, batch_size), "discount": torch.full((batch_size,), 0.9**4),
        "next_obs": torch.randn(batch_size, 5), "next_critic_obs": torch.randn(batch_size, 7),
        "terminated": torch.arange(batch_size) % 4 == 0,
    }


def ppo_rollout(learner, count=16):
    observation = torch.randn(count, 5)
    critic_observation = torch.randn(count, 7)
    with torch.no_grad():
        _, logp, pre_tanh = learner.actor.sample(observation)
        value = learner.value(critic_observation)
    return {
        "obs": observation, "critic_obs": critic_observation, "pre_tanh": pre_tanh,
        "logp": logp, "value": value, "reward": torch.linspace(-1, 1, count),
        "discount": torch.full((count,), 0.9**4), "next_value": torch.roll(value, -1),
        "terminated": torch.arange(count) == count - 1, "truncated": torch.arange(count) == 7,
    }


def test_sac_updates_actor_critics_temperature_and_polyak_targets():
    learner = SAC(5, 7, 3, {"hidden_dim": 16, "tau": 0.1})
    old_actor = learner.actor.mean.weight.detach().clone()
    old_target = [parameter.clone() for parameter in learner.target_critic.parameters()]
    old_alpha = learner.alpha.clone()
    metrics = learner.update(sac_batch())
    assert all(math.isfinite(value) for value in metrics.values())
    assert not torch.equal(old_actor, learner.actor.mean.weight)
    assert not torch.equal(old_alpha, learner.alpha)
    for before, source, target in zip(old_target, learner.critic.parameters(), learner.target_critic.parameters()):
        torch.testing.assert_close(target, 0.9 * before + 0.1 * source)
        assert target.grad is None
    assert learner.target_entropy == -3.0


def test_ppo_updates_with_exact_stored_inputs_and_logs_finite_metrics():
    learner = PPO(5, 7, 3, {"hidden_dim": 16, "epochs": 2, "minibatch_size": 8, "target_kl": None})
    rollout = ppo_rollout(learner)
    before = learner.actor.mean.weight.detach().clone()
    metrics = learner.update(rollout)
    assert metrics["optimizer_steps"] == 4
    assert all(math.isfinite(value) for value in metrics.values())
    assert not torch.equal(before, learner.actor.mean.weight)


def test_ppo_kl_guard_stops_before_applying_large_policy_update():
    learner = PPO(5, 7, 3, {"hidden_dim": 16, "target_kl": 0.001})
    rollout = ppo_rollout(learner)
    rollout["logp"] = rollout["logp"] - 2.0
    before = learner.actor.mean.weight.detach().clone()
    metrics = learner.update(rollout)
    assert metrics["early_stop"] == 1
    assert metrics["optimizer_steps"] == 0
    assert metrics["approx_kl"] > 0.001
    torch.testing.assert_close(before, learner.actor.mean.weight, rtol=0, atol=0)


@pytest.mark.parametrize("learner_class", [SAC, PPO])
def test_complete_learner_state_roundtrips_through_weights_only_torch_load(learner_class):
    config = {"hidden_dim": 16, "epochs": 1, "target_kl": None}
    original = learner_class(5, 7, 3, config)
    original.update(sac_batch() if learner_class is SAC else ppo_rollout(original))
    payload = io.BytesIO()
    torch.save(original.state_dict(), payload)
    payload.seek(0)
    restored = learner_class(5, 7, 3, config)
    restored.load_state_dict(torch.load(payload, weights_only=True))
    observation = torch.randn(4, 5)
    torch.testing.assert_close(original.actor.sample(observation, True)[0],
                               restored.actor.sample(observation, True)[0], rtol=0, atol=0)
    assert restored.updates == original.updates
    # Matching optimizer moments are necessary for continuing learning.
    if learner_class is SAC:
        torch.testing.assert_close(original.alpha, restored.alpha)
        assert restored.actor_optimizer.state_dict()["state"]
        assert restored.critic_optimizer.state_dict()["state"]
        assert restored.alpha_optimizer.state_dict()["state"]
    else:
        assert restored.optimizer.state_dict()["state"]
    continuation = sac_batch() if learner_class is SAC else ppo_rollout(original)
    torch.manual_seed(29)
    original_metrics = original.update(continuation)
    torch.manual_seed(29)
    restored_metrics = restored.update(continuation)
    assert original_metrics == restored_metrics
    for original_parameter, restored_parameter in zip(original.actor.parameters(), restored.actor.parameters()):
        torch.testing.assert_close(original_parameter, restored_parameter, rtol=0, atol=0)
