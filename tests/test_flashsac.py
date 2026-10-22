"""Residual FlashSAC: upstream-layer loading, residual init, targets and the pipeline."""

from __future__ import annotations

import io
import math
import subprocess
import sys

import pytest
import torch

from rl_vla.flashsac_vendor import flashsac_root

pytestmark = pytest.mark.skipif(
    not (flashsac_root() / "flash_rl" / "agents" / "flashSAC" / "layer.py").is_file(),
    reason="third_party/FlashSAC missing; run scripts/fetch_flashsac.sh")


@pytest.fixture(autouse=True)
def reproducible():
    torch.manual_seed(5)
    torch.set_num_threads(1)


def make(**overrides):
    from rl_vla.flashsac import FlashSAC
    config = {"actor_hidden_dim": 32, "critic_hidden_dim": 32, "critic_num_bins": 21} | overrides
    return FlashSAC(obs_dim=6, critic_dim=8, action_dim=3, config=config)


def batch(size=16, discount=0.9):
    return {
        "obs": torch.randn(size, 6), "next_obs": torch.randn(size, 6),
        "critic_obs": torch.randn(size, 8), "next_critic_obs": torch.randn(size, 8),
        "action": torch.rand(size, 3) * 1.8 - 0.9, "reward": torch.randn(size),
        "discount": torch.full((size,), discount), "terminated": torch.zeros(size),
    }


def test_upstream_layers_load_without_jax_or_package_init():
    code = ("import sys; from rl_vla.flashsac import FlashSAC; "
            "FlashSAC(4, 4, 2, {'actor_hidden_dim': 8, 'critic_hidden_dim': 8}); "
            "assert 'jax' not in sys.modules and 'omegaconf' not in sys.modules; "
            "assert sys.modules['flash_rl'].__rl_vla_stub__")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_actor_starts_at_base_with_initial_std_and_matches_entropy_target():
    learner = make()
    obs = torch.randn(10, 6)
    action, logp, pre_tanh = learner.actor.sample(obs, deterministic=True)
    assert torch.count_nonzero(action) == 0
    _, std = learner.actor.mean_and_std(obs)
    torch.testing.assert_close(std, torch.full((10, 3), math.exp(-3)))
    # Default sigma=0.05 ~= exp(-3): the initial policy already sits at the target.
    gaussian_entropy = 3 * (0.5 * math.log(2 * math.pi * math.e) - 3.0)
    assert abs(learner.target_entropy - gaussian_entropy) < 0.05
    _, logp, pre_tanh = learner.actor.sample(obs)
    torch.testing.assert_close(learner.actor.log_prob(obs, pre_tanh), logp)


def test_categorical_target_uses_per_sample_discount_and_terminal_mask():
    from rl_vla.flashsac import categorical_td_target
    atoms = torch.linspace(-5, 5, 11)
    point = torch.zeros(3, 11)
    point[:, 7] = 1.0  # All next-state mass on Z = 2.
    target = categorical_td_target(
        (point + 1e-30).log(), reward=torch.tensor([1.0, 1.0, 1.0]),
        discount=torch.tensor([0.5, 0.25, 0.5]), terminated=torch.tensor([0.0, 0.0, 1.0]),
        entropy_term=torch.zeros(3), min_v=-5.0, max_v=5.0)
    torch.testing.assert_close(target.sum(-1), torch.ones(3))
    expected = torch.tensor([1.0 + 0.5 * 2.0, 1.0 + 0.25 * 2.0, 1.0])
    torch.testing.assert_close((target * atoms).sum(-1), expected)


def test_updates_keep_unit_weight_norms_and_finite_losses():
    learner = make(actor_update_period=1)
    for _ in range(5):
        learner.observe(1.0, 0.9, False, False)
    for _ in range(4):
        metrics = learner.update(batch())
    assert all(math.isfinite(value) for value in metrics.values())
    assert "actor_loss" in metrics and learner.updates == 4
    weight = learner.critic.embedder.w.weight  # [num_qs, out, in]
    torch.testing.assert_close(weight.norm(dim=-1), torch.ones(weight.shape[:2]),
                               rtol=0, atol=1e-5)
    rows = learner.actor.mean_w.w.weight.norm(dim=-1)
    torch.testing.assert_close(rows, torch.ones_like(rows), rtol=0, atol=1e-5)
    assert learner.actor.mean_gain.abs().sum() > 0  # The residual head left zero.


def test_delayed_actor_updates():
    learner = make(actor_update_period=2)
    assert "actor_loss" in learner.update(batch())
    assert "actor_loss" not in learner.update(batch())
    assert "actor_loss" in learner.update(batch())


def test_reward_scaler_bounds_returns_and_resets_on_episode_end():
    from rl_vla.flashsac import DecisionReturnScaler
    scaler = DecisionReturnScaler(g_max=5.0)
    for _ in range(50):
        scaler.observe(10.0, 0.99, False)
    assert scaler.running_return / scaler.scale() <= 5.0 + 1e-6
    scaler.observe(10.0, 0.99, True)
    scaler.observe(1.0, 0.99, False)
    assert scaler.running_return == 1.0


def test_state_dict_round_trip_is_exact_and_weights_only_loadable():
    learner = make()
    learner.observe(2.0, 0.9, False, False)
    learner.update(batch())
    buffer = io.BytesIO()
    torch.save(learner.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored = make()
    restored.load_state_dict(state)
    obs = torch.randn(4, 6)
    torch.testing.assert_close(restored.actor.mean_and_std(obs), learner.actor.mean_and_std(obs))
    assert restored.reward_scaler.scale() == learner.reward_scaler.scale()
    assert restored.schedulers["critic"].last_epoch == learner.schedulers["critic"].last_epoch
    torch.manual_seed(0)
    a = restored.update(batch())
    torch.manual_seed(0)
    b = learner.update(batch())
    assert a == b


def test_toy_train_and_evaluate_pipeline(tmp_path):
    from rl_vla.config import load_config
    from rl_vla.evaluate import evaluate
    from rl_vla.train import train
    config = load_config("configs/flashsac_toy.toml")
    config["train"] |= {"total_decisions": 80, "learning_starts": 32, "batch_size": 16,
                        "log_every": 40, "save_every": 40}
    config["flashsac"] |= {"actor_hidden_dim": 32, "critic_hidden_dim": 32}
    summary = train(config, str(tmp_path / "run"))
    assert summary["updates"] == 80 - 32 + 1
    result = evaluate(config, str(tmp_path / "eval.json"), checkpoint=summary["checkpoint"],
                      episodes=2)
    assert result is not None
