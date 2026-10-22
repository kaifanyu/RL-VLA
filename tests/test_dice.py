"""Synthetic numerical checks only: no checkpoints, simulators, or GPU."""

import copy

import numpy as np
import pytest
import torch

from rl_vla.dice import config_from
from rl_vla.dice_core import DICE, discounted_returns, regularization_mask

SETTINGS = {
    "width": 16,
    "actor_lr": 1e-3,
    "critic_lr": 1e-3,
    "max_grad_norm": 1.0,
    "target_tau": 0.005,
    "filter_warmup": 1,
    "underestimation_threshold": -0.1,
    "bc_weight": 1.0,
}


def test_selective_gate_requires_both_conditions_and_retains_experts():
    q = torch.tensor([[2.0, 0.0], [2.0, 2.0], [2.0, 2.0]])
    qp = torch.ones_like(q)
    calibration = torch.tensor([-0.2, 0.0, -0.2])
    expert = torch.tensor([0.0, 0.0, 1.0])
    mask = regularization_mask(q, qp, calibration, step=2, warmup=1, threshold=-0.1, expert=expert)
    assert torch.equal(mask, torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))
    mask = regularization_mask(q, qp, calibration, step=1, warmup=1, threshold=-0.1, expert=expert)
    assert torch.equal(mask, torch.ones_like(q))


def test_missing_calibration_is_rejected():
    with pytest.raises(ValueError):
        regularization_mask(
            torch.ones(1, 2),
            torch.zeros(1, 2),
            torch.tensor([float("nan")]),
            step=2,
            warmup=1,
            threshold=-0.1,
            expert=torch.zeros(1),
        )


def test_chunk_returns_use_actual_duration_and_terminal_boundary():
    assert discounted_returns([0.0, 1.0], [0.9**5, 0.0]) == [0.9**5, 1.0]
    assert discounted_returns([1.0, 2.0, 100.0], [0.5, 0.0, 0.0]) == [2.0, 2.0, 100.0]


def test_initial_residual_exactly_zero_and_prior_has_no_gradient():
    model = DICE(3, 4, 2, SETTINGS)
    f, z, p = torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 2, requires_grad=True)
    a = model.action(f, z, p)
    assert torch.equal(a, p)
    a.sum().backward()
    assert p.grad is None


def test_selection_preserves_index():
    model = DICE(3, 4, 2, SETTINGS)
    f, z, p = torch.randn(4, 3), torch.randn(4, 4), torch.randn(4, 2)
    expected = int(model.q(f, p).min(0).values.argmax())
    action, index = model.select(f, z, p)
    assert index == expected
    assert torch.equal(action, p[index])


def test_numerical_update_is_finite_and_checkpoint_roundtrip():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = DICE(3, 4, 2, SETTINGS)
    before = copy.deepcopy(model.actor.state_dict())
    batch = {
        "features": torch.randn(2, 3),
        "actions": torch.randn(2, 2),
        "returns": torch.ones(2),
        "rewards": torch.ones(2),
        "discounts": torch.tensor([0.9, 0.0]),
        "expert": torch.tensor([0.0, 1.0]),
    }
    props = (torch.randn(2, 2, 3), torch.randn(2, 2, 4), torch.randn(2, 2, 2))
    nxt = (torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 2))
    result = model.update(batch, props, nxt, step=2)
    assert all(np.isfinite(value) for value in result.values())
    assert any(not torch.equal(before[k], v) for k, v in model.actor.state_dict().items())
    restored = DICE(3, 4, 2, SETTINGS)
    restored.load_state_dict(model.state_dict())
    assert restored.updates == 1
    assert torch.equal(restored.action(*nxt), model.action(*nxt))


def test_configuration_is_separate_from_existing_algorithms():
    config = config_from("configs/dice_pi05_libero.toml")
    assert config["algorithm"] == "dice"
    assert config["data"]["mode"] == "demonstrations"


def test_demonstration_chunking_normalizes_and_preserves_terminal_returns():
    from types import SimpleNamespace

    from rl_vla.dice_data import convert_episode

    def transform(raw):
        actions = raw["openpi"]["actions"]
        # Synthetic checkpoint transform: scale and pad, never guess model units.
        return {"actions": np.pad(actions * 2, ((0, 0), (0, 25)))}

    prior = SimpleNamespace(
        execute_steps=5,
        flow=SimpleNamespace(horizon=10, action_dim=32, _transform_observation=transform),
    )
    observations = [{"openpi": {"prompt": "synthetic"}} for _ in range(8)]
    episode = {
        "observations": observations,
        "actions": np.ones((7, 7)),
        "rewards": np.array([0.0] * 6 + [1.0]),
        "ending": "terminated",
    }
    records = convert_episode(episode, prior, 0.9)
    assert len(records) == 2
    assert records[0]["discounts"] == 0.9**5
    assert records[1]["discounts"] == 0.0
    assert np.isclose(records[0]["returns"], 0.9**6)
    assert np.isclose(records[1]["returns"], 0.9)
    np.testing.assert_array_equal(records[0]["actions"], np.full(35, 2.0))
    assert records[-1]["next_observation"] is observations[-1]


def test_standalone_runner_with_synthetic_prior_and_environment(tmp_path, monkeypatch):
    # Exercises collection/replay/save only on in-memory fixtures; no GPU or VLA.
    import rl_vla.dice as runner
    import rl_vla.dice_pi05 as adapter
    import rl_vla.rldt_env as env_module

    class Prior:
        feature_dim, noise_dim, action_dim = 3, 4, 2

        def __init__(self, *_args):
            self.metadata = {"synthetic": True}

        def propose(self, raw, count):
            return (
                np.full((count, 3), raw["value"], np.float32),
                np.zeros((count, 4), np.float32),
                np.zeros((count, 2), np.float32),
                np.zeros((count, 2), np.float32),
            )

        def decode(self, corrected, *_args):
            return np.asarray(corrected).reshape(2, 1)

    class Env:
        low, high = np.array([-1.0]), np.array([1.0])
        closed = False

        def reset(self, *, seed):
            self.steps = 0
            return {"value": 0.0}, {}

        def step(self, _action):
            self.steps += 1
            return {"value": float(self.steps)}, float(self.steps == 2), self.steps == 2, False, {}

        def close(self):
            self.closed = True

    env = Env()
    monkeypatch.setattr(runner, "check", lambda _c: None)
    monkeypatch.setattr(runner, "resource_check", lambda: None)
    monkeypatch.setattr(adapter, "FrozenPi05", Prior)
    monkeypatch.setattr(env_module, "make_primitive_env", lambda _c: env)
    c = config_from("configs/dice_pi05_libero.toml")
    c["train"].update(episodes=2, batch_size=2, updates_per_episode=1)
    c["data"]["mode"] = "online_only_ablation"
    output = tmp_path / "synthetic"
    runner.run(c, output)
    result = torch.load(output / "final.pt", weights_only=True)
    assert result["counters"] == {"episodes": 2, "decisions": 2, "updates": 1}
    assert result["prior"] == {"synthetic": True}
    assert env.closed
