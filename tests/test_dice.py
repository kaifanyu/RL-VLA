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
    provenance = {
        "task": {"task_key": "synthetic_0", "task_suite": "synthetic", "task_id": 0,
                 "instruction": "synthetic"},
        "environment": {"adapter": "synthetic.adapter", "protocol_id": f"sha256:{'a' * 64}",
                 "protocol_sha256": "a" * 64, "protocol_version": 1,
                         "split": "train"},
        "demonstration": {"source_id": "fixture-dataset", "source_split": "train"},
    }
    observations = [{"openpi": {"prompt": "synthetic"}} for _ in range(8)]
    episode = {
        "observations": observations,
        "actions": np.ones((7, 7)),
        "rewards": np.array([0.0] * 6 + [1.0]),
        "ending": "terminated",
        "provenance": provenance,
    }
    records = convert_episode(episode, prior, 0.9, provenance)
    assert len(records) == 2
    assert records[0]["discounts"] == 0.9**5
    assert records[1]["discounts"] == 0.0
    assert np.isclose(records[0]["returns"], 0.9**6)
    assert np.isclose(records[1]["returns"], 0.9)
    np.testing.assert_array_equal(records[0]["actions"], np.full(35, 2.0))
    assert records[-1]["next_observation"] is observations[-1]
    assert records[0]["provenance"] == provenance


def test_wrong_task_replay_provenance_is_rejected(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    from types import SimpleNamespace

    import rl_vla.dice as runner

    protocol = tmp_path / "protocol.json"
    protocol_data = {"version": 2, "tasks": {"libero_pro_task_a": {
        "suite": "libero_spatial", "task_id": 0,
    }}}
    protocol.write_text(json.dumps(protocol_data))
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    expected = {
        "task": {"task_key": "libero_pro_task_a", "task_suite": "libero_spatial",
                 "task_id": 0, "instruction": "pick up the cube"},
        "environment": {"adapter": "rl_vla.envs.libero_pro:make_env",
                 "protocol_id": f"sha256:{protocol_hash}",
                 "protocol_sha256": protocol_hash, "protocol_version": 2,
                         "split": "train"},
        "demonstration": {"source_id": "dataset-a", "source_split": "train"},
    }
    actual = copy.deepcopy(expected)
    actual["task"]["task_key"] = "libero_pro_task_b"
    archive_path = tmp_path / "wrong-task.mpk"
    archive_path.write_bytes(b"stub")
    payload = {
        "format": "dice-pi05-transitions-v1",
        "metadata": {"checkpoint_metadata_sha256": "checkpoint",
                     "normalization_assets_sha256": "normalization",
                     "execute_steps": 5, "gamma": 0.99},
        "provenance": actual,
        "transitions": [],
    }
    msgpack_numpy = SimpleNamespace(unpackb=lambda _value: payload)
    monkeypatch.setitem(sys.modules, "openpi_client", SimpleNamespace(msgpack_numpy=msgpack_numpy))
    config = {
        "data": {"mode": "demonstrations", "path": str(archive_path),
                 "provenance": expected},
        "train": {"replay_max_mb": 1}, "dice": {"gamma": 0.99},
        "env": {"factory": "rl_vla.envs.libero_pro:make_env", "kwargs": {
            "protocol": str(protocol), "task_key": "libero_pro_task_a", "split": "train"}},
    }
    prior = SimpleNamespace(metadata={"checkpoint_metadata_sha256": "checkpoint",
                                      "normalization_assets_sha256": "normalization"},
                            execute_steps=5)
    with pytest.raises(ValueError, match="task identity mismatch"):
        runner.load_demonstrations(config, prior)


def test_missing_replay_provenance_is_not_inferred():
    from rl_vla.dice import validate_replay_provenance

    with pytest.raises(TypeError, match="missing or malformed"):
        validate_replay_provenance(None, {})


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("task", "instruction", "different instruction"),
        ("environment", "adapter", "different.adapter:make_env"),
        ("environment", "protocol_sha256", "b" * 64),
        ("demonstration", "source_id", "different-dataset"),
        ("demonstration", "source_split", "validation"),
    ],
)
def test_replay_provenance_mismatches_are_rejected(section, key, value):
    from rl_vla.dice import validate_replay_provenance

    expected = {
        "task": {"task_key": "task_a", "task_suite": "libero_spatial", "task_id": 0,
                 "instruction": "pick up the cube"},
        "environment": {"adapter": "rl_vla.envs.libero_pro:make_env", "split": "train",
                        "protocol_id": f"sha256:{'a' * 64}", "protocol_sha256": "a" * 64,
                        "protocol_version": 2},
        "demonstration": {"source_id": "dataset-a", "source_split": "train"},
    }
    actual = copy.deepcopy(expected)
    actual[section][key] = value
    with pytest.raises(ValueError):
        validate_replay_provenance(actual, expected)


def test_live_environment_instruction_must_match_replay_provenance():
    from rl_vla.dice import validate_live_task

    provenance = {"task": {"instruction": "pick up the cube",
                            "task_suite": "libero_spatial", "task_id": 0}}
    with pytest.raises(ValueError, match="instruction differs"):
        validate_live_task(
            {"openpi": {"prompt": "open the drawer"}},
            {"task_suite": "libero_spatial", "task_id": 0},
            provenance,
        )
    with pytest.raises(ValueError, match="task identity differs"):
        validate_live_task(
            {"openpi": {"prompt": "pick up the cube"}},
            {"task_suite": "libero_spatial", "task_id": 1},
            provenance,
        )


def test_libero_pro_evaluation_requires_exact_protocol_states(tmp_path):
    import hashlib
    import json

    from rl_vla.dice_eval import evaluation_contract

    protocol = {
        "version": 2,
        "tasks": {
            "task_a": {
                "suite": "libero_spatial", "task_id": 0, "pro_state_count": 4,
                "train_state_ids": [0, 1], "validation_state_ids": [2],
                "test_state_ids": [3],
            }
        },
    }
    path = tmp_path / "protocol.json"
    encoded = json.dumps(protocol).encode()
    path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    provenance = {
        "task": {"task_key": "task_a", "task_suite": "libero_spatial", "task_id": 0,
                 "instruction": "pick up the cube"},
        "environment": {"adapter": "rl_vla.envs.libero_pro:make_env",
                         "protocol_id": f"sha256:{digest}", "protocol_sha256": digest,
                         "protocol_version": 2, "split": "train"},
        "demonstration": {"source_id": "demo-a", "source_split": "train"},
    }
    config = {
        "env": {"factory": "rl_vla.envs.libero_pro:make_env", "kwargs": {
            "protocol": str(path), "task_key": "task_a", "split": "validation"}},
        "data": {"provenance": provenance},
    }
    kwargs, state_ids, actual_hash, _ = evaluation_contract(config)
    assert kwargs["split"] == "validation"
    assert state_ids == [2]
    assert actual_hash == digest
    config["env"]["kwargs"]["task_key"] = "task_b"
    with pytest.raises(ValueError, match="Task key is absent"):
        evaluation_contract(config)


def test_baseline_and_dice_evaluation_use_identical_ordered_states(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace
    from typing import ClassVar

    import rl_vla.dice as runner
    import rl_vla.dice_core as core
    import rl_vla.dice_eval as evaluation
    import rl_vla.dice_pi05 as policy_adapter
    import rl_vla.rldt_env as env_adapter

    protocol = tmp_path / "protocol.json"
    protocol_data = {"version": 2, "tasks": {"task_a": {
        "suite": "libero_spatial", "task_id": 0, "pro_state_count": 6,
        "train_state_ids": [0, 1], "validation_state_ids": [2, 3],
        "test_state_ids": [4, 5],
    }}}
    protocol.write_text(json.dumps(protocol_data))
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    provenance = {
        "task": {"task_key": "task_a", "task_suite": "libero_spatial", "task_id": 0,
                 "instruction": "pick up the cube"},
        "environment": {"adapter": "rl_vla.envs.libero_pro:make_env", "split": "train",
                        "protocol_id": f"sha256:{protocol_hash}",
                        "protocol_sha256": protocol_hash, "protocol_version": 2},
        "demonstration": {"source_id": "demo-a", "source_split": "train"},
    }
    config = {
        "policy": {"checkpoint": "unused", "model_config": "pi05_libero"},
        "dice": {"selection_samples": 2, "selection_strategy": "max_q_min"},
        "data": {"provenance": provenance},
        "env": {"factory": "rl_vla.envs.libero_pro:make_env", "kwargs": {
            "protocol": str(protocol), "task_key": "task_a", "split": "validation"}},
    }

    class Prior:
        metadata: ClassVar = {"synthetic": True}
        feature_dim, noise_dim, action_dim = 1, 1, 1

        def __init__(self, *_args):
            self.flow = SimpleNamespace()

        def propose(self, _raw, count):
            return (np.zeros((count, 1), np.float32), np.zeros((count, 1), np.float32),
                    np.zeros((count, 1), np.float32), np.zeros((count, 1), np.float32))

        def decode(self, _action, _full, _raw):
            return np.zeros((1, 1), np.float32)

    class Learner:
        actor = SimpleNamespace(eval=lambda: None)
        critics = SimpleNamespace(eval=lambda: None)

        def __init__(self, *_args):
            pass

        def load_state_dict(self, _state):
            pass

        def select(self, _features, _noise, proposals, strategy):
            assert strategy == "max_q_min"
            return proposals[0], 0

    class Env:
        low, high = np.array([-1.0]), np.array([1.0])

        def __init__(self):
            self.state_ids = [2, 3]
            self.index = 0

        def reset(self, *, seed):
            state_id = self.state_ids[self.index]
            self.index += 1
            return {"openpi": {"prompt": "pick up the cube"}}, {
                "initial_state_id": state_id, "task_suite": "libero_spatial", "task_id": 0,
            }

        def step(self, _action):
            return {"openpi": {"prompt": "pick up the cube"}}, 0.0, True, False, {
                "is_success": False,
            }

        def close(self):
            pass

    monkeypatch.setattr(runner, "check", lambda _config: {"setup_ok": True})
    monkeypatch.setattr(runner, "resource_check", lambda: None)
    monkeypatch.setattr(evaluation, "torch", torch)
    monkeypatch.setattr(policy_adapter, "FrozenPi05", Prior)
    monkeypatch.setattr(core, "DICE", Learner)
    monkeypatch.setattr(env_adapter, "make_primitive_env", lambda _config: Env())
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {
        "format": "dice-pi05-v1", "config": config, "prior": Prior.metadata,
        "learner": {},
    })

    baseline = evaluation.evaluate(config, tmp_path / "baseline", episodes=2, seed=100000)
    trained = evaluation.evaluate(
        config, tmp_path / "dice", checkpoint="synthetic.pt", episodes=2, seed=100000,
    )
    assert baseline["state_ids"] == [2, 3]
    assert trained["state_ids"] == baseline["state_ids"]
    assert baseline["test_results_used_for_selection"] is False
    assert trained["test_results_used_for_selection"] is False


def test_prepare_dice_pro_requires_explicit_protocol_and_task(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts/prepare_dice_pro.py"
    spec = importlib.util.spec_from_file_location("prepare_dice_pro", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "version": 2,
        "tasks": {"task_a": {
            "suite": "libero_spatial", "task_id": 0, "pro_state_count": 4,
            "train_state_ids": [0, 1], "validation_state_ids": [2], "test_state_ids": [3],
        }},
    }))
    outputs = module.prepare(
        protocol, "task_a", tmp_path / "configs", "pick up the cube", "demo-a", "train",
        tmp_path / "replay.mpk",
    )
    assert [Path(path).name for path in outputs] == ["train.toml", "validation.toml", "test.toml"]
    assert all(Path(path).exists() for path in outputs)
    with pytest.raises(FileExistsError):
        module.prepare(
            protocol, "task_a", tmp_path / "configs", "pick up the cube", "demo-a", "train",
            tmp_path / "replay.mpk",
        )


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
    # A restart from an episode boundary matches an uninterrupted synthetic run.
    c["train"]["episodes"] = 4
    resumed, whole = tmp_path / "resumed", tmp_path / "whole"
    runner.run(c, resumed, resume=output / "resume.pt")
    runner.run(c, whole)
    a = torch.load(resumed / "final.pt", weights_only=True)
    b = torch.load(whole / "final.pt", weights_only=True)
    assert a["counters"] == b["counters"]
    for name in ("actor", "critics", "target"):
        for key in a["learner"][name]:
            assert torch.equal(a["learner"][name][key], b["learner"][name][key])

    from rl_vla.dice_eval import evaluate

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Evaluation must never update parameters")

    monkeypatch.setattr(DICE, "update", forbidden)
    for checkpoint, label in ((None, "base"), (str(whole / "final.pt"), "dice")):
        summary = evaluate(c, tmp_path / label, checkpoint=checkpoint, episodes=2, seed=100000)
        assert summary["success_rate"] == 1.0
        assert summary["optimizer_updates"] == 0
    assert env.closed


def test_multisample_target_averages_values_before_bootstrapping():
    torch.manual_seed(42)
    model = DICE(3, 4, 2, SETTINGS)
    batch = {
        "features": torch.randn(2, 3),
        "actions": torch.randn(2, 2),
        "returns": torch.ones(2),
        "rewards": torch.tensor([0.0, 1.0]),
        "discounts": torch.tensor([0.9, 0.0]),
        "expert": torch.zeros(2),
    }
    props = (torch.randn(2, 2, 3), torch.randn(2, 2, 4), torch.randn(2, 2, 2))
    nxt = (torch.randn(2, 3, 3), torch.randn(2, 3, 4), torch.randn(2, 3, 2))
    with torch.no_grad():
        nf, nz, np_ = (x.reshape(6, -1) for x in nxt)
        values = model.q(nf, model.action(nf, nz, np_), target=True).min(0).values
        target = batch["rewards"] + batch["discounts"] * values.reshape(2, 3).mean(1)
        expected = (model.q(batch["features"], batch["actions"]) - target).square().mean(1).sum()
    actual = model.update(batch, props, nxt, step=2)
    assert actual["critic_loss"] == pytest.approx(float(expected))


def test_offline_ratio_decays_and_clamps():
    from rl_vla.dice_runtime import expert_fraction

    data = {
        "expert_fraction": 0.5,
        "expert_fraction_end": 0.1,
        "expert_fraction_decay_decisions": 100,
    }
    assert expert_fraction(data, 0) == 0.5
    assert expert_fraction(data, 50) == pytest.approx(0.3)
    assert expert_fraction(data, 500) == pytest.approx(0.1)


def test_preflight_reports_fresh_host_without_loading_jax(tmp_path, monkeypatch):
    import rl_vla.dice as runner

    config = config_from("configs/dice_pi05_libero.toml")
    monkeypatch.chdir(tmp_path)
    result = runner.check(config)
    assert result["setup_ok"] is False
    assert result["training_started"] is False
    assert "third_party/openpi" in result["missing"]
