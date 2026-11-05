"""Numerical/runner contracts; official source oracle uses the pinned checkout."""

import ast
import copy
import runpy
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest
import torch

from rl_vla.dice import config_from
from rl_vla.dice_core import DICE, regularization_mask
from rl_vla.dice_replay import chunk_discount, chunk_reward, finish_episode, sample_rows
from test_dice import SETTINGS


def inputs():
    batch = {
        "features": torch.randn(2, 3),
        "actions": torch.randn(2, 2),
        "returns": torch.ones(2),
        "rewards": torch.tensor([1.0, 2.0]),
        "discounts": torch.tensor([0.9, 0.0]),
        "expert": torch.tensor([0.0, 1.0]),
    }
    props = (torch.randn(2, 16, 3), torch.randn(2, 16, 4), torch.randn(2, 16, 2))
    return batch, props


def test_paper_gate_uses_candidate_calibration_and_inclusive_boundaries():
    q = torch.tensor([[1.0, 3.0, 0.5]])
    qp = torch.tensor([[1.0, 2.0, 1.0]])
    calibration = q - 1.5
    result = regularization_mask(
        q,
        qp,
        calibration,
        step=2,
        warmup=0,
        threshold=-0.5,
        expert=torch.ones(1),
        mode="candidate",
        retain_experts=False,
    )
    assert torch.equal(result, torch.tensor([[0.0, 1.0, 1.0]]))
    # Replay calibration can allow the second candidate even when its Q is too high.
    code = regularization_mask(
        q,
        qp,
        torch.tensor([-1.0]),
        step=2,
        warmup=0,
        threshold=-0.5,
        expert=torch.zeros(1),
        mode="replay",
    )
    assert torch.equal(code, torch.tensor([[1.0, 0.0, 1.0]]))


def test_min_precedes_expectation_and_terminal_bootstrap_is_zero():
    model = DICE(3, 4, 2, {**SETTINGS, "critic_ensemble_size": 10})
    batch, props = inputs()

    def q(features, actions, target=False):
        if target:
            # The first two critics prefer opposite particles. mean(min)=0;
            # min(mean)=8 would be an incorrect optimistic reduction.
            alternating = torch.arange(len(actions)) % 2
            return torch.stack(
                [alternating * 16, (1 - alternating) * 16] + [torch.full_like(alternating, 100)] * 8
            ).float()
        return original(features, actions)

    original = model.q
    model.q = q
    actual = model.update(batch, props, props, step=2)
    assert actual["target_mean"] == pytest.approx(1.5)
    assert len(model.critics) == 10


def test_collection_clock_bursts_polyak_and_scheduler_roundtrip(tmp_path):
    from rl_vla.dice_runtime import save_training, restore_training

    settings = {
        **SETTINGS,
        "actor_update_freq": 2,
        "target_update_freq": 2,
        "delay_clock": "collection",
        "lr_scheduler": True,
        "lr_cycle_steps": 10,
        "lr_warmup_steps": 2,
        "min_lr": 1e-6,
    }
    model = DICE(3, 4, 2, settings)
    batch, props = inputs()
    old = copy.deepcopy(model.target.state_dict())
    for _ in range(3):
        model.update(batch, props, props, step=0)
    assert model.actor_updates == model.target_updates == 0
    assert all(torch.equal(old[k], v) for k, v in model.target.state_dict().items())
    model.update(batch, props, props, step=1)
    assert model.actor_updates == model.target_updates == 1
    for key, value in model.target.state_dict().items():
        expected = (
            old[key] * (1 - settings["target_tau"])
            + model.critics.state_dict()[key] * settings["target_tau"]
        )
        torch.testing.assert_close(value, expected)
    config = {
        "policy": {},
        "env": {},
        "data": {},
        "dice": settings,
        "train": {"episodes": 3, "save_every": 1},
    }
    prior = SimpleNamespace(metadata={"fixture": True})
    replay = deque([{"array": np.arange(5)}], maxlen=10)
    rng = np.random.default_rng(4)
    save_training(
        tmp_path / "resume.pt", model, config, prior, {"episodes": 1, "decisions": 2}, replay, rng
    )
    restored = DICE(3, 4, 2, settings)
    restored_rng = np.random.default_rng(99)
    restored_replay = deque(maxlen=10)
    assert restore_training(
        tmp_path / "resume.pt", restored, config, prior, restored_replay, restored_rng
    ) == (1, 2)
    assert restored_rng.random() == rng.random()
    np.testing.assert_array_equal(restored_replay[0]["array"], replay[0]["array"])
    for clock in (1, 2, 3):
        model.update(batch, props, props, step=clock)
        restored.update(batch, props, props, step=clock)
    for group in ("actor", "critics", "target"):
        for key, value in getattr(model, group).state_dict().items():
            assert torch.equal(value, getattr(restored, group).state_dict()[key])
    assert (
        model.actor_optimizer.param_groups[0]["lr"]
        == restored.actor_optimizer.param_groups[0]["lr"]
    )


def test_reward_units_nstep_no_duplicate_rewards_or_cross_episode():
    assert chunk_reward([1.0, 2.0], 0.9, "chunk") == 3
    assert chunk_reward([1.0, 2.0], 0.9, "primitive") == 2.8
    assert chunk_discount(0.9, 5, False, "chunk") == 0.9
    assert chunk_discount(0.9, 2, True, "chunk") == 0
    rows = [
        {"rewards": r, "discounts": d, "next_observation": i}
        for i, (r, d) in enumerate(zip([1.0, 2.0, 4.0, 8.0], [0.5, 0.5, 0.5, 0.0], strict=True))
    ]
    result = finish_episode(rows, 3)
    assert result[0]["rewards"] == 3.0  # 1 + .5*2 + .25*4
    assert result[0]["discounts"] == 0.125
    assert result[0]["returns"] == 4.0
    assert result[0]["next_observation"] == 2
    assert result[-1]["rewards"] == 8 and result[-1]["discounts"] == 0
    assert rows[0]["rewards"] == 1  # MC source remains unmodified
    with pytest.raises(ValueError, match="completed"):
        finish_episode(rows[:2], 3)


def test_fixed_replay_split_and_online_only_ablation():
    rng = np.random.default_rng(0)
    rows = sample_rows(["demo"], ["online"], 16, 0.7, rng, fixed_split=True)
    assert rows.count("demo") == 11  # official int(B*r), not a Bernoulli count
    assert sample_rows([], ["online"], 4, 0, rng, fixed_split=True) == ["online"] * 4


def test_regularizer_is_not_regression_to_demonstration_actions():
    # Actor sees states and fresh same-z prior pairs. Replay actions enter the TD
    # loss; in paper calibration mode they cannot become supervised actor targets.
    settings = {**SETTINGS, "calibration_mode": "candidate", "retain_expert_regularization": False}
    a = DICE(3, 4, 2, settings)
    b = DICE(3, 4, 2, settings)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    batch, props = inputs()
    different = {**batch, "actions": batch["actions"] + 100}
    a.update(batch, props, props, step=2)
    b.update(different, props, props, step=2)
    for key, value in a.actor.state_dict().items():
        assert torch.equal(value, b.actor.state_dict()[key])
    assert any(
        not torch.equal(v, b.critics.state_dict()[k]) for k, v in a.critics.state_dict().items()
    )


def test_official_critic_oracle_matches_ensemble_outputs_and_forced_min():
    root = Path("third_party/dice_rl_upstream")
    if not (root / "model/common/mlp.py").exists():
        pytest.skip("Pinned official checkout required for source oracle")
    mlp = runpy.run_path(str(root / "model/common/mlp.py"))["MLP"]
    tree = ast.parse((root / "model/rl/distill_residual_rl.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DistilledCritic")
    scope = {"nn": torch.nn, "torch": torch, "MLP": mlp, "List": List}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned-critic", "exec"), scope)
    official = scope["DistilledCritic"](
        3,
        2,
        horizon_steps=1,
        hidden_dims=[8] * 3,
        activation_type="GELU",
        use_layernorm=True,
        critic_ensemble_size=10,
    )
    local = DICE(
        3,
        4,
        2,
        {
            **SETTINGS,
            "width": 8,
            "activation": "GELU",
            "layernorm": True,
            "critic_ensemble_size": 10,
        },
    )
    for ours, theirs in zip(local.critics, official.Q_ensemble, strict=True):
        for ours_layer, theirs_layer in zip(
            [m for m in ours.modules() if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm))],
            [m for m in theirs.modules() if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm))],
            strict=True,
        ):
            ours_layer.load_state_dict(theirs_layer.state_dict())
    features, actions = torch.randn(4, 3), torch.randn(4, 2)
    expected = torch.stack(
        official(features[:, None], None, actions[:, None], return_all=True)
    ).squeeze(-1)
    torch.testing.assert_close(local.q(features, actions), expected)
    # return_mean=True is forcibly overridden in pinned code.
    torch.testing.assert_close(
        local.q(features, actions).min(0).values,
        official(features[:, None], None, actions[:, None], return_mean=True).squeeze(-1),
    )


def test_profiles_keep_k_roles_and_ablation_labels_separate():
    paper = config_from("configs/dice_paper_pi05_object0.toml")
    benchmark = config_from("configs/dice_audit_pi05_object0.toml")
    best = config_from("configs/dice_bestof16_pi05_object0.toml")
    assert paper["dice"]["critic_ensemble_size"] == 10
    assert paper["dice"]["k_train"] == paper["dice"]["k_bootstrap"] == 16
    assert paper["data"]["mode"] == "demonstrations" and not paper["ablations"]
    assert benchmark["data"]["mode"] == "online_only_ablation" and benchmark["ablations"]
    assert benchmark["evaluation"]["k"] == 1 and best["evaluation"]["k"] == 16
    assert benchmark["dice"] == best["dice"]
    full_chunk_groups = benchmark["train"]["max_steps"] // (
        benchmark["dice"]["execute_steps"] * benchmark["train"]["reference_parallel_envs"]
    )
    assert benchmark["dice"]["selection_warmup"] < full_chunk_groups
    assert benchmark["dice"]["filter_warmup"] < full_chunk_groups


def synthetic_runner(monkeypatch):
    import rl_vla.dice as runner
    import rl_vla.dice_pi05 as adapter
    import rl_vla.rldt_env as env_module

    class Prior:
        feature_dim, noise_dim, action_dim = 3, 4, 2

        def __init__(self, *_args):
            self.metadata = {"synthetic": True}

        def propose(self, raw, count):
            noise = torch.randn(count, 4).numpy()
            return (
                np.full((count, 3), raw["value"], np.float32),
                noise,
                noise[:, :2] * 0.1,
                np.zeros((count, 2), np.float32),
            )

        def decode(self, corrected, *_args):
            return np.asarray(corrected).reshape(2, 1)

    class Env:
        low, high = np.array([-1.0]), np.array([1.0])

        def reset(self, *, seed):
            self.steps = 0
            return {"value": 0.0}, {"initial_state_id": seed}

        def step(self, _action):
            self.steps += 1
            return {"value": float(self.steps)}, float(self.steps == 5), self.steps == 5, False, {}

        def close(self):
            pass

    monkeypatch.setattr(runner, "check", lambda _c: None)
    monkeypatch.setattr(runner, "resource_check", lambda: None)
    monkeypatch.setattr(adapter, "FrozenPi05", Prior)
    monkeypatch.setattr(env_module, "make_primitive_env", lambda _c: Env())
    c = config_from("configs/dice_audit_pi05_object0.toml")
    c["env"] = {"kind": "fixture"}
    c["dice"].update(
        width=8,
        execute_steps=2,
        k_train=3,
        k_bootstrap=4,
        k_collection=2,
        particles=3,
        selection_samples=2,
        selection_warmup=0,
    )
    c["train"].update(
        episodes=4,
        batch_size=2,
        max_steps=30,
        gradient_steps_per_chunk=3,
        reference_parallel_envs=2,
        milestones=[3, 5, 8, 10, 13],
        save_every=1,
    )
    return runner, c


def test_runner_save_resume_matches_uninterrupted_and_preserves_partial_masks(
    tmp_path, monkeypatch
):
    runner, c = synthetic_runner(monkeypatch)
    runner.run(c, tmp_path / "full")
    first = copy.deepcopy(c)
    first["train"]["episodes"] = 2
    runner.run(first, tmp_path / "first")
    runner.run(c, tmp_path / "resumed", resume=tmp_path / "first/resume.pt")
    full = torch.load(tmp_path / "full/resume.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed/resume.pt", weights_only=False)
    assert full["counters"] == resumed["counters"]
    assert full["numpy_rng"] == resumed["numpy_rng"]
    assert torch.equal(full["torch_rng"], resumed["torch_rng"])
    for group in ("actor", "critics", "target"):
        for key, value in full["learner"][group].items():
            assert torch.equal(value, resumed["learner"][group][key])
    assert full["counters"]["controller_steps"] == 20
    assert len(full["replay"]) == 12
    # The final one-command chunk has no critic/regularizer dependency on its tail.
    np.testing.assert_array_equal(full["replay"][2]["action_mask"], [1.0, 0.0])
    assert full["replay"][2]["discounts"] == 0
    assert len(full["learner"]["critics"]) == 10 * 14  # 4 linears + 3 norms per critic


def test_command_cap_snapshots_do_not_label_pending_episode_terminal(tmp_path, monkeypatch):
    import json

    runner, c = synthetic_runner(monkeypatch)
    c["train"]["max_steps"] = 13
    runner.run(c, tmp_path / "capped")
    value = torch.load(tmp_path / "capped/resume.pt", weights_only=False)
    assert value["counters"]["controller_steps"] == 13
    assert len(value["replay"]) == 6  # only the first two complete episodes
    for step in c["train"]["milestones"]:
        checkpoint = torch.load(tmp_path / f"capped/step_{step:06d}.pt", weights_only=True)
        assert checkpoint["counters"]["controller_steps"] == step
        assert checkpoint["counters"]["snapshot_phase"] == "post_command_pre_episode_updates"
    assert json.loads((tmp_path / "capped/partial_trajectory.json").read_text()) == {
        "commands": 3,
        "chunks": 2,
        "reason": "budget_stop",
        "included_in_mc_replay": False,
    }


def test_disabled_components_and_k_alias_conflicts_fail_explicitly(tmp_path):
    source = Path("configs/dice_paper_pi05_object0.toml").read_text()
    for bad in (
        source.replace("filter_enabled = true", "filter_enabled = false"),
        source.replace("k_train = 16", "k_train = 1"),
    ):
        path = tmp_path / "bad.toml"
        path.write_text(bad)
        with pytest.raises(ValueError):
            config_from(path)


def test_code_q_normalization_and_expert_actor_q_mask_match_formula():
    settings = {
        **SETTINGS,
        "q_normalization": True,
        "disable_expert_actor_q": True,
        "filter_enabled": False,
    }
    model = DICE(3, 4, 2, settings)
    with torch.no_grad():
        model.actor[-1].bias.fill_(0.2)
    batch, props = inputs()
    f, z, p = (x.reshape(32, -1) for x in props)
    with torch.no_grad():
        actions = model.action(f, z, p)
        q = model.q(f, actions).min(0).values.reshape(2, 16)
        expected = -q[0].mean() / (2 * q[0].abs().mean()) + settings["bc_weight"] * 0.04
    result = model.update(batch, props, props, step=2)
    assert result["actor_loss"] == pytest.approx(float(expected), abs=1e-6)


def test_unexecuted_tail_does_not_change_critic_or_actor_update():
    a = DICE(3, 4, 2, SETTINGS)
    b = DICE(3, 4, 2, SETTINGS)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    batch, props = inputs()
    batch["action_mask"] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    changed_batch = {**batch, "actions": batch["actions"].clone()}
    changed_batch["actions"][:, 1] += 100
    changed_props = tuple(x.clone() for x in props)
    changed_props[2][:, :, 1] += 100
    a.update(batch, props, props, step=2)
    b.update(changed_batch, changed_props, props, step=2)
    for group in ("actor", "critics", "target"):
        for key, value in getattr(a, group).state_dict().items():
            assert torch.equal(value, getattr(b, group).state_dict()[key])


def test_diagnostic_evaluation_reports_k_cost_and_cannot_select(tmp_path, monkeypatch):
    from rl_vla.dice_eval import evaluate

    runner, c = synthetic_runner(monkeypatch)
    runner.run(c, tmp_path / "train")
    c["env"]["kwargs"] = {"split": "validation"}
    summary = evaluate(
        c, tmp_path / "eval", checkpoint=tmp_path / "train/final.pt", episodes=1, diagnostic=True
    )
    assert summary["inference_k"] == 1
    assert summary["prior_candidates"] == 3
    assert summary["controller_steps"] == 5
    assert summary["diagnostic"] and not summary["selection_eligible"]
    assert summary["prior_seconds"] >= 0 and summary["selection_seconds"] >= 0


def test_benchmark_launcher_selects_validation_before_test_and_costs_n_separately(
    tmp_path, monkeypatch
):
    import importlib.util
    import json
    import sys

    path = Path("scripts/launch_dice_benchmark.py")
    spec = importlib.util.spec_from_file_location("dice_benchmark_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "benchmark"
    calls = []

    def fake_run(*args):
        calls.append(list(args))
        if args[0] == "evaluate":
            split = args[args.index("--split") + 1]
            k = args[args.index("--evaluation-k") + 1]
            if split in ("test", "source"):
                assert (output / "selection.json").exists()
            if split == "validation" and k == 1:
                assert not (output / "selection.json").exists()
                folder = Path(args[args.index("--output") + 1])
                folder.mkdir(parents=True)
                (folder / "summary.json").write_text(
                    json.dumps({"selection_eligible": True, "successes": 5, "episodes": 50})
                )

    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(sys, "argv", ["launch", "--output", str(output), "--best-of-n", "16"])
    module.main()
    selection = json.loads((output / "selection.json").read_text())
    assert selection["selected"]["controller_steps"] == 0  # all ties: earliest eligible
    assert selection["test_results_used_for_selection"] is False
    evaluation = [c for c in calls if c[0] == "evaluate"]
    assert len(evaluation) == 12  # 5 validation + 4 K1 test/retention + 3 separately costed N
    assert all(c[c.index("--evaluation-k") + 1] == 1 for c in evaluation[:9])
    assert all(c[c.index("--evaluation-k") + 1] == 16 for c in evaluation[9:])


def test_native_flow_time_runs_from_one_toward_zero():
    import jax
    import jax.numpy as jnp
    from rl_vla.rldt_pi05 import Pi05Flow

    flow = Pi05Flow()
    flow.horizon, flow.action_dim = 2, 3
    flow.velocity = lambda params, context, x, t, *, frozen=None: jnp.broadcast_to(
        t[:, None, None], x.shape
    )
    final, path = flow.sample(None, {"state": jnp.zeros((1, 3))}, jax.random.key(9), 4)
    # v=t distinguishes descending [1,.75,.5,.25] from an increasing-time solver.
    np.testing.assert_allclose(path[1], path[0] - 0.25, atol=1e-6)
    np.testing.assert_allclose(final, path[0] - 0.625, atol=1e-6)


@pytest.mark.parametrize(
    ("rewards", "ending", "message"),
    [
        ([0.0, 1.0, 1.0], "terminated", "first success"),
        ([0.0, 0.0, 0.5], "terminated", "binary"),
        ([0.0, 0.0, 1.0], "truncated", "ending disagrees"),
    ],
)
def test_profile_demo_rewards_match_single_success_online_objective(rewards, ending, message):
    from rl_vla.dice_data import convert_episode

    provenance = {
        "task": {"task_key": "t", "task_suite": "s", "task_id": 0, "instruction": "x"},
        "environment": {"adapter": "fixture", "split": "train"},
        "demonstration": {"source_id": "fixture", "source_split": "train"},
    }
    episode = {
        "provenance": provenance,
        "actions": np.zeros((3, 7)),
        "observations": [{"openpi": {"prompt": "x"}}] * 4,
        "rewards": rewards,
        "ending": ending,
    }
    with pytest.raises(ValueError, match=message):
        convert_episode(episode, SimpleNamespace(), 0.99, provenance, require_sparse_success=True)
