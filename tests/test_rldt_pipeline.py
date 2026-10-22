"""Numerical and CPU integration checks for RLDT replay, updates, and checkpointing."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
pytest.importorskip("gymnasium")

from rl_vla.rldt import Learner, evaluate, execute_chunk, load_config, train
from rl_vla.rldt_env import ProcessEnv, make_primitive_env
from rl_vla.rldt_replay import TrajectoryReplay, lambda_returns
from rl_vla.rldt_toy import ToyFlow


def test_lambda_returns_bootstrap_timeouts_without_crossing_resets():
    rewards = jnp.array([1.0, 2.0, 10.0, 20.0])
    # Row 1 is a timeout: bootstrap V(s_final), but stop the trajectory trace.
    # Row 3 is a true terminal: its large arbitrary next value is irrelevant.
    discounts = jnp.array([0.9, 0.9, 0.9, 0.0])
    next_values = jnp.array([3.0, 4.0, 5.0, 999.0])
    trace = jnp.array([1.0, 0.0, 1.0, 1.0])
    actual = lambda_returns(rewards, discounts, next_values, trace, 0.5)
    np.testing.assert_allclose(actual, [4.87, 5.6, 21.25, 20.0], rtol=1e-6)
    altered = lambda_returns(rewards.at[2:].set(10000.0), discounts, next_values, trace, 0.5)
    np.testing.assert_array_equal(actual[:2], altered[:2])


@pytest.mark.parametrize("td_lambda", [0.0, 1.0])
def test_lambda_returns_endpoints_and_forced_last_boundary(td_lambda):
    rewards = jnp.array([1.0, 2.0])
    discounts = jnp.array([0.5, 0.5])
    values = jnp.array([4.0, 8.0])
    # Last trace is deliberately 1: collection windows must still bootstrap.
    trace = jnp.ones(2)
    returns = jax.jit(lambda_returns)(rewards, discounts, values, trace, td_lambda)
    expected = [3.0, 6.0] if td_lambda == 0 else [4.0, 6.0]
    np.testing.assert_allclose(returns, expected)


class ScriptedEnv:
    def __init__(self, *, terminal=False, autoreset=False):
        self.low, self.high = np.array([-1.0]), np.array([1.0])
        self.terminal, self.autoreset = terminal, autoreset
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        index = len(self.actions)
        boundary = index == 2
        info = {"is_success": self.terminal and boundary}
        if self.autoreset:
            info["final_observation"] = {"state": index}
        return {"state": index}, float(index), boundary and self.terminal, (
            boundary and not self.terminal
        ), info


@pytest.mark.parametrize("terminal", [False, True])
def test_chunk_execution_uses_actual_length_and_preserves_final_observation(terminal):
    env = ScriptedEnv(terminal=terminal)
    raw, reward, terminated, truncated, info = execute_chunk(env, np.full((4, 1), 2.0), 0.5)
    assert raw == {"state": 2}
    assert reward == 2.0
    assert terminated == terminal and truncated != terminal
    assert info["executed_steps"] == 2 and info["raw_return"] == 3.0
    assert info["discount"] == (0.0 if terminal else 0.25)
    np.testing.assert_array_equal(env.actions, np.ones((2, 1)))


def test_chunk_execution_rejects_autoreset_and_nonfinite_actions():
    with pytest.raises(ValueError, match="Auto-reset"):
        execute_chunk(ScriptedEnv(autoreset=True), np.ones((4, 1)), 0.99)
    env = ScriptedEnv()
    with pytest.raises(FloatingPointError, match="Nonfinite policy"):
        execute_chunk(env, np.full((4, 1), np.nan), 0.99)
    assert not env.actions


def _segment(name, reward, length=1):
    return [{"name": name, "reward": reward}] * length


def test_replay_balance_fallback_and_whole_segment_eviction():
    replay = TrajectoryReplay(5, seed=3)
    old = _segment("old", 0.0, 3)
    positive = _segment("positive", 1.0, 2)
    zero = _segment("zero", 0.0, 3)
    for segment in (old, positive, zero):
        replay.add(segment)
    assert replay.size == 5
    assert list(replay.segments) == [positive, zero]
    selected = replay.sample(10, balanced=True)
    assert sum(segment is positive for segment in selected) == 5
    assert sum(segment is zero for segment in selected) == 5
    with pytest.raises(ValueError, match="capacity"):
        replay.add(_segment("too long", 0.0, 6))
    for reward in (0.0, 1.0):
        only_pool = TrajectoryReplay(2)
        segment = _segment("only", reward)
        only_pool.add(segment)
        assert all(item is segment for item in only_pool.sample(8, balanced=True))
    with pytest.raises(ValueError, match="empty replay"):
        TrajectoryReplay(2).sample(1)


def test_replay_keeps_configured_fresh_segment_fraction():
    replay = TrajectoryReplay(10)
    old = _segment("old", 0.0)
    fresh = _segment("fresh", 1.0)
    replay.add(old)
    selected = replay.sample(6, balanced=False, fresh=[fresh], fresh_fraction=1 / 3)
    assert sum(segment is fresh for segment in selected) == 2
    assert sum(segment is old for segment in selected) == 4


def test_replay_odd_old_count_samples_both_strata_and_zero_fresh_is_respected():
    replay = TrajectoryReplay(10, seed=9)
    positive, zero, fresh = (_segment("positive", 1.), _segment("zero", 0.),
                             _segment("fresh", 1.))
    for segment in (positive, zero, fresh):
        replay.add(segment)
    old = [replay.sample(2, fresh=[fresh], fresh_fraction=.33)[1] for _ in range(100)]
    assert any(segment is positive for segment in old)
    assert any(segment is zero for segment in old)
    assert all(segment is not fresh for segment in old)
    assert all(segment is not fresh for segment in replay.sample(
        20, fresh=[fresh], fresh_fraction=0))


def _tiny_config():
    root = Path(__file__).resolve().parents[1]
    config = copy.deepcopy(load_config(root / "configs/rldt_toy.toml"))
    config["policy"]["horizon"] = 2
    config["env"]["kwargs"].update(max_steps=3, success_radius=1e-6)
    config["train"].update(
        iterations=2, collect_decisions=4, critic_steps=1, actor_steps=1,
        critic_batch_size=2, actor_batch_size=1, critic_trajectories=1,
        save_every=1,
    )
    config["rldt"].update(
        ode_steps=2, execute_steps=2, particles=2, bootstrap_samples=1,
        critic_width=8, critic_warmup_iterations=1,
    )
    return config


def _assert_same_state(left, right):
    leaves_left, structure_left = jax.tree.flatten(left)
    leaves_right, structure_right = jax.tree.flatten(right)
    assert structure_left == structure_right
    for a, b in zip(leaves_left, leaves_right, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("clip_bootstrap", [False, True])
def test_critic_targets_average_particlewise_twin_minimum_and_polyak(clip_bootstrap):
    config = _tiny_config()
    config["rldt"].update(bootstrap_samples=2, clip_bootstrap=clip_bootstrap)
    learner = Learner(ToyFlow(config["policy"]), 4, config)
    # Exact linear twin targets Q1(a)=a and Q2(a)=3-a. At a={0,4},
    # mean(min(Q1,Q2))=-0.5; min(mean(Q1),mean(Q2))=1 would be incorrect.
    zero = jnp.zeros((8, 1))
    q = [[{"w": zero, "b": jnp.zeros(1)}] for _ in range(2)]
    target = [
        [{"w": zero.at[4, 0].set(1.0), "b": jnp.zeros(1)}],
        [{"w": zero.at[4, 0].set(-1.0), "b": jnp.array([3.0])}],
    ]
    next_actions = jnp.zeros((4, 2, 2)).at[:, 0, 0].set(jnp.array([0, 4, 100, -100]))
    data = {
        "features": jnp.zeros((2, 4)), "actions": jnp.zeros((2, 2, 2)),
        "next_features": jnp.zeros((4, 4)), "next_actions": next_actions,
        "rewards": jnp.array([2.0, 3.0]), "discounts": jnp.array([0.8, 0.0]),
        "trace": jnp.zeros(2),
    }
    updated, updated_target, _, metrics = learner._update_critic(
        q, target, learner.q_tx.init(q), data, jnp.arange(2),
    )
    expected = np.array([2.0 if clip_bootstrap else 1.6, 3.0])
    np.testing.assert_allclose(metrics["target_mean"], expected.mean())
    np.testing.assert_allclose(metrics["critic_loss"], np.square(expected).mean())
    rho = config["rldt"]["target_momentum"]
    for actual, old, new in zip(
        jax.tree.leaves(updated_target), jax.tree.leaves(target),
        jax.tree.leaves(updated), strict=True,
    ):
        np.testing.assert_allclose(actual, rho * old + (1 - rho) * new)


def test_toy_train_updates_saves_restores_and_evaluates(tmp_path):
    config = _tiny_config()
    summary = train(config, str(tmp_path / "run"))
    assert summary["decisions"] == 8
    assert summary["critic_updates"] == 2
    assert summary["actor_updates"] == 1
    assert summary["episodes"] >= 1
    assert summary["primitive_steps"] >= summary["decisions"]
    checkpoint = Path(summary["checkpoint"])
    assert (checkpoint / "actor.npz").is_file()
    assert (checkpoint / "learner.npz").is_file()
    assert (tmp_path / "run/iteration_000001/learner.npz").is_file()
    metrics = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert "actor_loss" not in metrics[0]["metrics"]
    assert metrics[1]["metrics"]["actor_update_norm"] > 0.0
    assert metrics[1]["metrics"]["actor_grad_norm"] > 0.0
    assert all(np.isfinite(value) for row in metrics for value in row["metrics"].values())

    flow = ToyFlow(config["policy"])
    loaded = flow.load_params(checkpoint)
    assert any(not np.array_equal(loaded[key], flow.params[key]) for key in loaded)
    learner = Learner(flow, 4, config)
    learner.restore(checkpoint)
    _assert_same_state(learner.params, loaded)
    # Moving teacher must equal the actor at the completed iteration boundary.
    _assert_same_state(learner.params, learner.reference)

    clone = Learner(ToyFlow(config["policy"]), 4, config)
    clone.restore(checkpoint)
    _assert_same_state(learner.state(), clone.state())
    observation = {"state": np.array([0.1, 0.2, -0.2, 0.4], np.float32)}
    # Restores include actor optimizer moments and RNG, not just actor weights.
    metric_a = learner.actor_step([observation])
    metric_b = clone.actor_step([observation])
    _assert_same_state(learner.state(), clone.state())
    for key in metric_a:
        np.testing.assert_array_equal(metric_a[key], metric_b[key])

    result = evaluate(config, str(tmp_path / "eval.json"), str(checkpoint), episodes=2, seed=77)
    repeated = evaluate(config, str(tmp_path / "eval_repeat.json"), str(checkpoint), episodes=2, seed=77)
    assert result["episodes"] == repeated["episodes"]
    assert len(result["episodes"]) == 2
    assert all(row["primitive_steps"] <= 3 for row in result["episodes"])
    assert all(np.isfinite(row["return"]) for row in result["episodes"])
    assert json.loads((tmp_path / "eval.json").read_text())["checkpoint"] == str(checkpoint)


def test_toy_sampling_detaches_trajectory_but_velocity_remains_trainable():
    flow = ToyFlow({"horizon": 2})
    context = flow.prepare([{"state": np.array([0.1, 0.2, 0.4, 0.5], np.float32)}])
    key = jax.random.PRNGKey(0)
    gradients = jax.grad(lambda params: sum(
        item.sum() for item in flow.sample(params, context, key, 4)
    ))(flow.params)
    for gradient in jax.tree.leaves(gradients):
        np.testing.assert_array_equal(gradient, 0.0)
    gradient = jax.grad(lambda params: flow.velocity(
        params, context, jnp.ones((1, 2, 2)), jnp.ones((1,))
    ).sum())(flow.params)
    assert any(np.any(value != 0.0) for value in jax.tree.leaves(gradient))


def test_process_environment_matches_local_environment_and_closes():
    pytest.importorskip("openpi_client")
    config = {"kind": "toy", "kwargs": {"max_steps": 2}}
    local = make_primitive_env(config)
    process = ProcessEnv(config | {"python": sys.executable})
    try:
        left, left_info = local.reset(seed=123)
        right, right_info = process.reset(seed=123)
        assert left_info == right_info
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        for _ in range(2):
            left = local.step(np.array([0.1, -0.1], np.float32))
            right = process.step(np.array([0.1, -0.1], np.float32))
            for key in left[0]:
                np.testing.assert_array_equal(left[0][key], right[0][key])
            assert left[1:] == tuple(right[1:])
    finally:
        local.close()
        process.close()
    assert process.process.poll() == 0
