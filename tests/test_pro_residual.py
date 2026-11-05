"""CPU tests for the LIBERO-PRO FlashSAC residual pipeline (fake simulator and base)."""

import numpy as np
import pytest

from rl_vla.pro_residual.residual import (
    ActionSpec,
    ArrayReplay,
    ObservationBuilder,
    episode_seed,
    noise_seed,
    original_seed,
)
from rl_vla.pro_residual.rollout import run_episode
from rl_vla.pro_residual.video import comparison_frames

SCALE = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.0]


def spec():
    return ActionSpec(SCALE, [-1.0] * 7, [1.0] * 7)


class FakeBase:
    """Chunk j of an episode has every coordinate equal to 0.01*call + 0.001*row (+ big x to force clips)."""

    feature_dim = 4

    def __init__(self):
        self.calls = []

    def infer(self, openpi_obs, noise_seed_value):
        call = len(self.calls)
        self.calls.append((int(openpi_obs["observation/state"][0] * 1000), noise_seed_value))
        chunk = np.zeros((10, 7), np.float32) + 0.01 * call + 0.001 * np.arange(10)[:, None]
        chunk[:, 0] = 1.2  # Outside the bounds: always clipped.
        chunk[:, 6] = -1.0
        return chunk, np.arange(4, dtype=np.float32) + call, {"action_ms": 1.0, "total_ms": 1.0}


class FakeSim:
    def __init__(self, success_at=None, max_steps=12):
        self.success_at, self.max_steps = success_at, max_steps
        self.actions = []

    def _obs(self):
        state = np.full(8, self.t / 1000.0, np.float32)
        image = np.zeros((4, 4, 3), np.uint8)
        return {"state": state, "openpi": {"observation/image": image, "observation/wrist_image": image,
                                           "observation/state": state, "prompt": "x"}}

    def _diag(self):
        return {"sim_time": 0.05 * self.t, "eef_pos": np.zeros(3), "eef_quat_xyzw": np.array([0, 0, 0, 1.0]),
                "gripper_qpos": np.zeros(2), "agentview": np.full((4, 4, 3), self.t, np.uint8), "objects": {}}

    def call(self, command, **kw):
        if command == "make":
            self.t = 0
            return {"state_index": kw["initial_state_id"], "max_steps": kw.get("max_steps") or self.max_steps,
                    "control_timestep": 0.05, "task_description": "x"}
        if command == "reset":
            self.t = 0
            return {"obs": self._obs(), "info": {}, "diag": self._diag()}
        if command == "step":
            self.actions.append(np.asarray(kw["action"]).copy())
            self.t += 1
            success = self.success_at is not None and self.t == self.success_at
            truncated = not success and self.t >= self.max_steps
            return {"obs": self._obs(), "reward": float(success), "terminated": success, "truncated": truncated,
                    "info": {"is_success": success, "native_done": False}, "diag": self._diag()}
        raise ValueError(command)


def builder():
    return ObservationBuilder(np.full(8, -1.0), np.full(8, 1.0), execute_steps=5, feature_dim=4)


def run(sim, actor=None, sink=None, record=True):
    base = FakeBase()
    result = run_episode(sim, base, spec(), builder(), execute_steps=5, gamma=0.99, make_request={},
                         seed=1, initial_state_id=0, episode_key="k", actor=actor, sink=sink, record=record)
    return result, base


def test_seed_protocols_match_existing_pipelines():
    from rl_vla.baseline import noise_seed as baseline_noise
    from rl_vla.robustness import episode_seed as queue_seed

    assert noise_seed("abc:1", 3) == baseline_noise("abc:1", 3)
    assert episode_seed("pro_libero_spatial_a05f47a612ad17c9", 7) == queue_seed("pro_libero_spatial_a05f47a612ad17c9", 7)
    assert original_seed(0, 8, 1) == 108001


def test_combination_bounds_scale_and_gripper():
    s = spec()
    assert s.action_dim == 6 and s.active.tolist() == [0, 1, 2, 3, 4, 5]
    base = np.array([0.95, -0.2, 0.0, 0.3, -0.99, 0.0, 0.7], np.float32)
    zero = s.combine(base, np.zeros(6))
    np.testing.assert_array_equal(zero.executed, np.clip(base, -1, 1))
    plus = s.combine(base, np.ones(6))
    np.testing.assert_allclose(plus.correction, SCALE)
    assert plus.executed[6] == base[6]  # Gripper follows the base exactly.
    assert plus.executed[0] == 1.0 and plus.clipped_high[0] and not plus.clipped_high[1]
    minus = s.combine(base, -np.ones(6))
    assert minus.executed[4] == -1.0 and minus.clipped_low[4]
    np.testing.assert_allclose(minus.requested, base - np.array(SCALE, np.float32))
    with pytest.raises(ValueError):
        s.combine(base, np.full(6, 1.5))
    disabled = s.combine(np.array([1.3, 0, 0, 0, 0, 0, -1.0], np.float32))
    assert disabled.executed[0] == 1.0 and disabled.clipped_high[0]


def test_schedule_and_replay_semantics_on_timeout():
    transitions = []
    sim = FakeSim(max_steps=10)  # Truncates exactly on a chunk boundary.
    result, _base = run(sim, actor=lambda x: np.full(6, 0.5, np.float32), sink=transitions.append)
    assert result["steps"] == 10 and result["termination_reason"] == "timeout"
    # Queries at env steps 0, 5, and a bootstrap-only query at 10.
    assert [r["env_step"] for r in result["inference_rows"]] == [0, 5, 10]
    assert [r["purpose"] for r in result["inference_rows"]] == ["execute", "execute", "bootstrap_only"]
    assert [r["noise_seed"] for r in result["inference_rows"]] == [noise_seed("k", i) for i in range(3)]
    # Executed actions come only from the first five rows of each chunk.
    executed = np.array(sim.actions)
    for t in range(10):
        call, row = divmod(t, 5)
        expected = 0.01 * call + 0.001 * row + 0.1 * 0.5  # y scale 0.1, v 0.5
        assert executed[t, 1] == pytest.approx(expected, abs=1e-6)
        assert executed[t, 0] == 1.0 and executed[t, 6] == -1.0
    assert [r["inference_index"] for r in result["step_rows"]] == [0] * 5 + [1] * 5
    assert len(transitions) == 10
    b = builder()
    for t, tr in enumerate(transitions):
        assert tr["discount"] == pytest.approx(0.99) and not tr["terminated"]
        assert tr["truncated"] == (t == 9)
        nxt = t + 1
        call, pos = divmod(nxt, 5)
        base_next = 0.01 * call + 0.001 * pos
        assert tr["next_obs"][8 + 1] == pytest.approx(base_next, abs=1e-6)   # next base action (y)
        assert tr["next_obs"][15 + pos] == 1.0                                # chunk position one-hot
        np.testing.assert_allclose(tr["action"], 0.5)
    np.testing.assert_allclose(transitions[4]["next_obs"], transitions[5]["obs"])
    assert len(result["frames"]) == result["steps"] + 1
    assert b.dim == transitions[0]["obs"].size


def test_terminal_success_does_not_bootstrap_or_query():
    transitions = []
    result, base = run(FakeSim(success_at=5, max_steps=30), actor=lambda x: np.zeros(6, np.float32),
                       sink=transitions.append)
    assert result["success"] and result["termination_reason"] == "success" and result["steps"] == 5
    assert len(base.calls) == 1  # No query after success at a chunk boundary.
    assert transitions[-1]["terminated"] and transitions[-1]["reward"] == 1.0
    assert not transitions[-1]["next_obs"].any()


def test_eval_timeout_mid_chunk_and_base_only():
    sim = FakeSim(max_steps=7)
    result, base = run(sim, actor=None, sink=None)
    assert result["steps"] == 7 and len(base.calls) == 2
    rows = result["step_rows"]
    assert all(r["residual_enabled"] == 0 and r["correction_y"] == 0.0 for r in rows)
    assert result["residual_metrics"]["any_clip_fraction"] == 1.0  # The base x is always clipped.


def test_flashsac_learner_starts_at_zero_and_updates(tmp_path):
    torch = pytest.importorskip("torch")
    from rl_vla.flashsac import FlashSAC

    b = builder()
    learner = FlashSAC(b.dim, b.dim, 6, {"actor_hidden_dim": 32, "critic_hidden_dim": 32, "critic_num_bins": 11,
                                         "actor_update_period": 1}, device="cpu")
    obs = torch.randn(1, b.dim)
    mean, _, _ = learner.actor.sample(obs, deterministic=True)
    assert torch.equal(mean, torch.zeros_like(mean))
    replay = ArrayReplay(64, b.dim, 6, seed=0)
    rng = np.random.default_rng(0)
    for i in range(64):
        replay.add(rng.normal(size=b.dim), rng.uniform(-1, 1, 6), float(i % 7 == 0), 0.99, rng.normal(size=b.dim),
                   i % 7 == 0, i % 11 == 0)
    actor_before = [p.detach().clone() for p in learner.actor.parameters()]
    critic_before = [p.detach().clone() for p in learner.critic.parameters()]
    for _ in range(3):
        learner.observe(1.0, 0.99, False, False)
        metrics = learner.update(replay.sample(32))
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, p) for a, p in zip(actor_before, learner.actor.parameters()))
    assert any(not torch.equal(a, p) for a, p in zip(critic_before, learner.critic.parameters()))
    replay.save(tmp_path / "r.npz")
    other = ArrayReplay(64, b.dim, 6, seed=1)
    other.load(tmp_path / "r.npz")
    np.testing.assert_array_equal(other.obs, replay.obs)
    np.testing.assert_array_equal(other.sample(8)["obs"].numpy(), replay.sample(8)["obs"].numpy())


def test_comparison_holds_last_frame():
    left = {"frames": [np.full((8, 8, 3), i, np.uint8) for i in range(3)], "label": "a", "success": True,
            "steps": 2, "max_steps": 9}
    right = {"frames": [np.full((8, 8, 3), i, np.uint8) for i in range(6)], "label": "b", "success": False,
             "timeout": True, "steps": 5, "max_steps": 9}
    frames = comparison_frames(left, right, scale=1)
    assert len(frames) == 6
    assert frames[0].shape[1] == 8 + 6 + 8


def test_episode_video_holds_latest_policy_input():
    from rl_vla.pro_residual.video import episode_frames

    frames = [np.full((8, 8, 3), k, np.uint8) for k in range(11)]
    inputs = {"front": [np.full((6, 6, 3), 100 + i, np.uint8) for i in range(3)],
              "wrist": [np.full((6, 6, 3), 200 + i, np.uint8) for i in range(3)],
              "call": [0, 1, 2], "env_step": [0, 5, 10]}
    out = episode_frames(frames, inputs, max_steps=220, scale=1)
    assert len(out) == len(frames)
    assert len({o.shape for o in out}) == 1
    header = 18 * 2 + 8
    # Right panel (after the 8-px agentview panel and a 6-px gap) shows the wrist image
    # consumed at or before frame k.
    assert out[0].shape[1] == 8 + 6 + 6
    for k, expected in ((0, 200), (4, 200), (5, 201), (9, 201), (10, 202)):
        assert out[k][header + 2, 8 + 6 + 2, 0] == expected, k


def test_state_set_collection_matches_rldt_rules(tmp_path):
    from rl_vla.pro_residual.states import assemble, collect_attempts, splits

    nq = 3
    published = np.array([[0.0, 9, 9, 9, 0], [0.0, 8, 8, 8, 0]])

    def attempt(index, state, error=""):
        path = tmp_path / f"{index:04d}.npz"
        np.savez(path, state=np.asarray(state, float), nq=nq if not error else 0, seed=100 + index, error=error)
        return path

    paths = [attempt(0, [0.5, 1, 1, 1, 7]),            # accepted
             attempt(1, [], error="State already satisfies goal"),
             attempt(2, [0.9, 9, 9, 9, 3]),            # qpos duplicate of a published state
             attempt(3, [0.1, 1, 1, 1, 2]),            # qpos duplicate of attempt 0 (time/other ignored)
             attempt(4, [0.0, 2, 2, 2, 0]),            # accepted
             attempt(5, [0.0, 3, 3, 3, 0])]            # not needed (count reached)
    generated, accepted, rejected = collect_attempts(paths, published, count=2)
    assert [a["generation_seed"] for a in accepted] == [100, 104]
    assert [r["generation_seed"] for r in rejected] == [101, 102, 103]
    combined, ordered = assemble(generated, accepted, published, split_seed=7)
    assert combined.shape == (4, 5) and np.array_equal(combined[2:], published)
    order = np.random.default_rng(7).permutation(2)
    assert [o["generation_seed"] for o in ordered] == [accepted[i]["generation_seed"] for i in order]
    assert splits(2, 1, 2) == {"train": [0, 1], "validation": [2], "test": [3, 4]}


def test_state_set_split_and_published_mapping(tmp_path):
    from rl_vla.pro_residual.spec import DEFAULTS, build_split, deep_merge, published_index
    from rl_vla.pro_residual.train import NO_MORE_VALIDATION, _next_validation

    rng = np.random.default_rng(0)
    states = rng.normal(size=(7, 5))
    path = tmp_path / "pro_states.npy"
    np.save(path, states)
    case = {"case_id": "pro_x", "name": "n", "category": "Position", "states": str(path), "states_sha256": "s",
            "state_set": {"published_offset": 4, "split_seed": 3,
                          "splits": {"train": [0, 1, 2], "validation": [3], "test": [4, 5, 6]}}}
    config = deep_merge(DEFAULTS, {"split": {"train": 3, "validation": 1, "test": 3}})
    split = build_split(config, case, {"available": False})
    assert (split["train"], split["validation"], split["test"]) == ([0, 1, 2], [3], [4, 5, 6])
    assert split["seed"] == 3 and split["unused"] == []
    assert [published_index(case, i) for i in (0, 3, 4, 6)] == [None, None, 0, 2]
    assert published_index({"case_id": "pro_x"}, 5) == 5
    bad = deep_merge(DEFAULTS, {"split": {"train": 2, "validation": 1, "test": 3}})
    with pytest.raises(ValueError):
        build_split(bad, case, {"available": False})
    schedule = {"validate_at_steps": [4000, 8000, 16000], "validate_every_steps": 4000}
    assert [_next_validation(schedule, s) for s in (0, 4100, 8000, 16050)] == [4000, 8000, 16000, NO_MORE_VALIDATION]
    assert _next_validation({"validate_at_steps": None, "validate_every_steps": 10}, 25) == 35


def test_budget_stop_ends_at_exact_command():
    sim, seen = FakeSim(max_steps=12), []
    base = FakeBase()
    result = run_episode(sim, base, spec(), builder(), execute_steps=5, gamma=0.99, make_request={}, seed=1,
                         initial_state_id=0, episode_key="k", actor=lambda x: np.zeros(6, np.float32),
                         sink=seen.append, stop=lambda: len(seen) >= 7)
    assert result["steps"] == len(seen) == len(sim.actions) == 7
    assert result["termination_reason"] == "budget_stop"
    assert not result["terminated"] and not result["truncated"] and not seen[-1]["truncated"]


def test_jax_noise_stream_matches_rldt_evaluation():
    import jax
    from types import SimpleNamespace

    from rl_vla.pro_residual.base import JAX_NOISE_PROTOCOL, NOISE_PROTOCOL, FrozenPi05Base

    fake = SimpleNamespace(noise_protocol=JAX_NOISE_PROTOCOL, noise_shape=(10, 32))
    stream = FrozenPi05Base.noise_stream(fake, 100007)
    key = jax.random.PRNGKey(100007)  # The chain of rl_vla.rldt.evaluate and Pi05Flow.sample.
    sample = jax.jit(lambda k: jax.random.normal(k, (1, 10, 32)))
    for _ in range(3):
        key, subkey = jax.random.split(key)
        np.testing.assert_allclose(stream()[0], np.asarray(sample(subkey))[0], rtol=0, atol=1e-6)
    assert FrozenPi05Base.noise_stream(SimpleNamespace(noise_protocol=NOISE_PROTOCOL), 1) is None


def test_contract_evaluation_seeds_follow_ordered_split():
    from rl_vla.pro_residual.experiment import Experiment

    exp = Experiment.__new__(Experiment)
    exp.contract = {"eval_seed_start": 100000}
    exp.split = {"validation": [100, 101, 102], "test": [150, 151]}
    exp.case = {"case_id": "c", "slug": "s", "states_sha256": "0" * 64,
                "state_set": {"published_offset": 150}}
    exp.pro_make_request = lambda: {}
    assert exp.evaluation_episode("validation", 102)["seed"] == 100002
    test = exp.evaluation_episode("test", 151)
    assert test["seed"] == 100001 and test["published_index"] == 1
    exp.split["retention"] = {"initial_state_ids": [0, 1, 2]}
    exp.original = {"available": True, "suite": "libero_object", "task_id": 0, "slug": "o", "name": "n",
                    "horizon": 280}
    exp.original_make_request = lambda: {}
    assert exp.evaluation_episode("retention", 2)["seed"] == 100002
    exp.contract = {}
    assert exp.evaluation_episode("test", 151)["seed"] == episode_seed("c", 1)
    assert exp.evaluation_episode("retention", 2)["seed"] == original_seed(1, 0, 2)


def test_contract_train_reset_matches_libero_pro_rule():
    ids = list(range(100))
    rule = [int(np.random.default_rng(0 + index).choice(ids)) for index in range(5)]
    assert rule == [int(np.random.default_rng(seed).choice(ids)) for seed in range(5)]
    assert len(set(rule)) > 1


@pytest.mark.parametrize("visible, expected", [("3,5", "--id=3"), ("", None), (None, None)])
def test_resource_check_watches_the_visible_gpu(monkeypatch, visible, expected):
    from rl_vla.pro_residual import experiment

    commands = []

    def fake_output(command, **kwargs):
        commands.append(command)
        return "1024, 40\n"

    monkeypatch.setattr(experiment.subprocess, "check_output", fake_output)
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    config = {"train": {"max_gpu_memory_gib": 46.0, "max_gpu_temperature_c": 77.0}}
    assert experiment.resource_check(config)["gpu_memory_gib"] == 1.0
    assert (expected in commands[0]) if expected else not any(c.startswith("--id") for c in commands[0])


def test_flow_backend_keys_match_rldt_evaluation():
    import jax
    from types import SimpleNamespace

    from rl_vla.pro_residual.base import FrozenPi05Flow

    stream = FrozenPi05Flow.noise_stream(SimpleNamespace(), 100003)
    key = jax.random.PRNGKey(100003)
    for _ in range(3):
        key, subkey = jax.random.split(key)
        np.testing.assert_array_equal(np.asarray(stream()[0]), np.asarray(subkey))


def test_flow_backend_requires_jax_noise():
    from rl_vla.pro_residual.spec import DEFAULTS, deep_merge, validate_config

    config = deep_merge(DEFAULTS, {"task": {"benchmark": "pro", "category": "Position", "task_id": 0},
                                   "base": {"backend": "pi05_flow"}})
    with pytest.raises(ValueError, match="requires base.noise_protocol"):
        validate_config(config)


def test_contract_retention_uses_rldt_source_environment():
    from rl_vla.pro_residual.experiment import Experiment

    exp = Experiment.__new__(Experiment)
    exp.config = {"simulator": {"max_steps": None}}
    exp.original = {"suite": "libero_object", "task_id": 0, "bddl": "b", "init_file": "s", "name": "n", "horizon": 280}
    exp.contract = {"retention_env": "source_protocol"}
    assert exp.retention_simulator() == "pro"
    assert exp.original_make_request() == {"kind": "source_protocol", "suite": "libero_object", "task_id": 0,
                                           "bddl": "b", "states": "s", "max_steps": 280}
    exp.contract = {}
    assert exp.retention_simulator() == "original" and exp.original_make_request()["kind"] == "original"
