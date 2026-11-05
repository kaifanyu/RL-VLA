"""Safety-critical campaign protocol and actor-schedule boundary checks."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_next_multigpu_job as runner


def test_completed_training_recovery_preserves_artifacts_and_rejects_changes(tmp_path):
    previous = tmp_path / "attempt_1"
    training = previous / "training"
    training.mkdir(parents=True)
    spec = {"job_id": "fixture", "mode": "train"}
    (previous / "job_spec.json").write_text(json.dumps(spec))
    (training / "metrics.jsonl").write_text("".join(json.dumps({"iterations": i}) + "\n" for i in range(1, 101)))
    (training / "summary.json").write_text(json.dumps({"iterations": 100, "decisions": 3200, "primitive_steps": 16000}))
    identities = {}
    for iteration in (25, 50, 75, 100):
        folder = training / f"iteration_{iteration:06d}"
        folder.mkdir()
        for name in ("expert.npz", "learner.npz", "rldt_pi05.json"):
            path = folder / name
            path.write_bytes(b"fixture checkpoint")
            stat = path.stat()
            identities[str(path.relative_to(previous))] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    spec["completed_training_reuse"] = {"attempt_directory": str(previous), "file_identity": identities,
        "sha256": {name: runner.digest(previous / name) for name in ("training/metrics.jsonl", "training/summary.json")}}
    output = tmp_path / "attempt_2"
    output.mkdir()
    runner.reuse_completed_training(spec, output)
    assert (output / "training").resolve() == training
    assert json.loads((output / "training_reuse.json").read_text())["new_training_updates"] == 0
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    spec["job_id"] = "different"
    with pytest.raises(ValueError, match="specification differs"):
        runner.reuse_completed_training(spec, invalid)
    spec["job_id"] = "fixture"
    (training / "iteration_000100/expert.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.reuse_completed_training(spec, invalid)
    assert not (invalid / "training").exists()


def report(*outcomes):
    return {"episodes": [{"success": value} for value in outcomes]}


def test_initialization_is_eligible_and_wins_earliest_tie():
    assert runner.select_iteration({0: report(True, False), 25: report(True, False),
                                    50: report(False, False), 100: report(True, False)}) == 0
    assert runner.select_iteration({0: report(False, False), 25: report(True, False),
                                    50: report(True, False)}) == 25
    with pytest.raises(ValueError, match="Initialization"):
        runner.select_iteration({25: report(True)})


def test_quarter_schedule_preserves_warmup_and_adam_state(monkeypatch):
    if importlib.util.find_spec("jax") is None:
        pytest.skip("Run with the OpenPI Python to exercise the optimizer")
    import jax.numpy as jnp
    import numpy as np
    import optax

    from rl_vla import rldt

    config = {"train": {"actor_steps": 4},
              "rldt": {"critic_warmup_iterations": 2, "actor_warmup_steps": 32,
                       "actor_lr": 2e-5, "max_grad_norm": 1},
              "campaign": {"actor_schedule": "quarter_after_iteration_25"}}

    class FixtureLearner:
        def __init__(self, flow, feature_dim, incoming):
            self.params = {"weight": jnp.array([1.], dtype=jnp.float32)}
            self.r = incoming["rldt"]

    monkeypatch.setattr(rldt, "Learner", FixtureLearner)
    evidence = runner.install_actor_schedule(config)
    learner = rldt.Learner(None, 1, config)
    assert evidence["first_reduced_update_index"] == 92
    native_schedule = optax.linear_schedule(2e-5 / 32, 2e-5, 32)
    native_tx = optax.chain(optax.clip_by_global_norm(1), optax.adam(native_schedule))
    native_state = native_tx.init(learner.params)
    gradients = {"weight": jnp.array([.3], dtype=jnp.float32)}
    for index in range(100):
        native_updates, native_state = native_tx.update(gradients, native_state, learner.params)
        candidate, learner.actor_opt = learner.actor_tx.update(gradients, learner.actor_opt, learner.params)
        expected_ratio = 1 if index < 92 else .25
        np.testing.assert_allclose(candidate["weight"], native_updates["weight"] * expected_ratio, rtol=1e-6)
        # The Adam moment/count state is identical, including across the boundary.
        np.testing.assert_allclose(learner.actor_opt[1][0].mu["weight"], native_state[1][0].mu["weight"])
        assert int(learner.actor_opt[1][0].count) == int(native_state[1][0].count) == index + 1


def test_success_timing_keeps_common_successes_separate():
    before = {"episodes": [{"seed": i, "reset": {"initial_state_id": i}, "success": success,
                             "primitive_steps": steps}
                            for i, (success, steps) in enumerate(((True, 100), (False, 280), (True, 180)))]}
    after = {"episodes": [{"seed": i, "reset": {"initial_state_id": i}, "success": success,
                            "primitive_steps": steps}
                           for i, (success, steps) in enumerate(((True, 90), (True, 120), (False, 280)))]}
    result = runner.compare(before, after)
    assert result["paired_common_success_count"] == 1
    assert result["paired_common_success_baseline_steps"] == 100
    assert result["paired_common_success_selected_steps"] == 90
    assert result["baseline_success_conditional_steps"] == 140
    assert result["selected_success_conditional_steps"] == 105
