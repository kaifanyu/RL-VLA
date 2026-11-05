"""Synthetic CPU serialization fixtures; these are not campaign results."""

from pathlib import Path
import importlib.util
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_vla.rldt import Learner, load_config
from rl_vla.rldt_toy import ToyFlow
from rl_vla.diagnostics.learner_restore import (
    check_restored_state, expected_updates, tree_records, tree_schema,
)


def fixture(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/rldt_toy.toml")
    flow = ToyFlow(config["policy"])
    learner = Learner(flow, 4, config)
    original = tree_records(flow.params)
    template = tree_schema(learner.state())
    expected = expected_updates(config, 3)
    # Set synthetic serialized counters without performing an optimizer step.
    def counts(tree, number):
        return jax.tree_util.tree_map_with_path(
            lambda path, value: jnp.asarray(number, value.dtype)
            if getattr(path[-1], "name", None) == "count" else value, tree)
    learner.actor_opt = counts(learner.actor_opt, expected["actor"])
    learner.q_opt = counts(learner.q_opt, expected["critic"])
    learner.params = jax.tree.map(lambda value: value + .125, learner.params)
    learner.reference = jax.tree.map(lambda value: value.copy(), learner.params)
    checkpoint = tmp_path / "checkpoint"
    learner.save(checkpoint)
    restored = Learner(flow, 4, config)
    restored.restore(checkpoint)
    return restored, checkpoint, template, original, expected


def test_actual_restore_checks_all_serialized_groups_and_moving_reference(tmp_path):
    learner, checkpoint, template, original, expected = fixture(tmp_path)
    result = check_restored_state(learner, checkpoint, template, original, expected)
    assert result["expert_params_exact"] and result["serialized_leaf_bytes_exact"]
    assert result["reference_equals_checkpoint_params"]
    assert not result["reference_equals_original_params"]
    assert result["expected_optimizer_updates"] == {"actor": 4, "critic": 12}
    assert len(result["optimizer_counts"]["actor"]) == 2
    assert result["prng_key"]["shape"] == [2] and result["prng_key"]["dtype"] == "uint32"
    assert tree_records(learner.flow.params) == original


def test_nonfinite_restored_critic_is_rejected(tmp_path):
    learner, checkpoint, template, original, expected = fixture(tmp_path)
    learner.q[0][0]["w"] = learner.q[0][0]["w"].at[0, 0].set(jnp.nan)
    with pytest.raises(ValueError, match="nonfinite"):
        check_restored_state(learner, checkpoint, template, original, expected)


def test_wrong_optimizer_count_is_rejected(tmp_path):
    learner, checkpoint, template, original, expected = fixture(tmp_path)
    with pytest.raises(ValueError, match="optimizer counts disagree"):
        check_restored_state(learner, checkpoint, template, original, {**expected, "actor": 5})


def test_expert_archive_mismatch_is_rejected(tmp_path):
    learner, checkpoint, template, original, expected = fixture(tmp_path)
    with np.load(checkpoint / "actor.npz") as archive:
        values = {key: archive[key].copy() for key in archive.files}
    values["b"][0] += .1
    np.savez(checkpoint / "actor.npz", **values)
    with pytest.raises(ValueError, match="expert archive"):
        check_restored_state(learner, checkpoint, template, original, expected)


def test_queue_summary_preserves_pending_failure_and_checkpoint_hashes(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_vla_trace_learner_restore_audits.py"
    spec = importlib.util.spec_from_file_location("restore_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initial = module.summary(tmp_path)
    assert initial["passed"] is None and initial["complete"] is False
    assert initial["required_checkpoints"] == 9
    artifact = tmp_path / "learner_restore_audit/object_0/iteration_000025.json"
    files = {"/fixture/checkpoint/learner.npz": {"bytes": 123, "sha256": "fixture"}}
    module.write(artifact, {"task": "object_0", "iteration": 25, "complete": True,
                           "passed": True, "checkpoint": "/fixture/checkpoint", "files": files})
    current = module.summary(tmp_path)
    assert current["completed_checkpoints"] == 1 and current["passed"] is None
    assert current["checkpoints"][0]["expected_checkpoint_files"] == files
    assert current["checkpoints"][0]["sha256"] == module.sha(artifact)
    module.write(artifact.with_name("iteration_000025.failed.json"), {"passed": False, "error": "fixture"})
    assert module.summary(tmp_path)["passed"] is False
