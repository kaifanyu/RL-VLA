"""Protect layout meaning, held-out data, cancellation and continuous online checks."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_policy import enforce, require_active_queue
from prepare_object_campaign_grasp import ITEMS, TASKS, freeze, permute_object1
from run_pro_task_grasp import cached_validation

from rl_vla import checkpoint_validation as validation


def test_six_cycle_moves_every_food_once_and_preserves_goal_basket_and_geometry():
    source = "(language cream cheese)\n(regions unchanged)\n(:init\n" + "\n".join(
        f"(On {item} floor_region_{i})" for i, item in enumerate(ITEMS))
    source += "\n(On basket_1 floor_bin_region))\n(:goal (In cream_cheese_1 basket_1))"
    changed, mapping = permute_object1(source)
    assert len(mapping) == 6
    assert all(row["before"] != row["after"] for row in mapping.values())
    assert {row["before"] for row in mapping.values()} == {row["after"] for row in mapping.values()}
    assert changed.split("(:init")[0] == source.split("(:init")[0]
    assert changed.split("(:goal")[1] == source.split("(:goal")[1]
    assert "(On basket_1 floor_bin_region)" in changed
    assert len(re.findall(r"\(On", changed)) == 7
    assert mapping["cream_cheese_1"]["after"] == "floor_region_5"


@pytest.mark.parametrize("key", ["object_2", "object_1_hard"])
def test_protocol_distinguishes_published_and_custom_test_states(tmp_path, key):
    published = np.arange(200).reshape(50, 4)
    states = tmp_path / "published.npy"
    np.save(states, published)
    item = {"path": str(states), "sha256": "fixture"}
    spec = TASKS[key] | {"key": key, "suite": "libero_object", "max_steps": 280}
    assets = {"spec": spec, "pro_source": "pinned", "condition": "custom" if key.endswith("hard") else "swap",
              "condition_suite": "suite", "test_scope": "custom_test" if key.endswith("hard") else "published_test"}
    assets.update({field: item for field in ("bddl", "published_states", "source_bddl", "source_states", "screen_states")})
    count = spec["generated_count"]
    generated = np.arange(count * 4).reshape(count, 4) + 1000
    freeze(tmp_path, assets, generated, [{"generation_seed": i} for i in range(count)], [])
    plan = json.loads((tmp_path / "protocol.json").read_text())
    actual = np.load(tmp_path / "pro_states.npy")
    assert actual.shape == (200, 4) and len(np.unique(actual, axis=0)) == 200
    task = plan["tasks"][key]
    assert task["train_state_ids"] == list(range(100))
    assert task["validation_state_ids"] == list(range(100, 150))
    assert task["test_state_ids"] == list(range(150, 200))
    assert plan["online_validation"]
    assert plan["validation_iterations"] == [25, 50, 100]
    if key == "object_2":
        np.testing.assert_array_equal(actual[150:], published)
    else:
        assert set(actual[:, 0]) == set(generated[:, 0])
        assert not set(actual[:, 0]) & set(published[:, 0])
        assert task["test_scope"] == "custom_test"


def test_user_policy_and_cancelled_queue_prevent_old_work(tmp_path):
    for method in ("baseline", "rldt", "flow_sde"):
        enforce(method, "libero_object")
        with pytest.raises(RuntimeError, match="paused"):
            enforce(method, "libero_spatial")
    with pytest.raises(RuntimeError, match="disabled"):
        enforce("flow_noise", "libero_object")
    (tmp_path / "CANCELLED.json").write_text("{}")
    with pytest.raises(RuntimeError, match="cancelled"):
        require_active_queue(tmp_path)


@pytest.mark.parametrize("method", ["rldt", "flow_sde"])
def test_online_check_uses_only_validation_and_retains_training_counters(tmp_path, monkeypatch, method):
    training = tmp_path / "training"
    training.mkdir()
    checkpoint = training / "iteration25"
    definition = {"validation_state_ids": list(range(100, 150))}
    experiment = {"method": method, "task_key": "object_2", "task_directory": str(tmp_path / "task"),
                  "iterations": [25, 50, 100], "protocol": {"online_validation": True, "tasks": {"object_2": definition}}}
    (tmp_path / "experiment.json").write_text(json.dumps(experiment))
    counters = {"decisions": 800, "primitive_steps": 3998, "elapsed_seconds": 42.0}

    def run(command, **kwargs):
        assert command[-2:] == ["--checkpoint", str(checkpoint)]
        assert kwargs["env"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.20"
        if method == "rldt":
            assert command[command.index("--config") + 1].endswith("validation.toml")
        else:
            assert command[command.index("--split") + 1] == "validation"
        destination = Path(command[command.index("--output") + 1])
        destination.write_text(json.dumps({"checkpoint": str(checkpoint), "success_rate": .4, "episodes": [
            {"seed": 100000 + i, "reset": {"initial_state_id": 100 + i}, "success": i < 20}
            for i in range(50)]}))

    monkeypatch.setattr(validation.subprocess, "run", run)
    elapsed = validation.run_checkpoint_validation(training, checkpoint, 25, counters, method)
    assert elapsed >= 0
    saved = json.loads((tmp_path / "checkpoint_validations.jsonl").read_text())
    assert saved["successes"] == 20 and saved["training_counters"] == counters
    assert cached_validation(tmp_path, "validation_000025", "validation", checkpoint, definition)["success_rate"] == .4
    with pytest.raises(ValueError, match="checkpoint differs"):
        cached_validation(tmp_path, "validation_000025", "validation", training / "wrong", definition)


def test_non_campaign_training_has_no_validation_side_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(validation.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not run"))
    assert validation.run_checkpoint_validation(tmp_path / "training", tmp_path / "ckpt", 1, {}, "rldt") == 0


@pytest.mark.parametrize("method", ["rldt", "flow_sde"])
def test_iteration_75_checkpoint_does_not_launch_validation(tmp_path, monkeypatch, method):
    (tmp_path / "experiment.json").write_text(json.dumps({"method": method, "iterations": [25, 50, 100],
        "protocol": {"online_validation": True, "validation_iterations": [25, 50, 100]}}))
    monkeypatch.setattr(validation.subprocess, "run", lambda *args, **kwargs: pytest.fail("75 must not evaluate"))
    assert validation.run_checkpoint_validation(tmp_path / "training", tmp_path / "checkpoint75", 75, {}, method) == 0
    assert list(tmp_path.iterdir()) == [tmp_path / "experiment.json"]


def test_baseline_handover_stops_only_obsolete_step_and_records_original_timing(tmp_path, monkeypatch):
    import run_object_campaign_grasp as campaign

    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setenv("SLURM_STEP_ID", "999")
    suite = tmp_path / "runs/campaign"
    output = suite / "methods/rldt/object_2"
    output.mkdir(parents=True)
    (suite / "queue").mkdir()
    (output / "baseline_validation.json").write_text("{}")
    (suite / "schedule_handover.json").write_text(json.dumps({"old_step": "1470624.97",
        "processes": [{"kind": "baseline", "pid": 999999999}]}))
    old_status = suite / "amendments/remove_validation_75/before" / output.relative_to(tmp_path) / "status.json"
    old_status.parent.mkdir(parents=True)
    old_status.write_text(json.dumps({"updated": "2026-10-28T21:21:06-04:00"}))
    commands = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda command, **kwargs: commands.append(command))
    campaign.finish_existing_baseline(suite)
    assert commands == [["scancel", "--signal=KILL", "1470624.97"]]
    assert json.loads((suite / "schedule_handover.json").read_text())["state"] == "completed"
    assert json.loads((output / "stage_timings.jsonl").read_text())["stage"] == "baseline_validation"
