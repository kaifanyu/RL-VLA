"""Integrity checks for dedicated-device validation and local campaign gates."""

import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rl_vla import checkpoint_validation
import run_vla_trace_objects as campaign


def test_dedicated_validation_leaves_parent_gpu_and_counters_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("RLDT_CPUSET", "0-7")
    monkeypatch.setenv("RL_VLA_VALIDATION_GPU", "4")
    monkeypatch.setenv("RL_VLA_VALIDATION_CPUSET", "32-39")
    monkeypatch.setenv("RL_VLA_EVAL_SHARDS", "3")
    plan = {"method": "rldt", "task_key": "object_0", "task_directory": str(tmp_path),
            "protocol": {"online_validation": True, "validation_iterations": [25, 50, 100],
                         "tasks": {"object_0": {"validation_state_ids": list(range(100, 150))}}}}
    (tmp_path/"experiment.json").write_text(json.dumps(plan))
    counters = {"iterations": 25, "decisions": 800, "primitive_steps": 3999}

    def fake_run(command, **kwargs):
        assert command[1].endswith("evaluate_vla_trace_sharded.py")
        assert "evaluate" not in command
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"
        assert kwargs["env"]["RLDT_CPUSET"] == "32-39"
        assert kwargs["env"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.55"
        destination = Path(command[command.index("--output")+1])
        destination.write_text(json.dumps({"success_rate": 0., "episodes": [
            {"seed": 100000+i, "reset": {"initial_state_id": 100+i}, "success": False}
            for i in range(50)]}))

    monkeypatch.setattr(checkpoint_validation.subprocess, "run", fake_run)
    checkpoint_validation.run_checkpoint_validation(tmp_path/"training", tmp_path/"checkpoint", 25, counters, "rldt")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["RLDT_CPUSET"] == "0-7"
    assert counters == {"iterations": 25, "decisions": 800, "primitive_steps": 3999}


def test_training_requires_real_attention_gate_and_exact_selection_schedule(tmp_path):
    task = tmp_path/"task"
    task.mkdir()
    (task/"protocol.json").write_text(json.dumps({"online_validation": True, "validation_iterations": [25, 50, 75, 100]}))
    gate = tmp_path/"gate.json"
    gate.write_text(json.dumps({"passed": True, "technical_only": True}))
    with pytest.raises(ValueError, match="only at"):
        campaign.freeze(task, tmp_path/"out", gate)
    (task/"protocol.json").write_text(json.dumps({"online_validation": True, "validation_iterations": [25, 50, 100]}))
    (task/"preparation_complete.json").write_text(json.dumps({"passed": True, "files": [], "source_and_task_assets": {}}))
    with pytest.raises(ValueError, match="real saved"):
        campaign.freeze(task, tmp_path/"out", gate)
    assert not (tmp_path/"out").exists()


def test_discontinuous_training_cannot_be_certified(tmp_path):
    train = tmp_path/"training"
    train.mkdir()
    (train/"manifest.json").write_text(json.dumps({"init_from": None, "frozen_vlm": True}))
    (train/"metrics.jsonl").write_text('\n'.join(json.dumps({"iterations": i}) for i in [1, 2, 4]))
    with pytest.raises(ValueError, match="discontinuous"):
        campaign.audit_training(tmp_path)
