"""Protocol checks for the upstream PPO experiment queue, without GPUs."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("pi_rl_task", ROOT / "scripts/run_pi_rl_task_grasp.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_readiness_rejects_missing_smoke(tmp_path):
    path = tmp_path / "gates.json"
    path.write_text(json.dumps({name: name != "flow_noise_smoke_passed" for name in runner.GATES}))
    with pytest.raises(RuntimeError, match="flow_noise_smoke_passed"):
        runner.require_readiness(path)


def test_actual_upstream_checkpoint_layout(tmp_path):
    cfg = {"runner": {"logger": {"experiment_name": "spatial_1_flow_sde"}}}
    assert runner.checkpoint_path(tmp_path, cfg, 25) == (
        tmp_path / "training/spatial_1_flow_sde/checkpoints/global_step_25")


def test_metrics_aggregate_workers_and_preserve_missing_diagnostics():
    metrics = [{"actor/policy_loss": 1, "critic/explained_variance": None},
               {"actor/policy_loss": 3, "actor/grad_norm": 2}]
    assert runner.flatten_numeric(metrics) == {"actor/policy_loss": [1., 3.], "actor/grad_norm": [2.]}


def test_frozen_source_change_is_rejected(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    runner.write_json(tmp_path / "input_contract.json", {str(source): runner.sha256(source)})
    runner.verify(tmp_path)
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="Frozen pi-RL input changed"):
        runner.verify(tmp_path)


@pytest.mark.parametrize("schedule", [[25, 50, 75, 100], [25, 50, 100]])
def test_selects_before_final_test_and_trains_only_once(tmp_path, monkeypatch, schedule):
    output = tmp_path / "run"
    output.mkdir()
    (output / "experiment.json").write_text(json.dumps({"validation_iterations": schedule}))
    cfg = {"runner": {"logger": {"experiment_name": "test"}}}
    for iteration in runner.ITERATIONS:
        runner.checkpoint_path(output, cfg, iteration).mkdir(parents=True)
    monkeypatch.setattr(runner, "freeze", lambda *args: ("spatial_1", {}, cfg))
    stages = []

    def stage(task, out, key, definition, name, split=None, checkpoint=None):
        stages.append(name)
        if name == "train":
            return {"iterations": 100, "decisions": 3200}
        if split == "test":
            assert json.loads((out / "selection.json").read_text())["iteration"] == 50
        rate = {"validation_000025": .5, "validation_000050": .8,
                "validation_000075": .8, "validation_000100": .7}.get(name, .5)
        return {"success_rate": rate, "episodes": []}

    monkeypatch.setattr(runner, "run_stage", stage)
    monkeypatch.setattr(runner, "training_analysis", lambda *args: [])
    monkeypatch.setattr(runner, "validation_plot", lambda *args: None)
    monkeypatch.setattr(runner, "compare", lambda *args: {})
    runner.run(tmp_path, output, "flow_sde", tmp_path / "gates.json")
    assert stages == ["baseline_validation", "train", *[f"validation_{i:06d}" for i in schedule],
                      "baseline_test", "selected_test", "baseline_source", "selected_source"]
    assert json.loads((output / "status.json").read_text())["state"] == "completed"
