"""Checkpoint selection and orchestration must not use the final test outcomes."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("pro_spatial_runner", scripts / "run_pro_spatial_grasp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation(successes, split="validation"):
    start = {"validation": 100, "test": 150, "source": 0}[split]
    return {"config": {"policy": {}, "env": {"split": split}, "rldt": {}},
            "success_rate": successes / 50, "episodes": [
                {"seed": 100000 + i, "success": i < successes, "primitive_steps": 50,
                 "reset": {"initial_state_id": start + i, "backend_seed": i,
                           "experiment_split": split}} for i in range(50)]}


def test_selection_uses_success_then_earliest_iteration(runner):
    reports = {100: evaluation(21), 25: evaluation(20), 50: evaluation(21), 75: evaluation(19)}
    assert runner.select_checkpoint(reports) == 50
    assert runner.select_checkpoint({25: evaluation(0), 50: evaluation(0)}) == 25
    with pytest.raises(ValueError, match="validation"):
        runner.select_checkpoint({25: evaluation(50, "test")})
    with pytest.raises(ValueError, match="exactly 50"):
        runner.select_checkpoint({25: {"episodes": []}})


def test_final_test_starts_only_after_selection_is_persisted(runner, monkeypatch, tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps({
        "validation_iterations": [25, 50, 75, 100], "selection": "success, then earliest",
        "scope": "test scope"}))
    monkeypatch.setattr(runner, "verify_contract", lambda _: None)
    monkeypatch.setattr(runner, "analyze", lambda _: None)
    monkeypatch.setattr(runner, "plot_validation", lambda *args: None)
    stages = []

    def stage(experiment, name, split, *, checkpoint=None, train=False):
        stages.append(name)
        if train:
            return {"iterations": 100, "actor_updates": 392, "critic_updates": 6400}
        if split != "validation":
            selected = json.loads((experiment / "selection.json").read_text())
            assert selected["iteration"] == 50
            if checkpoint is not None:
                assert checkpoint.name == "iteration_000050"
        success = {"baseline_validation": 40, "validation_000025": 5,
                   "validation_000050": 10, "validation_000075": 10,
                   "validation_000100": 8}.get(name, 15)
        return evaluation(success, split)

    monkeypatch.setattr(runner, "run_stage", stage)
    runner.run(tmp_path)
    assert stages == ["baseline_validation", "train", "validation_000025", "validation_000050",
                      "validation_000075", "validation_000100", "baseline_test", "selected_test",
                      "baseline_source", "selected_source"]
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "completed"
