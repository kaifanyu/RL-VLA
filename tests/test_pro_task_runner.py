"""Protect validation-only selection and paired outcome accounting across tasks."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("task_runner", scripts / "run_pro_task_grasp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(successes, split):
    start = {"validation": 100, "test": 150, "source": 0}[split]
    return {"config": {"policy": {}, "env": {"split": split}, "rldt": {}},
            "success_rate": successes / 50, "episodes": [
                {"seed": 100000 + i, "success": i < successes, "primitive_steps": 50,
                 "reset": {"initial_state_id": start + i, "experiment_split": split}}
                for i in range(50)]}


@pytest.mark.parametrize("schedule", [[25, 50, 75, 100], [25, 50, 100]])
def test_selection_is_saved_before_test_and_does_not_fall_back_to_baseline(runner, monkeypatch, tmp_path, schedule):
    output = tmp_path / "run"

    def freeze(*args):
        output.mkdir()
        (output / "experiment.json").write_text(json.dumps({"validation_iterations": schedule}))
        return "object_0", {}

    monkeypatch.setattr(runner, "freeze", freeze)
    monkeypatch.setattr(runner, "analyze", lambda _: None)
    monkeypatch.setattr(runner, "validation_plot", lambda *args: None)
    stages = []

    def stage(task, out, definition, name, split, checkpoint=None, train=False):
        stages.append(name)
        if train:
            return {"iterations": 100, "decisions": 3200}
        if split != "validation":
            assert json.loads((out / "selection.json").read_text())["iteration"] == 50
            if checkpoint:
                assert checkpoint.name == "iteration_000050"
        successes = {"baseline_validation": 50, "validation_000025": 10,
                     "validation_000050": 20, "validation_000075": 20,
                     "validation_000100": 15}.get(name, 30)
        return report(successes, split)

    monkeypatch.setattr(runner, "run_stage", stage)
    runner.run(tmp_path, output)
    assert stages[-4:] == ["baseline_test", "selected_test", "baseline_source", "selected_source"]
    assert ("validation_000075" in stages) == (75 in schedule)
    assert json.loads((output / "status.json").read_text())["state"] == "completed"


def test_comparison_records_regressions_and_rejects_mismatched_states(runner):
    before, after = report(40, "test"), report(39, "test")
    result = runner.compare(before, after, "test")
    assert result["new_successes"] == 0
    assert result["lost_successes"] == 1
    assert result["changed_episodes"][0]["episode"] == 40
    after["episodes"][0]["seed"] += 1
    with pytest.raises(ValueError, match="seed/reset"):
        runner.compare(before, after, "test")


def test_resume_reuses_only_complete_untrained_baseline(runner, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "input_contract.json").write_text("{}")
    (output / "experiment.json").write_text(json.dumps({"method": "rldt", "task_directory": str(tmp_path),
        "task_key": "object_2", "protocol": {"tasks": {"object_2": {
            "suite": "libero_object", "validation_state_ids": list(range(100, 150))}}}}))
    baseline = report(0, "validation") | {"checkpoint": None}
    (output / "baseline_validation.json").write_text(json.dumps(baseline))
    assert runner.resume_before_training(tmp_path, output)[2] == baseline
    baseline["episodes"][0]["seed"] += 1
    (output / "baseline_validation.json").write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="seeds differ"):
        runner.resume_before_training(tmp_path, output)
    (output / "training").mkdir()
    with pytest.raises(RuntimeError, match="training has started"):
        runner.resume_before_training(tmp_path, output)
