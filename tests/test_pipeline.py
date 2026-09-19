"""Exercise collection, learning, serialization and evaluation together on CPU."""

import json
import math
import subprocess
import sys
from copy import deepcopy
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from rl_vla.config import load_config
from rl_vla.evaluate import evaluate
from rl_vla.runtime import build_env, load_checkpoint, make_learner, seed_everything
from rl_vla.train import train

ROOT = Path(__file__).resolve().parents[1]


def assert_finite_tree(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite_tree(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


@pytest.fixture(scope="module", params=["sac", "ppo"])
def trained_run(request, tmp_path_factory):
    algorithm = request.param
    config = load_config(ROOT / "configs" / f"{algorithm}_toy.toml")
    config["train"].update({
        "seed": 23,
        "total_decisions": 72 if algorithm == "sac" else 64,
        "batch_size": 8,
        "learning_starts": 8,
        "rollout_steps": 16,
        "replay_capacity": 128,
        "log_every": 16,
        "save_every": 32,
    })
    config[algorithm].update({"hidden_dim": 32})
    if algorithm == "ppo":
        config[algorithm].update({"epochs": 2, "target_kl": 0.2})

    seed_everything(config["train"]["seed"], threads=1)
    env = build_env(config)
    try:
        env.reset(seed=config["train"]["seed"])
        initial = deepcopy(make_learner(config, env).state_dict())
    finally:
        env.close()

    directory = tmp_path_factory.mktemp(f"pipeline_{algorithm}") / "run"
    summary = train(config, str(directory))
    saved = load_checkpoint(directory / "final.pt")
    return config, directory, summary, initial, saved


def test_training_updates_weights_and_writes_reviewable_artifacts(trained_run):
    config, directory, summary, initial, saved = trained_run
    total = config["train"]["total_decisions"]
    assert summary["decisions"] == total
    assert total <= summary["primitive_steps"] <= total * config["chunk"]["execute_steps"]
    assert summary["episodes"] > 0
    assert summary["updates"] > 0
    assert saved["counters"]["decisions"] == total
    assert saved["learner"]["updates"] == summary["updates"]
    assert saved["config"] == config
    assert_finite_tree(saved)
    assert_finite_tree(summary)

    for component in ("actor", "critic" if config["algorithm"] == "sac" else "value"):
        assert any(
            not torch.equal(weight, initial[component][name])
            for name, weight in saved["learner"][component].items()
        ), f"{component} did not learn"

    assert (directory / "step_32.pt").is_file()
    assert (directory / "step_64.pt").is_file()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["config"] == config
    assert manifest["base_metadata"]["kind"] == "mock"
    assert manifest["dimensions"]["action"] > 0
    assert manifest["packages"]["torch"]
    assert_finite_tree(manifest)
    records = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert records[-1]["decisions"] == total
    assert records[-1]["metrics"]
    assert all(a["decisions"] <= b["decisions"] for a, b in pairwise(records))
    assert_finite_tree(records)
    assert json.loads((directory / "summary.json").read_text()) == summary


def test_reload_evaluation_repeats_local_sampling_seeds(trained_run, tmp_path):
    config, directory, _, _, _ = trained_run
    checkpoint = str(directory / "final.pt")
    first = evaluate(config, str(tmp_path / "first.json"), checkpoint, episodes=3, seed=404)
    second = evaluate(config, str(tmp_path / "second.json"), checkpoint, episodes=3, seed=404)
    assert first["controller"] == "residual"
    assert first["results"] == second["results"]
    assert [row["seed"] for row in first["results"]] == [404, 405, 406]
    assert 0 <= first["success_rate"] <= 1
    assert first["success_95pct_wilson"][0] <= first["success_rate"]
    assert first["success_rate"] <= first["success_95pct_wilson"][1]
    assert_finite_tree(first)


def test_base_evaluation_needs_no_learner_checkpoint(trained_run, tmp_path):
    config, _, _, _, _ = trained_run
    result = evaluate(config, str(tmp_path / "base.json"), episodes=3, seed=404)
    assert result["controller"] == "frozen_base"
    assert result["checkpoint"] is None
    assert len(result["results"]) == 3
    assert all(row["primitive_steps"] >= row["decisions"] > 0 for row in result["results"])
    assert_finite_tree(result)


def test_warm_start_keeps_learned_weights_and_restarts_collection(trained_run, tmp_path):
    config, directory, _, _, saved = trained_run
    resumed = deepcopy(config)
    resumed["train"]["total_decisions"] = 4 if config["algorithm"] == "sac" else 16
    resumed["train"]["learning_starts"] = 8
    checkpoint = str(directory / "final.pt")
    output = tmp_path / "warm_start"
    summary = train(resumed, str(output), init_from=checkpoint)
    restored = load_checkpoint(output / "final.pt")
    assert summary["decisions"] == resumed["train"]["total_decisions"]
    assert restored["learner"]["updates"] == saved["learner"]["updates"] + summary["updates"]
    if config["algorithm"] == "sac":
        # Fresh replay has not reached warmup, so loaded weights must survive exactly.
        assert summary["updates"] == 0
        for component in ("actor", "critic", "target_critic"):
            for name, weight in saved["learner"][component].items():
                torch.testing.assert_close(restored["learner"][component][name], weight, rtol=0, atol=0)
    else:
        assert summary["updates"] == 1
        assert any(
            not torch.equal(weight, saved["learner"]["actor"][name])
            for name, weight in restored["learner"]["actor"].items()
        )
    assert_finite_tree(restored)
    assert json.loads((output / "manifest.json").read_text())["init_from"] == checkpoint


def test_config_mismatch_rejected_before_checkpoint_is_applied(trained_run, tmp_path):
    config, directory, _, _, _ = trained_run
    changed = deepcopy(config)
    changed["base"]["identity"] = "different frozen policy"
    changed["train"]["total_decisions"] = 1
    checkpoint = str(directory / "final.pt")
    with pytest.raises(ValueError, match="Evaluation base"):
        evaluate(changed, str(tmp_path / "mismatch.json"), checkpoint, episodes=1)
    assert not (tmp_path / "mismatch.json").exists()
    with pytest.raises(ValueError, match="Warm-start base"):
        train(changed, str(tmp_path / "mismatch_run"), init_from=checkpoint)


def test_existing_artifacts_are_never_replaced(trained_run):
    config, directory, _, _, _ = trained_run
    checkpoint_bytes = (directory / "final.pt").read_bytes()
    with pytest.raises(FileExistsError):
        train(config, str(directory))
    with pytest.raises(FileExistsError):
        evaluate(config, str(directory / "summary.json"), episodes=1)
    assert (directory / "final.pt").read_bytes() == checkpoint_bytes


def test_cli_train_and_checkpoint_evaluate(tmp_path):
    output = tmp_path / "cli_run"
    config = str(ROOT / "configs" / "sac_toy.toml")
    trained = subprocess.run(
        [sys.executable, "-m", "rl_vla", "train", "--config", config,
         "--output", str(output), "--steps", "2", "--seed", "17"],
        cwd=ROOT, text=True, capture_output=True, timeout=60, check=False,
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["train"]["total_decisions"] == 2
    assert manifest["config"]["train"]["seed"] == 17
    destination = tmp_path / "cli_eval.json"
    evaluated = subprocess.run(
        [sys.executable, "-m", "rl_vla", "evaluate", "--config", config,
         "--checkpoint", str(output / "final.pt"), "--output", str(destination),
         "--episodes", "2", "--seed", "707", "--deterministic"],
        cwd=ROOT, text=True, capture_output=True, timeout=60, check=False,
    )
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    result = json.loads(destination.read_text())
    assert result["controller"] == "residual"
    assert result["residual_sampling"] == "mean"
    assert [row["seed"] for row in result["results"]] == [707, 708]
