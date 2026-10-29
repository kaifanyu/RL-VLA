"""State construction must stay disjoint, deterministic and resumable."""

import importlib.util
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def preparation(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "pro_suite_preparation", scripts / "prepare_pro_suite_grasp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_attempt(path, seed, state, *, nq=2, error=""):
    np.savez(path, seed=seed, state=np.asarray(state, dtype=np.float64), nq=nq, error=error)
    return path


def test_duplicate_physical_states_excluded_despite_time_difference(preparation, tmp_path):
    published = np.asarray([[0., 1., 2., 0.], [0., 3., 4., 0.]])
    paths = [
        save_attempt(tmp_path / "0.npz", 10, [12., 1., 2., 9.]),
        save_attempt(tmp_path / "1.npz", 11, [0., 5., 6., 0.]),
        save_attempt(tmp_path / "2.npz", 12, [12., 5., 6., 99.]),
        save_attempt(tmp_path / "3.npz", 13, [0., 7., 8., 0.]),
    ]
    states, accepted, rejected = preparation.collect_attempts(paths, published, count=2)
    assert [row["generation_seed"] for row in accepted] == [11, 13]
    assert [row["generation_seed"] for row in rejected] == [10, 12]
    np.testing.assert_array_equal(states[0], [0., 5., 6., 0.])
    assert all(row["attempt"]["sha256"] for row in accepted)


def test_failed_generation_attempts_preserved_in_audit(preparation, tmp_path):
    published = np.asarray([[0., 1., 2., 0.]])
    paths = [save_attempt(tmp_path / "0.npz", 10, [], error="state already satisfies goal"),
             save_attempt(tmp_path / "1.npz", 11, [0., 5., 6., 0.])]
    _, accepted, rejected = preparation.collect_attempts(paths, published, count=1)
    assert accepted[0]["generation_seed"] == 11
    assert rejected == [{"generation_seed": 10, "error": "state already satisfies goal"}]


def test_published_duplicate_physical_states_rejected(preparation, tmp_path):
    published = np.asarray([[0., 1., 2., 0.], [20., 1., 2., 9.]])
    paths = [save_attempt(tmp_path / "0.npz", 10, [0., 5., 6., 0.])]
    with pytest.raises(ValueError, match="Published states have duplicate"):
        preparation.collect_attempts(paths, published)


def test_fixed_selection_and_distinct_generation_seeds(preparation):
    assert list(preparation.TASKS) == ["spatial_1", "spatial_2", "object_0"]
    seed_ranges = []
    for spec in preparation.TASKS.values():
        seed_ranges.append(set(range(spec["generation_seed"], spec["generation_seed"] + 1000)))
    assert len(set.union(*seed_ranges)) == 3000
    assert preparation.TASKS["object_0"]["max_steps"] == 280
    mapping = {"libero_spatial": ["zero", "one", "two"],
               "libero_spatial_swap": ["zero", "one", "two"]}
    assert preparation.task_spec("spatial_2", mapping)["name"] == "two"
    mapping["libero_spatial_swap"][2] = "wrong"
    with pytest.raises(ValueError, match="indices disagree"):
        preparation.task_spec("spatial_2", mapping)


def test_protocol_freeze_preserves_published_test_and_recording(preparation, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/rldt_pi05_libero.toml").write_text(
        (root / "configs/rldt_pi05_libero.toml").read_text())
    upstream = tmp_path / "third_party/libero_pro/notebooks/generate_init_states.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# pinned upstream fixture\n")
    directory = tmp_path / "object_0"
    directory.mkdir()
    published = np.arange(200, dtype=np.float64).reshape(50, 4)
    published_path = tmp_path / "published.npy"
    np.save(published_path, published)
    generated = np.arange(600, dtype=np.float64).reshape(150, 4) + 10000
    accepted = [{"generation_seed": i, "qpos_sha256": str(i)} for i in range(150)]
    spec = preparation.TASKS["object_0"] | {
        "key": "object_0", "name": "soup_to_basket", "condition_suite": "libero_object_swap"}
    placeholder = preparation.record(published_path)
    assets = {"spec": spec, "name": spec["name"], "published_states": placeholder,
              "source_states": placeholder, "source_bddl": placeholder, "bddl": placeholder,
              "pro_source": str(tmp_path / "third_party/libero_pro"), "pro_commit": "pin"}
    preparation.freeze_task(directory, assets, generated, accepted, [])
    plan = json.loads((directory / "protocol.json").read_text())
    task = plan["tasks"]["object_0"]
    assert task["train_state_ids"] == list(range(100))
    assert task["validation_state_ids"] == list(range(100, 150))
    assert task["test_state_ids"] == list(range(150, 200))
    np.testing.assert_array_equal(np.load(directory / "pro_states.npy")[150:], published)
    assert len(np.unique(np.load(directory / "pro_states.npy"), axis=0)) == 200
    generation = json.loads((directory / "generation.json").read_text())
    assert sorted(row["generation_seed"] for row in generation["accepted"]) == list(range(150))
    for split in ("train", "validation", "test", "source"):
        config = tomllib.loads((directory / f"{split}.toml").read_text())
        assert config["env"]["kwargs"]["task_key"] == "object_0"
        assert config["env"]["kwargs"]["split"] == split
        assert config["train"]["iterations"] == 100
        assert config["train"]["save_every"] == 25
        assert config["recording"]["enabled"] is True


def test_refuse_resume_after_generator_or_assets_change(preparation, tmp_path, monkeypatch):
    assets_path = tmp_path / "assets.json"
    placeholder = tmp_path / "asset"
    placeholder.write_text("asset")
    item = preparation.record(placeholder)
    assets_path.write_text(json.dumps({key: item for key in
                                      ("bddl", "published_states", "source_bddl", "source_states")}))
    attempts = tmp_path / "generation_attempts"
    attempts.mkdir()
    (attempts / "context.json").write_text(json.dumps({"assets": "different"}))
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Cannot resume generation"):
        preparation.generate_task(tmp_path, workers=1)
