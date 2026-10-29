"""Persistent rollouts retain primitive-step data without changing policy behavior."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from rl_vla.trajectory_recording import TrajectoryRecorder


def _observation(value):
    return {"state": np.array([value, -value], np.float32), "openpi": {
        "observation/image": np.full((4, 4, 3), value, np.uint8),
        "prompt": "pick up the bowl",
    }}


def _read_chunk(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
        metadata = json.loads(archive["metadata_json"].item())
    return arrays, metadata


def test_recorder_losslessly_preserves_every_step_and_both_action_representations(tmp_path):
    recorder = TrajectoryRecorder(tmp_path / "data", {"checkpoint": "/original/policy"})
    initial = _observation(0)
    recorder.start_episode(initial, {"seed": 13, "reset": {"initial_state_id": 7}})
    normalized = np.arange(12, dtype=np.float32).reshape(3, 4)
    decoded = np.full((3, 2), 1.5, np.float32)
    recorder.begin_chunk(initial, normalized, decoded, {"sampling_key": np.array([1, 2])})
    # Mutating the source buffers after recording must not alter the stored trajectory.
    normalized[:] = 100
    initial["state"][:] = 100
    for value in (1, 2):
        observation = _observation(value)
        info = {"is_success": value == 2}
        recorder.record_step(np.ones(2), observation, float(value == 2), value == 2,
                             False, info)
        observation["openpi"]["observation/image"][:] = 100
        info["is_success"] = False
    recorder.end_chunk({"discounted_reward": .99})
    directory = recorder.end_episode({"success": True})
    recorder.close(reason="evaluation_complete")
    arrays, metadata = _read_chunk(Path(directory) / "chunk_000001.npz")
    assert arrays["executed_actions"].shape == (2, 2)
    np.testing.assert_array_equal(arrays["normalized_actions"], np.arange(12).reshape(3, 4))
    np.testing.assert_array_equal(arrays["decoded_actions"], 1.5)
    np.testing.assert_array_equal(arrays["rewards"], [0., 1.])
    np.testing.assert_array_equal(arrays["terminated"], [False, True])
    image_name = next(key for key, path in metadata["observation_schema"].items()
                      if path == ["openpi", "observation/image"])
    np.testing.assert_array_equal(arrays[image_name][:, 0, 0, 0], [0, 1, 2])
    state_name = next(key for key, path in metadata["observation_schema"].items()
                      if path == ["state"])
    np.testing.assert_array_equal(arrays[state_name], [[0, 0], [1, -1], [2, -2]])
    assert metadata["step_infos"][-1]["is_success"]
    assert metadata["decision"]["sampling_key"] == [1, 2]
    episode = json.loads((Path(directory) / "episode.json").read_text())
    assert episode["complete"] and episode["primitive_steps"] == 2
    assert episode["metadata"]["reset"]["initial_state_id"] == 7
    summary = json.loads((tmp_path / "data/summary.json").read_text())
    assert summary["primitive_steps"] == 2 and summary["chunks"] == 1


def test_recorder_closes_interrupted_partial_chunk_without_claiming_terminal(tmp_path):
    recorder = TrajectoryRecorder(tmp_path / "data", {})
    recorder.start_episode(_observation(0), {"seed": 1})
    recorder.begin_chunk(_observation(0), np.zeros((5, 2)), np.zeros((5, 2)), {})
    recorder.record_step(np.zeros(2), _observation(1), 0., False, False, {})
    recorder.close()
    arrays, metadata = _read_chunk(tmp_path / "data/episode_000001/chunk_000001.npz")
    assert len(arrays["executed_actions"]) == 1
    assert metadata["outcome"]["partial_chunk"]
    episode = json.loads((tmp_path / "data/episode_000001/episode.json").read_text())
    assert not episode["complete"] and episode["reason"] == "interrupted"
    with pytest.raises(FileExistsError):
        TrajectoryRecorder(tmp_path / "data", {})


@pytest.mark.parametrize("terminal", [False, True])
def test_execute_chunk_records_clipped_commands_and_final_observation(terminal):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    from rl_vla.rldt import execute_chunk

    class Environment:
        low, high = np.array([-1.]), np.array([1.])
        steps = 0

        def step(self, action):
            self.steps += 1
            boundary = self.steps == 2
            return _observation(self.steps), float(boundary), boundary and terminal, (
                boundary and not terminal), {"is_success": boundary and terminal}

    records = []
    env = Environment()
    raw, _, _, _, info = execute_chunk(
        env, np.full((5, 1), 2.), .99, record_step=lambda *args: records.append(args))
    assert info["executed_steps"] == len(records) == 2
    np.testing.assert_array_equal(np.stack([row[0] for row in records]), np.ones((2, 1)))
    np.testing.assert_array_equal(records[-1][1]["state"], raw["state"])
    assert records[-1][3] == terminal and records[-1][4] != terminal


def test_train_and_evaluate_recording_preserve_outcomes_and_all_step_accounting(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    pytest.importorskip("gymnasium")
    from rl_vla.rldt import evaluate, load_config, train

    config = load_config(Path(__file__).resolve().parents[1] / "configs/rldt_toy.toml")
    config["policy"]["horizon"] = 2
    config["env"]["kwargs"].update(max_steps=3, success_radius=1e-6)
    config["train"].update(iterations=1, collect_decisions=3, critic_steps=1, actor_steps=1,
                           critic_batch_size=2, actor_batch_size=1, critic_trajectories=1,
                           save_every=1)
    config["rldt"].update(ode_steps=2, execute_steps=2, particles=2, bootstrap_samples=1,
                          critic_width=8, critic_warmup_iterations=0)
    plain = train(config, tmp_path / "plain")
    recording = copy.deepcopy(config) | {"recording": {"enabled": True}}
    recorded = train(recording, tmp_path / "recorded")
    for key in ("decisions", "primitive_steps", "episodes", "successes", "actor_updates"):
        assert plain[key] == recorded[key]
    with (np.load(tmp_path / "plain/final/actor.npz", allow_pickle=False) as left,
          np.load(tmp_path / "recorded/final/actor.npz", allow_pickle=False) as right):
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])
    root = tmp_path / "recorded/trajectories"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["primitive_steps"] == recorded["primitive_steps"]
    assert summary["chunks"] == recorded["decisions"]
    episodes = [json.loads(path.read_text()) for path in sorted(root.glob("episode_*/episode.json"))]
    assert sum(episode["complete"] for episode in episodes) == recorded["episodes"]
    assert episodes[-1]["reason"] == "training_budget_exhausted"
    assert not episodes[-1]["complete"]
    before = evaluate(config, tmp_path / "before.json", episodes=2, seed=17)
    after = evaluate(recording, tmp_path / "after.json", episodes=2, seed=17)
    for left, right in zip(before["episodes"], after["episodes"], strict=True):
        assert left == {key: value for key, value in right.items() if key != "trajectory"}
    evaluation = json.loads((tmp_path / "after.trajectories/summary.json").read_text())
    assert evaluation["primitive_steps"] == sum(row["primitive_steps"] for row in after["episodes"])
    assert evaluation["reason"] == "evaluation_complete"
