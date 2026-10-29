"""RLinf bridge contracts, without importing the GPU learner or simulator."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

BRIDGE_PATH = Path(__file__).resolve().parents[1] / "integrations/rlinf_bridge/environment.py"
SPEC = importlib.util.spec_from_file_location("controlled_pro_test_module", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class Primitive:
    low, high = np.full(7, -1., np.float32), np.full(7, 1., np.float32)

    def __init__(self, config, *, terminal_after=2, max_steps=4):
        self.config = config
        self.terminal_after, self.max_steps = terminal_after, max_steps
        self.actions, self.seeds = [], []
        self.steps = 0
        self.closed = False

    def observation(self):
        state = np.full(8, self.steps, np.float32)
        return {"state": state, "openpi": {
            "observation/state": state,
            "observation/image": np.full((4, 4, 3), self.steps, np.uint8),
            "observation/wrist_image": np.full((4, 4, 3), self.steps + 1, np.uint8),
            "prompt": "move the bowl",
        }}

    def reset(self, *, seed):
        self.seeds.append(seed)
        self.steps = 0
        return self.observation(), {"initial_state_id": 100 + len(self.seeds) - 1,
                                    "backend_seed": seed + 99}

    def step(self, action):
        self.steps += 1
        self.actions.append(action.copy())
        terminal = self.terminal_after is not None and self.steps >= self.terminal_after
        return self.observation(), float(terminal), terminal, (
            self.steps >= self.max_steps and not terminal), {"is_success": terminal}

    def close(self):
        self.closed = True


def _config(tmp_path, *, split="train", recording=False):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"version": 2, "tasks": {"spatial_1": {
        "max_steps": 4, "train_state_ids": [0, 1], "validation_state_ids": [100, 101],
    }}}))
    cfg = {"protocol": str(path), "task_key": "spatial_1", "split": split,
           "simulator_python": "/simulator/python", "seed": 100000,
           "auto_reset": True, "ignore_terminations": False, "max_episode_steps": 4,
           "primitive_gamma": .9}
    if recording:
        cfg["recording_directory"] = str(tmp_path / "trajectories")
    return cfg


def test_success_stops_real_steps_pads_tensors_and_preserves_terminal_images(tmp_path, monkeypatch):
    instances = []

    def factory(config):
        instance = Primitive(config)
        instances.append(instance)
        return instance

    monkeypatch.setattr(bridge, "ProcessEnv", factory)
    env = bridge.ControlledProEnv(_config(tmp_path, recording=True), 1, 0, 1, None)
    try:
        initial, _ = env.reset()
        assert initial["states"].shape == (1, 8)
        assert initial["main_images"].dtype == torch.uint8
        env.set_policy_trace(np.zeros((10, 32), np.float32), {"actor_iteration": 25})
        obs, rewards, terminals, truncations, infos = env.chunk_step(torch.full((1, 5, 7), 2.))
        assert len(instances[0].actions) == 2
        np.testing.assert_array_equal(instances[0].actions, np.ones((2, 7)))
        torch.testing.assert_close(rewards, torch.tensor([[0., .9, 0., 0., 0.]]))
        assert terminals.tolist() == [[False, False, False, False, True]]
        assert not truncations.any()
        assert infos[-1]["executed_steps"].item() == 2
        assert infos[-1]["primitive_step_mask"].tolist() == [[True, True, False, False, False]]
        assert infos[-1]["final_observation"]["states"][0, 0].item() == 2
        assert obs[-1]["states"][0, 0].item() == 0
        assert infos[-1]["final_info"]["episode"]["success_once"].item()
        assert infos[-1]["final_info"]["episode"]["episode_len"].item() == 2
        assert instances[0].seeds == [100000, 100001]
        assert env.total_steps == 2 and env.total_decisions == 1
        with np.load(tmp_path / "trajectories/episode_000001/chunk_000001.npz",
                     allow_pickle=False) as data:
            assert data["normalized_actions"].shape == (10, 32)
            assert data["executed_actions"].shape == (2, 7)
            np.testing.assert_array_equal(data["rewards"], [0, 1])
        manifest = json.loads((tmp_path / "trajectories/manifest.json").read_text())
        assert manifest["metadata"]["primitive_environment"]["kwargs"]["split"] == "train"
    finally:
        env.close()


def test_timeout_bootstraps_true_final_observation_and_iteration_end_does_not_reset(
    tmp_path, monkeypatch,
):
    primitive = Primitive({}, terminal_after=None)
    monkeypatch.setattr(bridge, "ProcessEnv", lambda config: primitive)
    env = bridge.ControlledProEnv(_config(tmp_path), 1, 0, 1, None)
    try:
        env.reset()
        env.chunk_step(np.zeros((1, 2, 7), np.float32))
        env.update_reset_state_ids()
        assert primitive.seeds == [100000]
        assert env.elapsed_steps.tolist() == [2]
        obs, _, terminal, timeout, infos = env.chunk_step(np.zeros((1, 5, 7), np.float32))
        assert len(primitive.actions) == 4
        assert not terminal.any() and timeout.tolist() == [[False, False, False, False, True]]
        assert infos[-1]["final_observation"]["states"][0, 0].item() == 4
        assert obs[-1]["states"][0, 0].item() == 0
        assert not infos[-1]["final_info"]["episode"]["success_once"].item()
    finally:
        env.close()


def test_evaluation_enumerates_once_then_restarts_with_same_seeds(tmp_path, monkeypatch):
    instances = []

    def factory(config):
        instance = Primitive(config)
        instances.append(instance)
        return instance

    monkeypatch.setattr(bridge, "ProcessEnv", factory)
    env = bridge.ControlledProEnv(_config(tmp_path, split="validation"), 1, 0, 1, None)
    try:
        env.reset()
        env.chunk_step(np.zeros((1, 5, 7)))
        env.chunk_step(np.zeros((1, 5, 7)))
        _, reward, terminal, timeout, infos = env.chunk_step(np.zeros((1, 5, 7)))
        assert len(instances[0].actions) == 4
        assert instances[0].seeds == [100000, 100001]
        assert not reward.any() and not terminal.any() and not timeout.any()
        assert infos == [{}] * 5
        env.is_start = True
        env.reset()
        assert instances[0].closed and instances[1].seeds == [100000]
    finally:
        env.close()


def test_invalid_worker_layout_and_protocol_horizon_are_rejected(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr(bridge, "ProcessEnv", lambda config: pytest.fail("Must validate first"))
    with pytest.raises(ValueError, match="one environment"):
        bridge.ControlledProEnv(cfg, 2, 0, 1, None)
    with pytest.raises(ValueError, match="auto_reset"):
        bridge.ControlledProEnv(cfg | {"auto_reset": False}, 1, 0, 1, None)
    with pytest.raises(ValueError, match="frozen PRO protocol"):
        bridge.ControlledProEnv(cfg | {"max_episode_steps": 5}, 1, 0, 1, None)


def test_close_preserves_partial_episode_without_inventing_a_terminal(tmp_path, monkeypatch):
    primitive = Primitive({}, terminal_after=None)
    monkeypatch.setattr(bridge, "ProcessEnv", lambda config: primitive)
    env = bridge.ControlledProEnv(_config(tmp_path, recording=True), 1, 0, 1, None)
    env.reset()
    env.set_policy_trace(np.zeros((10, 32), np.float32), {"actor_iteration": 0})
    env.chunk_step(np.zeros((1, 2, 7), np.float32))
    env.close()
    directory = tmp_path / "trajectories"
    episode = json.loads((directory / "episode_000001/episode.json").read_text())
    assert episode["reason"] == "environment_closed"
    assert episode["primitive_steps"] == 2
    assert not episode["complete"]
    assert primitive.closed
    with np.load(directory / "episode_000001/chunk_000001.npz", allow_pickle=False) as data:
        assert data["executed_actions"].shape == (2, 7)
        assert not data["terminated"].any() and not data["truncated"].any()
