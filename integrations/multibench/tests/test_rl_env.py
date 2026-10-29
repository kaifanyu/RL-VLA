"""Paired benchmark bridge contract independent of GPU physics and policy loading."""
import json

import numpy as np
import pytest
import torch

from multibench import environment, rl_env


class Primitive:
    def __init__(self, width, terminal_after=2):
        self.low, self.high = np.full(width, -1., np.float32), np.ones(width, np.float32)
        self.steps = 0
        self.actions = []
        self.terminal_after = terminal_after

    def obs(self):
        state = np.full(14, self.steps, np.float32)
        return {"state": state, "openpi": {
            "observation/state": state, "prompt": "handover block",
            "observation/image": np.full((4, 4, 3), self.steps, np.uint8),
            "observation/wrist_image": np.full((4, 4, 3), 10, np.uint8),
            "observation/right_wrist_image": np.full((4, 4, 3), 20, np.uint8)}}

    def reset(self, seed):
        self.steps = 0
        return self.obs(), {"initial_state_id": seed, "is_success": False}

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        success = self.steps == self.terminal_after
        return self.obs(), float(success), success, False, {"is_success": success}

    def close(self):
        pass


@pytest.mark.parametrize("width", [7, 12, 14])
def test_chunk_termination_preserves_native_width_three_cameras_and_progress(tmp_path, monkeypatch, width):
    primitive = Primitive(width)
    monkeypatch.setattr(rl_env, "ProcessEnv", lambda _: primitive)
    monkeypatch.setattr(environment, "process_config", lambda *args: {})
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"tasks": {"robotwin": {
        "max_steps": 800, "train_state_ids": [1, 2], "validation_state_ids": [3]}}}))
    cfg = dict(benchmark="robotwin", protocol=str(protocol), split="train", seed=5,
               auto_reset=True, primitive_gamma=.9, record_trajectories=False,
               recording_directory=str(tmp_path / "interactions"))
    env = rl_env.ControlledEnv(cfg, 1, 0, 1, None)
    first, _ = env.reset()
    assert first["extra_view_images"][0, 0, 0, 0] == 20
    obs, rewards, terminated, truncated, infos = env.chunk_step(np.full((1, 5, width), 2., np.float32))
    assert len(primitive.actions) == 2
    np.testing.assert_equal(primitive.actions, np.ones((2, width)))
    torch.testing.assert_close(rewards, torch.tensor([[0., .9, 0., 0., 0.]]))
    assert terminated.tolist() == [[False, False, False, False, True]]
    assert not truncated.any()
    assert infos[-1]["primitive_step_mask"].tolist() == [[True, True, False, False, False]]
    assert infos[-1]["final_observation"]["states"][0, 0] == 2
    assert obs[-1]["states"][0, 0] == 0
    progress = json.loads((tmp_path / "interactions/progress.json").read_text())
    assert progress["primitive_steps"] == 2 and progress["successes"] == 1
    assert list((tmp_path / "interactions").iterdir()) == [tmp_path / "interactions/progress.json"]
    env.close()
