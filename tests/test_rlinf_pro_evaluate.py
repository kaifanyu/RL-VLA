"""Standalone evaluator checks with CPU policy/simulator substitutes."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
from rlinf_bridge.evaluate import evaluate, load_checkpoint, resolve_checkpoint


class Policy:
    config = SimpleNamespace(action_horizon=3, action_dim=8, action_chunk=2)

    def predict_action_batch(self, observation, *, mode, compute_values):
        assert mode == "eval" and compute_values is False
        assert observation["extra_view_images"] is None
        noise = torch.randn((1, 3, 8))
        actions = noise * .1
        return actions[:, :2, :7], {"forward_inputs": {
            "model_action": actions.reshape(1, -1),
            "chains": torch.stack([noise, actions], dim=1),
        }}

    @staticmethod
    def _input_transform(raw):
        return {"state": raw["observation/state"]}

    @staticmethod
    def _output_transform(raw):
        return {"actions": raw["actions"][:, :7]}


class Primitive:
    low, high = np.full(7, -1., np.float32), np.full(7, 1., np.float32)

    def __init__(self, config, first_length):
        self.config, self.first_length = config, first_length
        self.episode, self.steps = -1, 0
        self.actions, self.seeds = [], []
        self.closed = False

    def observation(self):
        state = np.full(8, self.steps, np.float32)
        return {"state": state, "openpi": {
            "observation/state": state, "prompt": "pick up the bowl",
            "observation/image": np.full((4, 4, 3), self.steps, np.uint8),
            "observation/wrist_image": np.full((4, 4, 3), self.steps + 1, np.uint8),
        }}

    def reset(self, *, seed):
        self.episode += 1
        self.steps = 0
        self.seeds.append(seed)
        return self.observation(), {"initial_state_id": 100 + self.episode, "backend_seed": seed + 7}

    def step(self, action):
        self.steps += 1
        self.actions.append(action.copy())
        terminal = self.steps == (self.first_length if self.episode == 0 else 1)
        return self.observation(), float(terminal), terminal, False, {"is_success": terminal}

    def close(self):
        self.closed = True


class Video:
    def __init__(self, path):
        self.path, self.frames = path, []

    def append(self, raw):
        self.frames.append(int(raw["state"][0]))

    def close(self):
        self.path.write_text(json.dumps(self.frames))


def _config(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"tasks": {"spatial_1": {
        "max_steps": 4, "validation_state_ids": [100, 101],
    }}}))
    config = {"actor": {"model": {"model_path": "/original/converted/checkpoint", "openpi": {
        "action_chunk": 2, "noise_method": "flow_sde", "num_steps": 10,
    }}}, "env": {"train": {"simulator_python": "/simulator/python"}}}
    return config, protocol


def test_full_recording_and_per_episode_seed_match_despite_previous_episode_length(tmp_path):
    config, protocol = _config(tmp_path)
    environments = []

    def factory(length):
        def create(spec):
            env = Primitive(spec, length)
            environments.append(env)
            return env
        return create

    def loader(config, checkpoint, device):
        return Policy(), {"test_device": device}

    baseline = evaluate(config, protocol, "spatial_1", "validation", tmp_path / "base.json",
                        episodes=2, device="cpu", model_loader=loader,
                        environment_factory=factory(1), video_factory=Video)
    trained = evaluate(config, protocol, "spatial_1", "validation", tmp_path / "trained.json",
                       checkpoint="/trained/iteration_25", episodes=2, device="cpu",
                       model_loader=loader, environment_factory=factory(3), video_factory=Video)
    assert baseline["config"] == trained["config"]
    assert baseline["checkpoint"] is None and trained["checkpoint"] == "/trained/iteration_25"
    assert baseline["complete_split"] and baseline["success_rate"] == 1
    assert environments[0].seeds == environments[1].seeds == [100000, 100001]
    assert all(env.closed for env in environments)
    assert len(environments[0].actions) == 2 and len(environments[1].actions) == 4
    base_chunk = tmp_path / "base.trajectories/episode_000002/chunk_000001.npz"
    trained_chunk = tmp_path / "trained.trajectories/episode_000002/chunk_000001.npz"
    with (np.load(base_chunk, allow_pickle=False) as left,
          np.load(trained_chunk, allow_pickle=False) as right):
        np.testing.assert_array_equal(left["normalized_actions"], right["normalized_actions"])
        assert left["decoded_actions"].shape == (3, 7)
        assert left["executed_actions"].shape == (1, 7)
        metadata = json.loads(left["metadata_json"].item())
        assert np.asarray(metadata["decision"]["initial_noise"]).shape == (3, 8)
    frames = json.loads((tmp_path / "trained.videos/episode_001.mp4").read_text())
    assert frames == [0, 1, 2, 3]
    with pytest.raises(FileExistsError):
        evaluate(config, protocol, "spatial_1", "validation", tmp_path / "base.json",
                 episodes=2, model_loader=loader)


def test_full_weights_checkpoint_load_is_strict_and_prefers_upstream_actor_layout(tmp_path):
    original = torch.nn.Linear(2, 3)
    directory = tmp_path / "actor/model_state_dict"
    directory.mkdir(parents=True)
    torch.save(original.state_dict(), directory / "full_weights.pt")
    fresh = torch.nn.Linear(2, 3)
    provenance = load_checkpoint(fresh, tmp_path)
    assert provenance["format"] == "weights" and len(provenance["sha256"]) == 64
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, fresh.state_dict()[key])
    torch.save({"wrong.weight": torch.ones(1)}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(fresh, tmp_path / "bad.pt")
    with pytest.raises(FileNotFoundError, match="full_weights"):
        resolve_checkpoint(tmp_path / "missing")


def test_dcp_load_requests_model_subtree_without_optimizer_or_rng(tmp_path):
    from torch.distributed import checkpoint as dcp

    original = torch.nn.Linear(2, 3)
    directory = tmp_path / "actor/dcp_checkpoint"
    dcp.save({"fsdp_checkpoint": {"model": original.state_dict(),
                                   "optimizers": {"unused_state": torch.ones(5)},
                                   "rng": {"unused_seed": 123}}}, checkpoint_id=str(directory))
    fresh = torch.nn.Linear(2, 3)
    provenance = load_checkpoint(fresh, tmp_path)
    assert provenance["format"] == "dcp" and len(provenance["metadata_sha256"]) == 64
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, fresh.state_dict()[key])
