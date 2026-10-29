"""CPU checks of the exact budget and already-computed PPO recording payloads."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
from rlinf_bridge import configure
from rlinf_bridge.configure import configuration
from rlinf_bridge.tracing import record_policy_trace


@pytest.mark.parametrize("method", ["flow_sde", "flow_noise"])
def test_configuration_uses_exact_chunk_budget_and_timeout_semantics(tmp_path, method, monkeypatch):
    monkeypatch.setattr(configure, "UPSTREAM", Path(__file__).parent / "fixtures/rlinf")
    task = tmp_path / "task"
    task.mkdir()
    (task / "protocol.json").write_text(json.dumps({"tasks": {
        "spatial_1": {"max_steps": 220},
    }}))
    cfg = configuration(task, tmp_path / method, method)
    env, model = cfg["env"]["train"], cfg["actor"]["model"]
    assert env["total_num_envs"] == env["rollout_epoch"] == cfg["rollout"]["pipeline_stage_num"] == 1
    assert env["max_steps_per_rollout_epoch"] // model["num_action_chunks"] == 32
    assert model["openpi"]["action_horizon"] == 10 and model["num_action_chunks"] == 5
    assert env["auto_reset"] and not env["ignore_terminations"]
    assert cfg["algorithm"]["reward_type"] == "chunk_level"
    assert cfg["algorithm"].get("bootstrap_type", "standard") == "standard"
    assert cfg["algorithm"]["gamma"] == env["primitive_gamma"] ** 5
    assert cfg["runner"]["max_steps"] == cfg["runner"]["max_epochs"] == 100
    assert cfg["runner"]["save_interval"] == 25
    assert cfg["runner"]["val_check_interval"] == -1
    assert cfg["actor"]["fsdp_config"]["use_orig_params"]
    assert cfg["actor"]["fsdp_config"]["save_full_model_weights"]
    assert cfg["actor"]["model"]["precision"] is None
    assert cfg["actor"]["model"]["enforce_native_scope"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tracing_keeps_full_normalized_chunk_and_does_not_mutate_ppo_inputs(dtype):
    recorded = []
    env = SimpleNamespace(set_policy_trace=lambda actions, metadata: recorded.append((actions, metadata)))
    action = torch.arange(320, dtype=dtype).reshape(1, -1)
    chains = torch.ones((1, 11, 10, 32))
    inputs = {"model_action": action, "chains": chains,
              "denoise_inds": torch.arange(10).reshape(1, -1)}
    before = copy.deepcopy(inputs)
    rollout = SimpleNamespace(forward_inputs=inputs, versions=torch.tensor([[25]]))
    record_policy_trace(env, rollout, {"openpi": {"action_horizon": 10, "action_dim": 32}})
    assert recorded[0][0].shape == (10, 32)
    assert recorded[0][1]["policy_versions"] == [[25]]
    assert np.asarray(recorded[0][1]["initial_noise"]).shape == (10, 32)
    for key, value in inputs.items():
        torch.testing.assert_close(value, before[key])
    action[:] = -1
    assert recorded[0][0][0, 1] == 1
    with pytest.raises(ValueError, match="full normalized"):
        record_policy_trace(env, SimpleNamespace(forward_inputs={}), {})
