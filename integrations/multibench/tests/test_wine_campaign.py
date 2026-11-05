"""Task restriction and fail-closed queue regressions, without GPUs or assets."""
import json
from pathlib import Path
import types
import sys

import gymnasium as gym
import numpy as np
import pytest

from multibench.robocasa import wine_campaign as campaign
from multibench import environment


class FakeNative(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (12,), dtype=np.float32)
    observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    def reset(self, *, seed=None, options=None):
        return np.zeros(1), {"initial_state_id": seed}


def test_protocol_restricts_native_object_and_preserves_paired_reset_seeds(tmp_path, monkeypatch):
    captured = []
    module = types.SimpleNamespace(make_env=lambda **kwargs: (captured.append(kwargs), FakeNative())[1])
    monkeypatch.setattr(environment.importlib, "import_module", lambda _: module)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"tasks": {"robocasa": {
        "max_steps": 750, "validation_seeds": [2000, 2001], "validation_state_ids": [2000, 2001],
        "native_kwargs": {"obj_groups": "wine", "expected_prompt": campaign.PROMPT}}}}))
    env = environment.make_env("robocasa", protocol, "validation")
    assert captured[0]["obj_groups"] == "wine"
    assert captured[0]["expected_prompt"] == campaign.PROMPT
    assert captured[0]["split"] == "target"
    assert [env.reset(seed=999)[1]["initial_state_id"] for _ in range(2)] == [2000, 2001]


def test_wrong_native_prompt_stops_before_policy_observation(monkeypatch):
    from multibench.robocasa.environment import RoboCasaEnv
    monkeypatch.setitem(sys.modules, "robocasa", types.ModuleType("robocasa"))
    registry = types.ModuleType("robocasa.utils.dataset_registry_utils")
    registry.get_task_horizon = lambda _: 750
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    captured = []
    native = types.SimpleNamespace(reset=lambda **kw: ({"annotation.human.task_description": "Pick the apple."}, {}))
    monkeypatch.setattr(gym, "make", lambda *args, **kwargs: (captured.append(kwargs), native)[1])
    env = RoboCasaEnv(obj_groups="wine", expected_prompt=campaign.PROMPT)
    with pytest.raises(RuntimeError, match="Native task prompt"):
        env.reset(seed=2000)
    assert captured[0]["obj_groups"] == "wine"


@pytest.mark.parametrize("failure", [None, "verify", "baseline_test"])
def test_queue_order_gpu_isolation_and_failure_propagation(tmp_path, monkeypatch, failure):
    (tmp_path / ".cache").mkdir()
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *args, **kwargs: "")
    campaign.write(tmp_path / "launch.json", {"source": str(source), "gpu_index": 7, "gpu_uuid": "GPU-seven", "cpus": [16]})
    campaign.write(tmp_path / "source_manifest.json", {})
    campaign.write(tmp_path / "experiment.json", {"protocol": {"validation_iterations": [25, 50, 100]}})
    calls = []
    class Process:
        pid = 123
        def __init__(self, command, **kwargs):
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-seven"
            self.name = Path(kwargs["stdout"].name).stem
            calls.append(self.name)
            if self.name == "verify":
                campaign.write(tmp_path / "verification.json", {"passed": True})
            if self.name == "training":
                (tmp_path / "checkpoint_validations.jsonl").write_text("".join(json.dumps({
                    "iteration": i, "successes": int(i == 50), "episodes": 5,
                    "checkpoint": str(tmp_path / "training" / f"iteration_{i:06d}"), "report": "x"}) + "\n" for i in (25, 50, 100)))
            if self.name == "selected_test":
                assert command[command.index("--checkpoint") + 1].endswith("iteration_000050")
        def wait(self):
            return 1 if self.name == failure else 0
    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    if failure:
        with pytest.raises(RuntimeError, match="exited 1"):
            campaign.run(tmp_path)
    else:
        campaign.run(tmp_path)
    order = ["verify", "baseline_validation", "baseline_test", "training", "selected_test"]
    assert calls == (order[:order.index(failure) + 1] if failure else order)
    status = json.loads((tmp_path / "queue_status.json").read_text())
    assert status["state"] == ("failed" if failure else "completed")
