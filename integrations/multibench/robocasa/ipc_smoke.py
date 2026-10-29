"""Exercise the actual cross-interpreter training simulator protocol."""
import json
from pathlib import Path
import time

import numpy as np

from multibench.environment import ROOT, process_config
from rl_vla.rldt_env import ProcessEnv


def main():
    output = Path(__file__).resolve().parent / "evidence"
    protocol = ROOT / "runs/multibench_20261028/campaign/tasks/robocasa/protocol.json"
    definition = json.loads(protocol.read_text())["tasks"]["robocasa"]
    cfg = process_config("robocasa", protocol, "validation")
    start = time.perf_counter()
    env = ProcessEnv(cfg)
    try:
        assert env.low.shape == env.high.shape == (12,)
        raw, info = env.reset(seed=100000)
        assert info["initial_state_id"] == definition["validation_state_ids"][0]
        assert info["backend_seed"] == definition["validation_seeds"][0]
        assert raw["state"].shape == (16,)
        for key in ("observation/image", "observation/wrist_image", "observation/right_wrist_image"):
            assert raw["openpi"][key].shape == (224, 224, 3)
            assert raw["openpi"][key].dtype == np.uint8
        np.savez(output / "ipc_observation.npz", **raw["openpi"])
        action = np.zeros(12, dtype=np.float32)
        action[[6, 11]] = -1
        _, reward, terminated, truncated, step_info = env.step(action)
        assert reward == float(step_info["is_success"])
        assert step_info["controller_steps"] == 1
        result = {"passed": True, "config": cfg, "reset_info": info, "step_info": step_info,
                  "reward": reward, "terminated": terminated, "truncated": truncated,
                  "elapsed_seconds": time.perf_counter()-start}
        (output / "ipc_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
