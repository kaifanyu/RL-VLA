"""Real simulator reset, step, RGB, determinism, and success-signal validation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from environment import Plate25Env, TASK_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-backend", default="cpu")
    parser.add_argument("--sim-backend", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    env = Plate25Env(sim_backend=args.sim_backend, render_backend=args.render_backend)
    try:
        records = []
        initial_states = []
        for i, seed in enumerate([20261029, 20261030, 20261029]):
            obs, info = env.reset(seed=seed)
            assert not info["is_success"], "A smoke reset already satisfies the task"
            assert env.observation_space.contains(obs), "Reset observation violates declared spaces"
            initial_states.append(obs["state"].copy())
            assert np.std(obs["openpi"]["observation/image"]) > 5, "Blank or failed camera render"
            Image.fromarray(obs["openpi"]["observation/image"]).save(args.output / f"reset_{i}.png")
            record = {"seed": seed, "episode_id": info["episode_id"], "prompt": obs["openpi"]["prompt"],
                      "state": obs["state"].tolist(), "image_shape": list(obs["openpi"]["observation/image"].shape),
                      "image_sha256": hashlib.sha256(obs["openpi"]["observation/image"].tobytes()).hexdigest(),
                      "reset_success": info["is_success"]}
            for _ in range(3):
                obs, reward, terminated, truncated, info = env.step(np.array([0., 0., 0., 0., 0., 0., 1.]))
                assert isinstance(info["is_success"], bool)
                assert np.isfinite(reward) and np.isfinite(obs["state"]).all()
                assert not terminated and not truncated
            Image.fromarray(obs["openpi"]["observation/image"]).save(args.output / f"after_three_steps_{i}.png")
            record.update({"controller_steps": info["controller_steps"], "final_success": info["is_success"]})
            records.append(record)
        np.testing.assert_allclose(initial_states[0], initial_states[2], rtol=0, atol=1e-6)
        assert records[0]["episode_id"] == records[2]["episode_id"] != records[1]["episode_id"]
        # Validate that the native success signal can turn true. This is a
        # simulator diagnostic only, never a rollout/training demonstration.
        from mani_skill.utils.structs.pose import Pose
        native = env.native.unwrapped
        object_name = native.carrot_names[int(native.select_carrot_ids[0])]
        plate_name = native.plate_names[int(native.select_plate_ids[0])]
        object_actor, plate_actor = native.objs_carrot[object_name], native.objs_plate[plate_name]
        goal_position = plate_actor.pose.p.clone()
        goal_position[:, 2] += (native.plate_bbox_world[:, 2] + native.carrot_bbox_world[:, 2]) / 2 + .002
        object_actor.set_pose(Pose.create_from_pq(goal_position, object_actor.pose.q))
        positive_success = False
        for _ in range(15):
            obs, reward, terminated, truncated, info = env.step(np.array([0., 0., 0., 0., 0., 0., 1.]))
            if info["is_success"]:
                positive_success = True
                assert reward == 1. and terminated and not truncated
                break
        assert positive_success, "Native success oracle failed an object-on-plate diagnostic"
        Image.fromarray(obs["openpi"]["observation/image"]).save(args.output / "diagnostic_goal.png")
        result = {"ready": True, "task": TASK_ID, "sim_backend": args.sim_backend,
                  "render_backend": args.render_backend, "records": records,
                  "synthetic_goal_success_checked": positive_success,
                  "render_device": env.render_device,
                  "robot": env.native.unwrapped.agent.uid,
                  "action_low": env.action_space.low.tolist(), "action_high": env.action_space.high.tolist(),
                  "total_num_trials": env.native.unwrapped.total_num_trials,
                  "elapsed_seconds": time.monotonic() - start,
                  "versions": {name: importlib.metadata.version(name)
                               for name in ["mani_skill", "sapien", "torch", "gymnasium", "numpy"]}}
        (args.output / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
