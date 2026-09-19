"""Container checks: actual learner/evaluation, optionally real LIBERO rendering."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from rl_vla.config import load_config
from rl_vla.evaluate import evaluate
from rl_vla.train import train


def check_toy():
    with tempfile.TemporaryDirectory(prefix="rl-vla-smoke-") as directory:
        root = Path(directory)
        for algorithm in ("sac", "ppo"):
            config = load_config(f"configs/{algorithm}_toy.toml")
            config["train"].update(total_decisions=128, learning_starts=16, batch_size=16)
            if algorithm == "ppo":
                config["train"]["rollout_steps"] = 32
            train(config, root / algorithm)
            checkpoint = root / algorithm / "final.pt"
            assert checkpoint.is_file(), f"Missing {algorithm} checkpoint"
            result = evaluate(config, root / f"{algorithm}.json", str(checkpoint),
                              episodes=2, seed=100_000, deterministic=True)
            assert len(result["results"]) == 2
            print(f"{algorithm}: trained 128 decisions, checkpoint loaded, evaluated 2 episodes")


def check_libero():
    from rl_vla.envs.libero import make_env

    env = make_env(initial_state_id=0)
    try:
        observation, info = env.reset(seed=7)
        assert env.observation_space.contains(observation)
        image = observation["openpi"]["observation/image"]
        assert image.shape == (224, 224, 3) and image.dtype == np.uint8
        assert image.std() > 0, "Renderer returned a blank image"
        observation, reward, terminated, truncated, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        assert env.observation_space.contains(observation)
        print(json.dumps({"libero_reset": info, "image_shape": list(image.shape),
                          "reward": reward, "terminated": terminated, "truncated": truncated}))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero", action="store_true", help="Check real simulator only")
    args = parser.parse_args()
    if args.libero:
        check_libero()
    else:
        check_toy()
