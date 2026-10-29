"""Exact-checkpoint model settings for upstream RLinf's two pi-RL methods.

The integration changes configuration only. Sampling, log probabilities, value
heads, and PPO losses remain the pinned upstream implementations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RLINF_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"
METHODS = ("flow_sde", "flow_noise")


def model_config(checkpoint: str | Path, method: str) -> dict:
    """Return the actor.model section; horizon and execution length differ."""
    if method not in METHODS:
        raise ValueError(f"Expected one of {METHODS}, got {method!r}")
    return {
        "model_type": "openpi",
        "model_path": str(Path(checkpoint).resolve()),
        "precision": None,
        "num_action_chunks": 5,
        "action_dim": 7,
        "is_lora": False,
        "lora_rank": 32,
        "use_proprio": True,
        "num_steps": 10,
        "add_value_head": True,
        "openpi": {
            "config_name": "pi05_libero",
            "pi05": True,
            "action_horizon": 10,
            "action_dim": 32,
            # This is explicit in the ORIGINAL pinned pi05_libero config too.
            "discrete_state_input": False,
            "num_images_in_input": 2,
            "action_chunk": 5,
            "action_env_dim": 7,
            "num_steps": 10,
            "train_expert_only": True,
            "noise_method": method,
            "noise_level": 0.5,
            "noise_anneal": False,
            "noise_logvar_range": [0.08, 0.16],
            "joint_logprob": method == "flow_noise",
            "double_layer": False,
            "ignore_last": False,
            "safe_get_logprob": False,
            "add_value_head": True,
            "value_after_vlm": True,
            "value_vlm_mode": "mean_token",
            "detach_critic_input": True,
            "use_dsrl": False,
            "is_nft": False,
        },
    }


def algorithm_overrides(method: str) -> dict:
    """The method-specific settings recommended by the official pi-RL recipe."""
    if method not in METHODS:
        raise ValueError(method)
    return {"entropy_bonus": 0.005 if method == "flow_noise" else 0.0}


def assert_native_model_settings(config: object) -> None:
    """Reject silently changing the original LIBERO inference construction."""
    expected = {
        "pi05": True, "action_horizon": 10, "action_dim": 32,
        "discrete_state_input": False,
        "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m",
    }
    for name, value in expected.items():
        actual = getattr(config, name)
        if actual != value:
            raise ValueError(f"Native model {name} changed: {actual!r} != {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        path = output / f"{method}.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps({
            "actor": {"model": model_config(args.checkpoint, method)},
            "algorithm": algorithm_overrides(method),
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
