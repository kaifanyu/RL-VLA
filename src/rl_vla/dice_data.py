"""Convert verified demonstration episodes into normalized DICE chunk replay.

Input uses existing LIBERO adapter observations. Dataset-specific image and
proprioception conversion must be verified before exporting this format.
"""

import argparse
from pathlib import Path

import numpy as np

from rl_vla.dice_core import discounted_returns


def convert_episode(episode, prior, gamma, provenance):
    from rl_vla.dice import validate_replay_provenance

    validate_replay_provenance(episode.get("provenance"), provenance)
    observations = episode["observations"]
    actions = np.asarray(episode["actions"], np.float32)
    rewards = np.asarray(episode["rewards"], np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError("Expected nonempty primitive LIBERO actions [T,7]")
    if rewards.shape != (len(actions),) or len(observations) != len(actions) + 1:
        raise ValueError("Need T rewards and T+1 observations, including final observation")
    if not np.isfinite(actions).all() or not np.isfinite(rewards).all():
        raise ValueError("Non-finite demonstration data")
    if episode.get("ending") not in ("terminated", "truncated"):
        raise ValueError("Need a verified finite-episode boundary")
    if any(
        not isinstance(raw, dict)
        or not isinstance(raw.get("openpi"), dict)
        or raw["openpi"].get("prompt") != provenance["task"]["instruction"]
        for raw in observations
    ):
        raise ValueError("Episode observation instruction does not match task provenance")
    records = []
    h, e = prior.flow.horizon, prior.execute_steps
    for start in range(0, len(actions), e):
        end = min(start + e, len(actions))
        future = actions[start : min(start + h, len(actions))]
        future = np.pad(future, ((0, h - len(future)), (0, 0)), mode="edge")
        raw = observations[start]
        if "openpi" not in raw:
            raise ValueError("Use the existing LIBERO/OpenPI observation adapter")
        transformed = prior.flow._transform_observation(
            {"openpi": {**raw["openpi"], "actions": future}}
        )
        normalized = np.asarray(transformed["actions"], np.float32)
        if normalized.shape != (h, prior.flow.action_dim) or not np.isfinite(normalized).all():
            raise ValueError("Checkpoint normalization returned unexpected actions")
        reward = float(sum(gamma**i * rewards[start + i] for i in range(end - start)))
        records.append(
            {
                "observation": raw,
                "next_observation": observations[end],
                "actions": normalized[:e, :7].reshape(-1).copy(),
                "rewards": reward,
                "discounts": 0.0 if end == len(actions) else gamma ** (end - start),
                "expert": 1.0,
                "provenance": provenance,
            }
        )
    returns = discounted_returns([r["rewards"] for r in records], [r["discounts"] for r in records])
    for r, value in zip(records, returns, strict=True):
        r["returns"] = value
    return records


def main():
    from openpi_client import msgpack_numpy

    from rl_vla.dice import check, config_from, resource_check
    from rl_vla.dice_pi05 import FrozenPi05

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dice_pi05_libero.toml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-split", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    c = config_from(args.config)
    if not check(c)["setup_ok"]:
        raise RuntimeError("DICE setup is incomplete")
    source = Path(args.input)
    limit = c["train"]["replay_max_mb"] * 1024**2
    if source.stat().st_size > limit:
        raise ValueError("Input exceeds replay memory budget")
    payload = msgpack_numpy.unpackb(source.read_bytes())
    if payload.get("format") != "dice-libero-episodes-v1" or not payload.get("episodes"):
        raise ValueError("Expected nonempty dice-libero-episodes-v1 archive")
    provenance = payload.get("provenance")
    from rl_vla.dice import config_provenance, validate_replay_provenance

    expected = config_provenance(c)
    validate_replay_provenance(provenance, expected)
    if provenance["task"]["task_key"] != args.task_key:
        raise ValueError("Explicit --task-key does not match demonstration provenance")
    if provenance["demonstration"]["source_id"] != args.source_id:
        raise ValueError("Explicit --source-id does not match demonstration provenance")
    if provenance["demonstration"]["source_split"] != args.source_split:
        raise ValueError("Explicit --source-split does not match demonstration provenance")
    resource_check()
    prior = FrozenPi05(c["policy"], c["dice"])
    records = []
    for episode in payload["episodes"]:
        records.extend(convert_episode(episode, prior, c["dice"]["gamma"], provenance))
    packed = msgpack_numpy.packb(
        {
            "format": "dice-pi05-transitions-v1",
            "metadata": {**prior.metadata, "gamma": c["dice"]["gamma"]},
            "provenance": provenance,
            "transitions": records,
        }
    )
    if len(packed) > limit:
        raise ValueError("Converted replay exceeds byte limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(packed)
    print(f"Prepared {len(records)} transitions; no optimizer updates")


if __name__ == "__main__":
    main()
