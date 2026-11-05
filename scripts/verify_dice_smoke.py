#!/usr/bin/env python
"""Verify real save/resume artifacts and frozen-prior invariance after an update.

Run in the OpenPI environment on an allocated GPU. Never changes checkpoint
files or simulator state. Reads trusted local resume.pt image replay.
"""

import argparse
import copy
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from rl_vla.dice import proposals, tensor
from rl_vla.dice_core import DICE
from rl_vla.dice_pi05 import FrozenPi05
from rl_vla.dice_runtime import restore_training, source_manifest


def digest(tensors):
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    first = torch.load(args.first, map_location="cpu", weights_only=False)
    resumed = torch.load(args.resumed, map_location="cpu", weights_only=False)
    assert first["prior"] == resumed["prior"]
    assert resumed["counters"]["controller_steps"] > first["counters"]["controller_steps"]
    assert resumed["learner"]["actor_updates"] > first["learner"]["actor_updates"]
    assert len(resumed["replay"]) > len(first["replay"])
    assert digest(first["learner"]["actor"]) != digest(resumed["learner"]["actor"])
    for name in ("actor_optimizer", "critic_optimizer"):
        first_steps = {int(v["step"]) for v in first["learner"][name]["state"].values()}
        resumed_steps = {int(v["step"]) for v in resumed["learner"][name]["state"].values()}
        expected_first = (
            first["learner"]["actor_updates"]
            if name == "actor_optimizer"
            else first["learner"]["updates"]
        )
        expected_resumed = (
            resumed["learner"]["actor_updates"]
            if name == "actor_optimizer"
            else resumed["learner"]["updates"]
        )
        assert first_steps == {expected_first} and resumed_steps == {expected_resumed}
    c = resumed["config"]
    prior = FrozenPi05(c["policy"], {**c["dice"], "seed": 917})
    learner = DICE(prior.feature_dim, prior.noise_dim, prior.action_dim, c["dice"])
    restore_config = copy.deepcopy(c)
    restore_config["train"]["episodes"] = resumed["counters"]["episodes"] + 1
    for path, saved in ((args.first, first), (args.resumed, resumed)):
        replay = deque(maxlen=c["train"]["replay_capacity"])
        rng = np.random.default_rng(321)
        restore_training(path, learner, restore_config, prior, replay, rng)
        for name in ("actor_optimizer", "critic_optimizer"):
            actual = getattr(learner, name).state_dict()
            expected = saved["learner"][name]
            assert actual["param_groups"] == expected["param_groups"]
            for index, state in expected["state"].items():
                for key, value in state.items():
                    assert torch.equal(actual["state"][index][key], value)
        assert torch.equal(torch.get_rng_state(), saved["torch_rng"])
        assert rng.bit_generator.state == saved["numpy_rng"]
        import jax

        np.testing.assert_array_equal(jax.random.key_data(prior.key), saved["prior_rng"])
        assert len(replay) == len(saved["replay"])
    rows = resumed["replay"][:2]
    key = prior.key
    before = prior.propose(rows[0]["observation"], 1)
    batch = {
        name: tensor(np.stack([row[name] for row in rows]))
        for name in (
            "features",
            "actions",
            "returns",
            "rewards",
            "discounts",
            "expert",
            "action_mask",
        )
    }
    actor_proposals = proposals(prior, [r["observation"] for r in rows], c["dice"]["k_train"])
    target_proposals = proposals(
        prior, [r["next_observation"] for r in rows], c["dice"]["k_bootstrap"]
    )
    actor_before = digest(learner.actor.state_dict())
    result = learner.update(batch, actor_proposals, target_proposals, step=5)
    assert actor_before != digest(learner.actor.state_dict())
    prior.key = key
    after = prior.propose(rows[0]["observation"], 1)
    assert all(np.array_equal(x, y) for x, y in zip(before, after, strict=True))
    # Decoder equivalence of an unedited prefix with the native full sample.
    _, _, prefix, full = before
    decoded = prior.decode(prefix[0], full[0], rows[0]["observation"])
    expected = prior.flow.decode(full, [rows[0]["observation"]])[0, : prior.execute_steps]
    assert np.array_equal(decoded, expected)
    # Best-of-N index uses the same latent/prior association as the selected action.
    f, z, p, _ = prior.propose(rows[0]["observation"], 16)
    action, index = learner.select(tensor(f), tensor(z), tensor(p))
    all_actions = learner.action(tensor(f), tensor(z), tensor(p))
    assert index == int(learner.q(tensor(f), all_actions).min(0).values.argmax())
    assert torch.equal(action, all_actions[index])
    report = {
        "first": str(args.first),
        "resumed": str(args.resumed),
        "first_counters": first["counters"],
        "resumed_counters": resumed["counters"],
        "replay_rows": [len(first["replay"]), len(resumed["replay"])],
        "frozen_features_noise_actions_bitwise_equal_after_update": True,
        "native_zero_residual_decode_equal": True,
        "best_of_16_index_matches_min_ensemble_oracle": True,
        "optimizer_moments_and_steps_restored": True,
        "restored_optimizer_tensors_and_numpy_torch_jax_rng_equal_saved": True,
        "post_restore_additional_update": result,
        "prior": prior.metadata,
        "sources": source_manifest(),
        "scope": "real pi0.5; two shortened LIBERO episodes; online-only smoke ablation; no performance claim",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("prior", "sources")}, indent=2))


if __name__ == "__main__":
    main()
