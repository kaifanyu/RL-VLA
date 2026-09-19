"""A loaded optimizer must not silently disagree with the run manifest."""

from copy import deepcopy

import pytest

from rl_vla.config import load_config
from rl_vla.train import train


@pytest.mark.parametrize("changed", ["actor_lr", "batch_size"])
def test_warm_start_rejects_effective_learner_change(tmp_path, changed):
    config = load_config("configs/ppo_toy.toml")
    config["train"].update(total_decisions=2, rollout_steps=2, batch_size=2)
    config["ppo"].update(hidden_dim=16, epochs=1)
    train(config, str(tmp_path / "original"))
    incompatible = deepcopy(config)
    if changed == "actor_lr":
        incompatible["ppo"]["actor_lr"] = 0.002
    else:
        incompatible["train"]["batch_size"] = 1
    with pytest.raises(ValueError, match="learner settings"):
        train(incompatible, str(tmp_path / "changed"), str(tmp_path / "original" / "final.pt"))

