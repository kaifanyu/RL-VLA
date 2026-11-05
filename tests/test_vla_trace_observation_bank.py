import numpy as np
import pytest

from rl_vla.diagnostics.observation_bank import save_entry, state_digest


def test_state_fingerprint_ignores_time_but_not_physics():
    assert state_digest([0., 1., 2.]) == state_digest([50., 1., 2.])
    assert state_digest([0., 1., 2.]) != state_digest([0., 1., 2.1])


def test_saved_bank_is_immutable_and_noise_is_repeatable(tmp_path):
    image = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    obs = {"openpi": {"observation/image": image, "observation/wrist_image": image,
                      "observation/state": np.zeros(8), "prompt": "pick up the soup"}}
    for name in ("a", "b"):
        save_entry(tmp_path / name, obs, {}, {}, np.zeros(10), noise_seed=10)
    np.testing.assert_array_equal(np.load(tmp_path / "a/initial_noise.npy"), np.load(tmp_path / "b/initial_noise.npy"))
    with pytest.raises(FileExistsError):
        save_entry(tmp_path / "a", obs, {}, {}, np.zeros(10), noise_seed=10)
