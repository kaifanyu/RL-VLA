"""Check state split isolation and strict evaluation enumeration without MuJoCo."""

import hashlib

import numpy as np
import pytest
from test_libero_adapter import FakeLibero

from rl_vla.envs.libero_pro import SplitLiberoEnv, checked_file


def make_split(split, ids):
    return SplitLiberoEnv(
        split=split, state_ids=ids, backend=FakeLibero(),
        initial_states=np.arange(30).reshape(10, 3), task_description="pick up the cube",
        resize_size=2, image_processor=lambda image, size: image,
    )


def test_training_never_samples_heldout_states_and_is_reproducible():
    first, second = make_split("train", [1, 3, 7]), make_split("train", [1, 3, 7])
    a = [first.reset(seed=i)[1]["initial_state_id"] for i in range(100)]
    b = [second.reset(seed=i)[1]["initial_state_id"] for i in range(100)]
    assert a == b
    assert set(a) == {1, 3, 7}


def test_evaluation_enumerates_each_frozen_state_and_rejects_overrun():
    env = make_split("heldout", [2, 5, 8])
    assert [env.reset(seed=100 + i)[1]["initial_state_id"] for i in range(3)] == [2, 5, 8]
    with pytest.raises(ValueError, match="more episodes"):
        env.reset(seed=104)
    with pytest.raises(ValueError, match="fixed"):
        env.reset(seed=1, options={"initial_state_id": 1})


def test_asset_changes_are_detected(tmp_path):
    path = tmp_path / "asset"
    path.write_bytes(b"original")
    record = {"path": str(path), "sha256": hashlib.sha256(b"original").hexdigest()}
    assert checked_file(record) == path
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        checked_file(record)
