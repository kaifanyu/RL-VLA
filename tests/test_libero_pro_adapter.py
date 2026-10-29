"""Check state split isolation and strict evaluation enumeration without MuJoCo."""

import hashlib

import numpy as np
import pytest
from test_libero_adapter import FakeLibero

from rl_vla.envs.libero_pro import SplitLiberoEnv, checked_file, validate_splits


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


@pytest.mark.parametrize("split", ["validation", "test", "source"])
def test_validation_and_test_enumerate_only_their_own_states(split):
    env = make_split(split, [4, 8])
    assert [env.reset(seed=i)[1]["initial_state_id"] for i in range(2)] == [4, 8]
    with pytest.raises(ValueError, match="more episodes"):
        env.reset(seed=2)


def test_three_way_split_rejects_overlap_duplicates_and_missing_states():
    task = {"pro_state_count": 6, "train_state_ids": [0, 1],
            "validation_state_ids": [2, 3], "test_state_ids": [4, 5]}
    validate_splits(task, 2)
    for invalid in ([3, 4, 5], [4, 4, 5], [4], [-1, 4, 5], [4, 5.0]):
        with pytest.raises(ValueError):
            validate_splits(task | {"test_state_ids": invalid}, 2)


def test_previous_two_way_protocol_still_works():
    validate_splits({"pro_state_count": 3, "train_state_ids": [0, 1],
                     "heldout_state_ids": [2]}, 1)
