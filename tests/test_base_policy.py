import numpy as np
import pytest

from rl_vla.base_policy import MockBasePolicy, OpenPIBasePolicy, make_base_policy


class FakeClient:
    def __init__(self, actions):
        self.actions = actions
        self.payload = None
        self.resets = 0

    def infer(self, payload):
        self.payload = payload
        return {"actions": self.actions}

    def get_server_metadata(self):
        return {"checkpoint": "test"}

    def reset(self):
        self.resets += 1


def test_mock_repeats_nominal_action():
    policy = make_base_policy({"kind": "mock"}, 2, 3)
    actual = policy.infer({"mock_base_action": [0.1, -0.2]})
    np.testing.assert_allclose(actual, [[0.1, -0.2]] * 3)
    with pytest.raises(ValueError, match="non-finite"):
        MockBasePolicy(2, 3).infer({"mock_base_action": [np.nan, 0]})


def test_openpi_passes_checkpoint_observation_and_preserves_chunk():
    client = FakeClient(np.ones((10, 7)))
    policy = OpenPIBasePolicy(7, 5, client=client)
    payload = {"observation/state": np.zeros(8), "prompt": "pick up object"}
    actual = policy.infer({"openpi": payload, "privileged": "not sent"})
    assert actual.shape == (10, 7)
    assert set(client.payload) == set(payload)
    actual[0, 0] = 123
    assert client.actions[0, 0] == 1
    assert policy.metadata["server"] == {"checkpoint": "test"}
    policy.reset()
    assert client.resets == 1


@pytest.mark.parametrize("actions", [np.zeros((3, 7)), np.zeros((10, 32)), np.zeros((1, 10, 7)), np.full((10, 7), np.inf)])
def test_openpi_rejects_wrong_or_nonfinite_chunk(actions):
    policy = OpenPIBasePolicy(7, 5, client=FakeClient(actions))
    with pytest.raises(ValueError):
        policy.infer({})


def test_explicit_action_indices_and_observation_mapping():
    client = FakeClient(np.arange(24).reshape(3, 8))
    policy = OpenPIBasePolicy(
        2, 3, client=client, action_indices=[2, 4], observation_map={"observation/state": "robot_state"}
    )
    actual = policy.infer({"robot_state": [1, 2], "private": "unused"})
    assert client.payload == {"observation/state": [1, 2]}
    np.testing.assert_array_equal(actual, [[2, 4], [10, 12], [18, 20]])


@pytest.mark.parametrize("indices", [[0], [0, 0], [-1, 0], [0.5, 1], [False, 1]])
def test_invalid_coordinate_mapping_fails_early(indices):
    with pytest.raises(ValueError, match="action_indices"):
        OpenPIBasePolicy(2, 3, action_indices=indices, client=FakeClient(np.zeros((3, 2))))
