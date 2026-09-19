"""LIBERO contract tests with an injected fake backend; no renderer or weights."""

import numpy as np
import pytest
import torch

from rl_vla.envs.libero import (
    SETTLING_ACTION,
    LiberoEnv,
    _load_initial_states,
    quaternion_to_axis_angle,
)


class FakeLibero:
    def __init__(self, success_at=None, done_at=None):
        self.success_at = success_at
        self.done_at = done_at
        self.front = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        self.wrist = self.front + 20
        self.seeds = []
        self.closed = False

    def seed(self, value):
        self.seeds.append(value)
        # Reproduce LIBERO's global RNG effect to verify local selection isolation.
        np.random.seed(value)

    def observation(self):
        return {
            "robot0_eef_pos": np.array([self.steps, 0, 0], dtype=np.float32),
            "robot0_eef_quat": np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)]),
            "robot0_gripper_qpos": np.array([0.01, -0.01]),
            "agentview_image": self.front.copy(),
            "robot0_eye_in_hand_image": self.wrist.copy(),
        }

    def reset(self):
        self.steps = 0
        self.actions = []
        return self.observation()

    def set_init_state(self, value):
        self.selected = value.copy()
        return self.observation()

    def check_success(self):
        return self.success_at is not None and self.steps >= self.success_at

    def step(self, action):
        self.steps += 1
        self.actions.append(np.array(action))
        done = self.check_success() or (self.done_at is not None and self.steps >= self.done_at)
        return self.observation(), 99.0, done, {}

    def close(self):
        self.closed = True


def build_env(**kwargs):
    backend = kwargs.pop("backend", FakeLibero())
    env = LiberoEnv(
        backend=backend,
        initial_states=np.arange(12).reshape(4, 3),
        task_description="pick up the cube",
        resize_size=2,
        image_processor=lambda image, size: image,
        **kwargs,
    )
    return env, backend


def test_reset_settles_with_official_action_and_maps_images_and_state():
    env, backend = build_env(initial_state_id=2)
    observation, info = env.reset(seed=9)
    assert backend.steps == 10 and env.steps == 0
    np.testing.assert_array_equal(backend.actions, np.tile(SETTLING_ACTION, (10, 1)))
    np.testing.assert_array_equal(backend.selected, [6, 7, 8])
    assert info["initial_state_id"] == 2 and info["settling_steps"] == 10
    np.testing.assert_array_equal(
        observation["openpi"]["observation/image"], backend.front[::-1, ::-1]
    )
    np.testing.assert_array_equal(
        observation["openpi"]["observation/wrist_image"], backend.wrist[::-1, ::-1]
    )
    np.testing.assert_allclose(observation["state"], [10, 0, 0, 0, 0, np.pi / 2, 0.01, -0.01])
    assert observation["openpi"]["prompt"] == "pick up the cube"
    assert env.observation_space.contains(observation)


def test_quaternion_xyzw_identity_and_clipped_roundoff_are_nonmutating():
    np.testing.assert_array_equal(quaternion_to_axis_angle([0, 0, 0, 1]), np.zeros(3))
    np.testing.assert_allclose(quaternion_to_axis_angle([1, 0, 0, 0]), [np.pi, 0, 0])
    q = np.array([0, 0, 0, 1.0000001])
    saved = q.copy()
    np.testing.assert_array_equal(quaternion_to_axis_angle(q), np.zeros(3))
    np.testing.assert_array_equal(q, saved)
    with pytest.raises(ValueError):
        quaternion_to_axis_angle([0, np.nan, 0, 1])


def test_success_is_verified_and_reward_is_once_only():
    env, _ = build_env(backend=FakeLibero(success_at=12))
    env.reset(seed=1)
    _, reward, terminated, truncated, _ = env.step(np.zeros(7))
    assert reward == 0 and not terminated and not truncated
    _, reward, terminated, truncated, info = env.step(np.zeros(7))
    assert reward == 1 and terminated and not truncated and info["is_success"]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(7))


def test_time_cutoff_preserves_final_physical_state_without_reward():
    env, backend = build_env(max_steps=2)
    env.reset(seed=1)
    env.step(np.zeros(7))
    observation, reward, terminated, truncated, info = env.step(np.zeros(7))
    assert reward == 0 and not terminated and truncated and not info["is_success"]
    assert observation["state"][0] == 12 and backend.steps == 12
    assert info["native_reward"] == 99.0  # Arbitrary native reward is not task truth.


def test_native_done_without_success_is_a_cutoff():
    env, _ = build_env(backend=FakeLibero(done_at=11))
    env.reset(seed=1)
    _, reward, terminated, truncated, info = env.step(np.zeros(7))
    assert reward == 0 and not terminated and truncated and info["native_done"]


def test_reset_seed_is_reproducible_with_local_rng_and_explicit_override():
    first, first_backend = build_env()
    second, second_backend = build_env()
    first_ids, second_ids = [], []
    for index in range(5):
        first_ids.append(first.reset(seed=17 if index == 0 else None)[1]["initial_state_id"])
        second_ids.append(second.reset(seed=17 if index == 0 else None)[1]["initial_state_id"])
    assert first_ids == second_ids and first_backend.seeds == second_backend.seeds
    assert first.reset(options={"initial_state_id": 3})[1]["initial_state_id"] == 3
    with pytest.raises(ValueError, match="initial_state_id"):
        first.reset(options={"initial_state_id": 4})
    first.close()
    assert first_backend.closed


def test_reset_success_is_rejected_instead_of_becoming_a_free_training_reward():
    env, _ = build_env(backend=FakeLibero(success_at=5))
    with pytest.raises(RuntimeError, match="settling"):
        env.reset(seed=0)


def test_modern_torch_restricted_initial_state_loader(tmp_path):
    states = np.arange(12, dtype=np.float64).reshape(3, 4)
    state_file = tmp_path / "test.pruned_init"
    torch.save(states, state_file)
    np.testing.assert_array_equal(_load_initial_states(state_file), states)
