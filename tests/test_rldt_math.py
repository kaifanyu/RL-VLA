"""Algorithm checks for the RLDT transport direction and pi0.5 time convention."""

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from rl_vla.rldt_math import actor_loss, svgd_field


def test_paper_kernel_matches_analytic_attraction():
    clean = jnp.array([[[-1.0], [1.0]]])
    field, metrics = svgd_field(clean, clean, jnp.ones_like(clean), kappa0=0.0)
    # h=2, so k(-1,+1)=exp(-1/2). Mean over both source particles.
    np.testing.assert_allclose(field, (1.0 + math.exp(-0.5)) / 2.0, rtol=1e-6)
    assert float(metrics["bandwidth"]) == 2.0


def test_repulsion_points_away_from_source_particles():
    # Q increases vertically; the horizontal field isolates kernel repulsion.
    clean = jnp.array([[[-1.0, 0.0], [1.0, 0.0]]])
    q_grad = jnp.array([[[0.0, 1.0], [0.0, 1.0]]])
    field, _ = svgd_field(clean, clean, q_grad, kappa0=0.5)
    assert float(field[0, 0, 0]) < 0.0
    assert float(field[0, 1, 0]) > 0.0
    np.testing.assert_allclose(field[0, :, 0].sum(), 0.0, atol=1e-7)
    np.testing.assert_allclose(field[0, :, 0], [-math.exp(-0.5) / 8, math.exp(-0.5) / 8])


def test_official_kernel_uses_lower_median_per_query():
    clean = jnp.array([[[0.0], [2.0], [4.0], [6.0]]])
    queries = jnp.array([[[1.0], [2.0], [4.0], [5.0]]])
    field, metrics = svgd_field(
        clean, queries, jnp.ones_like(clean), kappa0=0.0, kernel_mode="official"
    )
    squared = np.square(np.asarray(queries)[:, :, None] - np.asarray(clean)[:, None]).sum(-1)
    # Explicit lower median matches torch.median, not numpy/jax median for even K.
    h_squared = np.sort(squared, axis=-1)[..., 1:2] / (2 * np.log(5.0))
    expected = np.exp(-squared / (2 * h_squared)).mean(-1)[..., None]
    np.testing.assert_allclose(field, expected, rtol=1e-6)
    np.testing.assert_allclose(metrics["bandwidth"], np.sqrt(h_squared).mean(), rtol=1e-6)


@pytest.mark.parametrize("kernel_mode", ["paper", "official"])
@pytest.mark.parametrize("temperature_mode", ["scaled", "boltzmann"])
def test_coincident_particles_and_zero_q_gradient_are_finite(kernel_mode, temperature_mode):
    particles = jnp.ones((2, 8, 3))
    field, metrics = svgd_field(
        particles, particles, jnp.zeros_like(particles),
        kernel_mode=kernel_mode, temperature_mode=temperature_mode,
    )
    np.testing.assert_array_equal(field, 0.0)
    assert all(np.isfinite(value) for value in metrics.values())


def test_scaled_field_equals_kappa_times_boltzmann_field():
    clean = jnp.array([[[-1.0], [1.0]]])
    q_grad = jnp.array([[[2.0], [4.0]]])
    scaled, info = svgd_field(clean, clean, q_grad, kappa0=0.1)
    exact, _ = svgd_field(clean, clean, q_grad, kappa0=0.1, temperature_mode="boltzmann")
    np.testing.assert_allclose(scaled, exact * info["temperature"], rtol=1e-6)


def test_svgd_is_fully_detached():
    values = jnp.array([[[-1.0, 2.0], [1.0, 3.0]]])

    def objective(clean, queries, q_grad):
        return svgd_field(clean, queries, q_grad)[0].sum()

    gradients = jax.grad(objective, argnums=(0, 1, 2))(values, values * 0.9, values * 0.5)
    for gradient in gradients:
        np.testing.assert_array_equal(gradient, 0.0)


def test_gradient_descent_increases_expected_clean_action_in_pi05_time():
    latent = jnp.zeros((1, 1, 1))
    clean = jnp.zeros_like(latent)
    field = jnp.ones_like(latent)
    times = jnp.ones((1, 1))

    def objective(velocity):
        return actor_loss(
            velocity, latent, times, clean, field, jnp.zeros_like(latent),
            num_steps=8, consistency_weight=0.0, constraint_weight=0.0,
        )[0]

    velocity = jnp.zeros_like(latent)
    updated_velocity = velocity - 0.1 * jax.grad(objective)(velocity)
    expected_before = latent - times[..., None] * velocity
    expected_after = latent - times[..., None] * updated_velocity
    assert bool(jnp.all(expected_after > expected_before))


def test_intermediate_time_scaling_and_action_norms():
    # Two sampled steps, three action dimensions: sum events, mean samples.
    velocity = jnp.ones((1, 2, 3))
    latent = jnp.zeros_like(velocity)
    times = jnp.array([[1.0, 0.25]])
    loss, info = actor_loss(
        velocity, latent, times, latent, velocity, latent,
        num_steps=4, consistency_weight=0.5, constraint_weight=0.2,
    )
    np.testing.assert_allclose(info["rl_loss"], (3 / 4 + 3) / 2)
    np.testing.assert_allclose(info["consistency_loss"], 3 / 16)
    np.testing.assert_allclose(info["constraint_loss"], 3)
    np.testing.assert_allclose(loss, 1.875 + 0.5 * 3 / 16 + 0.2 * 3)

    def objective(v):
        return actor_loss(
            v, latent, times, latent, velocity, latent,
            num_steps=4, consistency_weight=0.0, constraint_weight=0.0,
        )[0]

    gradient = jax.grad(objective)(velocity)
    np.testing.assert_allclose(gradient[:, 1], 4 * gradient[:, 0])


def test_actor_gradient_does_not_follow_trajectory_critic_or_reference():
    values = jnp.ones((1, 2, 2))
    times = jnp.array([[1.0, 0.5]])

    def objective(velocity, latents, clean, field, reference):
        return actor_loss(
            velocity, latents, times, clean, field, reference, num_steps=8
        )[0]

    gradients = jax.grad(objective, argnums=(0, 1, 2, 3, 4))(
        values, values * 2, values * 0.5, values * 3, values * 0.2,
    )
    assert bool(jnp.any(gradients[0] != 0.0))
    for gradient in gradients[1:]:
        np.testing.assert_array_equal(gradient, 0.0)


def test_active_action_mask_removes_padded_dimensions_and_future_steps():
    velocity = jnp.ones((1, 2, 3, 4))
    latent = jnp.zeros_like(velocity)
    times = jnp.ones((1, 2))
    mask = jnp.zeros((3, 4)).at[:2, :2].set(1.0)

    def objective(v):
        return actor_loss(
            v, latent, times, latent, velocity, latent,
            num_steps=1, action_mask=mask,
        )[0]

    gradient = jax.grad(objective)(velocity)
    assert bool(jnp.all(gradient[..., :2, :2] != 0.0))
    np.testing.assert_array_equal(gradient[..., 2:, :], 0.0)
    np.testing.assert_array_equal(gradient[..., :, 2:], 0.0)
    # Padding must also not distort SVGD bandwidths or critic gradient magnitudes.
    particles = jnp.array([[[-1.0, -1000.0], [1.0, 1000.0]]])
    field, metrics = svgd_field(
        particles, particles, jnp.ones_like(particles), action_mask=jnp.array([1.0, 0.0])
    )
    np.testing.assert_array_equal(field[..., 1], 0.0)
    assert float(metrics["bandwidth"]) == 2.0
    assert float(metrics["q_grad_norm"]) == 1.0


def test_constraint_accepts_independent_interpolation_and_backpropagates_it():
    values = jnp.zeros((1, 2, 3))
    times = jnp.ones((1, 2))

    def objective(velocity, constraint_velocity):
        return actor_loss(
            velocity, values, times, values, values, values,
            num_steps=8, consistency_weight=0.0,
            constraint_weight=0.2, constraint_velocity=constraint_velocity,
        )[0]

    gradients = jax.grad(objective, argnums=(0, 1))(values, values + 2)
    np.testing.assert_array_equal(gradients[0], 0.0)
    np.testing.assert_allclose(gradients[1], 0.4)


def test_math_helpers_work_under_jit():
    clean = jnp.array([[[-1.0], [1.0]]])
    field, _ = jax.jit(svgd_field)(clean, clean, jnp.ones_like(clean))
    loss, _ = jax.jit(lambda v: actor_loss(
        v, clean, jnp.ones((1, 2)), clean, field, clean, num_steps=8,
    ))(clean)
    assert np.isfinite(loss)
