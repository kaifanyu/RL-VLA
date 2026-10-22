"""RLDT's transport field and local actor objective in pi0.5's time convention.

The paper integrates from noise at tau=0 to data at tau=1. OpenPI integrates
from noise at t=1 to data at t=0, so ``t = 1 - tau`` and ``v_openpi = -v_paper``.
Consequently, gradient *descent* on ``+v_openpi.dot(field)/(T*t)`` moves the
predicted clean action up the Q-derived transport field. No derivative of the
field, critic, ODE trajectory, or reference actor enters the actor update.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def svgd_field(
    clean: Array,
    queries: Array,
    q_grad: Array,
    *,
    kappa0: float = 0.001,
    bandwidth_floor: float = 1e-6,
    temperature_floor: float = 1e-8,
    temperature_mode: str = "scaled",
    kernel_mode: str = "paper",
    action_mask: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate detached SVGD ascent directions using clean source particles.

    Args:
        clean: Clean action particles, shape ``[B, K, D]``.
        queries: Expected clean actions at selected ODE steps, ``[B, K, D]``.
        q_grad: Gradients of the chosen critic with respect to ``clean``.
        kappa0: Adaptive temperature coefficient: kappa = kappa0 * mean(|dQ/da|).
        bandwidth_floor: Minimum RBF standard deviation, in action units. Use
            1e-4 to reproduce the official squared-bandwidth floor of 1e-8.
        temperature_mode: ``scaled`` reproduces the official implementation,
            ``mean(k*dQ/da + kappa*grad_source(k))``. For positive kappa this is
            kappa times the unnormalized Eq. (6) field and remains defined at
            kappa0=0. ``boltzmann`` uses dQ/da / max(kappa, temperature_floor).
            Neither mode additionally normalizes the field in RKHS norm.
        kernel_mode: ``paper`` sets h to the median off-diagonal clean-clean
            distance, as described in Sec. 4.1. ``official`` sets h^2 per query
            to median_j(|query-clean_j|^2)/(2*log(K+1)), as in the released code
            (the lower of the middle two values when K is even).
            Both use k(x,y) = exp(-|x-y|^2/(2*h^2)).
        action_mask: Mask broadcastable to ``[B,K,D]``; excludes unexecuted
            chunk steps and padded motor dimensions from distances and fields.

    The RBF derivative is taken with respect to the SOURCE: (query-clean)*k/h^2.
    It therefore repels nearby particles. Bandwidth and temperature are detached,
    as is the entire returned field. Metrics are means over states and particles.
    """
    if clean.ndim != 3 or queries.shape != clean.shape or q_grad.shape != clean.shape:
        raise ValueError("clean, queries, and q_grad must have identical [B,K,D] shapes")
    if clean.shape[1] < 1:
        raise ValueError("at least one source particle is required")
    if kappa0 < 0 or bandwidth_floor <= 0 or temperature_floor <= 0:
        raise ValueError("kappa0 must be nonnegative and numerical floors must be positive")
    if temperature_mode not in {"scaled", "boltzmann"}:
        raise ValueError(f"unknown temperature_mode: {temperature_mode}")
    if kernel_mode not in {"paper", "official"}:
        raise ValueError(f"unknown kernel_mode: {kernel_mode}")

    clean = jax.lax.stop_gradient(jnp.asarray(clean, dtype=jnp.float32))
    queries = jax.lax.stop_gradient(jnp.asarray(queries, dtype=jnp.float32))
    q_grad = jax.lax.stop_gradient(jnp.asarray(q_grad, dtype=jnp.float32))
    mask = 1.0 if action_mask is None else jax.lax.stop_gradient(
        jnp.asarray(action_mask, dtype=clean.dtype)
    )
    clean, queries, q_grad = clean * mask, queries * mask, q_grad * mask
    count = clean.shape[1]
    # Axes: state, query, source, action. Sources are always CLEAN actions.
    delta = queries[:, :, None, :] - clean[:, None, :, :]
    distance_squared = jnp.sum(jnp.square(delta), axis=-1)
    if kernel_mode == "official":
        # torch.median (the official implementation) chooses the lower middle
        # element for even K; jnp.median averages the two middle elements.
        h_squared = jnp.sort(distance_squared, axis=-1)[..., (count - 1) // 2, None]
        h_squared = h_squared / (2.0 * jnp.log(count + 1.0))
    elif count == 1:
        h_squared = jnp.full((clean.shape[0], 1, 1), bandwidth_floor**2)
    else:
        left, right = jnp.triu_indices(count, k=1)
        pair_distances = jnp.linalg.norm(clean[:, left] - clean[:, right], axis=-1)
        h_squared = jnp.square(jnp.median(pair_distances, axis=-1))[:, None, None]
    h_squared = jnp.maximum(h_squared, bandwidth_floor**2)
    kernel = jnp.exp(-distance_squared / (2.0 * h_squared))
    grad_norm = jnp.linalg.norm(q_grad, axis=-1)
    kappa = kappa0 * jnp.clip(jnp.mean(grad_norm, axis=-1, keepdims=True), 0.0, 1e5)
    attraction = jnp.mean(kernel[..., None] * q_grad[:, None, :, :], axis=2)
    repulsion = jnp.mean(kernel[..., None] * delta / h_squared[..., None], axis=2)
    if temperature_mode == "scaled":
        repulsion = kappa[..., None] * repulsion
    else:
        attraction = attraction / jnp.maximum(kappa[..., None], temperature_floor)
    field = (attraction + repulsion) * mask
    metrics = {
        "bandwidth": jnp.mean(jnp.sqrt(h_squared)),
        "temperature": jnp.mean(kappa),
        "field_norm": jnp.mean(jnp.linalg.norm(field, axis=-1)),
        "q_grad_norm": jnp.mean(grad_norm),
        "attraction_norm": jnp.mean(jnp.linalg.norm(attraction, axis=-1)),
        "repulsion_norm": jnp.mean(jnp.linalg.norm(repulsion, axis=-1)),
    }
    return jax.lax.stop_gradient(field), jax.tree.map(jax.lax.stop_gradient, metrics)


def actor_loss(
    velocity: Array,
    latents: Array,
    times: Array,
    clean: Array,
    field: Array,
    reference_velocity: Array,
    *,
    num_steps: int,
    consistency_weight: float = 0.5,
    constraint_weight: float = 0.2,
    action_mask: Array | None = None,
    constraint_velocity: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    """RLDT Eqs. (13)-(16), expressed for OpenPI's reverse-time velocity.

    ``times`` has the sample shape (usually ``[B,K]``); the action tensors share
    that prefix and one or more event dimensions (``[B,K,D]`` or ``[B,K,H,A]``).
    Valid sampled times are ``1, (T-1)/T, ..., 1/T``; terminal t=0 is excluded.
    They are lower-bounded at 1/T for numerical stability. The caller samples
    exactly these ODE grid points, not continuously interpolated states.

    The expected clean action is ``latents - t*velocity``. Action dimensions are
    SUMMED, then sample dimensions are averaged. ``action_mask`` broadcasts to
    the full action shape. Only current actor velocities receive gradients. Reference
    velocity must be evaluated using the outer-iteration snapshot of the actor.
    It is not permanently the original pretrained actor. Supply
    ``constraint_velocity`` to evaluate the constraint on a separate noise/data
    interpolation, as the released implementation does; ``reference_velocity``
    must then be evaluated at that same independently sampled latent/time.
    Otherwise the constraint uses the supplied trajectory ``velocity``.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if times.ndim < 1 or velocity.shape[: times.ndim] != times.shape:
        raise ValueError("times must be a nonempty sample-shape prefix of velocity")
    if velocity.ndim <= times.ndim:
        raise ValueError("velocity must have at least one action/event dimension")
    if any(item.shape != velocity.shape for item in (latents, clean, field, reference_velocity)):
        raise ValueError("all action tensors must have the same shape as velocity")
    if constraint_velocity is not None and constraint_velocity.shape != velocity.shape:
        raise ValueError("constraint_velocity must have the same shape as velocity")
    if consistency_weight < 0 or constraint_weight < 0:
        raise ValueError("regularization weights must be nonnegative")

    velocity = jnp.asarray(velocity, dtype=jnp.float32)
    constraint_velocity = velocity if constraint_velocity is None else jnp.asarray(
        constraint_velocity, dtype=velocity.dtype
    )
    latents, clean, field, reference_velocity = (
        jax.lax.stop_gradient(jnp.asarray(item, dtype=velocity.dtype))
        for item in (latents, clean, field, reference_velocity)
    )
    t = jax.lax.stop_gradient(jnp.asarray(times, dtype=velocity.dtype))
    t = jnp.maximum(t, 1.0 / num_steps)
    event_ndim = velocity.ndim - times.ndim
    t_action = t.reshape(t.shape + (1,) * event_ndim)
    event_axes = tuple(range(times.ndim, velocity.ndim))
    mask = 1.0 if action_mask is None else jax.lax.stop_gradient(
        jnp.asarray(action_mask, dtype=velocity.dtype)
    )
    expected = latents - t_action * velocity
    rl_loss = jnp.mean(jnp.sum(velocity * field * mask, axis=event_axes) / (num_steps * t))
    consistency_loss = jnp.mean(
        jnp.sum(jnp.square(expected - clean) * mask, axis=event_axes)
        / jnp.square(num_steps * t)
    )
    constraint_loss = jnp.mean(
        jnp.sum(jnp.square(constraint_velocity - reference_velocity) * mask, axis=event_axes)
    )
    loss = rl_loss + consistency_weight * consistency_loss + constraint_weight * constraint_loss
    return loss, {
        "actor_loss": loss,
        "rl_loss": rl_loss,
        "consistency_loss": consistency_loss,
        "constraint_loss": constraint_loss,
    }
