"""Check the real pi0.5 RLDT adapter against upstream OpenPI inference.

Run with the pinned OpenPI interpreter and PYTHONPATH=src.  This loads real
checkpoint weights and uses a synthetic observation to check software parity;
it does not measure task success.  GPU tests are explicitly invoked, never run
as part of the lightweight unit suite.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/openpi/pi05_libero")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--projection-only", action="store_true")
    parser.add_argument("--skip-gradient", action="store_true")
    parser.add_argument(
        "--save-expert",
        type=Path,
        help="Optional compact expert save/reload check; needs disk space",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)

    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from openpi.policies.policy import Policy

    from rl_vla.rldt_pi05 import Pi05Flow, is_expert_parameter

    started = time.monotonic()
    print("Loading native JAX pi0.5 checkpoint", flush=True)
    flow = Pi05Flow.from_config(
        {"checkpoint": str(args.checkpoint), "projection_only": args.projection_only}
    )
    rng = np.random.default_rng(args.seed)
    observation = {
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, np.float32),
        "prompt": "pick up the black bowl and place it on the plate",
    }
    raw = [{"openpi": observation}]
    key = jax.random.PRNGKey(args.seed)
    noise = jax.random.normal(key, (1, flow.horizon, flow.action_dim))
    print("Checking frozen-prefix sampler against ordinary OpenPI Policy.infer", flush=True)
    context = flow.prepare(raw)
    sample = jax.jit(
        lambda params, frozen, prepared, rng_key: flow.sample(
            params, prepared, rng_key, args.steps, frozen=frozen
        )
    )
    generated, path = sample(flow.params, flow.frozen, context, key)
    # Use the ordinary upstream policy on the SAME mixed-precision model.  It
    # recomputes its own prefix/cache and uses the original while-loop sampler.
    model = nnx.merge(flow.graphdef, flow.params, flow.frozen)
    policy = Policy(
        model, transforms=[flow._input_transform], sample_kwargs={"num_steps": args.steps}
    )
    reference = policy.infer(observation, noise=np.asarray(noise[0]))
    normalized = np.asarray(generated)[0]
    normalized_error = float(np.max(np.abs(normalized - reference["actions"])))
    # Scan's exact grid and upstream's repeated float32 subtraction can differ
    # by rounding before bfloat16 layers.  These tolerances detect broken cache,
    # attention masks, time direction, action transforms, and coordinate slicing.
    np.testing.assert_allclose(normalized, reference["actions"], atol=0.025, rtol=0.025)
    decoded = flow.decode(generated, raw)[0]
    reference_decoded = flow._output_transform(
        {
            "state": np.asarray(reference["state"]),
            "actions": reference["actions"].copy(),
        }
    )["actions"]
    np.testing.assert_allclose(decoded, reference_decoded, atol=0.025, rtol=0.025)
    assert path.shape == (args.steps, 1, flow.horizon, flow.action_dim)
    np.testing.assert_array_equal(path[0], noise)
    for path_key in flow.params.flat_state():
        assert is_expert_parameter(path_key, projection_only=args.projection_only)
    report = {
        "passed": True,
        "software_test_only": True,
        "metadata": flow.metadata,
        "devices": [str(device) for device in jax.devices()],
        "normalized_max_abs_error": normalized_error,
        "decoded_max_abs_error": float(np.max(np.abs(decoded - reference_decoded))),
        "parity_tolerances": {"atol": 0.025, "rtol": 0.025},
        "normalized_shape": list(generated.shape),
        "decoded_shape": list(decoded.shape),
        "trajectory_shape": list(path.shape),
        "features_shape": list(context["features"].shape),
    }
    if not args.skip_gradient:
        print("Checking finite expert gradient and changed velocity after update", flush=True)
        frozen_leaves = jax.tree.leaves(flow.frozen)
        x, t = path[args.steps // 2], jnp.full((1,), 0.5)

        def loss(params, frozen, prepared, latent, timestep):
            value = flow.velocity(params, prepared, latent, timestep, frozen=frozen)[
                ..., : flow.motor_dim
            ]
            return jnp.mean(jnp.square(value))

        value, grads = jax.jit(jax.value_and_grad(loss))(flow.params, flow.frozen, context, x, t)
        grad_norm = float(optax.global_norm(grads))
        if not np.isfinite(float(value)) or not np.isfinite(grad_norm) or grad_norm <= 0:
            raise AssertionError("Expert gradient must be finite and nonzero")
        updates = jax.tree.map(lambda gradient: -1e-4 * gradient, grads)
        updated = optax.apply_updates(flow.params, updates)
        velocity = jax.jit(flow.velocity)
        before = velocity(flow.params, context, x, t, frozen=flow.frozen)
        after = velocity(updated, context, x, t, frozen=flow.frozen)
        change = float(jnp.linalg.norm(after - before))
        if not np.isfinite(change) or change <= 0:
            raise AssertionError("Updated expert parameters did not change the flow velocity")
        if not all(
            a is b for a, b in zip(frozen_leaves, jax.tree.leaves(flow.frozen), strict=True)
        ):
            raise AssertionError("Frozen model arrays were replaced by the expert update")
        report.update(
            gradient_loss=float(value),
            expert_grad_norm=grad_norm,
            velocity_update_l2=change,
            frozen_arrays_unchanged=True,
        )
    if args.save_expert:
        print("Checking compact expert checkpoint save and restore", flush=True)
        saved = flow.save(args.save_expert, flow.params)
        restored = flow.load_params(saved)
        for original, loaded in zip(
            jax.tree.leaves(flow.params), jax.tree.leaves(restored), strict=True
        ):
            np.testing.assert_array_equal(original, loaded)
        report["expert_roundtrip"] = str(saved)
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
