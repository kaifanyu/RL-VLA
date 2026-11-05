"""Isolate interceptor equations, sampler carry, and optimized native projection HLO."""

import argparse
import fcntl
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from rl_vla.diagnostics.pi05_attention import (
    _Capture,
    _sample_from_noise,
    _token_metadata,
    key_keep_mask,
    load_observation,
    sample_with_intervention,
    transformed_observation,
)
from rl_vla.rldt_pi05 import Pi05Flow


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def comparison(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {"exact": bool(np.array_equal(a, b)), "max_abs_difference": float(np.max(np.abs(a-b))),
            "different_elements": int(np.count_nonzero(a != b)), "finite": bool(np.isfinite(a).all() and np.isfinite(b).all())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock_path = root / "runs/.vla_trace_gpu_locks/GPU-b64865f1-858a-43a6-62e4-e30649584262.lock"

    def guard():
        if any((run / "methods/rldt").glob("*/selection.json")):
            raise RuntimeError("Research selection ready; stop technical GPU work")
        headroom = int(Path("/sys/fs/cgroup/memory.max").read_text())-int(Path("/sys/fs/cgroup/memory.current").read_text())
        if headroom < 32*1024**3:
            raise RuntimeError("Insufficient host-memory headroom")

    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard()
        started = time.monotonic()
        checkpoint = run / "methods/rldt/object_0/training/iteration_000025"
        observation = run / "observation_bank/object_0/validation/state_000100/observation.npz"
        inputs = [Path(__file__), root / "src/rl_vla/rldt_pi05.py", root / "src/rl_vla/diagnostics/pi05_attention.py",
                  checkpoint / "expert.npz", checkpoint / "rldt_pi05.json", observation]
        hashes = {str(p): sha(p) for p in inputs}
        (output / "plan.json").write_text(json.dumps({"technical_only": True, "checkpoint": str(checkpoint),
            "observation": str(observation), "sampling": "Native PRNGKey(100000), first split subkey",
            "device": os.environ.get("CUDA_VISIBLE_DEVICES"), "memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "source_input_sha256": hashes, "scope": "No success scoring; exact native sampler compilation-shape isolation"}, indent=2)+"\n")
        flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
        raw = load_observation(observation)
        context = flow.prepare([raw])
        params = flow.load_params(checkpoint)
        _, key = jax.random.split(jax.random.PRNGKey(100000))
        noise = jax.random.normal(key, (1, flow.horizon, flow.action_dim))
        template = json.loads((run / "gate_b_actual/metadata.json").read_text())
        metadata = _token_metadata(transformed_observation(flow, raw), template["text_span"][0], flow.horizon)
        capture = _Capture(key_keep=key_keep_mask(metadata, "none"), capture_features=False)

        # Exact argument order and actions-only return of frozen rldt.evaluate.
        canonical = jax.jit(lambda p, c, k, f: flow.sample(p, c, k, 10, frozen=f)[0])
        helper_order = jax.jit(lambda p, f, c, k: flow.sample(p, c, k, 10, frozen=f)[0])
        full = jax.jit(lambda p, f, c, k: flow.sample(p, c, k, 10, frozen=f))

        def intercepted(p, f, c, k):
            with nn.intercept_methods(capture.interceptor):
                return flow.sample(p, c, k, 10, frozen=f)[0]

        intercepted_action = jax.jit(intercepted)
        transparent = _Capture(key_keep=None, capture_features=False)

        def no_mask(p, c, k, f):
            with nn.intercept_methods(transparent.interceptor):
                return flow.sample(p, c, k, 10, frozen=f)[0]

        no_mask_action = jax.jit(no_mask)
        attention_capture = _Capture(key_keep=None, capture_features=False)

        def attention_only(p, c, k, f):
            with attention_capture.intercept():
                return flow.sample(p, c, k, 10, frozen=f)[0]

        attention_action = jax.jit(attention_only)

        def one_step(p, c, x, index, f):
            t = jnp.full((x.shape[0],), 1.0-index/10, dtype=jnp.float32)
            return x-flow.velocity(p, c, x, t, frozen=f)/10

        one_step_jit = jax.jit(one_step)

        def prefix_sampler(count):
            def prefix(p, c, k, f):
                initial = jax.random.normal(k, (c["state"].shape[0], flow.horizon, flow.action_dim))

                def step(x, index):
                    t = jnp.full((x.shape[0],), 1.0-index/10, dtype=jnp.float32)
                    velocity = flow.velocity(p, c, x, t, frozen=f)
                    return x-velocity/10, x

                final, _ = jax.lax.scan(step, initial, jnp.arange(count, dtype=jnp.float32))
                return jax.lax.stop_gradient(final)

            return jax.jit(prefix)

        prefixes = {count: prefix_sampler(count) for count in range(1, 11)}
        explicit_action = jax.jit(lambda p, f, c, n: _sample_from_noise(flow, p, f, c, n, 10)[0])
        explicit_full = jax.jit(lambda p, f, c, n: _sample_from_noise(flow, p, f, c, n, 10))
        arrays = {}

        def save(name, operation):
            guard()
            value = operation()
            jax.block_until_ready(value)
            if isinstance(value, tuple):
                arrays[name+"_actions"], arrays[name+"_path"] = map(np.asarray, value)
            else:
                arrays[name+"_actions"] = np.asarray(value)
            np.savez(output / "arrays_in_progress.npz", **arrays)
            print(json.dumps({"completed": name, "elapsed_seconds": time.monotonic()-started,
                "vs_canonical": comparison(arrays["canonical_actions"], arrays[name+"_actions"])
                if "canonical_actions" in arrays else None}), flush=True)

        save("baseline_canonical", lambda: canonical(flow.params, context, key, flow.frozen))
        save("baseline_full", lambda: full(flow.params, flow.frozen, context, key))
        save("canonical", lambda: canonical(params, context, key, flow.frozen))
        save("helper_order", lambda: helper_order(params, flow.frozen, context, key))
        save("full", lambda: full(params, flow.frozen, context, key))
        save("intercepted_action", lambda: intercepted_action(params, flow.frozen, context, key))
        save("explicit_action", lambda: explicit_action(params, flow.frozen, context, noise))
        save("explicit_full", lambda: explicit_full(params, flow.frozen, context, noise))
        save("adapter_full", lambda: sample_with_intervention(flow, params, context, key, metadata, knockout="none"))
        save("transparent_no_mask_action", lambda: no_mask_action(params, context, key, flow.frozen))
        save("attention_only_action", lambda: attention_action(params, context, key, flow.frozen))
        jax.effects_barrier()
        attention = attention_capture.stacked(10*template["layers"])
        np.savez(output / "action_only_attention.npz", attention=attention)
        cursor, path = noise, []
        for index in range(10):
            guard()
            path.append(np.asarray(cursor))
            cursor = one_step_jit(params, context, cursor, jnp.asarray(index, jnp.float32), flow.frozen)
            jax.block_until_ready(cursor)
        arrays["one_step_reconstruction_actions"] = np.asarray(cursor)
        arrays["one_step_reconstruction_path"] = np.stack(path)
        for count, function in prefixes.items():
            save(f"prefix_{count:02d}", lambda function=function: function(params, context, key, flow.frozen))
        arrays["explicit_noise"] = np.asarray(noise)
        arrays["prefix_reconstruction_path"] = np.stack([np.asarray(noise)]+[arrays[f"prefix_{i:02d}_actions"] for i in range(1, 10)])
        np.savez(output / "arrays.npz", **arrays)
        checks = {name: comparison(arrays["canonical_actions"], arrays[name+"_actions"])
                  for name in ["helper_order", "full", "intercepted_action", "explicit_action", "explicit_full", "adapter_full",
                               "transparent_no_mask_action", "attention_only_action", "one_step_reconstruction", "prefix_10"]}
        checks.update(baseline_actions_only_vs_full=comparison(arrays["baseline_canonical_actions"], arrays["baseline_full_actions"]),
            native_full_vs_adapter_full=comparison(arrays["full_actions"], arrays["adapter_full_actions"]),
            full_path_initial_noise_vs_explicit_noise=comparison(arrays["full_path"][0], arrays["explicit_noise"]))
        hlo_files = {}
        for name, function, arguments in [
            ("canonical", canonical, (params, context, key, flow.frozen)),
            ("full", full, (params, flow.frozen, context, key)),
            ("intercepted_action", intercepted_action, (params, flow.frozen, context, key)),
            ("transparent_no_mask_action", no_mask_action, (params, context, key, flow.frozen)),
            ("explicit_action", explicit_action, (params, flow.frozen, context, noise)),
            ("one_step", one_step_jit, (params, context, noise, jnp.asarray(0, jnp.float32), flow.frozen)),
            ("prefix_10", prefixes[10], (params, context, key, flow.frozen)),
        ]:
            guard()
            lowered = function.lower(*arguments)
            for suffix, content in [("optimized_hlo.txt", lowered.compile().as_text()),
                                    ("stablehlo.txt", str(lowered.compiler_ir(dialect="stablehlo")))]:
                destination = output / f"{name}.{suffix}"
                destination.write_text(content)
                hlo_files[destination.name] = sha(destination)
        if any(sha(path) != value for path, value in hashes.items()):
            raise ValueError("Source/input changed during sampler isolation")
        result = {"complete": True, "technical_only": True, "checks": checks,
                  "checkpoint": str(checkpoint), "elapsed_seconds": time.monotonic()-started,
                  "optimized_and_stable_hlo_files": hlo_files,
                  "attention_capture_shape": attention.shape,
                  "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/1024**3,
                  "tolerance": "Exact equality; no relaxed threshold"}
        (output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
