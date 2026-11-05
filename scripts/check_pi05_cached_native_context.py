"""Read original native executable keys from a private cache snapshot; no shared writes."""

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import compilation_cache

from rl_vla.rldt_pi05 import Pi05Flow


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            value.update(block)
    return value.hexdigest()


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {"exact": bool(np.array_equal(a, b)), "max_abs_difference": float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
            "different_elements": int(np.count_nonzero(a != b))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", action="store_true", help="Compare profile environment without logical-key routing")
    parser.add_argument("--tasks", nargs="+", default=["object_0", "object_1", "object_2"])
    parser.add_argument("--iterations", nargs="+", type=int, default=[25])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.run, args.output = args.run.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "plan.json").exists():
        raise FileExistsError("Preserve existing technical attempt")
    private = Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).resolve()
    if not private.is_relative_to(args.output):
        raise ValueError("All executable/autotune cache IO must use this private snapshot")
    if not args.profile and os.environ.get("XLA_FLAGS", ""):
        raise ValueError("Original native key lookup cannot use alternate XLA flags")
    logical = root / ".cache/jax-rl-vla/xla_gpu_per_fusion_autotune_cache_dir"
    expected_private = private / "xla_gpu_per_fusion_autotune_cache_dir"
    original_key = compilation_cache.get_cache_key
    original_read = compilation_cache.get_executable_and_time
    cache_events = []

    def routed_key(module, devices, options, backend, *a, **kw):
        debug = options.executable_build_options.debug_options
        actual = debug.xla_gpu_per_fusion_autotune_cache_dir
        if actual != str(expected_private):
            raise ValueError(f"Unexpected private autotune cache: {actual}")
        try:
            # Only the key hash sees the old logical path. Restore before any
            # executable lookup or compilation, which uses private storage.
            debug.xla_gpu_per_fusion_autotune_cache_dir = str(logical)
            result = original_key(module, devices, options, backend, *a, **kw)
        finally:
            debug.xla_gpu_per_fusion_autotune_cache_dir = actual
        cache_events.append({"operation": str(module.operation.attributes["sym_name"]), "key": result,
                             "logical_key_path": str(logical), "physical_cache": str(private)})
        return result

    def logged_read(*a, **kw):
        result = original_read(*a, **kw)
        cache_events.append({"read_key": a[0] if a else kw["cache_key"], "hit": result[0] is not None})
        return result

    lock_path = root / "runs/.vla_trace_gpu_locks/GPU-b64865f1-858a-43a6-62e4-e30649584262.lock"
    with lock_path.open("a+") as lock, patch.object(compilation_cache, "get_executable_and_time", logged_read), \
            patch.object(compilation_cache, "get_cache_key", original_key if args.profile else routed_key):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {str(path): sha(path) for path in (Path(__file__), root / "src/rl_vla/rldt_pi05.py")}
        (args.output / "plan.json").write_text(json.dumps({"technical_only": True, "private_cache": str(private),
            "logical_native_key_routing": not args.profile, "source_sha256": sources,
            "environment": {key: os.environ.get(key) for key in ("XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES")}}, indent=2)+"\n")
        started = time.monotonic()
        flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
        sampler = jax.jit(lambda p, c, k, f: flow.sample(p, c, k, 10, frozen=f)[0])
        results = {}
        for task, iteration in itertools.product(args.tasks, args.iterations):
            destination = args.output / task / f"iteration_{iteration:06d}"
            destination.mkdir(parents=True)
            method = args.run / "methods/rldt" / task
            checkpoint = method / "training" / f"iteration_{iteration:06d}"
            report = json.loads((method / f"validation_{iteration:06d}.json").read_text())
            chunk = min(Path(report["episodes"][0]["trajectory"]).glob("chunk_*.npz"))
            with np.load(chunk, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                raw = {parts[1]: archive[field][0].copy() for field, parts in metadata["observation_schema"].items() if parts[0] == "openpi"}
                raw["prompt"] = str(raw["prompt"].item())
                recorded = archive["normalized_actions"][None].copy()
            key = jnp.asarray(metadata["decision"]["sampling_key"], dtype=jnp.uint32)
            params = flow.load_params(checkpoint)
            context = flow.prepare([raw])
            context_arrays, context_metadata = {}, []
            for index, (path, leaf) in enumerate(jax.tree_util.tree_flatten_with_path(context)[0]):
                value = np.asarray(leaf)
                name = f"leaf_{index:03d}"
                context_arrays[name] = value.astype(np.float32) if str(value.dtype) == "bfloat16" else value
                context_metadata.append({"name": name, "path": str(path), "shape": list(value.shape), "native_dtype": str(value.dtype)})
            np.savez(destination / "context.npz", **context_arrays)
            (destination / "context.json").write_text(json.dumps(context_metadata, indent=2)+"\n")
            actions = sampler(params, context, key, flow.frozen)
            jax.block_until_ready(actions)
            np.savez(destination / "arrays.npz", actions=np.asarray(actions), recorded_actions=recorded)
            (destination / "native_actions.optimized_hlo.txt").write_text(sampler.lower(params, context, key, flow.frozen).compile().as_text())
            (args.output / "cache_events.json").write_text(json.dumps(cache_events, indent=2)+"\n")
            # Obtain the exact already-prepared transformed prefix arguments.
            from rl_vla.diagnostics.pi05_attention import transformed_observation
            observation = transformed_observation(flow, raw)
            import jaxtyping
            typecheck_disabled = jaxtyping.config.jaxtyping_disable
            try:
                # Observation's dataclass type checker rejects JAX's ArgInfo
                # inspection placeholders. Disable only while exporting an
                # already executed JIT, without changing inference arithmetic.
                jaxtyping.config.update("jaxtyping_disable", True)
                prefix_hlo = flow._encode_jit.lower(flow.params, flow.frozen, observation).compile().as_text()
            finally:
                jaxtyping.config.update("jaxtyping_disable", typecheck_disabled)
            (destination / "native_prefix.optimized_hlo.txt").write_text(prefix_hlo)
            name = f"{task}:{iteration}"
            results[name] = {"recorded_oracle_comparison": compare(actions, recorded), "recorded_chunk": str(chunk),
                             "recorded_chunk_sha256": sha(chunk), "checkpoint_sha256": sha(checkpoint / "expert.npz")}
            print(json.dumps({"task": task, "iteration": iteration, **results[name], "elapsed_seconds": time.monotonic()-started}), flush=True)
            (args.output / "cache_events.json").write_text(json.dumps(cache_events, indent=2)+"\n")
        if any(sha(path) != digest for path, digest in sources.items()):
            raise ValueError("Source changed")
        (args.output / "summary.json").write_text(json.dumps({"complete": True, "technical_only": True,
            "results": results, "elapsed_seconds": time.monotonic()-started}, indent=2)+"\n")


if __name__ == "__main__":
    main()
