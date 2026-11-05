"""Technical trained-expert gates for actual diagnostic APIs with isolated tuning replay."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from check_pi05_autotune_replay import compare, sha

from rl_vla.diagnostics import pi05_attention as attention
from rl_vla.diagnostics.layer_knockout import (
    sample_with_layer_knockout,
    velocity_with_layer_knockout,
)
from rl_vla.rldt_pi05 import Pi05Flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--load-tuning", required=True, type=Path)
    parser.add_argument("--task", default="object_0")
    parser.add_argument("--iteration", type=int, default=25)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        raise FileExistsError("Preserve prior technical attempt")
    cache = Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).resolve()
    if not cache.is_relative_to(output):
        raise ValueError("An output-local private cache is mandatory")
    if any((run / "methods/rldt").glob("*/selection.json")):
        raise RuntimeError("Research selection ready; stop technical GPU work")
    headroom = int(Path("/sys/fs/cgroup/memory.max").read_text()) - int(Path("/sys/fs/cgroup/memory.current").read_text())
    if headroom < 32 * 1024**3:
        raise RuntimeError("Insufficient host-memory headroom")
    lock_path = root / "runs/.vla_trace_gpu_locks/GPU-b64865f1-858a-43a6-62e4-e30649584262.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        checkpoint = run / "methods/rldt" / args.task / f"training/iteration_{args.iteration:06d}"
        report_path = run / "methods/rldt" / args.task / f"validation_{args.iteration:06d}.json"
        report = json.loads(report_path.read_text())
        chunk = min(Path(report["episodes"][0]["trajectory"]).glob("chunk_*.npz"))
        with np.load(chunk, allow_pickle=False) as archive:
            chunk_meta = json.loads(str(archive["metadata_json"].item()))
            raw = {parts[1]: archive[field][0].copy() for field, parts in chunk_meta["observation_schema"].items()
                   if parts[0] == "openpi"}
            raw["prompt"] = str(raw["prompt"].item())
            recorded = archive["normalized_actions"][None].copy()
        key = jnp.asarray(chunk_meta["decision"]["sampling_key"], dtype=jnp.uint32)
        sources = [Path(__file__), root / "scripts/check_pi05_autotune_replay.py",
                   root / "src/rl_vla/diagnostics/pi05_attention.py", root / "src/rl_vla/diagnostics/layer_knockout.py",
                   root / "src/rl_vla/rldt_pi05.py", checkpoint / "expert.npz", checkpoint / "rldt_pi05.json",
                   report_path, chunk, args.load_tuning.resolve()]
        hashes = {str(path): sha(path) for path in sources}
        (output / "plan.json").write_text(json.dumps({"technical_only": True, "source_input_sha256": hashes,
            "scope": "Actual diagnostic APIs; only JIT compiler_options are scoped to this private process. Native runtime and shared caches are untouched.",
            "recorded_decision": chunk_meta["decision"], "checkpoint": str(checkpoint), "private_cache": str(cache)}, indent=2)+"\n")
        flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
        context = flow.prepare([raw])
        context_arrays = {}
        context_leaves = []
        for index, (path, leaf) in enumerate(jax.tree_util.tree_flatten_with_path(context)[0]):
            value = np.asarray(leaf)
            portable = value.astype(np.float32) if str(value.dtype) == "bfloat16" else value
            name = f"leaf_{index:03d}"
            context_arrays[name] = portable
            context_leaves.append({"name": name, "path": str(path), "shape": list(value.shape), "native_dtype": str(value.dtype)})
        np.savez(output / "context.npz", **context_arrays)
        (output / "context.json").write_text(json.dumps({"leaves": context_leaves}, indent=2)+"\n")
        params = flow.load_params(checkpoint)
        template = json.loads((run / "gate_b_actual/metadata.json").read_text())
        metadata = attention._token_metadata(attention.transformed_observation(flow, raw), template["text_span"][0], flow.horizon)
        metadata["layers"] = template["layers"]
        noise = jax.random.normal(key, (1, flow.horizon, flow.action_dim))
        options = {"xla_gpu_load_autotune_results_from": str(args.load_tuning.resolve()),
                   "xla_gpu_dump_autotune_results_to": str(output / "autotune.textproto"),
                   "xla_gpu_per_fusion_autotune_cache_dir": ""}
        original_jit = jax.jit

        def pinned_jit(function, *a, **kw):
            kw["compiler_options"] = {**options, **kw.get("compiler_options", {})}
            return original_jit(function, *a, **kw)

        arrays, checks = {"recorded_actions": recorded, "initial_noise": np.asarray(noise)}, {}

        def save(name, actual, expected):
            jax.block_until_ready(actual)
            arrays[name] = np.asarray(actual)
            checks[name] = {**compare(expected, actual), "finite": bool(np.isfinite(actual).all())}
            np.savez(output / "arrays_in_progress.npz", **arrays)
            print(json.dumps({"name": name, **checks[name], "elapsed_seconds": time.monotonic()-started}), flush=True)

        # The scoped patch only supplies supported compiler options to the unchanged
        # diagnostic JIT factories. All numerical functions are the actual source APIs.
        with patch.object(jax, "jit", pinned_jit):
            native_actions, path = attention.native_sample(flow, params, context, 0, initial_noise=noise)
            save("explicit_native_actions", native_actions, recorded)
            arrays["native_path"] = np.asarray(path)
            intervened, _ = attention.sample_with_intervention(flow, params, context, key, metadata)
            save("intervention_noop_actions", intervened, recorded)
            layer_actions, _ = sample_with_layer_knockout(flow, params, context, key, metadata)
            save("layer_noop_actions", layer_actions, recorded)
            for index in (0, 5, 9):
                x = path[index]
                t = jnp.full((1,), 1.0-jnp.asarray(index, jnp.float32)/10, dtype=jnp.float32)
                native = attention.native_velocity(flow, params, context, x, t)
                captured, probabilities, hidden = attention.capture_velocity(flow, params, context, x, t, metadata, return_hidden=True)
                projected = attention.project_suffix(flow, params, hidden)
                layer = velocity_with_layer_knockout(flow, params, context, x, t, metadata)
                arrays[f"step{index:02d}_native_velocity"] = native
                arrays[f"step{index:02d}_suffix_hidden"] = hidden
                save(f"step{index:02d}_capture_velocity", captured, native)
                save(f"step{index:02d}_projection", projected, captured)
                save(f"step{index:02d}_layer_velocity", layer, native)
                np.savez(output / f"step{index:02d}_attention.npz", attention=probabilities)
                next_x = np.asarray(native_actions if index == 9 else path[index+1])
                expected_next = np.asarray(x - jnp.asarray(native)/10)
                save(f"step{index:02d}_euler_path", next_x, expected_next)
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError("Source/input/tuning changed during technical proof")
        summary = {"complete": True, "technical_only": True, "checks": checks,
                   "passed": all(check["exact"] and check["finite"] for check in checks.values()),
                   "elapsed_seconds": time.monotonic()-started, "tolerance": "Bit exact; no relaxed threshold",
                   "recorded_oracle": str(chunk), "source_input_sha256": hashes}
        (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
