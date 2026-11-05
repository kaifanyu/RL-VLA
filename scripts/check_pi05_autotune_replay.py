"""Technical autotune export/replay using a private copy of the native JAX cache."""

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from rl_vla.diagnostics.pi05_attention import _Capture, _sample_from_noise, load_observation
from rl_vla.rldt_pi05 import Pi05Flow


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8*1024**2), b""):
            result.update(part)
    return result.hexdigest()


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {"exact": bool(np.array_equal(a, b)), "max_abs_difference": float(np.max(np.abs(a-b))),
            "different_elements": int(np.count_nonzero(a != b))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--load-tuning", type=Path)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    root = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        raise FileExistsError("Preserve prior technical attempt")
    cache = Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).resolve()
    if not cache.is_relative_to(output):
        raise ValueError("Only an output-local copied compilation cache is permitted")
    headroom = int(Path("/sys/fs/cgroup/memory.max").read_text()) - int(Path("/sys/fs/cgroup/memory.current").read_text())
    if headroom < 32 * 1024**3:
        raise RuntimeError("Insufficient host-memory headroom")
    lock_path = root / "runs/.vla_trace_gpu_locks/GPU-b64865f1-858a-43a6-62e4-e30649584262.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((run / "methods/rldt").glob("*/selection.json")):
            raise RuntimeError("Research selection is ready; stop technical GPU work")
        checkpoint = run / "methods/rldt/object_0/training/iteration_000025"
        observation = run / "observation_bank/object_0/validation/state_000100/observation.npz"
        sources = [Path(__file__), root / "src/rl_vla/rldt_pi05.py", root / "src/rl_vla/diagnostics/pi05_attention.py",
                   checkpoint / "expert.npz", checkpoint / "rldt_pi05.json", observation]
        if args.load_tuning:
            sources.append(args.load_tuning.resolve())
        hashes = {str(p): sha(p) for p in sources}
        (output / "plan.json").write_text(json.dumps({"technical_only": True, "source_input_sha256": hashes,
            "private_cache": str(cache), "global_precision_flags_changed": False,
            "load_tuning": str(args.load_tuning.resolve()) if args.load_tuning else None,
            "scope": "Only this diagnostic compilation may replay canonical algorithm choices"}, indent=2)+"\n")
        started = time.monotonic()
        flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
        raw = load_observation(observation)
        context = flow.prepare([raw])
        params = flow.load_params(checkpoint)
        _, key = jax.random.split(jax.random.PRNGKey(100000))
        canonical = jax.jit(lambda p, c, k, f: flow.sample(p, c, k, 10, frozen=f)[0])
        private_cache_native = np.asarray(canonical(params, context, key, flow.frozen))
        with np.load(run / "trained_sampler_compilation_v2/arrays.npz", allow_pickle=False) as archive:
            reference = archive["canonical_actions"].copy()
        report_path = run / "methods/rldt/object_0/validation_000025.json"
        report = json.loads(report_path.read_text())
        episode = report["episodes"][0]
        chunk = min(Path(episode["trajectory"]).glob("chunk_*.npz"))
        with np.load(chunk, allow_pickle=False) as archive:
            actual_actions = archive["normalized_actions"][None]
            chunk_metadata = json.loads(str(archive["metadata_json"].item()))
            input_equality = {}
            for field, nested_path in chunk_metadata["observation_schema"].items():
                if nested_path[0] == "openpi":
                    input_equality[nested_path[1]] = bool(np.array_equal(raw[nested_path[1]], archive[field][0]))
            input_equality["sampling_key"] = bool(np.array_equal(key, chunk_metadata["decision"]["sampling_key"]))
        if not input_equality or not all(input_equality.values()):
            raise ValueError(f"Technical observation differs from actual recorded decision: {input_equality}")
        if not np.array_equal(reference, actual_actions):
            raise ValueError("Saved canonical reference differs from actual recorded native evaluation")
        print(json.dumps({"saved_reference_equals_actual_recorded_native": True,
                          "private_cache_native_vs_recorded": compare(reference, private_cache_native)}), flush=True)
        (output / "canonical.optimized_hlo.txt").write_text(canonical.lower(params, context, key, flow.frozen).compile().as_text())
        options = {"xla_gpu_dump_autotune_results_to": str(output / "autotune.textproto")}
        if args.load_tuning:
            options["xla_gpu_load_autotune_results_from"] = str(args.load_tuning.resolve())
            # File replay is intentionally isolated from per-fusion UPDATE cache.
            options["xla_gpu_per_fusion_autotune_cache_dir"] = ""
        outputs = {"canonical": reference, "private_cache_native": private_cache_native}

        def execute(name, function, arguments):
            if any((run / "methods/rldt").glob("*/selection.json")):
                raise RuntimeError("Research selection appeared during technical tuning experiment")
            compiled = jax.jit(function, compiler_options=options)
            result = compiled(*arguments)
            jax.block_until_ready(result)
            action = result[0] if isinstance(result, tuple) else result
            outputs[name] = np.asarray(action)
            if isinstance(result, tuple):
                outputs[name+"_path"] = np.asarray(result[1])
            (output / f"{name}.optimized_hlo.txt").write_text(compiled.lower(*arguments).compile().as_text())
            np.savez(output / "arrays_in_progress.npz", **outputs)
            print(json.dumps({"name": name, "vs_canonical": compare(reference, action),
                              "elapsed_seconds": time.monotonic()-started}), flush=True)

        execute("recompiled_action", lambda p, c, k, f: flow.sample(p, c, k, 10, frozen=f)[0],
                (params, context, key, flow.frozen))
        if args.load_tuning:
            execute("full", lambda p, c, k, f: flow.sample(p, c, k, 10, frozen=f),
                    (params, context, key, flow.frozen))
            interceptor = _Capture(key_keep=None, capture_features=False)

            def captured(p, c, k, f):
                with interceptor.intercept():
                    return flow.sample(p, c, k, 10, frozen=f)

            execute("captured_full", captured, (params, context, key, flow.frozen))
            jax.effects_barrier()
            np.savez(output / "captured_attention.npz", attention=interceptor.stacked(180))
            noise = jax.random.normal(key, (1, flow.horizon, flow.action_dim))
            execute("explicit_full", lambda p, c, n, f: _sample_from_noise(flow, p, f, c, n, 10),
                    (params, context, noise, flow.frozen))
        checks = {name: compare(reference, value) for name, value in outputs.items()
                  if name not in {"canonical", "private_cache_native"} and not name.endswith("_path")}
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError("Source/input/tuning file changed during experiment")
        result = {"complete": True, "technical_only": True, "checks": checks,
                  "controls": {"private_cache_native_vs_recorded": compare(reference, private_cache_native)},
                  "all_exact": all(v["exact"] for v in checks.values()), "elapsed_seconds": time.monotonic()-started,
                  "private_cache": str(cache), "global_native_runtime_changed": False,
                  "recorded_reference_chunk": str(chunk), "recorded_reference_chunk_sha256": sha(chunk),
                  "recorded_decision_inputs_exact": input_equality,
                  "autotune_file": str(output / "autotune.textproto")}
        (output / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
