"""Independent native action-only replay from original executables in a private cache copy."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def context_hashes(context):
    import jax
    import numpy as np

    result = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(context)[0]:
        value = np.asarray(leaf)
        result.append({"path": str(path), "dtype": str(value.dtype), "shape": list(value.shape),
                       "sha256": hashlib.sha256(value.tobytes()).hexdigest()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--amendment", required=True, type=Path)
    args = parser.parse_args()
    args.run, args.output, args.amendment = args.run.resolve(), args.output.resolve(), args.amendment.resolve()
    cache = Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).resolve()
    allowed = args.run / "compiler_caches/native_trained_v2"
    if not cache.is_relative_to(allowed) or os.environ.get("XLA_FLAGS", ""):
        raise ValueError("Native oracle requires private cache and unchanged original XLA flags")
    if os.environ.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES") != "xla_gpu_per_fusion_autotune_cache_dir":
        raise ValueError("Native oracle requires original native cache configuration")
    reference_path = args.run / "native_cached_context_v2/cache_events.json"
    reference_events = json.loads(reference_path.read_text())
    required_keys = {row["read_key"] for row in reference_events if row.get("hit") is True
                     and (row["read_key"].startswith("jit__encode_prefix-") or row["read_key"].startswith("jit__lambda_-"))}
    if len(required_keys) != 2:
        raise ValueError("Expected one verified original native prefix and one action-only executable")
    executable_records = []
    for key in sorted(required_keys):
        original = ROOT / ".cache/jax-rl-vla" / (key+"-cache")
        copied = cache / (key+"-cache")
        digest = sha(original)
        if sha(copied) != digest:
            raise ValueError("Private native executable differs from original cache")
        executable_records.append({"cache_key": key, "path": str(original), "sha256": digest,
                                   "private_path": str(copied)})
    lock_path = ROOT / "runs/.vla_trace_gpu_locks" / f"{args.gpu_uuid}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        import jax
        import numpy as np
        from jax._src import compilation_cache

        from rl_vla.rldt_pi05 import Pi05Flow

        original_key, original_read = compilation_cache.get_cache_key, compilation_cache.get_executable_and_time
        observed_hits = set()
        events = []

        def routed_key(module, devices, options, backend, *a, **kw):
            debug = options.executable_build_options.debug_options
            actual = debug.xla_gpu_per_fusion_autotune_cache_dir
            if actual != str(cache / "xla_gpu_per_fusion_autotune_cache_dir"):
                raise ValueError("Unexpected physical cache path")
            try:
                debug.xla_gpu_per_fusion_autotune_cache_dir = str(ROOT / ".cache/jax-rl-vla/xla_gpu_per_fusion_autotune_cache_dir")
                key = original_key(module, devices, options, backend, *a, **kw)
            finally:
                debug.xla_gpu_per_fusion_autotune_cache_dir = actual
            events.append({"operation": str(module.operation.attributes["sym_name"]), "key": key})
            return key

        def logged_read(*a, **kw):
            key = a[0] if a else kw["cache_key"]
            result = original_read(*a, **kw)
            events.append({"read_key": key, "hit": result[0] is not None})
            if result[0] is not None:
                observed_hits.add(key)
            return result

        sources = {str(path): sha(path) for path in
                   (Path(__file__), ROOT / "src/rl_vla/rldt_pi05.py", args.amendment, reference_path)}
        with patch.object(compilation_cache, "get_cache_key", routed_key), \
                patch.object(compilation_cache, "get_executable_and_time", logged_read):
            flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
            native = jax.jit(lambda p, c, k, f: flow.sample(p, c, k, 10, frozen=f)[0])
            for job in args.jobs.split(","):
                task, iteration_text = job.split(":")
                iteration = int(iteration_text)
                if task not in ("object_0", "object_1", "object_2") or iteration not in (25, 50, 100):
                    raise ValueError("Unexpected native oracle checkpoint")
                destination = args.output / "native_oracle" / task / f"iteration_{iteration:06d}"
                if (destination / "summary.json").exists():
                    continue
                method = args.run / "methods/rldt" / task
                checkpoint = method / "training" / f"iteration_{iteration:06d}"
                report_path = method / f"validation_{iteration:06d}.json"
                inputs = {str(path): sha(path) for path in (checkpoint / "expert.npz", checkpoint / "rldt_pi05.json", report_path)}
                report = json.loads(report_path.read_text())
                params = flow.load_params(checkpoint)
                parameter_specs = [{"path": str(path), "shape": list(variable.value.shape),
                                    "dtype": str(variable.value.dtype), "sharding": str(variable.value.sharding)}
                                   for path, variable in params.flat_state().items()]
                if any(item["dtype"] != "float32" for item in parameter_specs):
                    raise ValueError("Unexpected trained expert parameter dtype")
                cases = []
                for episode_index in (0, 1):
                    episode = report["episodes"][episode_index]
                    state = episode["reset"]["initial_state_id"]
                    if state != 100+episode_index or episode["reset"]["experiment_split"] != "validation":
                        raise ValueError("Native oracle cases differ from predeclaration")
                    chunks = sorted(Path(episode["trajectory"]).glob("chunk_*.npz"))
                    key_stream = jax.random.PRNGKey(episode["seed"])
                    for decision in (0, 1):
                        chunk = chunks[decision]
                        chunk_hash = sha(chunk)
                        with np.load(chunk, allow_pickle=False) as archive:
                            metadata = json.loads(str(archive["metadata_json"].item()))
                            raw = {parts[1]: archive[field][0].copy() for field, parts in metadata["observation_schema"].items()
                                   if parts[0] == "openpi"}
                            raw["prompt"] = str(raw["prompt"].item())
                            historical = archive["normalized_actions"][None].copy()
                        key_stream, key = jax.random.split(key_stream)
                        if metadata["decision"]["decision"] != decision or not np.array_equal(key, metadata["decision"]["sampling_key"]):
                            raise ValueError("Recorded native key/decision mismatch")
                        context = flow.prepare([raw])
                        actions = np.asarray(native(params, context, key, flow.frozen))
                        if not required_keys.issubset(observed_hits):
                            raise ValueError("Current native oracle did not hit both verified original cached executables")
                        relevant_reads = [row for row in events if "read_key" in row and
                            (row["read_key"].startswith("jit__encode_prefix-") or row["read_key"].startswith("jit__lambda_-"))]
                        if any(row["read_key"] not in required_keys or row["hit"] is not True for row in relevant_reads):
                            raise ValueError("Native sampler/prefix recompiled or used an unverified executable")
                        if not np.isfinite(actions).all():
                            raise ValueError("Native oracle actions are nonfinite")
                        case_dir = destination / f"state_{state:06d}" / f"decision_{decision:06d}"
                        case_dir.mkdir(parents=True, exist_ok=False)
                        np.savez(case_dir / "arrays.npz", actions=actions, recorded_actions=historical,
                                 sampling_key=np.asarray(key), initial_noise=np.asarray(jax.random.normal(key, actions.shape)))
                        historical_comparison = {"exact": bool(np.array_equal(actions, historical)),
                            "finite": bool(np.isfinite(historical).all()),
                            "max_abs_difference": float(np.max(np.abs(actions.astype(np.float64)-historical.astype(np.float64)))),
                            "different_elements": int(np.count_nonzero(actions != historical))}
                        case = {"complete": True, "passed": True, "cache_hits_verified": True,
                            "task": task, "iteration": iteration, "state_id": state, "decision_index": decision,
                            "arrays_sha256": sha(case_dir / "arrays.npz"), "cached_executables": executable_records,
                            "native_oracle_source_sha256": sources[str(Path(__file__))],
                            "context_leaf_sha256": context_hashes(context), "parameter_specs": parameter_specs,
                            "input_specs": {name: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)} for name, value in raw.items()},
                            "recorded_chunk": str(chunk), "recorded_chunk_sha256": chunk_hash,
                            "historical_recorded_replay": historical_comparison,
                            "source_input_sha256": {**sources, **inputs, str(chunk): chunk_hash,
                                **{item["path"]: item["sha256"] for item in executable_records}},
                            "amendment": {"path": str(args.amendment), "sha256": sha(args.amendment)}}
                        if sha(chunk) != chunk_hash:
                            raise ValueError("Recorded input changed")
                        write(case_dir / "summary.json", case)
                        cases.append({"path": str(case_dir / "summary.json"), "sha256": sha(case_dir / "summary.json")})
                        print(json.dumps({"job": job, "state": state, "decision": decision,
                            "current_native_cached_oracle": True, "historical_recorded_replay": historical_comparison}), flush=True)
                if any(sha(path) != digest for path, digest in {**sources, **inputs}.items()):
                    raise ValueError("Native oracle source/input changed")
                for record in executable_records:
                    if sha(record["path"]) != record["sha256"] or sha(record["private_path"]) != record["sha256"]:
                        raise ValueError("Native cached executable changed")
                write(destination / "summary.json", {"complete": True, "passed": True, "cache_hits_verified": True,
                    "task": task, "iteration": iteration, "cases": cases, "cached_executables": executable_records,
                    "source_input_sha256": {**sources, **inputs}, "native_oracle_source_sha256": sources[str(Path(__file__))],
                    "amendment": {"path": str(args.amendment), "sha256": sha(args.amendment)}})
                write(args.output / "native_oracle/cache_events.json", events)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
