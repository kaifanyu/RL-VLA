"""Exact GPU6 descriptor-derived replay of predetermined recorded test inputs.

This is an unscored technical repair gate. It never runs an environment or changes
the original native runtime, recorded episodes, checkpoint, or compiler profile.
"""

import argparse
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_paths():
    return [Path(__file__).resolve(), *[ROOT / "src/rl_vla" / name for name in (
        "rldt_pi05.py", "diagnostics/pi05_attention.py", "diagnostics/layer_knockout.py",
        "diagnostics/compiler_profile_gpu6_v1.py")],
        ROOT / "scripts/check_pi05_trained_sampler_fidelity.py",
        ROOT / "scripts/check_pi05_native_baseline_oracle.py"]


def worker(args):
    from rl_vla.diagnostics.compiler_profile_gpu6_v1 import verify_effective_environment
    contract = json.loads(args.contract.read_text())
    audit = verify_effective_environment(args.profile, contract, check_versions=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from check_pi05_native_baseline_oracle import context_hashes
    from check_pi05_trained_sampler_fidelity import EXPECTED_CHECKS

    from rl_vla.diagnostics import pi05_attention as attention
    from rl_vla.diagnostics.layer_knockout import (
        sample_with_layer_knockout,
        velocity_with_layer_knockout,
    )
    from rl_vla.rldt_pi05 import Pi05Flow

    sources = {str(path): sha(path) for path in source_paths() + [args.profile, args.contract]}
    profile = json.loads(args.profile.read_text())
    tuning = args.profile.parent / profile["autotune_file"]["name"]
    sources[str(tuning)] = sha(tuning)
    flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
    base_files = [flow.checkpoint / "params/_METADATA", flow.checkpoint / "params/manifest.ocdbt"]
    base_files += [path for path in sorted((flow.checkpoint / "assets").rglob("*")) if path.is_file()]
    sources.update({str(path): sha(path) for path in base_files})
    canonical = jax.jit(lambda p, c, k, f: flow.sample(p, c, k, 10, frozen=f)[0])
    template = json.loads((args.run / "gate_b_actual/metadata.json").read_text())
    method = args.run / "methods/rldt/object_1"
    selection = json.loads((method / "selection.json").read_text())
    sources[str(method / "selection.json")] = sha(method / "selection.json")
    checkpoint_audit = json.loads((method / "checkpoint_audit.json").read_text())
    sources[str(method / "checkpoint_audit.json")] = sha(method / "checkpoint_audit.json")
    started = time.monotonic()

    def comparison(reference, actual):
        reference, actual = np.asarray(reference), np.asarray(actual)
        if reference.shape != actual.shape:
            raise ValueError("GPU6 replay shape differs from the recorded oracle")
        return {"exact": bool(np.array_equal(reference, actual)),
                "finite": bool(np.isfinite(reference).all() and np.isfinite(actual).all()),
                "max_abs_difference": float(np.max(np.abs(reference.astype(np.float64) - actual.astype(np.float64)))),
                "different_elements": int(np.count_nonzero(reference != actual))}

    for policy in args.policies.split(","):
        if policy not in POLICIES:
            raise ValueError("Unexpected GPU6 repair policy")
        baseline = policy == "pi05_libero"
        if profile["profile_id"] != ("native_baseline_gpu6_v1" if baseline else "native_trained_gpu6_v1"):
            raise ValueError("Each policy requires its own native role profile")
        iteration = None if baseline else int(policy.removeprefix("rldt_"))
        checkpoint = None if baseline else method / "training" / f"iteration_{iteration:06d}"
        params = flow.params if baseline else flow.load_params(checkpoint)
        checkpoint_inputs = {}
        if checkpoint:
            for name in ("expert.npz", "rldt_pi05.json"):
                path = checkpoint / name
                expected = checkpoint_audit["checkpoints"][str(iteration)]["files"][name]
                if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
                    raise ValueError("GPU6 proof checkpoint differs from passed main audit")
                checkpoint_inputs[str(path)] = sha(path)
        filename = "baseline_test.json" if baseline else (
            "selected_test.json" if selection["iteration"] == iteration else f"checkpoint_{iteration:06d}_test.json")
        report_path = method / filename
        report = json.loads(report_path.read_text())
        if len(report["episodes"]) != 50 or report["checkpoint"] != (str(checkpoint) if checkpoint else None):
            raise ValueError("Recorded main test report does not match policy identity")
        inputs = {**sources, **checkpoint_inputs, str(report_path): sha(report_path)}
        destination = args.output / policy
        destination.mkdir(parents=True, exist_ok=False)
        cases = []
        hlo_files = {}
        for episode_index in (0, 1):
            episode = report["episodes"][episode_index]
            state_id = episode["reset"]["initial_state_id"]
            if state_id != 150 + episode_index or episode["seed"] != 100000 + episode_index:
                raise ValueError("GPU6 proof cases differ from predetermined test identities")
            key_stream = jax.random.PRNGKey(episode["seed"])
            for decision in (0, 1):
                chunk = sorted(Path(episode["trajectory"]).glob("chunk_*.npz"))[decision]
                chunk_hash = sha(chunk)
                with np.load(chunk, allow_pickle=False) as archive:
                    chunk_metadata = json.loads(str(archive["metadata_json"].item()))
                    raw = {parts[1]: archive[name][0].copy()
                           for name, parts in chunk_metadata["observation_schema"].items() if parts[0] == "openpi"}
                    raw["prompt"] = str(raw["prompt"].item())
                    recorded = archive["normalized_actions"][None].copy()
                key_stream, key = jax.random.split(key_stream)
                if (chunk_metadata["decision"]["decision"] != decision
                        or not np.array_equal(key, chunk_metadata["decision"]["sampling_key"])):
                    raise ValueError("Native recorded sampling key differs from paired seed schedule")
                context = flow.prepare([raw])
                transformed = attention.transformed_observation(flow, raw)
                metadata = attention._token_metadata(transformed, template["text_span"][0], flow.horizon)
                metadata["layers"] = template["layers"]
                noise = jax.random.normal(key, (1, flow.horizon, flow.action_dim))
                case_dir = destination / f"state_{state_id:06d}" / f"decision_{decision:06d}"
                case_dir.mkdir(parents=True)
                arrays = {"recorded_actions": recorded, "initial_noise": np.asarray(noise), "sampling_key": np.asarray(key)}
                checks = {}

                def check(name, actual, expected, arrays=arrays, checks=checks):
                    jax.block_until_ready(actual)
                    arrays[name] = np.asarray(actual)
                    checks[name] = comparison(expected, actual)

                check("native_actions_only", canonical(params, context, key, flow.frozen), recorded)
                native, path = attention.native_sample(flow, params, context, 0, initial_noise=noise)
                check("explicit_native_actions", native, recorded)
                arrays["native_path"] = np.asarray(path)
                intervened, intervention_path = attention.sample_with_intervention(flow, params, context, key, metadata)
                check("intervention_noop_actions", intervened, recorded)
                check("intervention_noop_path", intervention_path, path)
                captured, capture_path, probabilities = attention.capture_sample(flow, params, context, 0, metadata, initial_noise=noise)
                check("captured_actions", captured, recorded)
                check("captured_path", capture_path, path)
                np.savez(case_dir / "sampler_attention.npz", attention=probabilities)
                layer_actions, layer_path = sample_with_layer_knockout(flow, params, context, key, metadata)
                check("layer_noop_actions", layer_actions, recorded)
                check("layer_noop_path", layer_path, path)
                for index in (0, 5, 9):
                    x = path[index]
                    t = jnp.full((1,), 1.0 - jnp.asarray(index, jnp.float32) / 10, dtype=jnp.float32)
                    velocity = attention.native_velocity(flow, params, context, x, t)
                    captured_velocity, probabilities, hidden = attention.capture_velocity(flow, params, context, x, t, metadata, return_hidden=True)
                    arrays[f"step{index:02d}_native_velocity"] = velocity
                    arrays[f"step{index:02d}_suffix_hidden"] = hidden
                    arrays[f"step{index:02d}_time"] = np.asarray(t)
                    check(f"step{index:02d}_capture_velocity", captured_velocity, velocity)
                    check(f"step{index:02d}_projection", attention.project_suffix(flow, params, hidden), captured_velocity)
                    check(f"step{index:02d}_layer_velocity", velocity_with_layer_knockout(flow, params, context, x, t, metadata), velocity)
                    check(f"step{index:02d}_euler_path", native if index == 9 else path[index + 1], np.asarray(x - jnp.asarray(velocity) / 10))
                    np.savez(case_dir / f"step{index:02d}_attention.npz", attention=probabilities)
                if set(checks) != EXPECTED_CHECKS:
                    raise ValueError("GPU6 proof must retain all twenty independent exact checks")
                np.savez(case_dir / "arrays.npz", **arrays)
                raw_hashes = {name: {"dtype": str(np.asarray(value).dtype), "shape": list(np.asarray(value).shape),
                                     "sha256": hashlib.sha256(np.asarray(value).tobytes()).hexdigest()} for name, value in raw.items()}
                case = {"complete": True, "passed": all(v["exact"] and v["finite"] for v in checks.values()),
                        "task": "object_1", "policy": policy, "iteration": iteration, "state_id": state_id,
                        "decision_index": decision, "seed": episode["seed"], "sampling_key": list(map(int, key)),
                        "recorded_chunk": str(chunk), "recorded_chunk_sha256": chunk_hash,
                        "recorded_report": str(report_path), "recorded_report_sha256": inputs[str(report_path)],
                        "recorded_oracle": "actual immutable main test normalized action chunk; exact acceptance, no tolerance",
                        "raw_input_arrays": raw_hashes, "context_leaf_sha256": context_hashes(context),
                        "checks": checks, "arrays_sha256": sha(case_dir / "arrays.npz"),
                        "arrays": {"path": str(case_dir / "arrays.npz"), "sha256": sha(case_dir / "arrays.npz")},
                        "recorded_action_replay": checks["native_actions_only"],
                        "sampler_attention_sha256": sha(case_dir / "sampler_attention.npz"),
                        "compiler_profile": contract, "source_input_sha256": {**inputs, str(chunk): chunk_hash}}
                write(case_dir / "summary.json", case)
                cases.append({"path": str(case_dir / "summary.json"), "sha256": sha(case_dir / "summary.json"), "passed": case["passed"]})
                print(json.dumps({"policy": policy, "state": state_id, "decision": decision, "passed": case["passed"],
                                  "failed": {k: v for k, v in checks.items() if not v["exact"] or not v["finite"]},
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
                if not case["passed"]:
                    raise ValueError("Derived GPU6 profile failed exact recorded-action/component fidelity")
                if episode_index == decision == 0:
                    graphs = {"native_actions": canonical.lower(params, context, key, flow.frozen).compile().as_text(),
                              "explicit_full": flow._diagnostic_native_sample_jits[(10, True)].lower(params, flow.frozen, context, noise).compile().as_text()}
                    for name, text in graphs.items():
                        hlo = destination / f"{name}.optimized_hlo.txt"
                        hlo.write_text(text)
                        hlo_files[str(hlo)] = sha(hlo)
                if sha(chunk) != chunk_hash:
                    raise ValueError("Original recorded oracle changed during proof")
        if any(sha(path) != digest for path, digest in inputs.items()):
            raise ValueError("GPU6 proof source/checkpoint/recording changed")
        verify_effective_environment(args.profile, contract, check_versions=True)
        write(destination / "summary.json", {"complete": True, "passed": True, "technical_only": True,
              "task": "object_1", "policy": policy, "iteration": iteration, "checkpoint": str(checkpoint) if checkpoint else None,
              "cases_expected": 4, "cases": cases, "checks_per_case": 20, "compiler_profile": contract,
              "compiler_profile_manifest_sha256": contract["manifest_sha256"],
              "autotune_sha256": contract["profile_sha256"],
              "compiler_audit": audit, "baseline_model_contract": flow.metadata, "source_input_sha256": inputs,
              "optimized_hlo_sha256": hlo_files, "scope": "Unscored actual recorded-input replay on GPU6; no environment or outcome evaluated"})


def parent(args):
    from rl_vla.diagnostics.compiler_profile_gpu6_v1 import (
        GPU_UUID,
        child_environment,
        load_profile,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    lock_path = ROOT / "runs/.vla_trace_gpu_locks" / f"{GPU_UUID}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = []
        for role, identifier, policies in (("baseline", "native_baseline_gpu6_v1", "pi05_libero"),
                                          ("trained", "native_trained_gpu6_v1", "rldt_25,rldt_50,rldt_100")):
            profile = load_profile(args.run / "compiler_profiles" / identifier / "manifest.json")
            env, contract = child_environment(profile, Path(profile["manifest"]["isolated_cache_root"]) / "fidelity_v1")
            env.update(XLA_PYTHON_CLIENT_MEM_FRACTION="0.80", XLA_PYTHON_CLIENT_PREALLOCATE="false")
            path = args.output / f"{role}_compiler_contract.json"
            write(path, contract)
            command = ["taskset", "-c", args.cpuset, str(ROOT / "third_party/openpi/.venv/bin/python"), str(Path(__file__).resolve()),
                       "--run", str(args.run), "--output", str(args.output), "--worker", "--policies", policies,
                       "--profile", profile["manifest_path"], "--contract", str(path)]
            with (args.output / f"{role}.log").open("x") as stream:
                process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                job = {"role": role, "pid": process.pid, "command": command, "compiler_profile": contract}
                jobs.append(job)
                write(args.output / "launch.json", {"physical_gpu_lock": str(lock_path), "source_sha256": sha(__file__), "jobs": jobs})
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                job["returncode"] = code
                write(args.output / "launch.json", {"physical_gpu_lock": str(lock_path), "source_sha256": sha(__file__), "jobs": jobs})
                if code:
                    raise subprocess.CalledProcessError(code, command)
        proofs = {policy: {"path": str(args.output / policy / "summary.json"), "sha256": sha(args.output / policy / "summary.json")}
                  for policy in POLICIES}
        if any(json.loads(Path(row["path"]).read_text()).get("passed") is not True for row in proofs.values()):
            raise ValueError("GPU6 numerical gate requires all four policy proofs")
        write(args.output / "summary.json", {"complete": True, "passed": True, "technical_only": True,
              "task": "object_1", "policies": list(POLICIES), "proofs": proofs, "total_cases": 16, "total_exact_checks": 320,
              "source_sha256": sha(__file__), "concurrency_gate": "separate proof required", "jobs": jobs})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpuset", default="56-63")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--policies")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--contract", type=Path)
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    worker(args) if args.worker else parent(args)
