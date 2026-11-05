#!/usr/bin/env python3
"""Replay recorded native sampling keys and trace actual rollout decision phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from run_vla_trace_interventions import checked_selection
from run_vla_trace_grid import checkpoint_audit_contract, verify_checkpoint_audit, native_timestep

from rl_vla.diagnostics.compiler_dispatch import load_profile, verify_effective_environment
from rl_vla.diagnostics.trained_fidelity import policy_fidelity_contract, verify_task_fidelity
from rl_vla.diagnostics.pi05_attention import (
    camera_views,
    capture_prefix,
    capture_sample,
    capture_velocity,
    load_observation,
    native_velocity,
)
from rl_vla.rldt_pi05 import Pi05Flow

POLICIES = ("pi05_libero", "rldt_000025", "rldt_000050", "rldt_000100")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_phase_records(summary, records):
    if not summary.get("complete") or not summary.get("sensor_timed_masks_exact_policy_rgb"):
        raise ValueError("Rollout bank lacks complete sensor-aligned replay evidence")
    if len(records) != summary["phase_decisions"] or not records:
        raise ValueError("Rollout phase manifest count differs from its complete summary")
    plan = summary["plan"]
    expected = sorted(set(plan["phases"].values()))
    if [record["decision_index"] for record in records] != expected:
        raise ValueError("Rollout records differ from the declared phase decisions")
    for record in records:
        for key in ("task", "split", "policy", "checkpoint", "initial_state_id", "episode"):
            if record[key] != plan[key]:
                raise ValueError(f"Rollout record identity differs from phase plan: {key}")
        names = {
            name for name, index in plan["phases"].items() if index == record["decision_index"]
        }
        if set(record["phase_names"]) != names:
            raise ValueError("Rollout record phase labels differ from phase plan")
    if plan["policy"] not in POLICIES:
        raise ValueError("Unknown native rollout policy alias")
    return plan


def max_error(reference, actual):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape:
        raise ValueError(f"Replay shape mismatch: {reference.shape} versus {actual.shape}")
    if not np.isfinite(actual).all():
        raise ValueError("Replayed values are not finite")
    return float(np.max(np.abs(reference.astype(np.float64) - actual.astype(np.float64))))


def trace_bank(flow, bank, output, *, steps, replay_only=False, selection=None,
               compiler_profile=None, compiler_contract=None, fidelity_contract=None, source_guard=None,
               checkpoint_contract=None):
    def compiler_guard():
        if compiler_profile is not None:
            verify_effective_environment(compiler_profile, compiler_contract, check_versions=True)
        if fidelity_contract is not None:
            verify_task_fidelity(fidelity_contract)
        if source_guard is not None:
            source_guard()
        if checkpoint_contract is not None:
            verify_checkpoint_audit(checkpoint_contract)

    compiler_guard()
    bank, output = Path(bank).resolve(), Path(output).resolve()
    source_summary = json.loads((bank / "summary.json").read_text())
    manifest = bank / "manifest.jsonl"
    manifest_hash = sha(manifest)
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    plan = validate_phase_records(source_summary, records)
    task, split, policy, checkpoint = (
        plan[key] for key in ("task", "split", "policy", "checkpoint")
    )
    if (policy == "pi05_libero") != (checkpoint is None):
        raise ValueError("Native baseline/checkpoint identity is inconsistent")
    if compiler_profile is not None:
        expected_profile = "native_trained_v2" if checkpoint else "native_baseline_v1"
        if compiler_profile["manifest"]["profile_id"] != expected_profile:
            raise ValueError("Rollout policy requires its own native compiler profile")
    selection_hash = None
    if split == "test":
        if compiler_profile is None:
            raise ValueError("Held-out rollout traces require the canonical compiler profile")
        if selection is None:
            raise ValueError("Held-out rollout traces require a locked selection")
        checked_selection(selection)
        if Path(selection).parent.name != task:
            raise ValueError("Selection belongs to a different task")
        if fidelity_contract is None:
            fidelity_contract = policy_fidelity_contract(Path(selection).resolve().parents[3], task, compiler_profile)
        if fidelity_contract is None or fidelity_contract["task"] != task:
            raise ValueError("Held-out rollout traces require completed task numerical proofs")
        if checkpoint_contract is None:
            checkpoint_contract = checkpoint_audit_contract(Path(selection).resolve().parent)
        compiler_guard()
        selection_hash = sha(selection)
        if checkpoint:
            iteration = int(policy.removeprefix("rldt_"))
            expected = Path(selection).parent / "training" / f"iteration_{iteration:06d}"
            if Path(checkpoint).resolve() != expected.resolve():
                raise ValueError(
                    "Recorded checkpoint is not the declared policy from this task run"
                )
    if checkpoint:
        if getattr(flow, "_rollout_trace_checkpoint", None) != checkpoint:
            flow._rollout_trace_params = flow.load_params(Path(checkpoint))
            flow._rollout_trace_checkpoint = checkpoint
        params = flow._rollout_trace_params
    else:
        params = flow.params
    checkpoint_hash = (
        sha(Path(checkpoint) / "expert.npz")
        if checkpoint
        else flow.metadata["checkpoint_manifest_sha256"]
    )
    if not hasattr(flow, "_rollout_trace_sample"):
        flow._rollout_trace_sample = jax.jit(
            lambda p, f, c, key: flow.sample(p, c, key, steps=10, frozen=f)
        )
    sample = flow._rollout_trace_sample
    output.mkdir(parents=True, exist_ok=False)
    adapter = Path(__file__).resolve().parents[1] / "src/rl_vla/diagnostics/pi05_attention.py"
    (output / "source_hashes.json").write_text(
        json.dumps({str(path): sha(path) for path in (Path(__file__), adapter)}, indent=2) + "\n"
    )
    started, rows = time.monotonic(), []
    for record in records:
        compiler_guard()
        frame_start = time.monotonic()
        if not record.get("exact_recorded_pixels_state_verified"):
            raise ValueError("Recorded policy pixels/state were not verified by simulator replay")
        source = Path(record["observation_path"])
        source_hash = sha(source)
        masks = source.parent / "masks.npz"
        masks_hash = sha(masks)
        if sha(record["source_chunk"]) != record["source_chunk_sha256"]:
            raise ValueError("Original recorded action chunk changed after phase-bank construction")
        raw = load_observation(source)
        with np.load(source, allow_pickle=False) as archive:
            key = np.asarray(archive["sampling_key"])
            recorded_actions = archive["recorded_normalized_actions"][None]
            recorded_decoded = archive["recorded_decoded_actions"][None]
        if key.shape != (2,) or key.dtype != np.uint32:
            raise ValueError("Expected the recorded native uint32[2] sampling key")
        destination = output / f"decision_{record['decision_index']:06d}"
        destination.mkdir()
        (destination / "observation.npz").symlink_to(source)
        (destination / "masks.npz").symlink_to(masks)
        context, prefix, metadata = capture_prefix(flow, raw, capture_attention=not replay_only)
        actions, path = sample(params, flow.frozen, context, jnp.asarray(key))
        normalized, path = np.asarray(actions), np.asarray(path)
        decoded = flow.decode(actions, [raw])
        checks = {
            "recorded_normalized_action_max_abs_difference": max_error(
                recorded_actions, normalized
            ),
            "recorded_decoded_action_max_abs_difference": max_error(recorded_decoded, decoded),
        }
        if any(checks.values()):
            raise RuntimeError(f"Native recorded actions failed exact replay: {checks}")
        if not replay_only:
            captured, capture_path, attention = capture_sample(
                flow, params, context, 0, metadata, initial_noise=path[0]
            )
            checks.update(
                no_op_sample_max_abs_difference=max_error(normalized, captured),
                no_op_path_max_abs_difference=max_error(path, capture_path),
            )
            if any(checks.values()):
                raise RuntimeError(
                    f"Native rollout no-op attention capture changed sampling: {checks}"
                )
            np.savez_compressed(destination / "prefix.npz", **prefix)
            np.savez_compressed(destination / "own_trajectory_attention.npz", attention=attention)
            for step in steps:
                t = native_timestep(step)
                original = native_velocity(flow, params, context, path[step], t)
                velocity, attention = capture_velocity(
                    flow, params, context, path[step], t, metadata
                )
                error = max_error(original, velocity)
                if error != 0:
                    raise RuntimeError(
                        f"Rollout expert capture changed velocity at step{step}: {error}"
                    )
                checks[f"no_op_velocity_step{step:02d}_max_abs_difference"] = error
                np.savez_compressed(
                    destination / f"expert_step{step:02d}_none.npz",
                    attention=attention,
                    velocity=velocity,
                    x_t=path[step],
                    t=np.asarray(t),
                    query_indices=np.arange(flow.horizon),
                    key_indices=np.arange(attention.shape[-1]),
                )
                np.savez_compressed(
                    destination / f"views_step{step:02d}_none.npz",
                    **camera_views(attention, metadata),
                )
        np.savez_compressed(
            destination / "actions.npz",
            normalized=normalized,
            decoded=decoded,
            own_path=path,
            initial_noise=path[0],
            sampling_key=key,
        )
        metadata.update(
            schema="rl-vla-native-rollout-attention-v1",
            source_record=record,
            instruction=raw["prompt"],
            base=flow.metadata,
            expert=checkpoint,
            checkpoint_sha256=checkpoint_hash,
            source_observation=str(source),
            source_observation_sha256=source_hash,
            source_masks_sha256=masks_hash,
            initial_noise_protocol={
                "source": "exact recorded native sampling_key",
                "sampling_key": key.tolist(),
                "saved_in": "actions.npz:initial_noise",
                "generator": "native JAX random.normal",
            },
            comparison_context="rollout-conditioned own-policy observation and denoising path; not matched cross-policy inputs",
            checks=checks,
            compiler_profile=compiler_contract,
            policy_fidelity=fidelity_contract,
            checkpoint_audit_contract=checkpoint_contract,
            elapsed_seconds=time.monotonic() - frame_start,
            action_units="decoded LIBERO OSC/gripper controller commands, not metres",
        )
        if sha(source) != source_hash or sha(masks) != masks_hash:
            raise ValueError("Phase observation/masks changed during trace capture")
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        row = {
            "decision_index": record["decision_index"],
            "phase_names": record["phase_names"],
            "checks": checks,
            "passed": True,
            "output": str(destination),
            "observation_sha256": source_hash,
            "mask_sha256": masks_hash,
        }
        rows.append(row)
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(
            f"{task}/{policy}: decision {record['decision_index']} exact native replay; {metadata['elapsed_seconds']:.2f}s",
            flush=True,
        )
    if sha(manifest) != manifest_hash or (selection_hash and sha(selection) != selection_hash):
        raise ValueError("Manifest or locked selection changed during rollout traces")
    if checkpoint and sha(Path(checkpoint) / "expert.npz") != checkpoint_hash:
        raise ValueError("Expert checkpoint changed during rollout traces")
    compiler_guard()
    result = {
        "complete": True,
        "passed": True,
        "bank": str(bank),
        "bank_manifest_sha256": manifest_hash,
        "task": task,
        "policy": policy,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "split": split,
        "phase_decisions": len(records),
        "phase_checks": rows,
        "replay_only": replay_only,
        "elapsed_seconds": time.monotonic() - started,
        "selection_sha256": selection_hash,
        "compiler_profile": compiler_contract,
        "policy_fidelity": fidelity_contract,
        "checkpoint_audit_contract": checkpoint_contract,
        "scope": "actual rollout-conditioned phases, exact recorded sampling keys and own-policy latents",
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bank", type=Path, help="One complete episode phase-bank directory")
    source.add_argument(
        "--run", type=Path, help="Wait for all task/policy phase banks; reuse one native backbone"
    )
    parser.add_argument("--tasks", default="object_0,object_1,object_2")
    parser.add_argument("--stage", choices=["baseline", "trained"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--compiler-profile", type=Path)
    parser.add_argument("--compiler-contract", type=Path)
    parser.add_argument("--steps", default="0,5,9")
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Exact native action software gate, omits attention exports",
    )
    args = parser.parse_args()
    if bool(args.compiler_profile) != bool(args.compiler_contract):
        parser.error("Provide both compiler profile and parent environment contract")
    if args.run and args.compiler_profile is None:
        parser.error("Research rollout traces require the verified canonical compiler profile")
    if args.run and args.stage is None:
        parser.error("Research rollout traces require a separate baseline or trained process stage")
    compiler_profile = load_profile(args.compiler_profile) if args.compiler_profile else None
    compiler_contract = json.loads(args.compiler_contract.read_text()) if args.compiler_contract else None
    if args.run:
        expected_profile = "native_baseline_v1" if args.stage == "baseline" else "native_trained_v2"
        if compiler_profile["manifest"]["profile_id"] != expected_profile:
            raise ValueError("Rollout stage and compiler profile are incompatible")
    source_paths = [Path(__file__).resolve(), Path(__file__).resolve().parent / "run_vla_trace_grid.py",
                    Path(__file__).resolve().parent / "run_vla_trace_interventions.py",
                    Path(__file__).resolve().parents[1] / "src/rl_vla/diagnostics/trained_fidelity.py",
                    Path(__file__).resolve().parents[1] / "src/rl_vla/diagnostics/compiler_dispatch.py"]
    if compiler_profile is not None:
        source_paths += [Path(compiler_profile[key]) for key in ("manifest_path", "profile_path", "helper_path")]
        source_paths.append(args.compiler_contract)
    source_hashes = {str(path): sha(path) for path in source_paths}

    def source_guard():
        if any(sha(path) != expected for path, expected in source_hashes.items()):
            raise ValueError("Rollout trace source/profile/contract changed after launch")

    def compiler_guard():
        source_guard()
        if compiler_profile is not None:
            verify_effective_environment(compiler_profile, compiler_contract, check_versions=True)

    compiler_guard()
    steps = [int(step) for step in args.steps.split(",")]
    if not steps or len(set(steps)) != len(steps) or any(step < 0 or step > 9 for step in steps):
        parser.error("Flow steps must be unique and lie between 0 and 9")
    flow = None
    if args.bank:
        bank_summary = json.loads((args.bank / "summary.json").read_text())
        fidelity = None
        checkpoint_contract = None
        if bank_summary.get("plan", {}).get("split") == "test":
            if args.selection is None or compiler_profile is None:
                parser.error("Held-out bank requires a locked selection and compiler profile")
            task = Path(args.selection).parent.name
            fidelity = policy_fidelity_contract(Path(args.selection).resolve().parents[3], task, compiler_profile)
            if fidelity is None:
                raise ValueError("Held-out bank numerical proof is still pending")
            checkpoint_contract = checkpoint_audit_contract(Path(args.selection).resolve().parent)
        flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
        trace_bank(
            flow,
            args.bank,
            args.output,
            steps=steps,
            replay_only=args.replay_only,
            selection=args.selection,
            compiler_profile=compiler_profile,
            compiler_contract=compiler_contract,
            fidelity_contract=fidelity,
            source_guard=source_guard,
            checkpoint_contract=checkpoint_contract,
        )
        if checkpoint_contract is not None:
            verify_checkpoint_audit(checkpoint_contract, full_hash=True)
        return
    control = args.output / "stages" / args.stage
    control.mkdir(parents=True, exist_ok=False)
    policies = ("pi05_libero",) if args.stage == "baseline" else POLICIES[1:]
    pending = [(task, policy) for task in args.tasks.split(",") for policy in policies]
    checkpoint_contracts = {}
    write_json(
        control / "plan.json",
        {
            "tasks": args.tasks.split(","),
            "policies": list(policies),
            "stage": args.stage,
            "state_ids": list(range(150, 155)),
            "selection": "first five fixed test episodes; no outcome filtering; phase rules in rollout_phase_predeclaration.json",
            "scope": "actual rollout-conditioned own-policy traces, not fixed-input cross-policy comparisons",
            "steps": steps,
            "replay_only": args.replay_only,
            "source_sha256": sha(__file__),
            "input_sha256": source_hashes,
            "compiler_profile": compiler_contract,
        },
    )
    while pending:
        compiler_guard()
        for task in {task for task, _ in pending}:
            status = args.run / "methods/rldt" / task / "status.json"
            if status.exists() and json.loads(status.read_text()).get("state") == "failed":
                raise RuntimeError(f"Upstream task failed; cannot await rollout bank: {status}")
        for task, policy in list(pending):
            fidelity = policy_fidelity_contract(args.run, task, compiler_profile)
            if fidelity is None:
                continue
            source = args.run / "rollout_observation_bank" / task / policy / "summary.json"
            if not source.exists():
                continue
            record = json.loads(source.read_text())
            if not record.get("complete"):
                continue
            selection = args.run / "methods/rldt" / task / "selection.json"
            checked_selection(selection)
            if sha(selection) != record["selection_sha256"]:
                raise ValueError("Locked selection differs from rollout-bank selection")
            banks = [Path(bank) for bank in record["banks"]]
            if record["task"] != task or record["policy"] != policy or record["episodes"] != 5:
                raise ValueError(
                    "Completed phase-bank identity differs from the queued task/policy"
                )
            if any(bank.resolve().parent != source.resolve().parent for bank in banks):
                raise ValueError("Phase bank belongs to another task/policy directory")
            if [bank.name for bank in banks] != [f"state_{i:06d}" for i in range(150, 155)]:
                raise ValueError(
                    "Rollout phase banks must retain all predetermined first5 test episodes"
                )
            if flow is None:
                if task not in checkpoint_contracts:
                    checkpoint_contracts[task] = checkpoint_audit_contract(selection.resolve().parent)
                flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
            if task not in checkpoint_contracts:
                checkpoint_contracts[task] = checkpoint_audit_contract(selection.resolve().parent)
            verify_checkpoint_audit(checkpoint_contracts[task])
            destination = args.output / task / policy
            results = [
                trace_bank(
                    flow,
                    bank,
                    destination / bank.name,
                    steps=steps,
                    replay_only=args.replay_only,
                    selection=selection,
                    compiler_profile=compiler_profile,
                    compiler_contract=compiler_contract,
                    fidelity_contract=fidelity,
                    source_guard=source_guard,
                    checkpoint_contract=checkpoint_contracts[task],
                )
                for bank in banks
            ]
            write_json(
                destination / "summary.json",
                {
                    "complete": True,
                    "passed": True,
                    "task": task,
                    "policy": policy,
                    "episodes": 5,
                    "phase_decisions": sum(row["phase_decisions"] for row in results),
                    "banks": [str(destination / bank.name) for bank in banks],
                    "selection_sha256": sha(selection),
                    "scope": "rollout-conditioned own-policy phases",
                    "compiler_profile": compiler_contract,
                    "policy_fidelity": fidelity,
                },
            )
            pending.remove((task, policy))
        write_json(control / "status.json", {"complete": not pending, "pending": pending})
        if pending:
            print(f"Waiting for complete rollout phase banks: {pending}", flush=True)
            time.sleep(30)
    compiler_guard()
    for contract in checkpoint_contracts.values():
        verify_checkpoint_audit(contract, full_hash=True)
    write_json(
        control / "summary.json",
        {
            "complete": True,
            "passed": True,
            "tasks": args.tasks.split(","),
            "policies": list(policies),
            "stage": args.stage,
            "compiler_profile": compiler_contract,
        },
    )


if __name__ == "__main__":
    main()
