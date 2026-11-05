#!/usr/bin/env python3
"""Wait for locked selections, then trace predetermined matched cases at 0/25/50/100."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.compiler_dispatch import load_profile, verify_effective_environment
from rl_vla.diagnostics.trained_fidelity import policy_fidelity_contract, verify_task_fidelity
from rl_vla.diagnostics.grid_stages import (
    aggregate_task, read_stage, publish_stage, verify_stage, save_context, load_context,
)


def _load_native():
    if "jax" in globals():
        return
    import jax
    import jax.numpy as jnp
    from run_vla_trace_interventions import masked_observation
    from rl_vla.diagnostics.interventions import action_delta
    from rl_vla.diagnostics.layer_knockout import (
        capture_layer_velocity, sample_with_layer_knockout, velocity_with_layer_knockout,
    )
    from rl_vla.diagnostics.pi05_attention import (
        Pi05VLMReference, array_sha256, camera_views, capture_prefix, capture_sample,
        capture_velocity, key_keep_mask, load_observation, native_sample, native_velocity,
        project_suffix, sample_with_intervention, transformed_observation,
    )
    from rl_vla.rldt_pi05 import Pi05Flow
    globals().update(locals())


def __getattr__(name):
    if name in {"Pi05Flow", "Pi05VLMReference"}:
        _load_native()
        return globals()[name]
    raise AttributeError(name)


def checked_selection(path):
    from run_vla_trace_interventions import checked_selection as check
    return check(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_summary(path, value):
    """Publish completion only after the whole JSON artifact is durable to readers."""
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def requested_steps(value):
    steps = [int(step) for step in value.split(",")]
    if not steps or len(set(steps)) != len(steps) or any(not 0 <= step < 10 for step in steps):
        raise ValueError("ODE steps must be unique indices between 0 and 9")
    return steps


def native_timestep(step):
    """Use the native float32 index expression, not Python double arithmetic."""
    import jax.numpy as jnp

    return jnp.full((1,), 1.0 - jnp.asarray(step, dtype=jnp.float32) / 10, dtype=jnp.float32)


def checkpoint_audit_contract(method):
    """Verify all requested snapshots before model loading or held-out capture."""
    checked_selection(method / "selection.json")
    audit_path = method / "checkpoint_audit.json"
    audit_hash = sha(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit.get("passed") is not True or set(audit.get("checkpoints", {})) != {"25", "50", "100"}:
        raise ValueError("Grid requires the passed full 25/50/100 checkpoint audit")
    files = {}
    for iteration in (25, 50, 100):
        for name in ("expert.npz", "rldt_pi05.json"):
            path = method / "training" / f"iteration_{iteration:06d}" / name
            expected = audit["checkpoints"][str(iteration)]["files"][name]
            before = path.stat()
            measured = sha(path)
            after = path.stat()
            identity = (after.st_size, after.st_mtime_ns, after.st_ino)
            if (
                identity != (before.st_size, before.st_mtime_ns, before.st_ino)
                or after.st_size != expected["bytes"]
                or measured != expected["sha256"]
            ):
                raise ValueError(f"Grid snapshot differs from passed checkpoint audit: {path}")
            files[str(path)] = {"identity": identity, "sha256": measured}
    contract = {"audit_path": str(audit_path), "audit_sha256": audit_hash, "files": files}
    verify_checkpoint_audit(contract)
    return contract


def verify_checkpoint_audit(contract, *, full_hash=False):
    if sha(contract["audit_path"]) != contract["audit_sha256"]:
        raise ValueError("Checkpoint audit changed during grid diagnostics")
    for filename, expected in contract["files"].items():
        path = Path(filename)
        value = path.stat()
        if (value.st_size, value.st_mtime_ns, value.st_ino) != tuple(expected["identity"]):
            raise ValueError(f"Audited grid snapshot changed: {path}")
        if full_hash and sha(path) != expected["sha256"]:
            raise ValueError(f"Audited grid snapshot hash changed: {path}")


def parameter_drift(reference, params):
    """Aggregate exact expert/projection differences, preserving scanned layer IDs."""
    groups = {}
    for path, var in params.flat_state().items():
        before = np.asarray(reference.flat_state()[path].value)
        after = np.asarray(var.value)
        name = "/".join(map(str, path))
        scanned = "layers" in path
        slices = (
            [
                (f"expert_layer_{layer:02d}", before[layer], after[layer])
                for layer in range(before.shape[0])
            ]
            if scanned
            else [(str(path[0]), before, after)]
        )
        for group, a, b in slices:
            delta = b.astype(np.float64) - a.astype(np.float64)
            row = groups.setdefault(
                group,
                {"squared_l2": 0.0, "reference_squared_l2": 0.0, "parameters": 0, "tensors": []},
            )
            row["squared_l2"] += float(np.square(delta).sum())
            row["reference_squared_l2"] += float(np.square(a.astype(np.float64)).sum())
            row["parameters"] += int(a.size)
            row["tensors"].append(name)
    for row in groups.values():
        row["l2"] = float(np.sqrt(row.pop("squared_l2")))
        row["reference_l2"] = float(np.sqrt(row.pop("reference_squared_l2")))
        row["relative_l2"] = row["l2"] / max(row["reference_l2"], 1e-12)
    return groups


def frozen_hash(flow):
    digest = hashlib.sha256()
    for path, value in sorted(flow.frozen.flat_state().items()):
        array = np.asarray(value.value)
        digest.update(str(path).encode())
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def open_loop_interventions(
    flow,
    params,
    raw,
    context,
    metadata,
    noise,
    reference_actions,
    source,
    target,
    distractor,
    output,
    *,
    save_inputs=False,
):
    """Matched observation/noise causal effects; no rollout success claim."""
    _load_native()
    with np.load(source.parent / "masks.npz", allow_pickle=False) as archive:
        masks = {name: archive[name] for name in archive.files}
    bank_record = metadata["bank_record"]
    run_root = Path(bank_record["protocol"]).parents[2]
    auxiliary = (
        run_root
        / "observation_bank_aux"
        / bank_record["task"]
        / bank_record["split"]
        / source.parent.name
    )

    auxiliary_metadata = json.loads((auxiliary / "metadata.json").read_text())
    auxiliary_file = auxiliary / "old_target_region.npz"
    if (
        sha(auxiliary_file) != auxiliary_metadata["artifact_sha256"]
        or auxiliary_metadata["observation_sha256"] != bank_record["observation_sha256"]
    ):
        raise ValueError("Old-target auxiliary mask identity/hash differs from frozen observation")
    with np.load(auxiliary_file, allow_pickle=False) as archive:
        masks.update({name: archive[name] for name in archive.files})
    wrapped = {"openpi": raw, "diagnostic_masks": masks}
    reference_decoded = flow.decode(reference_actions, [raw])[0]
    conditions = ["none", "no_image", "no_text"] + [
        f"{region}:{mode}"
        for region in ("target", "distractor", "random", "robot", "background", "old_target")
        for mode in ("background_fill", "mosaic")
    ]
    records, action_arrays, input_arrays = [], {}, {}
    for condition in conditions:
        changed, severity = masked_observation(
            wrapped, condition, target=target, distractor=distractor, seed=300000
        )
        changed_context = flow.prepare([changed]) if ":" in condition else context
        actions, _ = sample_with_intervention(
            flow,
            params,
            changed_context,
            jax.random.key(0),
            metadata,
            knockout=condition if condition in {"no_image", "no_text"} else "none",
            initial_noise=noise,
        )
        normalized = np.asarray(actions)
        decoded = flow.decode(actions, [changed])[0]
        if condition == "none" and not np.array_equal(normalized, reference_actions):
            raise RuntimeError("Open-loop no-op changed canonical sampled actions")
        label = condition.replace(":", "_")
        if save_inputs:
            for camera in ("image", "wrist_image"):
                input_arrays[f"{label}__{camera}"] = changed["openpi"][f"observation/{camera}"]
        action_arrays[label + "__normalized"] = normalized[0]
        action_arrays[label + "__decoded"] = decoded
        records.append(
            {
                "condition": condition,
                "normalized_delta": action_delta(reference_actions[0], normalized[0]),
                "decoded_controller_delta": action_delta(reference_decoded, decoded),
                "severity": severity,
            }
        )
    np.savez(output / "open_loop_actions.npz", **action_arrays)
    if save_inputs:
        np.savez(output / "masked_inputs.npz", **input_arrays)
    (output / "open_loop_summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "conditions": records,
                "noise_sha256": metadata["noise_sha256"],
                "observation_sha256": metadata["bank_record"]["observation_sha256"],
                "masked_inputs": "masked_inputs.npz" if save_inputs else None,
                "old_target_position_control": "camera projection of original BDDL target region extruded to the target object's native geometry height; separate from distractor instance",
                "old_target_auxiliary_metadata": str(auxiliary / "metadata.json"),
                "old_target_auxiliary_sha256": auxiliary_metadata["artifact_sha256"],
                "scope": "open-loop action displacement at matched pixels/state/prompt/initial latent; no closed-loop success score",
                "decoded_action_units": "LIBERO OSC pose/gripper command coordinates, clipped to [-1,1] at execution; not metres or metric Cartesian displacement",
            },
            indent=2,
        )
        + "\n"
    )


def layer_interventions(
    flow, params, raw, context, metadata, noise, reference_actions, baseline_path, steps, output
):
    """True native layer-window and single-layer causal ablations, secondary set."""
    _load_native()
    depth = int(metadata["layers"])
    windows = {
        "early": list(range(6)),
        "middle": list(range(6, 12)),
        "late": list(range(12, 18)),
        **{f"layer_{i:02d}": [i] for i in range(depth)},
    }
    if depth != 18:
        raise ValueError("Predeclared layer windows require the validated18-layer model")
    no_op, _no_op_path = sample_with_layer_knockout(
        flow,
        params,
        context,
        jax.random.key(0),
        metadata,
        knockout="none",
        layers=[],
        initial_noise=noise,
    )
    if not np.array_equal(np.asarray(no_op), reference_actions):
        raise RuntimeError("Layer intervention adapter changed no-op sampled actions")
    reference_decoded = flow.decode(reference_actions, [raw])[0]
    reference_velocities = {
        step: native_velocity(
            flow,
            params,
            context,
            baseline_path[step],
            native_timestep(step),
        )
        for step in steps
    }
    rows, arrays = [], {}
    for modality in ("no_image", "no_text"):
        blocked = ~key_keep_mask(metadata, modality)
        for name, layers in windows.items():
            actions, _ = sample_with_layer_knockout(
                flow,
                params,
                context,
                jax.random.key(0),
                metadata,
                knockout=modality,
                layers=layers,
                initial_noise=noise,
            )
            actions = np.asarray(actions)
            decoded = flow.decode(actions, [raw])[0]
            velocity_records = []
            for step in steps:
                t = native_timestep(step)
                if step == steps[0]:
                    velocity, attention = capture_layer_velocity(
                        flow,
                        params,
                        context,
                        baseline_path[step],
                        t,
                        metadata,
                        knockout=modality,
                        layers=layers,
                    )
                    blocked_mass = float(np.max(attention[np.asarray(layers)][..., blocked]))
                    if blocked_mass != 0:
                        raise RuntimeError("Layer knockout leaked attention to blocked keys")
                else:
                    velocity = np.asarray(
                        velocity_with_layer_knockout(
                            flow,
                            params,
                            context,
                            baseline_path[step],
                            t,
                            metadata,
                            knockout=modality,
                            layers=layers,
                        )
                    )
                velocity_records.append(
                    {
                        "step": step,
                        "t": float(np.asarray(t)[0]),
                        "normalized_velocity_delta": action_delta(
                            reference_velocities[step][0], velocity[0]
                        ),
                    }
                )
                arrays[f"{modality}__{name}__velocity_step{step:02d}"] = velocity
            arrays[f"{modality}__{name}__normalized_actions"] = actions[0]
            arrays[f"{modality}__{name}__decoded_actions"] = decoded
            rows.append(
                {
                    "mode": modality,
                    "window": name,
                    "layers_zero_based": layers,
                    "normalized_action_delta": action_delta(reference_actions[0], actions[0]),
                    "decoded_controller_delta": action_delta(reference_decoded, decoded),
                    "blocked_attention_max": blocked_mass,
                    "fixed_common_latent_velocity_effects": velocity_records,
                }
            )
    np.savez(output / "layer_knockout_arrays.npz", **arrays)
    (output / "layer_knockout_summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "no_op_action_max_abs_difference": 0.0,
                "conditions": rows,
                "scope": "secondary causal native attention-mask intervention, first5 predetermined cases",
                "sampling": "same persisted initial noise; each intervened sampler follows its own ten-step trajectory",
                "velocity_comparison": "fixed baseline latent/time for each policy and intervention",
                "decoded_action_units": "LIBERO OSC/gripper controller commands, not metres",
                "native_mask_coordinates": {
                    "camera_spans": metadata["camera_spans"],
                    "text_span": metadata["text_span"],
                    "prefix_tokens": metadata["prefix_tokens"],
                    "action_tokens": metadata["action_tokens"],
                },
            },
            indent=2,
        )
        + "\n"
    )


def trace_task(flow, reference_flow, root, task, output, *, stage, split, count, steps, checkpoint_contract,
               compiler_profile=None, compiler_contract=None, fidelity_contract=None, source_guard=None):
    _load_native()
    dependencies = [read_stage(output, "baseline")] if stage == "trained" else []
    for dependency in dependencies:
        verify_stage(dependency, full_hash=True)

    def compiler_guard():
        if compiler_profile is not None:
            verify_effective_environment(compiler_profile, compiler_contract, check_versions=True)
            if fidelity_contract is None:
                raise ValueError("Research capture requires the matching native policy numerical proof")
            verify_task_fidelity(fidelity_contract)
        for dependency in dependencies:
            verify_stage(dependency)
        if source_guard is not None:
            source_guard()

    compiler_guard()
    method = root / "methods/rldt" / task
    verify_checkpoint_audit(checkpoint_contract)
    selection = method / "selection.json"
    checked_selection(selection)
    selection_hash = sha(selection)
    bank = root / "observation_bank" / task / split
    all_records = [
        json.loads(line)
        for line in (bank / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    protocol_path = root / "tasks" / task / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    expected_ids = protocol["tasks"][task][f"{split}_state_ids"]
    if [record["initial_state_id"] for record in all_records] != expected_ids:
        raise ValueError("Observation bank does not enumerate the exact frozen protocol IDs")
    records = all_records[:count]
    if len(records) != count:
        raise ValueError(f"Requested {count} fixed cases, but bank has only {len(records)}")
    bank_summary = json.loads((bank / "summary.json").read_text())
    if not bank_summary.get("complete") or bank_summary["manifest_sha256"] != sha(
        bank / "manifest.jsonl"
    ):
        raise ValueError("Observation bank manifest changed or is incomplete")
    auxiliary_root = root / "observation_bank_aux" / task / split
    if not json.loads((auxiliary_root / "summary.json").read_text()).get("complete"):
        raise ValueError("Old-target auxiliary projection bank is incomplete")
    protocol_hash = sha(protocol_path)
    for record in records:
        source = Path(record["observation_path"])
        if (
            record["task"] != task
            or record["split"] != split
            or record["protocol_sha256"] != protocol_hash
        ):
            raise ValueError("Observation bank task/split/protocol identity differs")
        for filename, field in [
            (source, "observation_sha256"),
            (source.parent / "masks.npz", "masks_sha256"),
            (source.parent / "initial_noise.npy", "initial_noise_sha256"),
        ]:
            if sha(filename) != record[field]:
                raise ValueError(f"Immutable bank artifact changed: {filename}")
        auxiliary = auxiliary_root / source.parent.name
        auxiliary_metadata = json.loads((auxiliary / "metadata.json").read_text())
        if auxiliary_metadata["artifact_sha256"] != sha(auxiliary / "old_target_region.npz"):
            raise ValueError("Old-target projection artifact changed")
        if auxiliary_metadata["observation_sha256"] != record["observation_sha256"]:
            raise ValueError("Old-target projection does not match saved observation")
        if (
            auxiliary_metadata["source_bddl_sha256"]
            != protocol["tasks"][task]["source"]["bddl"]["sha256"]
        ):
            raise ValueError("Old-target projection uses a different original BDDL region")
    policies = {
        "pi05_libero": None,
        **{f"rldt_{i:06d}": method / "training" / f"iteration_{i:06d}" for i in (25, 50, 100)},
    }
    policies = {name: path for name, path in policies.items()
                if (path is None) == (stage == "baseline")}
    output.mkdir(parents=True, exist_ok=True)
    stage_plan = {"task": task, "split": split, "state_ids": [r["initial_state_id"] for r in records],
                  "steps": steps, "selection_sha256": selection_hash,
                  "checkpoint_audit_sha256": checkpoint_contract["audit_sha256"],
                  "manifest_sha256": bank_summary["manifest_sha256"],
                  "profile_id": compiler_contract["profile_id"], "compiler_profile": compiler_contract,
                  "policy_fidelity": fidelity_contract}
    if stage == "trained":
        for key in ("task", "split", "state_ids", "steps", "selection_sha256", "checkpoint_audit_sha256", "manifest_sha256"):
            if stage_plan[key] != dependencies[0]["summary"][key]:
                raise ValueError(f"Baseline/trained comparison contract differs: {key}")
    if stage == "baseline":
        if (output / "case_plan.json").exists():
            raise FileExistsError("Baseline case plan already exists; do not overwrite partial or completed traces")
        write_summary(output / "case_plan.json", {
            "task": task, "split": split,
            "state_ids": [r["initial_state_id"] for r in records],
            "selection_sha256": selection_hash,
            "checkpoint_audit_contract": checkpoint_contract,
            "compiler_profile": compiler_contract,
            "policy_fidelity": fidelity_contract,
            "manifest_sha256": bank_summary["manifest_sha256"],
            "protocol_sha256": protocol_hash,
            "checkpoint_iterations": [0, 25, 50, 100], "flow_steps": steps,
            "frame_rule": "first predetermined bank IDs; initial decision; no outcome-based choice",
            "noise": "persisted observation-adjacent initial_noise.npy, common to all policies",
            "comparison": "common baseline x_t at each flow timestep; own trajectories separate",
            "scope": "exploratory fixed-input diagnostics after selection lock",
        })
    # Reuse the exact baseline prefix and common latent path for every expert.
    contexts = []
    projection_references = {}
    bddl = Path(protocol["tasks"][task]["pro"]["bddl"]["path"]).read_text()
    distractor = (
        re.search(r"\(On\s+(\w+)\s+floor_target_object_region\)", bddl)
        .group(1)
        .removesuffix("_1")
        .replace("_", " ")
    )
    for frame_index, record in enumerate(records):
        if source_guard is not None:
            source_guard()
        source = Path(record["observation_path"])
        raw = load_observation(source)
        shared = output / "shared" / f"state_{record['initial_state_id']:06d}"
        if stage == "trained":
            context = load_context(shared, lambda value, dtype: jnp.asarray(value, dtype=dtype))
            metadata = json.loads((shared / "metadata.json").read_text())
            if metadata["bank_record"] != record:
                raise ValueError("Persisted baseline context names a different bank observation")
            with np.load(shared / "common_latents.npz", allow_pickle=False) as archive:
                noise = archive["initial_noise"]
                baseline_path = jnp.asarray(archive["baseline_path"])
            if not np.array_equal(noise, np.load(source.parent / "initial_noise.npy", allow_pickle=False)):
                raise ValueError("Persisted baseline noise differs from immutable observation bank")
            if frame_index < 5:
                projection_references[record["initial_state_id"]] = {}
                for step in steps:
                    with np.load(shared / f"projection_reference_step{step:02d}.npz", allow_pickle=False) as archive:
                        projection_references[record["initial_state_id"]][step] = (archive["hidden"], archive["velocity"])
            contexts.append((record, raw, context, metadata, noise, baseline_path, shared))
            continue
        context, features, metadata = capture_prefix(flow, raw)
        noise_path = source.parent / "initial_noise.npy"
        noise = np.load(noise_path, allow_pickle=False)
        baseline_actions, baseline_path = native_sample(
            flow, flow.params, context, 0, initial_noise=noise
        )
        shared.mkdir(parents=True)
        save_context(shared, context)
        np.savez(shared / "prefix.npz", **features)
        np.savez(
            shared / "common_latents.npz",
            initial_noise=noise,
            baseline_path=np.asarray(baseline_path),
            baseline_actions=np.asarray(baseline_actions),
        )
        if frame_index < 5:
            projection_references[record["initial_state_id"]] = {}
            for step in steps:
                t = native_timestep(step)
                velocity, _, hidden = capture_velocity(
                    flow, flow.params, context, baseline_path[step], t, metadata, return_hidden=True
                )
                reconstructed = project_suffix(flow, flow.params, hidden)
                if not np.array_equal(velocity, reconstructed):
                    raise RuntimeError(
                        "Native projection did not exactly reconstruct baseline velocity from captured hidden states"
                    )
                projection_references[record["initial_state_id"]][step] = (hidden, velocity)
                np.savez(
                    shared / f"projection_reference_step{step:02d}.npz",
                    hidden=hidden,
                    velocity=velocity,
                )
        metadata.update(
            {
                "bank_record": record,
                "noise_path": str(noise_path),
                "noise_sha256": sha(noise_path),
                "source_observation": str(source),
                "selection_sha256": selection_hash,
            }
        )
        (shared / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        changed = dict(
            raw, prompt=raw["prompt"].replace(record["target"].replace("_", " "), distractor)
        )
        if changed["prompt"] == raw["prompt"]:
            raise RuntimeError("Counterfactual noun replacement did not change instruction")
        _, cf_features, cf_metadata = capture_prefix(flow, changed)
        np.savez(shared / "counterfactual_prefix.npz", **cf_features)
        (shared / "counterfactual_metadata.json").write_text(
            json.dumps(
                {
                    **cf_metadata,
                    "instruction": changed["prompt"],
                    "offline_only": True,
                    "goal_predicate_not_modified": True,
                },
                indent=2,
            )
            + "\n"
        )
        base_output = output / "pi05_base" / f"state_{record['initial_state_id']:06d}"
        base_output.mkdir(parents=True)
        for label, payload in [("prefix", raw), ("counterfactual_prefix", changed)]:
            _, base_features, base_metadata = capture_prefix(reference_flow, payload)
            np.savez(base_output / f"{label}.npz", **base_features)
            (base_output / f"{label}_metadata.json").write_text(
                json.dumps(
                    {
                        **base_metadata,
                        "base": reference_flow.metadata,
                        "instruction": payload["prompt"],
                        "bank_record": record,
                        "action_contract": "N/A — incompatible action contract",
                        "offline_only": True,
                    },
                    indent=2,
                )
                + "\n"
            )
        (base_output / "metadata.json").symlink_to((base_output / "prefix_metadata.json").resolve())
        contexts.append((record, raw, context, metadata, noise, baseline_path, shared))
    if stage == "trained":
        compiler_guard()
        compiler_precision_controls(flow, output, contexts[:5], steps,
                                    baseline_stage=dependencies[0], compiler_contract=compiler_contract)
    canonical_encode = jax.jit(flow._encode_prefix)
    for policy, checkpoint in policies.items():
        compiler_guard()
        verify_checkpoint_audit(checkpoint_contract)
        params = flow.params if checkpoint is None else flow.load_params(checkpoint)
        policy_output = output / policy
        policy_output.mkdir()
        drift = parameter_drift(flow.params, params) if checkpoint else {}
        (policy_output / "weight_drift.json").write_text(json.dumps(drift, indent=2) + "\n")
        checkpoint_hash = (
            sha(checkpoint / "expert.npz")
            if checkpoint
            else flow.metadata["checkpoint_manifest_sha256"]
        )
        equality_records = []
        for record, raw, context, metadata, noise, baseline_path, shared in contexts:
            destination = policy_output / f"state_{record['initial_state_id']:06d}"
            destination.mkdir()
            (destination / "prefix.npz").symlink_to((shared / "prefix.npz").resolve())
            (destination / "counterfactual_prefix.npz").symlink_to(
                (shared / "counterfactual_prefix.npz").resolve()
            )
            (destination / "counterfactual_metadata.json").symlink_to(
                (shared / "counterfactual_metadata.json").resolve()
            )
            checked_context = canonical_encode(
                params, flow.frozen, transformed_observation(flow, raw)
            )
            jax.block_until_ready(checked_context)
            prefix_error = max(
                float(np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))
                for a, b in zip(
                    jax.tree.leaves(context), jax.tree.leaves(checked_context), strict=True
                )
            )
            if prefix_error != 0:
                raise RuntimeError("Expert checkpoint changed supposedly frozen prefix/cache")
            baseline_params = flow.params
            try:
                flow.params = params
                _, checkpoint_features, checkpoint_prefix_metadata = capture_prefix(
                    flow, raw, capture_attention=False
                )
            finally:
                flow.params = baseline_params
            equal_feature = (
                checkpoint_prefix_metadata["prefix_features_sha256"]
                == metadata["prefix_features_sha256"]
            )
            if not equal_feature:
                raise RuntimeError("Checkpoint's spatial prefix feature hash differs from baseline")
            equality_records.append(
                {
                    "observation_sha256": record["observation_sha256"],
                    "baseline_feature_sha256": metadata["prefix_features_sha256"],
                    "checkpoint_feature_sha256": checkpoint_prefix_metadata[
                        "prefix_features_sha256"
                    ],
                    "prefix_max_abs_difference": prefix_error,
                    "passed": equal_feature,
                }
            )
            del checkpoint_features
            own_actions, own_path, own_attention = capture_sample(
                flow, params, context, 0, metadata, initial_noise=noise
            )
            if stage == "baseline":
                with np.load(shared / "common_latents.npz", allow_pickle=False) as archive:
                    if not np.array_equal(own_actions, archive["baseline_actions"]) or not np.array_equal(own_path, archive["baseline_path"]):
                        raise RuntimeError("Captured baseline sampler differs from persisted native comparison path")
            np.savez(destination / "own_trajectory_attention.npz", attention=own_attention)
            np.savez(
                destination / "actions.npz",
                normalized=own_actions,
                decoded=flow.decode(own_actions, [raw]),
                own_path=own_path,
                baseline_path=np.asarray(baseline_path),
                initial_noise=noise,
            )
            open_loop_interventions(
                flow,
                params,
                raw,
                context,
                metadata,
                noise,
                own_actions,
                Path(record["observation_path"]),
                record["target"] + "_1",
                distractor.replace(" ", "_") + "_1",
                destination,
                save_inputs=checkpoint is None
                and record["initial_state_id"] == records[0]["initial_state_id"],
            )
            if record["initial_state_id"] in projection_references:
                layer_interventions(
                    flow,
                    params,
                    raw,
                    context,
                    metadata,
                    noise,
                    own_actions,
                    baseline_path,
                    steps,
                    destination,
                )
            checks = {
                "frozen_prefix_max_abs_difference": prefix_error,
                "frozen_kv_cache_sha256": [array_sha256(x) for x in checked_context["kv_cache"]],
            }
            for step in steps:
                t = native_timestep(step)
                x = baseline_path[step]
                native = native_velocity(flow, params, context, x, t)
                for knockout in ("none", "no_image", "no_text"):
                    secondary = (
                        knockout == "none" and record["initial_state_id"] in projection_references
                    )
                    captured = capture_velocity(
                        flow,
                        params,
                        context,
                        x,
                        t,
                        metadata,
                        knockout=knockout,
                        return_hidden=secondary,
                    )
                    velocity, attention = captured[:2]
                    if knockout == "none" and not np.array_equal(native, velocity):
                        raise RuntimeError("Captured checkpoint velocity changed native output")
                    if secondary:
                        hidden_checkpoint = captured[2]
                        hidden_base, base_velocity = projection_references[
                            record["initial_state_id"]
                        ][step]
                        combinations = {
                            "baseH_baseW": project_suffix(flow, flow.params, hidden_base),
                            "baseH_checkpointW": project_suffix(flow, params, hidden_base),
                            "checkpointH_baseW": project_suffix(
                                flow, flow.params, hidden_checkpoint
                            ),
                            "checkpointH_checkpointW": project_suffix(
                                flow, params, hidden_checkpoint
                            ),
                        }
                        if (stage == "baseline" and not np.array_equal(
                            combinations["baseH_baseW"], base_velocity
                        )) or not np.array_equal(combinations["checkpointH_checkpointW"], velocity):
                            raise RuntimeError("Native output projection swap fidelity failed")
                        if stage == "trained":
                            # Preserve both profiles' baseline-weight calculations. Only the
                            # later baseline process may certify baseline-weight endpoints.
                            np.savez(destination / f"projection_inputs_step{step:02d}.npz",
                                     hidden_base=hidden_base, hidden_checkpoint=hidden_checkpoint,
                                     baseH_baseW=base_velocity,
                                     baseH_checkpointW=combinations["baseH_checkpointW"],
                                     checkpointH_checkpointW=combinations["checkpointH_checkpointW"],
                                     baseH_baseW_under_trained_profile=combinations["baseH_baseW"],
                                     checkpointH_baseW_under_trained_profile=combinations["checkpointH_baseW"],
                                     checkpoint_native_velocity=velocity)
                        else:
                            write_projection(destination, step, hidden_base, hidden_checkpoint,
                                             combinations, profile_ownership={name: "native_baseline_v1" for name in combinations})
                    np.savez(
                        destination / f"expert_step{step:02d}_{knockout}.npz",
                        attention=attention,
                        velocity=velocity,
                        x_t=np.asarray(x),
                        t=np.asarray(t),
                    )
                    np.savez(
                        destination / f"views_step{step:02d}_{knockout}.npz",
                        **camera_views(attention, metadata),
                    )
            (destination / "metadata.json").write_text(
                json.dumps(
                    {
                        **metadata,
                        "policy": policy,
                        "checkpoint": str(checkpoint) if checkpoint else None,
                        "checkpoint_sha256": checkpoint_hash,
                        "checks": checks,
                        "shared_prefix_artifact": str(shared / "prefix.npz"),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                json.dumps(
                    {
                        "task": task,
                        "policy": policy,
                        "state": record["initial_state_id"],
                        "completed": True,
                    }
                ),
                flush=True,
            )
        write_summary(
            policy_output / "prefix_equality.json",
            {
                "complete": True,
                "passed": True,
                "checkpoint_sha256": checkpoint_hash,
                "frozen_backbone_sha256": flow._diagnostic_frozen_sha256,
                "observations": equality_records,
                "scope": "exact frozen-backbone identity and matched diagnostic spatial-feature equality",
            },
        )
        del params
    if sha(selection) != selection_hash:
        raise RuntimeError("Locked checkpoint selection changed during diagnostics")
    verify_checkpoint_audit(checkpoint_contract, full_hash=True)
    compiler_guard()
    owned = [output / policy for policy in policies]
    if stage == "baseline":
        owned += [output / "shared", output / "pi05_base", output / "case_plan.json"]
    else:
        owned += [output / "compiler_controls"]
    publish_stage(output, stage, plan=stage_plan, owned_paths=owned,
                  dependencies=[dependency["path"] for dependency in dependencies])


def compiler_precision_controls(flow, output, contexts, steps, *, baseline_stage, compiler_contract):
    """Unchanged baseline weights under the trained compiler; diagnostic only."""
    controls = output / "compiler_controls"
    controls.mkdir()
    rows = []
    for record, _raw, context, metadata, _noise, baseline_path, shared in contexts:
        state = record["initial_state_id"]
        destination = controls / f"state_{state:06d}"
        destination.mkdir()
        for step in steps:
            t = native_timestep(step)
            velocity, attention, hidden = capture_velocity(flow, flow.params, context, baseline_path[step], t,
                                                          metadata, return_hidden=True)
            native = native_velocity(flow, flow.params, context, baseline_path[step], t)
            reconstructed = project_suffix(flow, flow.params, hidden)
            if not np.array_equal(velocity, native) or not np.array_equal(velocity, reconstructed):
                raise RuntimeError("Compiler-only control failed same-profile native fidelity")
            baseline_capture = output / "pi05_libero" / f"state_{state:06d}" / f"expert_step{step:02d}_none.npz"
            baseline_projection = shared / f"projection_reference_step{step:02d}.npz"
            with np.load(baseline_capture, allow_pickle=False) as archive:
                base_attention, base_velocity = archive["attention"], archive["velocity"]
                if not np.array_equal(archive["x_t"], baseline_path[step]) or not np.array_equal(archive["t"], t):
                    raise ValueError("Compiler-only control does not use the original baseline latent/time")
            with np.load(baseline_projection, allow_pickle=False) as archive:
                base_hidden = archive["hidden"]
                if not np.array_equal(archive["velocity"], base_velocity):
                    raise ValueError("Baseline fixed-input capture and projection reference disagree")
            arrays = destination / f"step{step:02d}.npz"
            np.savez(arrays, attention=attention, hidden=hidden, velocity=velocity,
                     x_t=np.asarray(baseline_path[step]), t=np.asarray(t))
            differences = {}
            for name, before, after in (("attention", base_attention, attention), ("hidden", base_hidden, hidden),
                                        ("velocity", base_velocity, velocity)):
                difference = np.asarray(after, np.float64) - np.asarray(before, np.float64)
                differences[name] = {"max_abs": float(np.max(np.abs(difference))),
                                     "l2": float(np.linalg.norm(difference)),
                                     "different_elements": int(np.count_nonzero(difference)), "elements": int(difference.size)}
            value = {"complete": True, "task": record["task"], "state_id": state, "step": step,
                     "weights": "unchanged pi05_libero baseline", "observation_sha256": record["observation_sha256"],
                     "baseline_profile": baseline_stage["summary"]["compiler_profile"],
                     "control_profile": compiler_contract, "differences": differences,
                     "same_profile_native_velocity_max_abs_difference": 0.0,
                     "same_profile_native_projection_max_abs_difference": 0.0,
                     "arrays": {"path": str(arrays), "sha256": sha(arrays)},
                     "reference_artifacts": [{"path": str(path), "sha256": sha(path)} for path in
                                             (baseline_capture, baseline_projection, shared / "context.npz", shared / "common_latents.npz")],
                     "scope": "First5 predetermined fixed cases only; same baseline weights/context/x_t/time, different verified compiler profile. No action sampler, policy condition, behavioral success, or correction of all50 cases."}
            path = destination / f"step{step:02d}.json"
            write_summary(path, value)
            rows.append({"state_id": state, "step": step, "path": str(path), "sha256": sha(path)})
    write_summary(controls / "summary.json", {"complete": True, "states": [item[0]["initial_state_id"] for item in contexts],
                                             "steps": steps, "cases": rows, "control_profile": compiler_contract,
                                             "baseline_profile": baseline_stage["summary"]["compiler_profile"],
                                             "scope": "Compiler-only full attention/hidden/velocity control on first5 predetermined cases; never substitutes for baseline research outputs."})


def write_projection(destination, step, hidden_base, hidden_checkpoint, combinations, *,
                     profile_ownership, precision_controls=None, source_contract=None):
    prefix = destination / f"projection_ablation_step{step:02d}"
    if prefix.with_suffix(".npz").exists() or prefix.with_suffix(".json").exists():
        raise FileExistsError("Refusing to overwrite an existing final projection comparison")
    controls = precision_controls or {}
    np.savez(prefix.with_suffix(".npz"), hidden_base=hidden_base, hidden_checkpoint=hidden_checkpoint,
             **combinations, **controls)
    base_velocity = combinations["baseH_baseW"]
    write_summary(prefix.with_suffix(".json"), {
        "complete": True, "native_reconstruction_max_abs_difference": 0.0,
        "scope": "secondary open-loop final hidden/projection swap; first5 predetermined states; fixed common baseline latent/time",
        "hidden_capture": "native final normalized expert suffix, BF16 values exported float32",
        "projection": "actual NNX action_out_proj with each checkpoint kernel AND bias",
        "units": "normalized flow velocity, not decoded controller displacement",
        "profile_ownership": profile_ownership,
        "source_contract": source_contract,
        "l2_difference_from_baseH_baseW": {name: float(np.linalg.norm(value - base_velocity)) for name, value in combinations.items()},
        "same_hidden_baseline_weights_compiler_effect": {
            name: {"l2": float(np.linalg.norm(value - combinations[name.removesuffix('_under_trained_profile')])),
                   "max_abs": float(np.max(np.abs(value - combinations[name.removesuffix('_under_trained_profile')])))}
            for name, value in controls.items()
        },
        "interpretation": "Cross-policy projection swaps include verified policy-specific compiler arithmetic; same-hidden baseline-weight controls quantify this numerical component separately from parameter changes.",
    })


def baseline_projection_task(flow, root, task, output, *, checkpoint_contract, compiler_profile,
                             compiler_contract, fidelity_contract, source_guard):
    _load_native()
    dependencies = [read_stage(output, stage) for stage in ("baseline", "trained")]
    for dependency in dependencies:
        verify_stage(dependency, full_hash=True)
    baseline, trained = [dependency["summary"] for dependency in dependencies]
    for key in ("task", "split", "state_ids", "steps", "selection_sha256", "checkpoint_audit_sha256", "manifest_sha256"):
        if baseline[key] != trained[key]:
            raise ValueError(f"Projection stage inputs disagree: {key}")
    if sha(root / "methods/rldt" / task / "selection.json") != baseline["selection_sha256"]:
        raise ValueError("Projection stage selection changed")
    if checkpoint_contract["audit_sha256"] != baseline["checkpoint_audit_sha256"]:
        raise ValueError("Projection stage checkpoint audit changed")
    owned = []
    for policy in ("rldt_000025", "rldt_000050", "rldt_000100"):
        source_guard()
        verify_effective_environment(compiler_profile, compiler_contract, check_versions=True)
        verify_task_fidelity(fidelity_contract)
        for dependency in dependencies:
            verify_stage(dependency)
        verify_checkpoint_audit(checkpoint_contract)
        for state in baseline["state_ids"][:5]:
            destination = output / policy / f"state_{state:06d}"
            for step in baseline["steps"]:
                inputs = destination / f"projection_inputs_step{step:02d}.npz"
                with np.load(inputs, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                baseline_endpoint = project_suffix(flow, flow.params, arrays["hidden_base"])
                if not np.array_equal(baseline_endpoint, arrays["baseH_baseW"]):
                    raise RuntimeError("Baseline projection postpass failed exact native endpoint reconstruction")
                if not np.array_equal(arrays["checkpointH_checkpointW"], arrays["checkpoint_native_velocity"]):
                    raise RuntimeError("Trained projection endpoint changed in file exchange")
                combinations = {name: arrays[name] for name in ("baseH_baseW", "baseH_checkpointW", "checkpointH_checkpointW")}
                combinations["checkpointH_baseW"] = project_suffix(flow, flow.params, arrays["hidden_checkpoint"])
                controls = {name: arrays[name] for name in ("baseH_baseW_under_trained_profile", "checkpointH_baseW_under_trained_profile")}
                write_projection(destination, step, arrays["hidden_base"], arrays["hidden_checkpoint"], combinations,
                                 profile_ownership={"baseH_baseW": "native_baseline_v1", "checkpointH_baseW": "native_baseline_v1",
                                                    "baseH_checkpointW": "native_trained_v2", "checkpointH_checkpointW": "native_trained_v2"},
                                 precision_controls=controls, source_contract={"path": str(inputs), "sha256": sha(inputs)})
                owned.extend(destination / f"projection_ablation_step{step:02d}.{suffix}" for suffix in ("npz", "json"))
    source_guard()
    verify_checkpoint_audit(checkpoint_contract, full_hash=True)
    verify_effective_environment(compiler_profile, compiler_contract, check_versions=True)
    verify_task_fidelity(fidelity_contract)
    for dependency in dependencies:
        verify_stage(dependency, full_hash=True)
    plan = {key: baseline[key] for key in ("task", "split", "state_ids", "steps", "selection_sha256", "checkpoint_audit_sha256", "manifest_sha256")}
    plan.update(profile_id=compiler_contract["profile_id"], compiler_profile=compiler_contract, policy_fidelity=fidelity_contract)
    publish_stage(output, "baseline_projection", plan=plan, owned_paths=owned,
                  dependencies=[dependency["path"] for dependency in dependencies])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["baseline", "trained", "baseline_projection", "aggregate"], required=True)
    parser.add_argument("--compiler-profile", type=Path)
    parser.add_argument("--compiler-contract", type=Path)
    parser.add_argument("--tasks", default="object_0,object_1,object_2")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--steps", default="0,5,9")
    args = parser.parse_args()
    if not 1 <= args.count <= 50:
        parser.error("count must be between 1 and 50")
    try:
        steps = requested_steps(args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    tasks = args.tasks.split(",")
    if len(tasks) != len(set(tasks)) or not set(tasks) <= {"object_0", "object_1", "object_2"}:
        parser.error("tasks must be unique object_0/object_1/object_2 identifiers")
    args.output = args.output.resolve()
    args.run_root = args.run_root.resolve()
    if args.stage == "aggregate":
        for task in tasks:
            aggregate_task(args.output / task)
        return  # Parent alone publishes campaign-wide completion after all tasks.
    if args.compiler_profile is None or args.compiler_contract is None:
        parser.error("Native stages require startup compiler profile and exact parent contract")
    compiler_profile = load_profile(args.compiler_profile)
    compiler_contract = json.loads(args.compiler_contract.read_text())
    verify_effective_environment(compiler_profile, compiler_contract, check_versions=True)
    expected_profile = "native_trained_v2" if args.stage == "trained" else "native_baseline_v1"
    if compiler_profile["manifest"]["profile_id"] != expected_profile:
        raise ValueError(f"Stage {args.stage} requires {expected_profile} in a separate process")
    repository = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__),
                    *[repository / "src/rl_vla/diagnostics" / name for name in (
                        "pi05_attention.py", "layer_knockout.py", "grid_stages.py", "compiler_dispatch.py",
                        "compiler_profile.py", "compiler_profile_baseline.py", "trained_fidelity.py")],
                    repository / "src/rl_vla/rldt_pi05.py",
                    Path(__file__).parent / "run_vla_trace_interventions.py",
                    args.compiler_profile, args.compiler_contract]
    launch_sources = {str(path.resolve()): sha(path) for path in source_paths}
    for task in tasks:
        controls = args.output / task / "stages" / args.stage
        controls.mkdir(parents=True, exist_ok=False)
        write_summary(controls / "plan.json", {
            "stage": args.stage, "task": task, "split": args.split, "count": args.count, "steps": steps,
            "rule": "first N immutable bank IDs, no outcome selection",
            "compiler_profile": compiler_contract, "source_sha256": launch_sources,
            "native_policy_process_separation": True,
            "common_inputs": "Baseline process persists exact prefix context, initial noise, common baseline latent path and projection references; trained process reads immutable artifacts.",
            "projection_swap_subset": "first5 predetermined states; baseline-weight combinations completed in baseline-profile postpass; same-hidden compiler effect retained separately",
            "layer_knockout_subset": "unchanged first5 states; early/middle/late and each individual layer; image/text masks",
        })

    def source_guard():
        if any(sha(path) != expected for path, expected in launch_sources.items()):
            raise ValueError("Grid source/profile/contract changed after launch")

    flow = reference_flow = None
    pending = list(tasks)
    while pending:
        source_guard()
        verify_effective_environment(compiler_profile, compiler_contract, check_versions=True)
        for task in pending:
            status = args.run_root / "methods/rldt" / task / "status.json"
            if status.exists() and json.loads(status.read_text()).get("state") == "failed":
                raise RuntimeError(f"Upstream task failed; cannot await its diagnostics: {status}")
        required_stages = [] if args.stage == "baseline" else ["baseline"]
        if args.stage == "baseline_projection":
            required_stages.append("trained")
        ready = [task for task in pending if
                 (args.run_root / "methods/rldt" / task / "selection.json").is_file()
                 and (args.run_root / "methods/rldt" / task / "checkpoint_audit.json").is_file()
                 and (args.run_root / "observation_bank" / task / args.split / "summary.json").is_file()
                 and (args.run_root / "observation_bank_aux" / task / args.split / "summary.json").is_file()
                 and all((args.run_root / "methods/rldt" / task / "training" / f"iteration_{i:06d}/expert.npz").is_file() for i in (25, 50, 100))
                 and all((args.output / task / "stages" / stage / "summary.json").is_file() for stage in required_stages)]
        fidelity_contracts = {task: policy_fidelity_contract(args.run_root, task, compiler_profile) for task in ready}
        ready = [task for task in ready if fidelity_contracts[task] is not None]
        if not ready:
            print(f"Waiting for locked selection, checkpoint audit, predecessor stages and matching native proof: {pending}", flush=True)
            time.sleep(30)
            continue
        checkpoint_contracts = {task: checkpoint_audit_contract(args.run_root / "methods/rldt" / task) for task in ready}
        for task in ready:
            for stage in required_stages:
                verify_stage(read_stage(args.output / task, stage), full_hash=True)
        # All audit/proof gates are checked before any native model is loaded.
        if flow is None:
            _load_native()
            flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
            flow._diagnostic_frozen_sha256 = frozen_hash(flow)
            if args.stage == "baseline":
                reference_flow = Pi05VLMReference.from_checkpoint(Path("checkpoints/openpi/pi05_base"))
        for task in ready:
            shared = dict(checkpoint_contract=checkpoint_contracts[task], compiler_profile=compiler_profile,
                          compiler_contract=compiler_contract, fidelity_contract=fidelity_contracts[task], source_guard=source_guard)
            if args.stage == "baseline_projection":
                baseline_projection_task(flow, args.run_root, task, args.output / task, **shared)
            else:
                trace_task(flow, reference_flow, args.run_root, task, args.output / task,
                           stage=args.stage, split=args.split, count=args.count, steps=steps, **shared)
            pending.remove(task)
    source_guard()


if __name__ == "__main__":
    main()
