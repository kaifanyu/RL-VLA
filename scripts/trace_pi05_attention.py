#!/usr/bin/env python3
"""Export native JAX attention on one saved matched observation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.pi05_attention import (
    Pi05VLMReference,
    camera_views,
    capture_prefix,
    capture_sample,
    capture_velocity,
    load_observation,
    native_sample,
    native_velocity,
    synthetic_observation,
)
from rl_vla.rldt_pi05 import Pi05Flow


def extract_bank(flow, args, bank, output_root):
    started = time.monotonic()
    output_root.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in bank.read_text().splitlines() if line.strip()]
    for index, record in enumerate(records):
        state_id = int(record["initial_state_id"])
        source = Path(
            record.get(
                "observation_path",
                bank.parent / f"state_{state_id:06d}" / "observation.npz",
            )
        )
        raw = load_observation(source)
        if args.prompt is not None:
            raw["prompt"] = args.prompt
        frame_start = time.monotonic()
        _, features, metadata = capture_prefix(
            flow, raw, capture_attention=False, canonical_context=index == 0
        )
        output = output_root / f"state_{state_id:06d}"
        output.mkdir(exist_ok=False)
        np.savez(output / "prefix.npz", **features)
        metadata.update(
            {
                "base": flow.metadata,
                "source_observation": str(source),
                "source_record": record,
                "instruction": raw["prompt"],
                "frame_seconds": time.monotonic() - frame_start,
                "frozen_prefix_policy_equivalence": "pi05_libero and all task-specific RLDT experts use identical frozen prefix weights; expert weights are not read by prefix encoding",
            }
        )
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (output_root / "progress.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "index": index,
                        "initial_state_id": state_id,
                        "output": str(output),
                        "seconds": metadata["frame_seconds"],
                        "prefix_features_sha256": metadata["prefix_features_sha256"],
                    }
                )
                + "\n"
            )
        print(
            f"Features {index + 1}/{len(records)}, state {state_id}, {metadata['frame_seconds']:.2f}s",
            flush=True,
        )
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "completed": True,
                "observations": len(records),
                "base": flow.metadata,
                "elapsed_seconds": time.monotonic() - started,
                "bank": str(bank),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observation", type=Path)
    p.add_argument(
        "--bank-root",
        type=Path,
        help="Wait for complete task/split bank manifests and extract all with one model",
    )
    p.add_argument("--tasks", default="object_0,object_1,object_2")
    p.add_argument("--splits", default="train,validation,test")
    p.add_argument(
        "--bank",
        type=Path,
        help="Observation-bank manifest JSONL; one model load, reusable compiled functions",
    )
    p.add_argument(
        "--synthetic", action="store_true", help="Technical fidelity only; never a research scene"
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--expert", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", default=100000, type=int)
    p.add_argument(
        "--noise",
        type=Path,
        help="Persisted initial Gaussian latent; defaults to observation-adjacent initial_noise.npy when present",
    )
    p.add_argument(
        "--prompt",
        help="Offline counterfactual instruction; never changes a simulator goal or supplies a success score",
    )
    p.add_argument("--steps", default="0,5,9")
    p.add_argument("--features-only", action="store_true")
    p.add_argument(
        "--raw-base",
        action="store_true",
        help="Independent image/text-only reference; no action contract",
    )
    p.add_argument("--knockouts", default="none,no_image,no_text")
    args = p.parse_args()
    if sum([bool(args.observation), bool(args.bank), bool(args.bank_root), args.synthetic]) != 1:
        p.error("Select exactly one of --observation, --bank, --bank-root, or --synthetic")
    if (args.bank or args.bank_root) and not args.features_only:
        p.error(
            "--bank currently requires --features-only; full traces use predetermined individual frames"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    adapter_source = (
        Path(__file__).resolve().parents[1] / "src/rl_vla/diagnostics/pi05_attention.py"
    )
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), adapter_source]
    }
    (args.output / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    started = time.monotonic()
    config = {"model_config": "pi05_libero"}
    if args.checkpoint:
        config["checkpoint"] = str(args.checkpoint)
    print("Loading native checkpoint", flush=True)
    if args.raw_base:
        if args.expert:
            p.error("--raw-base cannot load LIBERO RLDT experts")
        flow = Pi05VLMReference.from_checkpoint(
            args.checkpoint or Path("checkpoints/openpi/pi05_base")
        )
    else:
        flow = Pi05Flow.from_config(config)
    params = flow.load_params(args.expert) if args.expert else flow.params
    if args.bank:
        extract_bank(flow, args, args.bank, args.output)
        return
    if args.bank_root:
        pending = [
            (task, split) for task in args.tasks.split(",") for split in args.splits.split(",")
        ]
        while pending:
            ready = [
                (task, split)
                for task, split in pending
                if (args.bank_root / task / split / "summary.json").exists()
            ]
            if not ready:
                print(f"Waiting for complete observation banks: {pending}", flush=True)
                time.sleep(15)
                continue
            for task, split in ready:
                bank = args.bank_root / task / split / "manifest.jsonl"
                extract_bank(flow, args, bank, args.output / task / split)
                pending.remove((task, split))
        (args.output / "summary.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "tasks": args.tasks,
                    "splits": args.splits,
                    "base": flow.metadata,
                    "elapsed_seconds": time.monotonic() - started,
                },
                indent=2,
            )
            + "\n"
        )
        return
    raw = load_observation(args.observation) if args.observation else synthetic_observation()
    if args.prompt is not None:
        raw["prompt"] = args.prompt
    np.savez(
        args.output / "observation.npz",
        image=raw["observation/image"],
        wrist_image=raw["observation/wrist_image"],
        state=raw["observation/state"],
        prompt=raw["prompt"],
    )
    print("Capturing prefix", flush=True)
    context, features, metadata = capture_prefix(
        flow, raw, capture_attention=not args.features_only
    )
    np.savez_compressed(args.output / "prefix.npz", **features)
    metadata.update(
        {
            "schema": "rl-vla-native-attention-v1",
            "synthetic_technical_test": args.synthetic,
            "instruction": raw["prompt"],
            "seed": args.seed,
            "expert": str(args.expert) if args.expert else None,
            "base": flow.metadata,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_observation": str(args.observation) if args.observation else None,
            "source_sha256": source_hashes,
        }
    )
    noise_path = args.noise
    if (
        noise_path is None
        and args.observation
        and (args.observation.parent / "initial_noise.npy").exists()
    ):
        noise_path = args.observation.parent / "initial_noise.npy"
    initial_noise = None if noise_path is None else np.load(noise_path, allow_pickle=False)
    recorded_actions = None
    recorded_sampling_key = None
    if args.observation and not args.raw_base:
        with np.load(args.observation, allow_pickle=False) as recorded:
            if "sampling_key" in recorded:
                import jax
                import jax.numpy as jnp

                recorded_sampling_key = np.asarray(recorded["sampling_key"], dtype=np.uint32)
                if recorded_sampling_key.shape != (2,):
                    raise ValueError("Recorded native sampling_key must be uint32[2]")
                if args.noise is not None:
                    raise ValueError(
                        "Cannot override a recorded rollout sampling key with different noise"
                    )
                initial_noise = np.asarray(
                    jax.random.normal(
                        jnp.asarray(recorded_sampling_key), (1, flow.horizon, flow.action_dim)
                    )
                )
                if "recorded_normalized_actions" in recorded:
                    recorded_actions = np.asarray(
                        recorded["recorded_normalized_actions"], dtype=np.float32
                    )
                    if recorded_actions.ndim == 2:
                        recorded_actions = recorded_actions[None]
    if initial_noise is not None and (
        initial_noise.shape != (1, 10, 32) or not np.isfinite(initial_noise).all()
    ):
        raise ValueError("Expected finite persisted initial noise [1,10,32]")
    metadata["initial_noise_protocol"] = (
        {
            "source": str(noise_path),
            "sha256": hashlib.sha256(noise_path.read_bytes()).hexdigest(),
            "generator": "persisted bank latent; see bank metadata",
            "used_for": "common baseline path and each policy own path",
        }
        if noise_path
        else {
            "source": "JAX random.normal(PRNGKey(seed))",
            "seed": args.seed,
            "saved_in": "actions.npz:initial_noise",
        }
    )
    if args.raw_base:
        metadata["initial_noise_protocol"] = (
            "N/A — VLM-only diagnostic does not sample actions or consume flow noise"
        )
    elif recorded_sampling_key is not None:
        metadata["initial_noise_protocol"] = {
            "source": "sampling_key stored in rollout observation.npz",
            "sampling_key": recorded_sampling_key.tolist(),
            "generator": "native JAX random.normal from exact recorded key",
            "used_for": "own-policy rollout replay and offline common baseline latent path",
            "comparison_context": "rollout-conditioned observation; do not label as initial fixed-state bank",
        }
    checks = {"all_prefix_features_finite": bool(np.isfinite(features["prefix_features"]).all())}
    if not args.features_only and not args.raw_base:
        import jax
        import jax.numpy as jnp

        print("Checking native context and sampler", flush=True)
        native_context = flow.prepare([raw])
        checks["context_max_abs_difference"] = max(
            float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))
            for a, b in zip(jax.tree.leaves(context), jax.tree.leaves(native_context), strict=True)
        )
        _baseline_actions, path = native_sample(
            flow, flow.params, context, args.seed, initial_noise=initial_noise
        )
        final_actions, own_path = native_sample(
            flow, params, context, args.seed, initial_noise=initial_noise
        )
        if recorded_actions is not None:
            if recorded_actions.shape != np.asarray(final_actions).shape:
                raise ValueError("Recorded action chunk shape differs from replayed policy")
            replay_error = float(np.max(np.abs(recorded_actions - np.asarray(final_actions))))
            checks["recorded_rollout_action_max_abs_difference"] = replay_error
            if replay_error != 0:
                raise RuntimeError(
                    f"Recorded rollout actions did not replay exactly ({replay_error}); verify checkpoint and preprocessing"
                )
        print("Checking complete captured no-op sample", flush=True)
        captured_actions, captured_path, trajectory_attention = capture_sample(
            flow, params, context, args.seed, metadata, initial_noise=initial_noise
        )
        checks["no_op_sample_max_abs_difference"] = float(
            np.max(np.abs(captured_actions - np.asarray(final_actions)))
        )
        checks["no_op_path_max_abs_difference"] = float(
            np.max(np.abs(captured_path - np.asarray(own_path)))
        )
        np.savez_compressed(
            args.output / "own_trajectory_attention.npz", attention=trajectory_attention
        )
        np.savez_compressed(
            args.output / "actions.npz",
            normalized=np.asarray(final_actions),
            decoded=flow.decode(final_actions, [raw]),
            baseline_path=np.asarray(path),
            own_path=np.asarray(own_path),
            initial_noise=np.asarray(path[0]),
        )
        checks["actions_finite"] = bool(np.isfinite(np.asarray(final_actions)).all())
        checks["no_op_velocity_max_abs_difference"] = 0.0
        checks["attention_row_sum_max_error"] = 0.0
        checks["masked_camera_mass_max"] = 0.0
        for step in map(int, args.steps.split(",")):
            t = jnp.full((1,), 1.0 - jnp.asarray(step, jnp.float32) / 10, dtype=jnp.float32)
            x = path[step]
            original = native_velocity(flow, params, context, x, t)
            for knockout in args.knockouts.split(","):
                print(f"Capturing expert step {step}, intervention {knockout}", flush=True)
                velocity, attention = capture_velocity(
                    flow, params, context, x, t, metadata, knockout=knockout
                )
                if knockout == "none":
                    error = float(np.max(np.abs(velocity - original)))
                    checks["no_op_velocity_max_abs_difference"] = max(
                        checks["no_op_velocity_max_abs_difference"], error
                    )
                    checks["attention_row_sum_max_error"] = max(
                        checks["attention_row_sum_max_error"],
                        float(np.max(abs(attention.sum(-1) - 1))),
                    )
                    for span in metadata["camera_spans"]:
                        if not span["valid"]:
                            checks["masked_camera_mass_max"] = max(
                                checks["masked_camera_mass_max"],
                                float(np.max(np.abs(attention[..., span["start"] : span["end"]]))),
                            )
                np.savez_compressed(
                    args.output / f"expert_step{step:02d}_{knockout}.npz",
                    attention=attention,
                    velocity=velocity,
                    x_t=np.asarray(x),
                    t=np.asarray(t),
                    query_indices=np.arange(flow.horizon),
                    key_indices=np.arange(attention.shape[-1]),
                )
                np.savez_compressed(
                    args.output / f"views_step{step:02d}_{knockout}.npz",
                    **camera_views(attention, metadata),
                )
        checks["passed"] = bool(
            checks["all_prefix_features_finite"]
            and checks["actions_finite"]
            and checks["context_max_abs_difference"] == 0.0
            and checks["no_op_velocity_max_abs_difference"] == 0.0
            and checks["no_op_sample_max_abs_difference"] == 0.0
            and checks["no_op_path_max_abs_difference"] == 0.0
            and checks["attention_row_sum_max_error"] < 0.01
            and checks["masked_camera_mass_max"] == 0.0
        )
    metadata["elapsed_seconds"] = time.monotonic() - started
    metadata["checks"] = checks
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "gate_summary.json").write_text(
        json.dumps(
            {
                **checks,
                "technical_only": args.synthetic,
                "observation": str(args.observation) if args.observation else None,
                "metadata": str(args.output / "metadata.json"),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(checks, indent=2), flush=True)
    if checks.get("passed") is False:
        raise RuntimeError("Attention capture fidelity gate failed; inspect metadata.json")


if __name__ == "__main__":
    main()
