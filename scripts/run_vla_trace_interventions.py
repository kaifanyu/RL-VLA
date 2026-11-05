#!/usr/bin/env python3
"""Paired native-mask/pixel interventions on predeclared frozen LIBERO cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from pathlib import Path

import jax
import numpy as np

from rl_vla.diagnostics.compiler_dispatch import load_profile, verify_effective_environment
from rl_vla.diagnostics.trained_fidelity import policy_fidelity_contract, verify_task_fidelity
from rl_vla.diagnostics.interventions import matched_area_control, replace_region
from rl_vla.diagnostics.pi05_attention import (
    _token_metadata,
    native_sample,
    sample_with_intervention,
    transformed_observation,
)
from rl_vla.rldt import evaluation_frame, execute_chunk, load_config
from rl_vla.rldt_env import make_primitive_env
from rl_vla.rldt_pi05 import Pi05Flow


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_selection(path):
    """Require this campaign's locked validation-only selection before test use."""
    path = Path(path)
    record = json.loads(path.read_text())
    counts = record.get("validation_counts", {})
    if set(counts) != {"25", "50", "100"} or any(
        type(v) is not int or not 0 <= v <= 50 for v in counts.values()
    ):
        raise ValueError("Selection requires exactly the 25/50/100 validation success counts")
    selected = min((25, 50, 100), key=lambda i: (-counts[str(i)], i))
    if (
        record.get("iteration") != selected
        or not record.get("locked")
        or record.get("test_results_seen") is not False
    ):
        raise ValueError("Selection is not a locked, validation-only, earliest-tie selection")
    if (
        Path(record["checkpoint"]).resolve()
        != (path.parent / "training" / f"iteration_{selected:06d}").resolve()
    ):
        raise ValueError("Selected checkpoint belongs to a different task/run")
    return record


def instance_names(protocol, task_key):
    task = json.loads(Path(protocol).read_text())["tasks"][task_key]
    bddl = Path(task["pro"]["bddl"]["path"]).read_text()
    target = re.search(r"\(In\s+(\w+)\s+basket_1_contain_region", bddl).group(1)
    distractor = re.search(r"\(On\s+(\w+)\s+floor_target_object_region\)", bddl).group(1)
    return target, distractor


def masked_observation(raw, condition, *, target, distractor, seed):
    """Change only observed pixels using current simulator-owned masks."""
    if condition in {"none", "no_image", "no_text"}:
        return raw, {}
    region, mode = condition.split(":", 1)
    if "diagnostic_masks" not in raw:
        raise RuntimeError("Simulator did not supply current chunk-boundary instance masks")
    masks = raw["diagnostic_masks"]
    changed = {**raw, "openpi": dict(raw["openpi"])}
    severity = {}
    for camera in ("image", "wrist_image"):
        target_mask = np.asarray(masks[f"{camera}__{target}"], bool)
        background = np.asarray(masks[f"{camera}__unowned_background"], bool)
        if region == "target":
            mask = target_mask
        elif region == "distractor":
            mask = np.asarray(masks[f"{camera}__{distractor}"], bool)
        elif region == "old_target":
            mask = np.asarray(masks[f"{camera}__old_target_prism"], bool)
        elif region == "random":
            # Exclude all named food objects/robot; exact target-area match.
            mask = matched_area_control(target_mask, background, seed=seed)
        elif region == "robot":
            candidates = [
                np.asarray(value, bool)
                for name, value in masks.items()
                if name.startswith(camera + "__")
                and (
                    "robot" in name.lower()
                    or "gripper" in name.lower()
                    or "panda" in name.lower()
                    or "mount" in name.lower()
                )
            ]
            if not candidates:
                raise RuntimeError("No simulator robot/gripper instance masks")
            mask = np.logical_or.reduce(candidates)
        elif region == "background":
            mask = background
        else:
            raise ValueError(f"Unknown pixel intervention region: {region}")
        original = raw["openpi"][f"observation/{camera}"]
        changed["openpi"][f"observation/{camera}"] = replace_region(original, mask, mode)
        delta = changed["openpi"][f"observation/{camera}"].astype(np.float32) - original
        severity[camera] = {
            "mask_pixels": int(mask.sum()),
            "mask_fraction": float(mask.mean()),
            "pixel_rms_0_255": float(np.sqrt(np.mean(delta**2))),
        }
    return changed, severity


def evaluate_condition(
    flow, params, config, args, condition, output, token_template, target, distractor
):
    output.mkdir(parents=True, exist_ok=False)
    env_config = copy.deepcopy(config["env"])
    pixel = ":" in condition
    if pixel:
        env_config["factory"] = "rl_vla.diagnostics.interventions:make_env"
        env_config["kwargs"].update(
            mask_every=int(config["rldt"]["execute_steps"]), limit=args.episodes
        )
    env = make_primitive_env(env_config)
    results = []
    start = time.monotonic()
    native = jax.jit(
        lambda p, f, c, key: flow.sample(p, c, key, int(config["rldt"]["ode_steps"]), frozen=f)[0]
    )
    try:
        with (output / "episodes.jsonl").open("x") as log:
            for episode in range(args.episodes):
                key = jax.random.PRNGKey(args.seed + episode)
                raw, reset = env.reset(seed=args.seed + episode)
                frames = [evaluation_frame(raw)] if episode < args.record_episodes else []
                steps = 0
                chunks, severities, telemetry = [], [], []
                primitive_index = 0

                def record_step(
                    executed_action,
                    observation,
                    reward,
                    terminated,
                    truncated,
                    step_info,
                    _telemetry=telemetry,
                ):
                    nonlocal primitive_index
                    primitive_index += 1
                    if "object_telemetry" in step_info:
                        _telemetry.append(
                            {
                                "primitive_steps": primitive_index,
                                "action": np.asarray(executed_action).tolist(),
                                **step_info["object_telemetry"],
                            }
                        )

                for decision in range(10000):
                    key, subkey = jax.random.split(key)
                    policy_input, severity = masked_observation(
                        raw,
                        condition,
                        target=target,
                        distractor=distractor,
                        seed=args.seed + episode * 10000 + decision,
                    )
                    context = flow.prepare([policy_input])
                    token_metadata = _token_metadata(
                        transformed_observation(flow, policy_input),
                        token_template["text_span"][0],
                        flow.horizon,
                    )
                    if token_metadata["prefix_tokens"] != context["prefix_mask"].shape[1]:
                        raise RuntimeError("Native token coordinates differ from validated gate")
                    knockout = condition if condition in {"no_image", "no_text"} else "none"
                    actions, _ = sample_with_intervention(
                        flow,
                        params,
                        context,
                        subkey,
                        token_metadata,
                        steps=int(config["rldt"]["ode_steps"]),
                        knockout=knockout,
                    )
                    if episode == 0 and decision == 0:
                        no_op, _ = sample_with_intervention(
                            flow,
                            params,
                            context,
                            subkey,
                            token_metadata,
                            steps=int(config["rldt"]["ode_steps"]),
                            knockout="none",
                        )
                        reference = native(params, flow.frozen, context, subkey)
                        error = float(np.max(np.abs(np.asarray(no_op) - np.asarray(reference))))
                        explicit_noise = jax.random.normal(
                            subkey, (1, flow.horizon, flow.action_dim)
                        )
                        explicit_actions, _ = native_sample(
                            flow,
                            params,
                            context,
                            0,
                            steps=int(config["rldt"]["ode_steps"]),
                            initial_noise=explicit_noise,
                        )
                        explicit_error = float(
                            np.max(np.abs(np.asarray(explicit_actions) - np.asarray(reference)))
                        )
                        (output / "no_op.json").write_text(
                            json.dumps(
                                {
                                    "passed": error == 0 and explicit_error == 0,
                                    "max_abs_difference": error,
                                    "persisted_noise_recurrence_max_abs_difference": explicit_error,
                                }
                            )
                            + "\n"
                        )
                        if error != 0 or explicit_error != 0:
                            raise RuntimeError(
                                "Closed-loop intervention sampler no-op changed native actions"
                            )
                    decoded = flow.decode(actions, [policy_input])[0]
                    chunks.append(np.asarray(actions)[0])
                    severities.append(severity)
                    raw, _, terminated, truncated, info = execute_chunk(
                        env,
                        decoded[: int(config["rldt"]["execute_steps"])],
                        float(config["rldt"]["gamma"]),
                        record_step=record_step,
                    )
                    steps += info["executed_steps"]
                    if frames:
                        frames.append(evaluation_frame(raw))
                    if terminated or truncated or (args.technical_smoke and decision + 1 >= args.smoke_decisions):
                        break
                else:
                    raise RuntimeError("No episode boundary after 10000 decisions")
                row = {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "reset": reset,
                    "initial_state_id": reset["initial_state_id"],
                    "condition": condition,
                    "checkpoint": str(args.checkpoint) if args.checkpoint else "pi05_libero",
                    "success": bool(info.get("is_success", False))
                    if not args.technical_smoke
                    else None,
                    "primitive_steps": steps,
                    "target": target,
                    "distractor": distractor,
                    "intervention_severity": severities,
                    "technical_smoke": args.technical_smoke,
                    "selection_sha256": sha(args.selection) if args.selection else None,
                    "scope": "predeclared exploratory first-N frozen cases; no checkpoint reselection",
                }
                np.savez(
                    output / f"episode_{episode:03d}_actions.npz", normalized=np.asarray(chunks)
                )
                (output / f"episode_{episode:03d}_telemetry.json").write_text(
                    json.dumps(telemetry) + "\n"
                )
                if frames:
                    import imageio.v3 as iio

                    video = output / f"episode_{episode:03d}.mp4"
                    iio.imwrite(
                        video, np.stack(frames), fps=20 / int(config["rldt"]["execute_steps"])
                    )
                    row["video"] = str(video)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                results.append(row)
                print(
                    json.dumps(
                        {
                            "condition": condition,
                            "episode": episode + 1,
                            "success": row["success"],
                            "steps": steps,
                        }
                    ),
                    flush=True,
                )
    finally:
        env.close()
    summary = {
        "completed": True,
        "complete": True,
        "condition": condition,
        "episodes": results,
        "successes": sum(bool(row["success"]) for row in results)
        if not args.technical_smoke
        else None,
        "n": len(results),
        "elapsed_seconds": time.monotonic() - start,
        "technical_smoke": args.technical_smoke,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--compiler-profile", type=Path)
    parser.add_argument("--compiler-contract", type=Path)
    parser.add_argument("--token-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument(
        "--technical-smoke", action="store_true", help="One decision only, no success score"
    )
    parser.add_argument("--smoke-decisions", type=int, default=1, help="Diagnostic-only decision cap when --technical-smoke is set")
    parser.add_argument(
        "--conditions",
        default="none,no_image,no_text,target:background_fill,distractor:background_fill,random:background_fill,target:mosaic,distractor:mosaic,random:mosaic",
    )
    args = parser.parse_args()
    if bool(args.compiler_profile) != bool(args.compiler_contract):
        parser.error("Provide both compiler profile and parent environment contract")
    if not args.technical_smoke and args.compiler_profile is None:
        parser.error("Research diagnostics require the verified canonical compiler profile")
    compiler_profile = load_profile(args.compiler_profile) if args.compiler_profile else None
    if compiler_profile is not None:
        expected_profile = "native_trained_v2" if args.checkpoint else "native_baseline_v1"
        if compiler_profile["manifest"]["profile_id"] != expected_profile:
            raise ValueError("Native baseline and loaded checkpoint require distinct compiler profiles")
    compiler_contract = json.loads(args.compiler_contract.read_text()) if args.compiler_contract else None
    fidelity_contract = None
    source_hashes = {}

    def compiler_guard():
        if compiler_profile is not None:
            verify_effective_environment(compiler_profile, compiler_contract, check_versions=True)
        if fidelity_contract is not None:
            verify_task_fidelity(fidelity_contract)
        if any(sha(path) != expected for path, expected in source_hashes.items()):
            raise ValueError("Intervention source/input changed after launch")

    compiler_guard()
    config = load_config(args.config)
    split = config["env"]["kwargs"]["split"]
    if split == "train":
        parser.error("Closed-loop diagnostics require an enumerated evaluation split")
    if split == "test" and (args.selection is None or not args.selection.is_file()):
        parser.error("Frozen checkpoint selection must exist before test diagnostics")
    if args.selection:
        checked_selection(args.selection)
    if not 1 <= args.episodes <= 50:
        parser.error("episodes must be between1 and50")
    if args.smoke_decisions < 1 or (args.smoke_decisions != 1 and not args.technical_smoke):
        parser.error("--smoke-decisions requires --technical-smoke and a positive count")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = config["env"]["kwargs"]["protocol"]
    task_key = config["env"]["kwargs"]["task_key"]
    if not args.technical_smoke:
        if args.selection is None or args.selection.parent.name != task_key:
            raise ValueError("Research interventions require the task's locked selection")
        fidelity_contract = policy_fidelity_contract(args.selection.resolve().parents[3], task_key, compiler_profile)
        if fidelity_contract is None:
            raise ValueError("Trained-checkpoint recorded-action proofs are pending")
        verify_task_fidelity(fidelity_contract)
    plan = json.loads(Path(protocol).read_text())
    target, distractor = instance_names(protocol, task_key)
    declared = {
        "task": task_key,
        "split": split,
        "state_ids": plan["tasks"][task_key][f"{split}_state_ids"][: args.episodes],
        "conditions": args.conditions.split(","),
        "seeds": list(range(args.seed, args.seed + args.episodes)),
        "noise_protocol": "JAX PRNGKey(seed+episode), split once per decision as native evaluate",
        "checkpoint": str(args.checkpoint) if args.checkpoint else "pi05_libero",
        "config_sha256": sha(args.config),
        "protocol_sha256": sha(protocol),
        "target": target,
        "distractor": distractor,
        "selection_sha256": sha(args.selection) if args.selection else None,
        "technical_smoke": args.technical_smoke,
        "smoke_decisions": args.smoke_decisions if args.technical_smoke else None,
        "no_counterfactual_goal_scoring": True,
        "compiler_profile": compiler_contract,
        "policy_fidelity": fidelity_contract,
        "source_sha256": {
            str(path): sha(path)
            for path in [
                Path(__file__),
                Path(__file__).resolve().parents[1] / "src/rl_vla/diagnostics/pi05_attention.py",
                Path(__file__).resolve().parents[1] / "src/rl_vla/diagnostics/interventions.py",
                Path(__file__).resolve().parents[1] / "src/rl_vla/diagnostics/compiler_profile.py",
                Path(__file__).resolve().parents[1] / "src/rl_vla/diagnostics/compiler_profile_baseline.py",
                Path(__file__).resolve().parents[1] / "src/rl_vla/diagnostics/compiler_dispatch.py",
                Path(__file__).resolve().parents[1] / "src/rl_vla/diagnostics/trained_fidelity.py",
                args.config, Path(protocol), args.token_metadata,
            ]
        },
    }
    source_hashes = declared["source_sha256"]
    if args.compiler_contract:
        source_hashes[str(args.compiler_contract)] = sha(args.compiler_contract)
    if args.selection:
        source_hashes[str(args.selection)] = sha(args.selection)
    (args.output / "case_plan.json").write_text(json.dumps(declared, indent=2) + "\n")
    flow = Pi05Flow.from_config(config["policy"])
    params = flow.load_params(args.checkpoint) if args.checkpoint else flow.params
    token_template = json.loads(args.token_metadata.read_text())
    summaries = []
    for condition in declared["conditions"]:
        compiler_guard()
        summaries.append(
            evaluate_condition(
                flow,
                params,
                config,
                args,
                condition,
                args.output / condition.replace(":", "_"),
                token_template,
                target,
                distractor,
            )
        )
    compiler_guard()
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "completed": True,
                "complete": True,
                "case_plan": declared,
                "conditions": [{k: v for k, v in s.items() if k != "episodes"} for s in summaries],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
