#!/usr/bin/env python3
"""Versioned strict-control Object1 resume; unsupported controls remain NA."""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
import run_vla_trace_interventions_gpu6_v1 as original
from run_vla_trace_interventions_gpu6_v1 import (
    Pi05Flow,
    _token_metadata,
    checked_selection,
    evaluation_frame,
    execute_chunk,
    instance_names,
    load_config,
    load_profile,
    make_primitive_env,
    masked_observation,
    native_sample,
    policy_fidelity_contract,
    sample_with_intervention,
    transformed_observation,
    verify_effective_environment,
    verify_task_fidelity,
)

from rl_vla.diagnostics.structural_na_resume import (
    CONDITIONS,
    NONZERO_PROOFS,
    StructuralNA,
    condition_summary,
    record,
    sha,
    strict_support,
    validate_amendment,
    verify_acceptance,
    verify_record,
    write,
)

ROOT = Path(__file__).resolve().parents[1]


def install_process_journal(output):
    """Record native env/ffmpeg children without changing Popen arguments."""
    popen = subprocess.Popen
    journal = output / "owned_processes.jsonl"
    with journal.open("x"):
        pass

    def tracked(*args, **kwargs):
        process = popen(*args, **kwargs)
        try:
            raw = Path(f"/proc/{process.pid}/stat").read_text()
            fields = raw[raw.rindex(")") + 2:].split()
            identity = {"pid": process.pid, "start_ticks": int(fields[19]),
                        "pgid": int(fields[2]), "ppid": int(fields[1]), "state": fields[0]}
            with journal.open("a") as stream:
                stream.write(json.dumps(identity) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # Never leave an unjournaled child live if provenance recording fails.
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise
        return process

    subprocess.Popen = tracked


def save_structural_evidence(output, episode, decision, condition, raw, reset,
                             key, subkey, chunks, telemetry, severities, support, args):
    folder = output / f"episode_{episode:03d}_structural_na"
    folder.mkdir(exist_ok=False)
    observation_path = folder / "observation.npz"
    observation = raw["openpi"]
    current_state = (telemetry[-1] if telemetry else reset["object_telemetry"])["simulator_state"]
    np.savez(observation_path,
             image=observation["observation/image"],
             wrist_image=observation["observation/wrist_image"],
             state=observation["observation/state"], prompt=np.asarray(observation["prompt"]),
             sampling_key=np.asarray(subkey), key_after_split=np.asarray(key),
             simulator_state=np.asarray(current_state),
             normalized_action_prefix=np.asarray(chunks).reshape(-1, 10, 32),
             initial_noise=np.asarray(jax.random.normal(subkey, (1, 10, 32))))
    masks_path = folder / "masks.npz"
    np.savez(masks_path, **raw["diagnostic_masks"])
    telemetry_path = folder / "partial_telemetry.json"
    write(telemetry_path, telemetry)
    severity_path = folder / "partial_intervention_severity.json"
    write(severity_path, severities)
    evidence = {
        "complete": True, "status": "structural_NA", "task": "object_1",
        "policy": args.policy, "condition": condition, "episode": episode,
        "target": args.target, "distractor": args.distractor,
        "initial_state_id": 150 + episode, "seed": args.seed + episode,
        "decision": decision, "random_mask_seed": args.seed + episode * 10000 + decision,
        "support": support, "first_failing_camera": next(x["camera"] for x in support if not x["exact_match_possible"]),
        "primitive_steps_before_undefined_input": len(telemetry),
        "normalized_chunks_before_undefined_input": len(chunks),
        "success": None, "outcome_available": False, "reset": reset,
        "mask_metadata": raw["diagnostic_mask_metadata"],
        "artifacts": [record(p) for p in (observation_path, masks_path, telemetry_path, severity_path)],
        "amendment": args.amendment_record,
        "definition": "Required unique target-area pixels within unowned background do not exist. No fallback mask/input/action/outcome is substituted.",
        "state_timing": "Full simulator state after the preceding primitive; camera masks/RGB retain their exact native sensor-time alignment metadata.",
    }
    path = folder / "summary.json"
    write(path, evidence)
    return record(path)


def evaluate_condition(
    flow, params, config, args, condition, output, token_template, target, distractor
):
    output.mkdir(parents=True, exist_ok=False)
    if not args.technical_smoke:
        for item in args.reuse_condition["episode_artifacts"]:
            for name in ("actions", "telemetry", "video"):
                source = Path(item[name]["path"])
                (output / source.name).symlink_to(source)
        old_no_op = Path(args.reuse_condition["original_directory"]) / "no_op.json"
        if args.reuse_condition["rows"] and old_no_op.exists():
            (output / "no_op.json").symlink_to(old_no_op)
    env_config = copy.deepcopy(config["env"])
    pixel = ":" in condition
    if pixel:
        env_config["factory"] = "rl_vla.diagnostics.interventions:make_env"
        env_config["kwargs"].update(
            mask_every=int(config["rldt"]["execute_steps"]), limit=args.episodes
        )
    env = make_primitive_env(env_config)
    results = []
    reused = args.reuse_condition["rows"] if not args.technical_smoke else []
    resume_checked = False
    start = time.monotonic()
    native = jax.jit(
        lambda p, f, c, key: flow.sample(p, c, key, int(config["rldt"]["ode_steps"]), frozen=f)[0]
    )
    try:
        with (output / "episodes.jsonl").open("x") as log:
            for episode in range(args.episodes):
                key = jax.random.PRNGKey(args.seed + episode)
                raw, reset = env.reset(seed=args.seed + episode)
                if reset["initial_state_id"] != 150 + episode:
                    raise ValueError("Resume selected another frozen state")
                if args.technical_smoke and episode != args.proof_episode_index:
                    continue
                if episode < len(reused):
                    if reset != reused[episode]["reset"]:
                        raise ValueError("Skipped completed episode reset differs from original")
                    results.append(reused[episode])
                    log.write(args.reuse_condition["raw_lines"][episode] + "\n")
                    log.flush()
                    continue
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

                structural = None
                for decision in range(10000):
                    key, subkey = jax.random.split(key)
                    try:
                        strict_support(raw, condition, target)
                    except StructuralNA as unavailable:
                        structural = save_structural_evidence(
                            output, episode, decision, condition, raw, reset, key, subkey,
                            chunks, telemetry, severities, unavailable.support, args)
                        break
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
                    if not resume_checked:
                        resume_checked = True
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
                        (output / ("resume_no_op.json" if (output / "no_op.json").exists() else "no_op.json")).write_text(
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
                    if not args.technical_smoke and structural is None
                    else None,
                    "primitive_steps": steps,
                    "target": target,
                    "distractor": distractor,
                    "intervention_severity": severities,
                    "technical_smoke": args.technical_smoke,
                    "selection_sha256": sha(args.selection) if args.selection else None,
                    "scope": "predeclared exploratory first-N frozen cases; no checkpoint reselection",
                }
                if structural is not None:
                    row.update(status="structural_NA", structural_evidence=structural,
                               availability_reason="insufficient_unowned_background_for_exact_target_area")
                elif not args.technical_smoke:
                    row["status"] = "completed"
                np.savez(
                    output / f"episode_{episode:03d}_actions.npz", normalized=np.asarray(chunks).reshape(-1, 10, 32)
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
    if not args.technical_smoke:
        summary = condition_summary(condition, results, amendment=args.amendment_record,
                                    reuse_manifest=args.reuse_manifest_record,
                                    elapsed=time.monotonic() - start)
        write(output / "summary.json", summary)
        return summary
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
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Stopping owned resume environment/video children")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--compiler-profile", type=Path, required=True)
    parser.add_argument("--compiler-contract", type=Path, required=True)
    parser.add_argument("--token-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("pi05_libero", "rldt_25", "rldt_50", "rldt_100"), required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--resume-proof", type=Path)
    parser.add_argument("--root-acceptance", type=Path)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--smoke-decisions", type=int, default=2)
    parser.add_argument("--proof-episode-index", type=int, choices=(0, 2, 5, 7, 8), default=0)
    args = parser.parse_args()
    if args.seed != 100000 or args.record_episodes < args.episodes:
        parser.error("Original seed and all-episode recording are required")
    if args.technical_smoke:
        if args.episodes != args.proof_episode_index + 1 or args.smoke_decisions != 2:
            parser.error("Technical replay resets original indices and executes two decisions")
        if args.proof_episode_index not in (0, NONZERO_PROOFS[args.policy][0]):
            parser.error("Only the predeclared last-completed interrupted-condition proof case is allowed")
    elif args.episodes != 10 or args.resume_proof is None or args.root_acceptance is None or args.proof_episode_index != 0:
        parser.error("Research resume requires ten frozen cases and a reviewed exact replay proof")
    if sha(Path(original.__file__)) != "808fd5f2cfd88e15362ed6f590033743e7c820bbea83d8a93c68ea785f98ced3":
        raise ValueError("Original numerical evaluator changed")
    checked_selection(args.selection)
    config = load_config(args.config)
    if config["env"]["kwargs"]["task_key"] != "object_1" or config["env"]["kwargs"]["split"] != "test":
        raise ValueError("Resume is scoped to Object1 frozen test protocol")
    if int(config["rldt"]["ode_steps"]) != 10 or int(config["rldt"]["execute_steps"]) != 5:
        raise ValueError("Original native sampling/execute schedule changed")
    run = args.selection.resolve().parents[3]
    expected = None if args.policy == "pi05_libero" else run / "methods/rldt/object_1/training" / f"iteration_{int(args.policy.split('_')[1]):06d}"
    if (args.checkpoint.resolve() if args.checkpoint else None) != expected:
        raise ValueError("Policy/checkpoint identity mismatch")
    profile = load_profile(args.compiler_profile)
    expected_profile = "native_baseline_gpu6_v1" if expected is None else "native_trained_gpu6_v1"
    if profile["manifest"]["profile_id"] != expected_profile:
        raise ValueError("Wrong native policy/device compiler profile")
    contract = json.loads(args.compiler_contract.read_text())
    verify_effective_environment(profile, contract, check_versions=True)
    fidelity = policy_fidelity_contract(run, "object_1", profile)
    if fidelity is None:
        raise ValueError("Original GPU6 fidelity proof missing")
    verify_task_fidelity(fidelity)
    reuse = json.loads(args.reuse_manifest.read_text())
    if (reuse.get("task") != "object_1" or reuse.get("completed_episodes") != 256
            or reuse.get("whole_conditions_complete") != 23):
        raise ValueError("Resume does not bind the preserved failed attempt")
    args.reuse_manifest_record = record(args.reuse_manifest)
    args.amendment_record = record(args.amendment)
    amendment = json.loads(args.amendment.read_text())
    validate_amendment(amendment, execution=not args.technical_smoke)
    sources = {str(p): record(p) for p in (
        Path(__file__).resolve(), Path(original.__file__).resolve(),
        ROOT / "src/rl_vla/diagnostics/structural_na_resume.py",
        args.config, args.selection, args.compiler_contract, args.token_metadata,
        args.reuse_manifest, args.amendment,
    )}
    if not args.technical_smoke:
        from rl_vla.diagnostics.structural_na_resume import verify_resume_proof
        sources.update(verify_resume_proof(args.resume_proof, run=run))
        accepted = verify_acceptance(args.root_acceptance, amendment=args.amendment,
                                     reuse_manifest=args.reuse_manifest, resume_proof=args.resume_proof)
        sources[accepted["path"]] = accepted

    def guard(full=False):
        verify_effective_environment(profile, contract, check_versions=True)
        verify_task_fidelity(fidelity, full_hash=full)
        for item in [*sources.values(), *reuse["inputs"].values()]:
            verify_record(item, full_hash=full)

    guard()
    args.output.mkdir(parents=True, exist_ok=False)
    install_process_journal(args.output)
    target, distractor = instance_names(config["env"]["kwargs"]["protocol"], "object_1")
    args.target, args.distractor = target, distractor
    conditions = (NONZERO_PROOFS[args.policy][1],) if args.technical_smoke and args.proof_episode_index != 0 else CONDITIONS
    indices = [args.proof_episode_index] if args.technical_smoke else list(range(10))
    plan = {"version": "strict_background_structural_na_resume_v1", "task": "object_1", "policy": args.policy,
            "checkpoint": str(expected) if expected else "pi05_libero", "conditions": list(conditions),
            "state_ids": [150 + i for i in indices], "seeds": [100000 + i for i in indices],
            "reset_progression_state_ids": list(range(150, 150 + args.episodes)),
            "reuse_manifest": args.reuse_manifest_record, "amendment": args.amendment_record,
            "source_sha256": {p: item["sha256"] for p, item in sources.items()},
            "compiler_profile": contract, "technical_smoke": args.technical_smoke,
            "smoke_decisions": 2 if args.technical_smoke else None,
            "noise_protocol": "Unchanged JAX PRNGKey(100000+original_episode_index), split once per decision",
            "random_mask_seed": "100000+original_episode_index*10000+decision",
            "state_selection": "Reset every original index in order; reused reset metadata must match exactly"}
    write(args.output / "case_plan.json", plan)
    flow = Pi05Flow.from_config(config["policy"])
    params = flow.load_params(args.checkpoint) if args.checkpoint else flow.params
    token_template = json.loads(args.token_metadata.read_text())
    summaries = []
    for condition in conditions:
        guard()
        args.reuse_condition = reuse["policies"][args.policy]["conditions"][condition]
        cell = args.output / condition.replace(":", "_")
        if args.reuse_condition["whole_condition_complete"] and not args.technical_smoke:
            cell.symlink_to(args.reuse_condition["original_directory"], target_is_directory=True)
            summary = json.loads((cell / "summary.json").read_text())
        else:
            summary = evaluate_condition(flow, params, config, args, condition, cell, token_template, target, distractor)
        summaries.append({k: v for k, v in summary.items() if k != "episodes"})
    guard(full=True)
    write(args.output / "summary.json", {"complete": True, "completed": True,
          "execution_coverage_complete": True, "case_plan": plan, "conditions": summaries,
          "primary_outcomes_all_available": all(s.get("primary_outcome_available", True) for s in summaries),
          "canonical_completion_published": False})


if __name__ == "__main__":
    main()

