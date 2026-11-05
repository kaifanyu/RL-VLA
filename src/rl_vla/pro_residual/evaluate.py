"""Paired base-only vs base+residual evaluation with full recording."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from .rollout import run_episode, save_episode_artifacts, write_json

SUMMARY_FIELDS = ["eval_id", "split", "policy", "checkpoint_tag", "checkpoint_path", "episode_id",
                  "initial_state_id", "env_seed", "noise_key", "task_slug", "success", "termination_reason",
                  "steps", "max_steps", "inference_calls", "mean_abs_correction", "rms_residual_v",
                  "any_clip_fraction", "wall_seconds", "metadata", "video", "steps_csv",
                  "comparison_video", "non_protocol_overrides"]


def wilson(successes: int, total: int, z: float = 1.959964) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def checkpoint_tag(payload: dict | None) -> str:
    return "ckptnone" if payload is None else f"step{int(payload['counters']['env_steps']):08d}"


def load_residual_checkpoint(path: Path, learner, device: str) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "flashsac-pro-residual-v1":
        raise ValueError(f"Unexpected checkpoint format in {path}")
    learner.load_state_dict(payload["learner"])
    return payload


def episode_metadata(exp, *, split, policy, episode_id, initial_state_id, episode, result, residual_checkpoint,
                     eval_id, overrides, deterministic) -> dict:
    task = exp.original if episode["source"] == "original" else exp.case
    make = result["make_info"]
    return {
        "episode_id": episode_id, "eval_id": eval_id, "split": split, "policy": policy,
        "task": {"source": episode["source"], "name": task["name"], "task_id": task["task_id"],
                 "suite": task["suite"], "slug": episode["slug"],
                 "case_id": task.get("case_id"), "perturbation": task.get("category", "none (original LIBERO)"),
                 "perturbation_suffix": task.get("perturbation_suffix"),
                 "instruction": make["task_description"], "goal_state": make["goal_state"],
                 "bddl": task.get("bddl"), "bddl_sha256": task.get("bddl_sha256"),
                 "init_file": task.get("states") or task.get("init_file"),
                 "init_sha256": task.get("states_sha256") or task.get("init_sha256"),
                 "initial_state_id": initial_state_id, "simulator_state_index": make["state_index"],
                 "published_state_index": episode.get("published_index"),
                 "state_set": task.get("state_set", {}).get("directory") if episode["source"] == "pro" else None},
        "seeds": {"env_seed": episode["seed"], "base_noise_key": episode["noise_key"],
                  "base_noise_protocol": exp.config["base"]["noise_protocol"],
                  "noise_seeds": [r["noise_seed"] for r in result["inference_rows"]],
                  "residual_sampling": ("deterministic mean" if deterministic else "stochastic")
                  if policy != "base" else "disabled", "backend_seed": result["reset_info"].get("backend_seed")},
        "base_checkpoint": str(exp.base.checkpoint), "residual_checkpoint": residual_checkpoint,
        "resolved_config": str(exp.run_dir / "config/resolved_config.json"),
        "split_manifest": str(exp.run_dir / "config/split_manifest.json"),
        "timing": exp.timing(), "action_spec": exp.spec.describe(),
        "outcome": {"success": result["success"], "termination_reason": result["termination_reason"],
                    "terminated": result["terminated"], "truncated": result["truncated"],
                    "steps": result["steps"], "max_steps": result["max_steps"],
                    "inference_calls": result["inference_calls"],
                    "executed_inference_calls": result["executed_inference_calls"],
                    "wall_seconds": result["wall_seconds"]},
        "residual_metrics": result["residual_metrics"],
        "simulator": {k: make[k] for k in ("controller", "controller_output_max", "control_timestep",
                                            "control_freq", "settle_steps", "objects")},
        "non_protocol_overrides": overrides,
    }


def _append_summary(path: Path, rows: list[dict]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _variant_metrics(rows: list[dict], results: list[dict]) -> dict:
    n = len(rows)
    wins = sum(r["success"] for r in rows)
    lengths = [r["steps"] for r in rows]
    success_lengths = [r["steps"] for r in rows if r["success"]]
    reasons = {}
    for r in rows:
        reasons[r["termination_reason"]] = reasons.get(r["termination_reason"], 0) + 1
    corrections = np.array([res["residual_metrics"]["mean_abs_correction"] for res in results])
    clips = np.array([res["residual_metrics"]["clip_fraction_per_dim"] for res in results])
    return {"episodes": n, "successes": wins, "success_rate": wins / n if n else None,
            "success_95pct_wilson": wilson(wins, n), "termination_counts": reasons,
            "timeouts": reasons.get("timeout", 0),
            "other_terminations": n - reasons.get("timeout", 0) - reasons.get("success", 0),
            "mean_episode_length": float(np.mean(lengths)) if n else None,
            "median_episode_length": float(np.median(lengths)) if n else None,
            "mean_success_length": float(np.mean(success_lengths)) if success_lengths else None,
            "mean_abs_correction_per_dim": corrections.mean(0).tolist() if n else None,
            "mean_rms_residual_v": float(np.mean([res["residual_metrics"]["rms_residual_v_active"] for res in results])) if n else None,
            "any_clip_fraction_mean": float(np.mean([res["residual_metrics"]["any_clip_fraction"] for res in results])) if n else None,
            "clip_fraction_per_dim_mean": clips.mean(0).tolist() if n else None}


def _queue_comparable(row: dict) -> bool:
    """Queue episodes used osmesa and the suite horizon; only an osmesa renderer override keeps that."""
    if not row["non_protocol_overrides"]:
        return True
    overrides = json.loads(row["non_protocol_overrides"])
    return set(overrides) == {"renderer"} and overrides["renderer"]["evaluated_with"] == "osmesa"


def _queue_reproduction(exp, split: str, base_rows: list[dict]) -> dict | None:
    """Compare base-only outcomes with the robustness queue's recorded episodes."""
    from .spec import ROOT

    if split == "retention":
        directory = ROOT / "runs/evaluation/pi05_libero_spatial_20260928/episodes"
        def name(i):
            return f"{exp.original['suite']}_task{exp.original['task_id']:02d}_init{i:03d}.json"
    elif exp.case.get("frozen_protocol"):
        directory = Path(exp.case["frozen_protocol"]).parent / "episodes"

        def name(i):
            published = exp.published_index(i)  # None: generated state, not in the queue
            return None if published is None else f"{exp.case['case_id']}_init{published:03d}.json"
    else:
        return None
    if exp.case["suite"] != "libero_spatial" and split == "retention":
        return None
    compared, mismatches = 0, []
    for row in base_rows:
        filename = name(row["initial_state_id"])
        if filename is None or not (directory / filename).exists() or not _queue_comparable(row):
            continue
        path = directory / filename
        saved = json.loads(path.read_text())
        compared += 1
        if saved["success"] != row["success"] or saved["primitive_steps"] != row["steps"] or saved["seed"] != row["env_seed"]:
            mismatches.append({"initial_state_id": row["initial_state_id"], "queue": [saved["success"], saved["primitive_steps"]],
                               "here": [row["success"], row["steps"]]})
    return {"compared": compared, "matched": compared - len(mismatches), "mismatches": mismatches,
            "source": str(directory)}


def evaluate(exp, sim_factory, *, split: str, initial_states: list[int], variants: tuple[str, ...],
             learner=None, checkpoint_payload: dict | None = None, checkpoint_path: str | None = None,
             eval_id: str, deterministic: bool = True, record: bool = True, device: str = "cpu",
             overrides: dict | None = None, comparison: bool = True) -> dict:
    from .experiment import actor_fn
    from .video import comparison_frames, write_mp4

    overrides = overrides or {}
    existing = exp.run_dir / "evaluation_metrics.json"
    if existing.exists() and eval_id in json.loads(existing.read_text()):
        raise FileExistsError(f"Evaluation id {eval_id} already exists in {existing}; choose a new --eval-id")
    if "residual" in variants and learner is None:
        raise ValueError("Residual evaluation needs a learner checkpoint")
    tag = checkpoint_tag(checkpoint_payload)
    rows = {v: [] for v in variants}
    results = {v: [] for v in variants}
    record_cfg = exp.config["record"]
    source = exp.retention_simulator() if split == "retention" else "pro"
    sim = sim_factory(source)
    started = time.time()
    try:
        for initial_state_id in initial_states:
            episode = exp.evaluation_episode(split, initial_state_id)
            episode_id = f"{eval_id}-init{initial_state_id:03d}"
            pair = {}
            for policy in variants:
                actor = actor_fn(learner, device, deterministic) if policy == "residual" else None
                result = run_episode(sim, exp.base, exp.spec, exp.builder, execute_steps=exp.execute_steps,
                                     gamma=exp.gamma, make_request=episode["make"], seed=episode["seed"],
                                     initial_state_id=initial_state_id, episode_key=episode["noise_key"],
                                     actor=actor, record=record, objects=bool(record_cfg["object_poses"]))
                ckpt = tag if policy == "residual" else "ckptnone"
                prefix = f"{episode['slug']}_{split}_{policy}_{ckpt}_{episode_id}"
                metadata = episode_metadata(exp, split=split, policy=policy, episode_id=episode_id,
                                            initial_state_id=initial_state_id, episode=episode, result=result,
                                            residual_checkpoint=checkpoint_path if policy == "residual" else None,
                                            eval_id=eval_id, overrides=overrides, deterministic=deterministic)
                files = save_episode_artifacts(exp.run_dir, prefix, result, metadata, record=record,
                                               policy_inputs_mode=record_cfg["policy_inputs"])
                row = {"eval_id": eval_id, "split": split, "policy": policy, "checkpoint_tag": ckpt,
                       "checkpoint_path": checkpoint_path if policy == "residual" else "",
                       "episode_id": episode_id, "initial_state_id": initial_state_id, "env_seed": episode["seed"],
                       "noise_key": episode["noise_key"], "task_slug": episode["slug"],
                       "success": result["success"], "termination_reason": result["termination_reason"],
                       "steps": result["steps"], "max_steps": result["max_steps"],
                       "inference_calls": result["inference_calls"],
                       "mean_abs_correction": float(np.mean(result["residual_metrics"]["mean_abs_correction"])),
                       "rms_residual_v": result["residual_metrics"]["rms_residual_v_active"],
                       "any_clip_fraction": result["residual_metrics"]["any_clip_fraction"],
                       "wall_seconds": round(result["wall_seconds"], 2), "metadata": files["metadata"],
                       "video": files.get("video", ""), "steps_csv": files["steps_csv"],
                       "comparison_video": "", "non_protocol_overrides": json.dumps(overrides) if overrides else ""}
                print(f"[{eval_id}] {split} init {initial_state_id:03d} {policy:8s} success={result['success']} "
                      f"steps={result['steps']} reason={result['termination_reason']} "
                      f"wall={result['wall_seconds']:.0f}s", flush=True)
                rows[policy].append(row)
                results[policy].append({"residual_metrics": result["residual_metrics"]})
                pair[policy] = (result, row)
            if comparison and record and {"base", "residual"} <= set(pair):
                sides = []
                for policy, label in (("base", "base pi0.5 only"), ("residual", f"pi0.5 + FlashSAC residual ({tag})")):
                    result, _ = pair[policy]
                    sides.append({"frames": result["frames"], "label": label, "success": result["success"],
                                  "timeout": result["termination_reason"] == "timeout", "steps": result["steps"],
                                  "max_steps": result["max_steps"]})
                path = exp.run_dir / "videos" / f"{episode['slug']}_{split}_compare_{tag}_{episode_id}_agentview.mp4"
                write_mp4(path, comparison_frames(sides[0], sides[1]), 1.0 / pair["base"][0]["control_timestep"])
                for policy in ("base", "residual"):
                    pair[policy][1]["comparison_video"] = str(path)
            _append_summary(exp.run_dir / "evaluation_summary.csv", [pair[p][1] for p in variants])
    finally:
        sim.close()
    metrics = {"eval_id": eval_id, "split": split, "initial_states": initial_states,
               "checkpoint_tag": tag, "checkpoint_path": checkpoint_path,
               "residual_sampling": "deterministic mean" if deterministic else "stochastic",
               "started_unix_time": started, "finished_unix_time": time.time(),
               "non_protocol_overrides": overrides,
               "variants": {v: _variant_metrics(rows[v], results[v]) for v in variants}}
    if {"base", "residual"} <= set(variants):
        outcome = {"both_succeed": 0, "base_only_succeeds": 0, "residual_only_succeeds": 0, "both_fail": 0}
        for b, r in zip(rows["base"], rows["residual"], strict=True):
            key = ("both_succeed" if b["success"] and r["success"] else "base_only_succeeds" if b["success"]
                   else "residual_only_succeeds" if r["success"] else "both_fail")
            outcome[key] += 1
        metrics["paired_outcomes"] = outcome
        metrics["pairing"] = ("Same task, BDDL, initial state, env seed, step limit, preprocessing, base checkpoint, "
                              "sampling protocol and chunk schedule; base noise is identical per inference index. "
                              "After the first correction the observations differ, so later base chunks are "
                              "conditioned on different states (same noise, different outputs). Simulator physics "
                              "is deterministic given state and action.")
    if "base" in variants:
        metrics["base_queue_reproduction"] = _queue_reproduction(exp, split, rows["base"])
    path = exp.run_dir / "evaluation_metrics.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    if eval_id in merged:
        raise FileExistsError(f"Evaluation id {eval_id} already exists in {path}")
    merged[eval_id] = metrics
    write_json(path, merged)
    return metrics
