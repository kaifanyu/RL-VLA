"""FlashSAC residual training on the train split; validation selects checkpoints."""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .experiment import actor_fn, resolve_device, resource_check
from .residual import ArrayReplay, derived_seed
from .rollout import run_episode, save_episode_artifacts, write_json

FORMAT = "flashsac-pro-residual-v1"


class CsvLog:
    def __init__(self, path: Path, fields: list[str]):
        self.path, self.fields = path, fields
        self.exists = path.exists()

    def write(self, row: dict) -> None:
        with self.path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields, extrasaction="ignore")
            if not self.exists:
                writer.writeheader()
                self.exists = True
            writer.writerow(row)


TRAIN_METRIC_FIELDS = ["env_steps", "episodes", "updates", "replay_size", "elapsed_seconds", "critic_loss",
                       "actor_loss", "entropy", "temperature_loss", "alpha", "q", "target_q", "reward_scale", "lr"]
EPISODE_FIELDS = ["episode", "initial_state_id", "env_seed", "noise_key", "success", "termination_reason", "steps",
                  "env_steps_total", "updates_total", "mean_abs_correction", "rms_residual_v", "any_clip_fraction",
                  "wall_seconds", "recorded_prefix"]
VALIDATION_FIELDS = ["env_steps", "variant", "episodes", "successes", "success_rate", "mean_steps",
                     "timeouts", "mean_abs_correction", "any_clip_fraction", "is_best", "checkpoint"]


def _save(path: Path, payload: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_state(episode_rng) -> dict:
    import torch

    state = {"episode_rng": json.dumps(episode_rng.bit_generator.state), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict, episode_rng) -> None:
    import torch

    episode_rng.bit_generator.state = json.loads(state["episode_rng"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _validate(exp, sim, learner, device, env_steps, variant, record) -> tuple[dict, list[dict]]:
    states = exp.split_states("validation")
    limit = exp.config["train"]["validation_states"]
    if limit != "all":
        states = states[:int(limit)]
    rows = []
    for initial_state_id in states:
        episode = exp.evaluation_episode("validation", initial_state_id)
        actor = actor_fn(learner, device, deterministic=True) if variant == "residual" else None
        result = run_episode(sim, exp.base, exp.spec, exp.builder, execute_steps=exp.execute_steps, gamma=exp.gamma,
                             make_request=episode["make"], seed=episode["seed"], initial_state_id=initial_state_id,
                             episode_key=episode["noise_key"], actor=actor, record=record)
        if record:
            prefix = (f"{episode['slug']}_validation_{variant}_step{env_steps:08d}_"
                      f"val{env_steps:08d}-init{initial_state_id:03d}")
            save_episode_artifacts(exp.run_dir, prefix, result,
                                   {"episode_id": prefix, "split": "validation", "policy": variant,
                                    "timing": exp.timing(), "seeds": {"env_seed": episode["seed"],
                                                                      "base_noise_key": episode["noise_key"]}},
                                   record=True, policy_inputs_mode="npz")
        rows.append({"initial_state_id": initial_state_id, "success": result["success"], "steps": result["steps"],
                     "reason": result["termination_reason"],
                     "mean_abs_correction": float(np.mean(result["residual_metrics"]["mean_abs_correction"])),
                     "any_clip_fraction": result["residual_metrics"]["any_clip_fraction"]})
    n = len(rows)
    summary = {"env_steps": env_steps, "variant": variant, "episodes": n,
               "successes": sum(r["success"] for r in rows),
               "success_rate": sum(r["success"] for r in rows) / n,
               "mean_steps": float(np.mean([r["steps"] for r in rows])),
               "timeouts": sum(r["reason"] == "timeout" for r in rows),
               "mean_abs_correction": float(np.mean([r["mean_abs_correction"] for r in rows])),
               "any_clip_fraction": float(np.mean([r["any_clip_fraction"] for r in rows]))}
    return summary, rows


def _better(candidate: dict, best: dict | None) -> bool:
    # Most validation successes; ties keep the earlier checkpoint (RLDT selection rule, step 0 included).
    return best is None or candidate["success_rate"] > best["success_rate"]


NO_MORE_VALIDATION = 2**62


def _next_validation(settings: dict, env_steps: int) -> int:
    """Next validation point: explicit ``validate_at_steps`` if given, else periodic."""
    schedule = settings.get("validate_at_steps")
    if schedule:
        later = [int(s) for s in schedule if int(s) > env_steps]
        return min(later) if later else NO_MORE_VALIDATION
    return env_steps + int(settings["validate_every_steps"])


def train(exp, sim_factory, *, resume: bool, smoke: bool) -> dict:
    import torch

    from rl_vla.runtime import seed_everything

    config, run_dir = exp.config, exp.run_dir
    from rl_vla.attempt_accounting import reject_restart
    reject_restart(config, resume)
    settings = config["train"]
    device = resolve_device(settings["device"])
    seed = int(settings["seed"])
    seed_everything(seed, int(settings["torch_threads"]))
    exp.ensure_split(create=not resume)
    train_states = exp.split_states("train")
    forbidden = set(exp.split["test"]) | set(exp.split["validation"])
    if forbidden & set(train_states):
        raise RuntimeError("Train split overlaps validation/test")
    base = exp.load_base()
    flashsac = config["flashsac"]
    if resume:  # Keep the original schedule even if the budget is extended.
        saved = json.loads((run_dir / "config/resolved_config.json").read_text())["config"]["flashsac"]
        if "lr_decay_updates" in saved:
            flashsac["lr_decay_updates"] = saved["lr_decay_updates"]
    flashsac.setdefault("lr_decay_updates", int(settings["total_env_steps"]) * int(settings["updates_per_step"]))
    learner = exp.make_learner(device)
    replay = ArrayReplay(int(settings["replay_capacity"]), exp.builder.dim, exp.spec.action_dim,
                         seed=derived_seed(seed, "replay"))
    episode_rng = np.random.default_rng(derived_seed(seed, "episodes"))
    counters = {"env_steps": 0, "episodes": 0, "updates": 0, "recorded_train_episodes": 0,
                "next_validation_at": 0 if settings["validate_at_start"] else _next_validation(settings, 0),
                "next_checkpoint_at": int(settings["checkpoint_every_steps"]), "resumes": 0}
    best, base_validation = None, None
    contract = config.get("contract", {})
    strict = contract.get("budget") == "strict"
    checkpoints = run_dir / "checkpoints"
    fingerprints_path = run_dir / "logs/frozen_base_fingerprints.json"
    if resume:
        exp.check_resume_config()
        latest = checkpoints / "latest.pt"
        payload = torch.load(latest, map_location="cpu", weights_only=True)  # RNG states must stay on CPU
        if payload["format"] != FORMAT:
            raise ValueError("Unexpected checkpoint format")
        learner.load_state_dict(payload["learner"])
        counters = dict(payload["counters"])
        counters["resumes"] = counters.get("resumes", 0) + 1
        best, base_validation = payload.get("best"), payload.get("base_validation")
        _restore_rng(payload["rng"], episode_rng)
        replay_path = checkpoints / "replay_latest.npz"
        if replay_path.exists():
            replay.load(replay_path)
            if replay.size != payload["replay_size"]:
                raise ValueError("Replay file does not match the checkpoint")
        elif payload["replay_size"]:
            print("WARNING: replay file missing; resuming with an empty replay", flush=True)
        print(f"Resumed at env_steps={counters['env_steps']} episodes={counters['episodes']} "
              f"updates={counters['updates']} replay={replay.size}", flush=True)
    else:
        exp.write_config()
    fingerprints = json.loads(fingerprints_path.read_text()) if fingerprints_path.exists() else {"records": []}
    start_print = {"when": "train_start", "env_steps": counters["env_steps"], "unix_time": time.time(),
                   "parameters": base.parameter_fingerprint(), "norm_stats": base.norm_stats_fingerprint()}
    if fingerprints["records"] and (fingerprints["records"][0]["parameters"] != start_print["parameters"] or
                                    fingerprints["records"][0]["norm_stats"] != start_print["norm_stats"]):
        raise RuntimeError("Base checkpoint differs from the one this run started with")
    expected_base = contract.get("base_parameter_sha256")
    if expected_base and start_print["parameters"]["sha256"] != expected_base:
        raise RuntimeError(f"Base parameters {start_print['parameters']['sha256']} differ from the benchmark contract")
    fingerprints["records"].append(start_print)
    write_json(fingerprints_path, fingerprints)

    logs = run_dir / "logs"
    metric_log = CsvLog(logs / "train_metrics.csv", TRAIN_METRIC_FIELDS)
    episode_log = CsvLog(logs / "train_episodes.csv", EPISODE_FIELDS)
    validation_log = CsvLog(logs / "validation_metrics.csv", VALIDATION_FIELDS)
    started = time.perf_counter()
    total = int(settings["total_env_steps"])
    batch_size, utd = int(settings["batch_size"]), int(settings["updates_per_step"])
    learning_starts, log_every = int(settings["learning_starts"]), int(settings["log_every_steps"])
    last_metrics = {}
    last_storage_check = 0.0

    def save_checkpoint(name: str) -> Path:
        payload = {"format": FORMAT, "learner": learner.state_dict(), "counters": dict(counters),
                   "best": best, "base_validation": base_validation, "rng": _rng_state(episode_rng),
                   "replay_size": replay.size, "config_hash": json.loads((run_dir / "config/resolved_config.json")
                                                                         .read_text())["config_hash"],
                   "obs_layout": exp.builder.layout(), "action_spec": exp.spec.describe(),
                   "split_states_sha256": exp.split["states_sha256"], "case_id": exp.case["case_id"],
                   "base_parameter_sha256": start_print["parameters"]["sha256"], "smoke": smoke,
                   "saved_unix_time": time.time()}
        path = checkpoints / name
        _save(path, payload)
        _save(checkpoints / "latest.pt", payload)
        if settings["save_replay"]:
            replay.save(checkpoints / "replay_latest.npz")
        return path

    def run_validation(target) -> None:
        """Validate and checkpoint at the current step without advancing any training RNG."""
        nonlocal base_validation, best
        accounting = bool(os.environ.get("RL_VLA_ATTEMPT_DIR"))
        if accounting:
            target.call("accounting_phase", phase="validation")
        devices = [torch.device(device).index or 0] if device.startswith("cuda") else []
        with torch.random.fork_rng(devices=devices):
            record_val = bool(config["record"]["validation_videos"])
            base_rows = None
            if base_validation is None:
                base_validation, base_rows = _validate(exp, target, learner, device, counters["env_steps"], "base",
                                                       record_val)
                write_json(logs / "validation_base_episodes.json", base_rows)
                validation_log.write({**base_validation, "is_best": "", "checkpoint": ""})
            if base_rows is not None and counters["updates"] == 0 and config["simulator"].get("renderer", "osmesa") == "osmesa":
                # The untrained actor's mean is exactly zero and osmesa is repeatable, so the residual
                # pass would replay the base pass episode for episode.
                summary, rows = {**base_validation, "variant": "residual"}, base_rows
                print("VALIDATION residual at step 0 reuses the base pass (zero residual, osmesa)", flush=True)
            else:
                summary, rows = _validate(exp, target, learner, device, counters["env_steps"], "residual", record_val)
        if accounting:
            target.call("accounting_phase", phase="training")
        improved = _better(summary, best)
        name = f"step_{counters['env_steps']:08d}.pt"
        counters["next_validation_at"] = _next_validation(settings, counters["env_steps"])
        if strict:
            counters["next_checkpoint_at"] = counters["env_steps"] + int(settings["checkpoint_every_steps"])
        if improved:
            best = {**summary, "checkpoint": str(checkpoints / name)}
        path = save_checkpoint(name)
        if improved:
            shutil.copyfile(path, checkpoints / "best.pt")
            write_json(checkpoints / "best.json", best)
        validation_log.write({**summary, "is_best": int(improved), "checkpoint": str(path)})
        write_json(logs / f"validation_episodes_step{counters['env_steps']:08d}.json", rows)
        print(f"VALIDATION env_steps={counters['env_steps']} success={summary['successes']}/{summary['episodes']} "
              f"(base {base_validation['successes']}/{base_validation['episodes']}) best={int(improved)}", flush=True)

    validation_sim = None

    def sink(transition: dict) -> None:
        nonlocal last_metrics, validation_sim
        replay.add(**transition)
        learner.observe(transition["reward"], transition["discount"], transition["terminated"], transition["truncated"])
        counters["env_steps"] += 1
        if counters["env_steps"] >= learning_starts and len(replay) >= batch_size:
            for _ in range(utd):
                last_metrics = learner.update(replay.sample(batch_size, device=device))
                counters["updates"] += 1
                bad = [k for k, v in last_metrics.items() if not np.isfinite(v)]
                if bad:
                    raise FloatingPointError(f"Non-finite FlashSAC metrics {bad} at update {counters['updates']}")
        if counters["env_steps"] % log_every == 0:
            metric_log.write({**counters, "replay_size": len(replay),
                              "elapsed_seconds": round(time.perf_counter() - started, 1), **last_metrics})
        if strict and counters["env_steps"] >= counters["next_validation_at"]:
            # Exact milestone inside a pending episode: a second simulator leaves that episode untouched.
            validation_sim = validation_sim or sim_factory("pro")
            run_validation(validation_sim)

    stop = (lambda: counters["env_steps"] >= total) if strict else None
    sim = sim_factory("pro")
    try:
        while True:
            if counters["env_steps"] >= counters["next_validation_at"]:
                run_validation(sim)
            if counters["env_steps"] >= total:
                break
            if time.time() - last_storage_check > 3600:
                resource_check(config, storage=True)
                last_storage_check = time.time()
            else:
                resource_check(config)
            index = counters["episodes"]
            if contract.get("train_reset") == "seed_plus_episode":
                # Same rule as rl_vla.envs.libero_pro: seed + episode, then a uniform train state.
                env_seed = seed + index
                initial_state_id = int(np.random.default_rng(env_seed).choice(train_states))
            else:
                initial_state_id = int(episode_rng.choice(train_states))
                env_seed = derived_seed(seed, "train-episode", index)
            key = f"train:{seed}:{index}"
            every = int(config["record"]["train_every_episodes"])
            record = (every > 0 and index % every == 0
                      and counters["recorded_train_episodes"] < int(config["record"]["max_train_recordings"]))
            result = run_episode(sim, base, exp.spec, exp.builder, execute_steps=exp.execute_steps, gamma=exp.gamma,
                                 make_request=exp.pro_make_request(), seed=env_seed, initial_state_id=initial_state_id,
                                 episode_key=key, actor=actor_fn(learner, device, deterministic=False), sink=sink,
                                 record=record, objects=record and bool(config["record"]["object_poses"]), stop=stop)
            prefix = ""
            if record:
                prefix = (f"{exp.case['slug']}_train_residual_step{counters['env_steps']:08d}_"
                          f"train-ep{index:06d}-init{initial_state_id:03d}")
                save_episode_artifacts(run_dir, prefix, result,
                                       {"episode_id": f"train-ep{index:06d}-init{initial_state_id:03d}", "split": "train",
                                        "policy": "residual(stochastic, training)", "timing": exp.timing(),
                                        "task": {"case_id": exp.case["case_id"], "name": exp.case["name"],
                                                 "instruction": result["make_info"]["task_description"],
                                                 "initial_state_id": initial_state_id},
                                        "seeds": {"env_seed": env_seed, "base_noise_key": key},
                                        "base_checkpoint": str(base.checkpoint),
                                        "outcome": {k: result[k] for k in ("success", "termination_reason", "steps")}},
                                       record=True, policy_inputs_mode=config["record"]["policy_inputs"])
                counters["recorded_train_episodes"] += 1
            counters["episodes"] += 1
            episode_log.write({"episode": index, "initial_state_id": initial_state_id, "env_seed": env_seed,
                               "noise_key": key, "success": result["success"],
                               "termination_reason": result["termination_reason"], "steps": result["steps"],
                               "env_steps_total": counters["env_steps"], "updates_total": counters["updates"],
                               "mean_abs_correction": float(np.mean(result["residual_metrics"]["mean_abs_correction"])),
                               "rms_residual_v": result["residual_metrics"]["rms_residual_v_active"],
                               "any_clip_fraction": result["residual_metrics"]["any_clip_fraction"],
                               "wall_seconds": round(result["wall_seconds"], 1), "recorded_prefix": prefix})
            print(f"TRAIN ep={index} init={initial_state_id} success={result['success']} steps={result['steps']} "
                  f"env_steps={counters['env_steps']}/{total} updates={counters['updates']} "
                  f"critic_loss={last_metrics.get('critic_loss', float('nan')):.4f}", flush=True)
            if not strict and counters["env_steps"] >= counters["next_checkpoint_at"]:
                counters["next_checkpoint_at"] = counters["env_steps"] + int(settings["checkpoint_every_steps"])
                save_checkpoint(f"step_{counters['env_steps']:08d}.pt")
        final = save_checkpoint("final.pt")
    finally:
        sim.close()
        if validation_sim is not None:
            validation_sim.close()
    end_print = {"when": "train_end", "env_steps": counters["env_steps"], "unix_time": time.time(),
                 "parameters": base.parameter_fingerprint(), "norm_stats": base.norm_stats_fingerprint()}
    fingerprints["records"].append(end_print)
    fingerprints["unchanged"] = all(r["parameters"] == fingerprints["records"][0]["parameters"] and
                                    r["norm_stats"] == fingerprints["records"][0]["norm_stats"]
                                    for r in fingerprints["records"])
    write_json(fingerprints_path, fingerprints)
    if not fingerprints["unchanged"]:
        raise RuntimeError("pi0.5 parameters or normalization statistics changed during training")
    summary = {**counters, "final_checkpoint": str(final), "best": best, "base_validation": base_validation,
               "elapsed_seconds": time.perf_counter() - started, "smoke": smoke,
               "base_unchanged": fingerprints["unchanged"], "device": device}
    write_json(run_dir / "logs/train_summary.json", summary)
    return summary
