"""One episode through the shared base/residual execution pipeline.

Timing (identical to the robustness queue and rl_vla.baseline):
  * The frozen base is queried at reset and then every ``execute_steps`` (5)
    environment steps. Each query returns a 10-action chunk; only the first 5 are
    executed, then the chunk is discarded and the base is queried again at the new
    observation. A query also happens at a timeout that falls on a chunk boundary
    (training only, to build the bootstrap observation); it is never executed.
  * The residual actor acts at EVERY environment step, conditioned on the current
    proprio state, the base action about to be executed, its chunk position and the
    frozen features from the most recent base query.
  * One replay transition = one executed environment step (0.05 s simulated):
    (obs_t, v_t, r_t, gamma, obs_{t+1}, terminated, truncated). obs_{t+1} contains
    the next base action that will actually be executed: the next queued action of
    the same chunk, or the first action of the freshly sampled chunk at a boundary.
  * Success -> terminated (no bootstrap). Step limit -> truncated (bootstraps).
    A native ``done`` without success is a truncation as in LiberoEnv.
  * Base noise for call k of an episode is seeded by noise_seed(episode_key, k),
    so base-only and residual variants of a paired episode receive identical noise
    for each call index.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .residual import ActionSpec, ObservationBuilder, noise_seed
from .spec import ACTION_NAMES


def termination_reason(terminated: bool, truncated: bool, steps: int, max_steps: int, native_done: bool) -> str:
    if terminated:
        return "success"
    if truncated and steps >= max_steps:
        return "timeout"
    if truncated and native_done:
        return "native_done_without_success"
    return "unknown"


def run_episode(sim, base, spec: ActionSpec, builder: ObservationBuilder, *, execute_steps: int,
                gamma: float, make_request: dict, seed: int, initial_state_id: int, episode_key: str,
                actor: Callable[[np.ndarray], np.ndarray] | None = None,
                sink: Callable[[dict], None] | None = None, record: bool = False,
                objects: bool = False, stop: Callable[[], bool] | None = None) -> dict:
    """Run one episode. ``actor=None`` disables the residual (base-only).

    ``stop`` is checked after every executed command; True ends the episode there, without
    a terminal or truncation flag, as ``budget_stop``.
    """
    started = time.perf_counter()
    make_info = sim.call("make", seed=seed, initial_state_id=initial_state_id, **make_request)
    max_steps = int(make_info["max_steps"])
    dt = float(make_info["control_timestep"])
    reset = sim.call("reset", seed=seed, initial_state_id=make_info["state_index"], record=record, objects=objects)
    obs, diag = reset["obs"], reset["diag"]
    frames = [diag["agentview"]] if record else []
    policy_inputs = {"front": [], "wrist": [], "call": [], "env_step": []}
    inference_rows, step_rows, object_rows = [], [], []
    if objects:
        object_rows.append({"env_step": 0, "sim_time": diag["sim_time"], **_flatten_objects(diag["objects"])})
    calls = 0
    make_stream = getattr(base, "noise_stream", None)
    next_noise = make_stream(seed) if make_stream is not None else None

    def query(observation, env_step, sim_time, purpose):
        nonlocal calls
        if next_noise is None:
            seed_value = noise_seed(episode_key, calls)
            actions, features, timing = base.infer(observation["openpi"], seed_value)
        else:
            noise, seed_value = next_noise()
            actions, features, timing = base.infer(observation["openpi"], seed_value, noise=noise)
        inference_rows.append({"inference_index": calls, "env_step": env_step, "sim_time": sim_time,
                               "noise_seed": seed_value, "purpose": purpose,
                               "action_ms": round(timing["action_ms"], 3), "total_ms": round(timing["total_ms"], 3),
                               **{f"chunk{j}_{n}": float(actions[j, i]) for j in range(execute_steps)
                                  for i, n in enumerate(ACTION_NAMES)}})
        if record:
            policy_inputs["front"].append(np.asarray(observation["openpi"]["observation/image"]))
            policy_inputs["wrist"].append(np.asarray(observation["openpi"]["observation/wrist_image"]))
            policy_inputs["call"].append(calls)
            policy_inputs["env_step"].append(env_step)
        calls += 1
        return actions[:execute_steps].copy(), features

    chunk, features = query(obs, 0, diag["sim_time"], "execute")
    chunk_call, position = 0, 0
    x = builder.build(obs["state"], chunk[0], 0, features)
    total_reward, terminated, truncated, info = 0.0, False, False, {}
    stopped = False
    for t in range(max_steps + 1):
        base_action = chunk[position]
        v = actor(x) if actor is not None else None
        combo = spec.combine(base_action, v)
        sim_time_before = diag["sim_time"]
        if stop is not None and stop():
            raise RuntimeError("Training command budget exhausted before simulator step")
        reply = sim.call("step", action=combo.executed, record=record, objects=objects)
        next_obs, diag = reply["obs"], reply["diag"]
        reward, terminated, truncated, info = reply["reward"], reply["terminated"], reply["truncated"], reply["info"]
        total_reward += reward
        if record:
            frames.append(diag["agentview"])
        if objects:
            object_rows.append({"env_step": t + 1, "sim_time": diag["sim_time"], **_flatten_objects(diag["objects"])})
        next_x, next_position, used_call = None, position + 1, chunk_call
        if not terminated:
            if next_position == execute_steps and (not truncated or sink is not None):
                chunk, features = query(next_obs, t + 1, diag["sim_time"], "bootstrap_only" if truncated else "execute")
                chunk_call, next_position = calls - 1, 0
            if next_position < execute_steps:
                next_x = builder.build(next_obs["state"], chunk[next_position], next_position, features)
        if sink is not None:
            sink({"obs": x, "action": (combo.residual[spec.active] if v is not None else np.zeros(spec.action_dim, np.float32)),
                  "reward": float(reward), "discount": float(gamma),
                  "next_obs": next_x if next_x is not None else np.zeros_like(x),
                  "terminated": bool(terminated), "truncated": bool(truncated)})
        row = {"env_step": t, "frame_before": t, "frame_after": t + 1,
               "sim_time_before": sim_time_before, "sim_time_after": diag["sim_time"],
               "episode_time_after_s": round((t + 1) * dt, 6), "inference_index": used_call,
               "chunk_position": position, "residual_enabled": int(actor is not None)}
        for name, values in (("base", combo.base), ("residual_v", combo.residual), ("correction", combo.correction),
                             ("requested", combo.requested), ("executed", combo.executed)):
            row.update({f"{name}_{n}": float(values[i]) for i, n in enumerate(ACTION_NAMES)})
        row.update({f"clip_{n}": int(combo.clipped_low[i] or combo.clipped_high[i]) for i, n in enumerate(ACTION_NAMES)})
        row["any_clipped"] = int(np.any(combo.clipped_low | combo.clipped_high))
        row.update({"reward": reward, "terminated": int(terminated), "truncated": int(truncated),
                    "success": int(bool(info.get("is_success", False)))})
        row.update({f"eef_pos_{a}": float(v_) for a, v_ in zip("xyz", diag["eef_pos"], strict=True)})
        row.update({f"eef_quat_{a}": float(v_) for a, v_ in zip("xyzw", diag["eef_quat_xyzw"], strict=True)})
        row.update({f"gripper_qpos_{i}": float(v_) for i, v_ in enumerate(diag["gripper_qpos"])})
        row.update({f"state_{i}": float(v_) for i, v_ in enumerate(next_obs["state"])})
        step_rows.append(row)
        if terminated or truncated:
            break
        if stop is not None and stop():
            stopped = True
            break
        x, position = next_x, next_position
    else:
        raise RuntimeError("Simulator did not enforce its step limit")
    steps = len(step_rows)
    reason = ("budget_stop" if stopped else
              termination_reason(terminated, truncated, steps, max_steps, bool(info.get("native_done", False))))
    success = bool(info.get("is_success", False))
    if success != bool(terminated) or total_reward != float(success):
        raise RuntimeError("Success, termination and sparse reward disagree")
    corrections = np.array([[r[f"correction_{n}"] for n in ACTION_NAMES] for r in step_rows])
    residual_v = np.array([[r[f"residual_v_{n}"] for n in ACTION_NAMES] for r in step_rows])
    clips = np.array([[r[f"clip_{n}"] for n in ACTION_NAMES] for r in step_rows])
    return {
        "success": success, "terminated": bool(terminated), "truncated": bool(truncated),
        "termination_reason": reason, "steps": steps, "max_steps": max_steps, "return": total_reward,
        "inference_calls": calls, "executed_inference_calls": sum(r["purpose"] == "execute" for r in inference_rows),
        "control_timestep": dt, "make_info": make_info, "reset_info": reset["info"],
        "wall_seconds": time.perf_counter() - started,
        "residual_metrics": {
            "mean_abs_correction": np.abs(corrections).mean(0).tolist(),
            "max_abs_correction": np.abs(corrections).max(0).tolist(),
            "rms_residual_v_active": float(np.sqrt(np.mean(residual_v[:, spec.active] ** 2))),
            "any_clip_fraction": float(clips.max(1).mean()),
            "clip_fraction_per_dim": clips.mean(0).tolist()},
        "step_rows": step_rows, "inference_rows": inference_rows, "object_rows": object_rows,
        "frames": frames, "policy_inputs": policy_inputs,
    }


def _flatten_objects(objects: dict) -> dict:
    row = {}
    for name, pose in objects.items():
        for label, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose, strict=True):
            row[f"{name}_{label}"] = float(value)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields += [k for k in row if k not in fields]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")
    temporary.replace(path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def save_episode_artifacts(run_dir: Path, prefix: str, result: dict, metadata: dict, *,
                           record: bool, policy_inputs_mode: str = "npz") -> dict:
    """Write videos/, metadata/ and logs/ files sharing one prefix; return their paths."""
    from .video import write_mp4

    paths = {"metadata": run_dir / "metadata" / f"{prefix}.json",
             "steps_csv": run_dir / "logs" / f"{prefix}_steps.csv",
             "inference_csv": run_dir / "logs" / f"{prefix}_inference.csv"}
    write_csv(paths["steps_csv"], result["step_rows"])
    write_csv(paths["inference_csv"], result["inference_rows"])
    if result["object_rows"]:
        paths["objects_csv"] = run_dir / "logs" / f"{prefix}_objects.csv"
        write_csv(paths["objects_csv"], result["object_rows"])
    if record:
        from .video import episode_frames

        fps = 1.0 / result["control_timestep"]
        inputs = result["policy_inputs"]
        # One video per episode: agentview | policy wrist input, at simulated-time fps.
        paths["video"] = run_dir / "videos" / f"{prefix}_combined.mp4"
        write_mp4(paths["video"], episode_frames(result["frames"], inputs, max_steps=result["max_steps"]), fps)
        if "npz" in policy_inputs_mode:
            paths["policy_inputs_npz"] = run_dir / "videos" / f"{prefix}_policyinputs.npz"
            np.savez_compressed(paths["policy_inputs_npz"], front=np.stack(inputs["front"]),
                                wrist=np.stack(inputs["wrist"]), inference_index=np.asarray(inputs["call"]),
                                env_step=np.asarray(inputs["env_step"]))
    metadata = dict(metadata)
    metadata["files"] = {k: str(v) for k, v in paths.items()}
    metadata["alignment"] = {
        "agentview_frames": len(result["frames"]), "env_steps": result["steps"],
        "video_layout": "left: simulator agentview; right: latest pi0.5 wrist input (held between queries); front input only in npz",
        "rule": ("video frame k = observation after k executed env steps (k=0 initial scene after "
                 "settling); steps CSV row env_step=t has frame_before=t, frame_after=t+1; policy input "
                 "image i was consumed by inference_index i at the env_step recorded in the inference CSV"),
        "video_fps": 1.0 / result["control_timestep"],
        "video_fps_matches_sim": "fps = 1 / control_timestep (20 Hz control, 0.05 s per env step)"}
    write_json(paths["metadata"], metadata)
    return {k: str(v) for k, v in paths.items()}
