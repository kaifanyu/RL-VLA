"""Verification suite run against the real checkpoint and simulator (no training)."""

from __future__ import annotations

import importlib.util
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .residual import noise_seed
from .rollout import run_episode, write_json
from .spec import ROOT


def _serve_baseline_module():
    spec = importlib.util.spec_from_file_location("serve_baseline", ROOT / "scripts/serve_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _InProcessEvaluationClient:
    """Mimics rl_vla.baseline.EvaluationClient + serve_baseline.EvaluationPolicy, minus the socket."""

    def __init__(self, evaluation_policy):
        self.policy = evaluation_policy
        self.episode_key = "unset"
        self.calls = 0

    def get_server_metadata(self):
        return {"noise_protocol": "numpy-pcg64-float32-standard-normal-v1"}

    def reset(self):
        self.calls = 0

    def infer(self, observation):
        payload = dict(observation)
        payload["evaluation_noise_seed"] = noise_seed(self.episode_key, self.calls)
        self.calls += 1
        result = self.policy.infer(payload)
        return {"actions": np.asarray(result["actions"])}


class _SimGymEnv(gym.Env):
    """Gymnasium view of the simulator child, for the original ResidualChunkEnv."""

    def __init__(self, sim, low, high):
        self.sim = sim
        self.action_space = spaces.Box(np.asarray(low, np.float32), np.asarray(high, np.float32), dtype=np.float32)
        self.trace = []

    def reset(self, *, seed=None, options=None):
        reply = self.sim.call("reset", seed=seed, initial_state_id=options["initial_state_id"])
        return reply["obs"], reply["info"]

    def step(self, action):
        reply = self.sim.call("step", action=np.asarray(action, np.float32))
        self.trace.append({"action": np.asarray(action, np.float32).copy(), "state": np.asarray(reply["obs"]["state"]).copy()})
        return reply["obs"], reply["reward"], reply["terminated"], reply["truncated"], reply["info"]


def run_checks(exp, sim_factory, *, parity_steps: int, reproduce_states: list[int], output_name: str) -> dict:
    from rl_vla.base_policy import OpenPIBasePolicy
    from rl_vla.env_adapter import ResidualChunkEnv

    if exp.config["base"].get("backend") == "pi05_flow":
        raise SystemExit("check compares OpenPI policy.infer with the queue; use backend = 'openpi_policy'")
    report = {"started_unix_time": time.time(), "case": exp.case, "checks": {}}
    checks = report["checks"]
    split = exp.ensure_split(create=True)
    base = exp.load_base()
    before = {"parameters": base.parameter_fingerprint(), "norm_stats": base.norm_stats_fingerprint()}
    sim = sim_factory("pro")
    try:
        # 1. Base policy and intended PRO environment.
        state_id = split["train"][0]
        episode = exp.evaluation_episode("train", state_id)
        make = sim.call("make", seed=episode["seed"], initial_state_id=state_id, **exp.pro_make_request())
        reset = sim.call("reset", seed=episode["seed"], initial_state_id=make["state_index"], record=True, objects=True)
        obs = reset["obs"]
        image = obs["openpi"]["observation/image"]
        checks["environment"] = {
            "passed": bool(image.std() > 0 and make["task_description"]),
            "case_id": exp.case["case_id"], "task_name": exp.case["name"], "bddl": exp.case["bddl"],
            "instruction_from_bddl": make["task_description"], "goal_state": make["goal_state"],
            "objects": make["objects"], "max_steps": make["max_steps"], "control_timestep": make["control_timestep"],
            "controller": make["controller"], "controller_output_max": make["controller_output_max"],
            "action_low": np.asarray(make["low"]).tolist(), "action_high": np.asarray(make["high"]).tolist(),
            "policy_image_shape": list(image.shape), "external_frame_shape": list(reset["diag"]["agentview"].shape),
            "state": np.asarray(obs["state"]).tolist(), "object_poses_available_for_logging": sorted(reset["diag"]["objects"])}

        # 2a. Same observation + same noise: identical to the server's EvaluationPolicy.
        serve = _serve_baseline_module()
        reference = serve.EvaluationPolicy(base.policy, base.train_config.model)
        seed_value = noise_seed(episode["noise_key"], 0)
        mine, features, _ = base.infer(obs["openpi"], seed_value)
        theirs = np.asarray(reference.infer({**obs["openpi"], "evaluation_noise_seed": seed_value})["actions"], np.float32)
        again, features2, _ = base.infer(obs["openpi"], seed_value)
        other, _, _ = base.infer(obs["openpi"], seed_value + 1)
        checks["policy_call_parity"] = {
            "passed": bool(np.array_equal(mine, theirs) and np.array_equal(mine, again)
                           and np.array_equal(features, features2) and not np.array_equal(mine, other)),
            "max_abs_diff_vs_serve_baseline": float(np.abs(mine - theirs).max()),
            "repeat_identical": bool(np.array_equal(mine, again)), "features_repeat_identical": bool(np.array_equal(features, features2)),
            "different_noise_changes_actions": bool(not np.array_equal(mine, other)),
            "chunk_shape": list(mine.shape), "feature_dim": int(features.size), "features_finite": bool(np.isfinite(features).all()),
            "base_action_range_first_chunk": [float(mine.min()), float(mine.max())]}

        # 2b. Residual-disabled pipeline vs the original ResidualChunkEnv + OpenPIBasePolicy path.
        client = _InProcessEvaluationClient(reference)
        client.episode_key = episode["noise_key"]
        gym_env = _SimGymEnv(sim, make["low"], make["high"])
        chunk_env = ResidualChunkEnv(gym_env, OpenPIBasePolicy(7, 10, client=client),
                                     {"horizon": 10, "execute_steps": exp.execute_steps, "gamma": exp.gamma,
                                      "residual_scale": [0.1] * 6 + [0.0]})
        sim.call("make", seed=episode["seed"], initial_state_id=state_id, **exp.pro_make_request())
        chunk_env.reset(seed=episode["seed"], options={"initial_state_id": make["state_index"]})
        for _ in range(max(1, parity_steps // exp.execute_steps)):
            *_, terminated, truncated, _ = chunk_env.step(np.zeros(chunk_env.action_dim, np.float32))
            if terminated or truncated:
                break
        original_trace = gym_env.trace
        request = dict(exp.pro_make_request(), max_steps=len(original_trace))
        result = run_episode(sim, base, exp.spec, exp.builder, execute_steps=exp.execute_steps, gamma=exp.gamma,
                             make_request=request, seed=episode["seed"], initial_state_id=state_id,
                             episode_key=episode["noise_key"], actor=None)
        names = ("x", "y", "z", "rx", "ry", "rz", "gripper")
        executed = np.array([[r[f"executed_{n}"] for n in names] for r in result["step_rows"]])
        states = np.array([[r[f"state_{i}"] for i in range(8)] for r in result["step_rows"]])
        old_actions = np.array([t["action"] for t in original_trace])
        old_states = np.array([t["state"] for t in original_trace])
        checks["pipeline_parity_residual_disabled"] = {
            "passed": bool(executed.shape == old_actions.shape and np.abs(executed - old_actions).max() <= 1e-6
                           and np.abs(states - old_states).max() <= 1e-6),
            "env_steps_compared": len(original_trace), "tolerance": 1e-6,
            "max_abs_executed_action_diff": float(np.abs(executed - old_actions).max()),
            "max_abs_state_diff": float(np.abs(states - old_states).max()),
            "reference": "rl_vla.env_adapter.ResidualChunkEnv(zero residual) + OpenPIBasePolicy + serve_baseline.EvaluationPolicy"}

        # 4. Action bounds / scaling / gripper on real base actions.
        spec = exp.spec
        rows = []
        for action in mine[:exp.execute_steps]:
            zero = spec.combine(action, np.zeros(spec.action_dim))
            plus = spec.combine(action, np.ones(spec.action_dim))
            minus = spec.combine(action, -np.ones(spec.action_dim))
            rows.append(bool(np.array_equal(zero.executed, np.clip(action, spec.low, spec.high))
                             and np.allclose(plus.correction, spec.scale) and np.allclose(minus.correction, -spec.scale)
                             and plus.executed[6] == np.clip(action[6], spec.low[6], spec.high[6])
                             and np.all(plus.executed <= spec.high) and np.all(minus.executed >= spec.low)
                             and np.array_equal(plus.clipped_high, action + spec.scale > spec.high)))
        checks["action_semantics_on_real_base_actions"] = {"passed": all(rows), "action_spec": spec.describe(),
                                                            "actor_action_dim": spec.action_dim}

        # 2c. Full base-only episodes reproduce the robustness queue's recorded outcomes.
        reproduction = []
        for initial_state_id in reproduce_states:
            ep = exp.evaluation_episode("train", initial_state_id)
            res = run_episode(sim, base, exp.spec, exp.builder, execute_steps=exp.execute_steps, gamma=exp.gamma,
                              make_request=exp.pro_make_request(), seed=ep["seed"], initial_state_id=initial_state_id,
                              episode_key=ep["noise_key"], actor=None)
            from pathlib import Path

            protocol = exp.case.get("frozen_protocol")
            published = exp.published_index(initial_state_id)
            path = (Path(protocol).parent / "episodes" / f"{exp.case['case_id']}_init{published:03d}.json"
                    if protocol and published is not None else None)
            saved = None
            if path is not None and path.exists():
                import json

                saved = json.loads(path.read_text())
            reproduction.append({"initial_state_id": initial_state_id, "split": next(
                (k for k in ("train", "validation", "test") if initial_state_id in split[k]), "unused"),
                "here": {"success": res["success"], "steps": res["steps"], "seed": ep["seed"]},
                "queue": None if saved is None else {"success": saved["success"], "steps": saved["primitive_steps"],
                                                     "seed": saved["seed"]},
                "match": None if saved is None else (saved["success"] == res["success"]
                                                     and saved["primitive_steps"] == res["steps"]
                                                     and saved["seed"] == ep["seed"])})
            print(f"reproduction init {initial_state_id}: {reproduction[-1]}", flush=True)
        checks["queue_episode_reproduction"] = {"passed": all(r["match"] for r in reproduction if r["match"] is not None)
                                                and any(r["match"] is not None for r in reproduction) if reproduction else None,
                                                "episodes": reproduction}
    finally:
        sim.close()
    # 3. Frozen weights and normalization statistics.
    after = {"parameters": base.parameter_fingerprint(), "norm_stats": base.norm_stats_fingerprint()}
    checks["base_frozen"] = {"passed": before == after, "before": before, "after": after}
    report["passed"] = all(c["passed"] for c in checks.values() if c["passed"] is not None)
    report["finished_unix_time"] = time.time()
    write_json(exp.run_dir / "logs" / f"{output_name}.json", report)
    return report
