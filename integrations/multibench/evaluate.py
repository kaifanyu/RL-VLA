"""Paired primitive-step evaluation for the isolated three-benchmark campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class NativePolicy:
    def __init__(self, config, checkpoint=None):
        import jax
        from rl_vla.rldt import make_flow
        self.jax = jax
        self.flow = make_flow(config["policy"])
        self.params = self.flow.load_params(checkpoint) if checkpoint else self.flow.params
        self.execute_steps = int(config["rldt"]["execute_steps"])
        self.sample = jax.jit(lambda params, context, key, frozen: self.flow.sample(
            params, context, key, int(config["rldt"]["ode_steps"]), frozen=frozen)[0])
        self.metadata = getattr(self.flow, "metadata", {})

    def reset(self, seed):
        self.key = self.jax.random.PRNGKey(seed)

    def predict(self, raw):
        self.key, subkey = self.jax.random.split(self.key)
        actions = self.sample(self.params, self.flow.prepare([raw]), subkey, self.flow.frozen)
        decoded = np.asarray(self.flow.decode(actions, [raw]))[0]
        return np.asarray(actions)[0], decoded, decoded[:self.execute_steps], {"sampling_key": np.asarray(subkey).tolist()}


class TorchPolicy:
    def __init__(self, config, checkpoint=None):
        from multibench.torch_model import load_model, predict_chunk
        self.model, self.metadata = load_model(config, checkpoint=checkpoint, device="cuda")
        self.predict_chunk = predict_chunk
        self.execute_steps = int(self.model.config.action_chunk)

    def reset(self, seed):
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def predict(self, raw):
        normalized, decoded, motor, noise = self.predict_chunk(self.model, raw)
        return normalized, decoded, motor, {"initial_noise": np.asarray(noise).tolist()}


def evaluate(config, protocol, benchmark, split, output, *, method, checkpoint=None,
             episodes=50, seed=100000, record=False, policy_factory=None, environment_factory=None):
    if method not in {"rldt", "flow_sde"} or split not in {"validation", "test"}:
        raise ValueError("Unsupported method or evaluation split")
    plan = json.loads(Path(protocol).read_text()) if not isinstance(protocol, dict) else protocol
    definition = plan["tasks"][benchmark]
    ids, reset_seeds = definition[f"{split}_state_ids"], definition[f"{split}_seeds"]
    if not 1 <= episodes <= len(ids) or len(ids) != len(reset_seeds) or len(set(ids)) != len(ids):
        raise ValueError("Evaluation must use unique states from its frozen split")
    path = Path(output)
    log_path, trajectory_path = path.with_suffix(".episodes.jsonl"), path.with_suffix(".trajectories")
    if any(p.exists() for p in [path, log_path, trajectory_path]):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    policy = (policy_factory or (NativePolicy if method == "rldt" else TorchPolicy))(config, checkpoint)
    model_seconds = time.monotonic() - start
    if environment_factory is None:
        from multibench.environment import process_config
        from rl_vla.rldt_env import ProcessEnv
        environment_factory = lambda: ProcessEnv(process_config(benchmark, protocol, split))
    before_env = time.monotonic()
    env = environment_factory()
    environment_seconds = time.monotonic() - before_env
    recorder = None
    reason, rows = "interrupted", []
    try:
        if record:
            from rl_vla.trajectory_recording import TrajectoryRecorder
            recorder = TrajectoryRecorder(trajectory_path, {
                "phase": "evaluation", "benchmark": benchmark, "method": method,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "model_metadata": policy.metadata, "native_camera_semantics": True})
        with log_path.open("x") as log:
            for episode in range(episodes):
                episode_start = time.monotonic()
                policy_seed = seed + episode
                policy.reset(policy_seed)
                raw, reset_info = env.reset(seed=policy_seed)
                if reset_info["initial_state_id"] != ids[episode]:
                    raise ValueError("Simulator did not enumerate the frozen evaluation state IDs")
                policy.reset(policy_seed)
                if recorder:
                    recorder.start_episode(raw, {"seed": policy_seed, "reset": reset_info})
                steps, decisions, total = 0, 0, 0.
                success, terminated, truncated = False, False, False
                inference_seconds = simulation_seconds = 0.
                while not (terminated or truncated):
                    t = time.monotonic()
                    normalized, decoded, motor, sampling = policy.predict(raw)
                    inference_seconds += time.monotonic() - t
                    normalized, decoded, motor = map(np.asarray, (normalized, decoded, motor))
                    if motor.shape != (policy.execute_steps, len(env.low)) or not all(np.isfinite(a).all() for a in [normalized, decoded, motor]):
                        raise ValueError("Nonfinite actions or incompatible action dimensions")
                    if recorder:
                        recorder.begin_chunk(raw, normalized, decoded, {"decision": decisions, **sampling})
                    chunk_return = 0.
                    for action in motor:
                        action = np.clip(action, env.low, env.high)
                        t = time.monotonic()
                        raw, reward, terminated, truncated, info = env.step(action)
                        simulation_seconds += time.monotonic() - t
                        if "final_observation" in info or "final_obs" in info:
                            raise ValueError("Evaluation environments must not autoreset")
                        if not np.isfinite(reward):
                            raise ValueError("Nonfinite evaluation reward")
                        success |= bool(info.get("is_success", False))
                        steps += 1
                        total += float(reward)
                        chunk_return += float(reward)
                        if recorder:
                            recorder.record_step(action, raw, reward, terminated, truncated, info)
                        if terminated or truncated:
                            break
                        if steps >= int(definition["max_steps"]):
                            raise RuntimeError("Simulator failed to enforce the frozen horizon")
                    if recorder:
                        recorder.end_chunk({"raw_return": chunk_return, "info": info})
                    decisions += 1
                row = {"seed": policy_seed, "environment_seed": reset_seeds[episode], "reset": reset_info,
                       "success": success, "return": total, "primitive_steps": steps,
                       "controller_steps": steps, "policy_decisions": decisions,
                       "terminated": bool(terminated), "truncated": bool(truncated),
                       "elapsed_seconds": time.monotonic() - episode_start,
                       "inference_seconds": inference_seconds, "simulation_seconds": simulation_seconds}
                if recorder:
                    row["trajectory"] = recorder.end_episode(row)
                rows.append(row)
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
        report = {"config": config, "benchmark": benchmark, "method": method,
                  "split": split, "checkpoint": str(checkpoint) if checkpoint else None,
                  "baseline": checkpoint is None, "episodes": rows,
                  "successes": sum(r["success"] for r in rows),
                  "success_rate": float(np.mean([r["success"] for r in rows])),
                  "controller_steps": sum(r["controller_steps"] for r in rows),
                  "policy_decisions": sum(r["policy_decisions"] for r in rows),
                  "elapsed_seconds": time.monotonic() - start,
                  "model_load_seconds": model_seconds, "environment_init_seconds": environment_seconds,
                  "episode_seconds": sum(r["elapsed_seconds"] for r in rows),
                  "model_metadata": policy.metadata, "complete_split": episodes == len(ids),
                  "sampler": "flow_ode", "sampling_backend": "jax" if method == "rldt" else "torch",
                  "sampling_note": "Paired environment states and per-backend RNG seeds; JAX and Torch noise differs."}
        write_json(path, report)
        reason = "evaluation_complete"
        return report
    finally:
        try:
            if recorder:
                recorder.close(reason=reason)
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["rldt", "flow_sde"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--benchmark", choices=["robotwin", "robocasa", "maniskill"], required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.method == "rldt":
        from rl_vla.rldt import load_config
        config = load_config(args.config)
    else:
        from omegaconf import OmegaConf
        config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    report = evaluate(config, args.protocol, args.benchmark, args.split, args.output,
                      method=args.method, checkpoint=args.checkpoint, episodes=args.episodes,
                      seed=args.seed, record=args.record)
    print(json.dumps({"output": args.output, "success_rate": report["success_rate"]}), flush=True)


if __name__ == "__main__":
    main()
