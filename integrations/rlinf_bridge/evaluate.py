"""Standalone, paired ODE evaluation of the official RLinf pi-RL models.

Run with the RLinf Python environment and PYTHONPATH including integrations,
src, and third_party/rlinf. The simulator runs in the separate .venv-pro child.
"""

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from rl_vla.rldt_env import ProcessEnv
from rl_vla.trajectory_recording import TrajectoryRecorder
from rlinf_bridge.environment import ControlledProEnv


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def resolve_checkpoint(path):
    """Recognize actual upstream full-model and distributed checkpoint layouts."""
    path = Path(path).resolve()
    if path.is_file():
        return "weights", path
    for relative in ("model_state_dict/full_weights.pt", "actor/model_state_dict/full_weights.pt",
                     "full_weights.pt", "model.pt", "actor/model.pt"):
        candidate = path / relative
        if candidate.is_file():
            return "weights", candidate
    for candidate in (path / "dcp_checkpoint", path / "actor/dcp_checkpoint", path):
        if (candidate / ".metadata").is_file():
            return "dcp", candidate
    raise FileNotFoundError(
        f"No full_weights.pt/model.pt or DCP model found at {path}. "
        "Use actor.fsdp_config.save_full_model_weights=true; local_shard checkpoints "
        "alone are not accepted as full unsharded models.")


def load_checkpoint(model, path):
    """Strictly load policy/value/noise parameters; never load optimizer pickle."""
    kind, resolved = resolve_checkpoint(path)
    if kind == "weights":
        state = torch.load(resolved, map_location="cpu", weights_only=True, mmap=True)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
            raise ValueError("Expected a tensor-only full model state dictionary")
        model.load_state_dict(state, strict=True)
        return {"format": kind, "path": str(resolved), "sha256": _sha256(resolved),
                "bytes": resolved.stat().st_size}

    from torch.distributed import checkpoint as dcp

    # The pinned upstream Checkpoint.state_dict saves this exact nested model
    # subtree. Request only model tensors, without optimizer or RNG objects.
    state = {"fsdp_checkpoint": {"model": model.state_dict()}}
    dcp.load(state_dict=state, checkpoint_id=str(resolved))
    model.load_state_dict(state["fsdp_checkpoint"]["model"], strict=True)
    return {"format": kind, "path": str(resolved),
            "metadata_sha256": _sha256(resolved / ".metadata"),
            "files": [{"name": file.name, "bytes": file.stat().st_size}
                      for file in sorted(resolved.iterdir()) if file.is_file()]}


def seed_episode(seed):
    """Reset every sampling RNG independently of preceding episode lengths."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(config, checkpoint=None, device="cuda"):
    from omegaconf import OmegaConf
    from pi_rl.model_configs import assert_native_model_settings
    from pi_rl.model_contract import enforce_trainable_scope
    from rlinf.models.embodiment.openpi import get_model

    model_config = copy.deepcopy(config["actor"]["model"])
    converted = Path(model_config["model_path"]) / "conversion_manifest.json"
    if not converted.is_file():
        raise FileNotFoundError(f"Missing native-checkpoint conversion provenance: {converted}")
    seed_episode(0)
    print("Loading converted pi05_libero using the official RLinf model...", flush=True)
    model = get_model(OmegaConf.create(model_config))
    assert_native_model_settings(model.config)
    counts = enforce_trainable_scope(model)
    provenance = load_checkpoint(model, checkpoint) if checkpoint else None
    model.eval().to(device)
    return model, {
        "parameter_counts": counts, "conversion_manifest": json.loads(converted.read_text()),
        "conversion_manifest_sha256": _sha256(converted), "loaded_checkpoint": provenance,
        "torch_version": torch.__version__, "device": str(device),
    }


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def predict_chunk(model, raw):
    """Official ODE inference plus full normalized/decoded chunks for recording."""
    observation = ControlledProEnv._wrap_obs(raw) | {"extra_view_images": None}
    with torch.inference_mode():
        motor, outputs = model.predict_action_batch(observation, mode="eval", compute_values=False)
    motor = _as_numpy(motor)[0]
    shape = (int(model.config.action_horizon), int(model.config.action_dim))
    normalized = _as_numpy(outputs["forward_inputs"]["model_action"])[0].reshape(shape)
    # Upstream output_transform returns only the execution prefix. Its underlying
    # transforms can also decode the complete saved normalized chunk for SFT.
    processed = model._input_transform(copy.deepcopy(raw["openpi"]))
    decoded = np.asarray(model._output_transform({
        "state": processed["state"], "actions": normalized.copy(),
    })["actions"], np.float32)
    if motor.shape != (int(model.config.action_chunk), 7):
        raise ValueError(f"Unexpected decoded execution prefix: {motor.shape}")
    if not np.allclose(decoded[:len(motor)], motor, rtol=1e-6, atol=1e-6):
        raise ValueError("Full-chunk decoding disagrees with official executed-action decoding")
    if not np.isfinite(normalized).all() or not np.isfinite(decoded).all():
        raise FloatingPointError("Nonfinite model actions")
    initial_noise = _as_numpy(outputs["forward_inputs"]["chains"])[0, 0]
    return normalized, decoded, motor, initial_noise


class PrimitiveVideo:
    """Stream every processed camera frame; no full-episode video buffer."""

    def __init__(self, path):
        import imageio.v2 as imageio

        self.writer = imageio.get_writer(str(path), fps=20, codec="libx264", quality=8,
                                         macro_block_size=None)

    def append(self, raw):
        cameras = raw["openpi"]
        self.writer.append_data(np.concatenate((cameras["observation/image"],
                                               cameras["observation/wrist_image"]), axis=1))

    def close(self):
        self.writer.close()


def evaluate(config, protocol, task_key, split, output, *, checkpoint=None, episodes=50,
             seed=100000, device="cuda", model_loader=load_model, environment_factory=ProcessEnv,
             video_factory=PrimitiveVideo):
    """Save the complete paired evaluation, with model/checkpoint outside its config."""
    if split not in {"validation", "test", "source"}:
        raise ValueError("Evaluation split must be validation, test, or source")
    protocol = Path(protocol).resolve()
    plan = json.loads(protocol.read_text())
    task = plan["tasks"][task_key]
    state_ids = task[f"{split}_state_ids"]
    if not 1 <= episodes <= len(state_ids):
        raise ValueError("Requested episodes must fit the frozen split")
    path = Path(output).resolve()
    trajectory_path = path.with_suffix(".trajectories")
    video_path = path.with_suffix(".videos")
    episode_path = path.with_suffix(".episodes.jsonl")
    if any(candidate.exists() for candidate in (path, trajectory_path, video_path, episode_path)):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = copy.deepcopy(config["actor"]["model"])
    execution_steps = int(model_config["openpi"]["action_chunk"])
    environment_config = config["env"].get("eval", config["env"].get("train", {}))
    simulator_python = environment_config.get("simulator_python")
    if not simulator_python:
        raise ValueError("env.eval.simulator_python or env.train.simulator_python is required")
    env_spec = {"kind": "external", "factory": "rl_vla.envs.libero_pro:make_env",
                "python": str(simulator_python),
                "kwargs": {"protocol": str(protocol), "task_key": task_key, "split": split}}
    report_config = {
        "algorithm": "pi_rl", "policy": model_config, "env": env_spec,
        "method": model_config["openpi"]["noise_method"],
        # Existing paired-analysis helper checks policy/env/rldt. Null explicitly
        # identifies that this is not an RLDT learner; policy contains the ODE settings.
        "rldt": None,
        "inference": {"mode": "eval", "sampler": "flow_ode", "execute_steps": execution_steps,
                      "episode_seed": seed, "rng_backend": "pytorch",
                      "rng_note": "Seeds match across Torch baseline and Torch-trained models. "
                      "They do not produce identical noise to the JAX RLDT backend."},
    }
    model, model_metadata = model_loader(config, checkpoint, device)
    env = environment_factory(env_spec)
    recorder = None
    video = None
    reason = "interrupted"
    results = []
    started = time.monotonic()
    try:
        recorder = TrajectoryRecorder(trajectory_path, {
            "config": report_config, "checkpoint": str(checkpoint) if checkpoint else None,
            "model_metadata": model_metadata, "protocol_sha256": _sha256(protocol),
            "phase": "evaluation", "episodes": episodes,
        })
        video_path.mkdir()
        with episode_path.open("x") as log:
            for episode in range(episodes):
                episode_seed = seed + episode
                seed_episode(episode_seed)
                raw, reset_info = env.reset(seed=episode_seed)
                if reset_info["initial_state_id"] != state_ids[episode]:
                    raise ValueError("Simulator reset does not match frozen split order")
                # Environment reset runs in a child; reseeding here also makes the
                # policy independent of any future in-process factory implementation.
                seed_episode(episode_seed)
                recorder.start_episode(raw, {"seed": episode_seed, "reset": reset_info})
                filename = video_path / f"episode_{episode + 1:03d}.mp4"
                video = video_factory(filename)
                video.append(raw)
                steps, total, decision = 0, 0., 0
                terminated = truncated = False
                print(f"pi-RL {split} episode {episode + 1}/{episodes}, seed {episode_seed}",
                      flush=True)
                while not (terminated or truncated):
                    normalized, decoded, motor, initial_noise = predict_chunk(model, raw)
                    if len(motor) != execution_steps:
                        raise ValueError("Loaded model execution length differs from evaluation config")
                    recorder.begin_chunk(raw, normalized, decoded, {
                        "decision": decision, "planned_execute_steps": execution_steps,
                        "episode_seed": episode_seed, "initial_noise": initial_noise.tolist(),
                    })
                    chunk_return = 0.
                    for action in motor:
                        command = np.clip(action, env.low, env.high)
                        raw, reward, terminated, truncated, info = env.step(command)
                        if "final_observation" in info or "final_obs" in info:
                            raise ValueError("Primitive evaluation environments must not autoreset")
                        if not np.isfinite(reward):
                            raise FloatingPointError("Nonfinite evaluation reward")
                        recorder.record_step(command, raw, reward, terminated, truncated, info)
                        video.append(raw)
                        steps += 1
                        total += reward
                        chunk_return += reward
                        if terminated or truncated:
                            break
                    recorder.end_chunk({"raw_return": chunk_return, "info": info})
                    decision += 1
                    if steps >= int(task["max_steps"]) and not (terminated or truncated):
                        raise RuntimeError("Environment exceeded the frozen episode horizon")
                video.close()
                video = None
                result = {"seed": episode_seed, "reset": reset_info,
                          "success": bool(info.get("is_success", False)), "return": float(total),
                          "primitive_steps": steps, "video": str(filename),
                          "video_note": "Every controller-step front/wrist policy image, 20 fps; "
                          "includes reset and final frame. Lossless images also in trajectory NPZs."}
                result["trajectory"] = recorder.end_episode(result)
                results.append(result)
                log.write(json.dumps(result, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(result), flush=True)
        report = {"config": report_config, "checkpoint": str(checkpoint) if checkpoint else None,
                  "episodes": results, "success_rate": float(np.mean([r["success"] for r in results])),
                  "model_metadata": model_metadata, "complete_split": episodes == len(state_ids),
                  "protocol_sha256": _sha256(protocol), "elapsed_seconds": time.monotonic() - started}
        _write_json(path, report)
        reason = "evaluation_complete"
        return report
    finally:
        try:
            if video is not None:
                video.close()
            if recorder is not None:
                recorder.close(reason=reason)
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--split", choices=["validation", "test", "source"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    torch.set_num_threads(1)
    report = evaluate(config, args.protocol, args.task_key, args.split, args.output,
                      checkpoint=args.checkpoint, episodes=args.episodes, seed=args.seed,
                      device=args.device)
    print(json.dumps({"output": args.output, "success_rate": report["success_rate"],
                      "episodes": len(report["episodes"])}), flush=True)


if __name__ == "__main__":
    main()
