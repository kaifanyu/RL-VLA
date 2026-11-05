"""Record a complete frozen native-JAX pi0.5 episode, one frame per action.

Run with the pinned OpenPI Python, PYTHONPATH=src, and OSMesa environment.
This uses the same transforms and sampler as the repository's RLDT baseline.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
from PIL import Image, ImageDraw

from rl_vla.rldt import load_config, make_flow
from rl_vla.rldt_env import make_primitive_env


def frame(raw, *, step, decision, policy_label="pi05_libero", outcome="running"):
    views = raw["openpi"]
    pixels = np.concatenate(
        [views["observation/image"], views["observation/wrist_image"]], axis=1
    )
    canvas = Image.new("RGB", (pixels.shape[1], pixels.shape[0] + 64), (32, 38, 46))
    canvas.paste(Image.fromarray(pixels), (0, 64))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f"Frozen {policy_label} | native JAX | OSMesa", fill="white")
    draw.text((6, 19), f"step {step} | decision {decision} | {outcome}", fill="white")
    prompt = views["prompt"]
    for row, offset in enumerate(range(0, min(len(prompt), 144), 72)):
        draw.text((6, 34 + row * 13), prompt[offset:offset + 72], fill="white")
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rldt_pi05_libero.toml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--policy", choices=("pi05_libero", "pi05_base"), default="pi05_libero",
                        help="Base uses unchanged general weights and LIBERO calibration statistics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    if args.policy == "pi05_base":
        config["policy"] = {
            "kind": "pi05", "model_config": "pi05_base",
            "checkpoint": "checkpoints/openpi/pi05_base",
            "normalization_assets": "checkpoints/openpi/pi05_libero/assets",
            "observation_action_adapter": "libero", "weights_updated": False,
        }
    config["env"]["kwargs"]["initial_state_id"] = args.initial_state
    # Keep the venv executable path: resolving its symlink bypasses pyvenv.cfg
    # and starts the bare interpreter without the simulator's packages.
    config["env"]["python"] = str(Path(config["env"]["python"]).absolute())
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    started = time.monotonic()
    print("Loading frozen native policy", flush=True)
    if args.policy == "pi05_base":
        from openpi.models.pi0_config import Pi0Config
        from openpi.policies.policy_config import create_trained_policy
        from openpi.training import checkpoints, config as openpi_config

        # Retain the released base horizon and discrete state input. Only the
        # robot observation/action mapping and calibration come from LIBERO.
        model_config = Pi0Config(pi05=True, action_horizon=50, action_dim=32,
                                 discrete_state_input=True)
        train_config = dataclasses.replace(openpi_config.get_config("pi05_libero"), model=model_config)
        assets = Path(config["policy"]["normalization_assets"])
        asset_id = "physical-intelligence/libero"
        stats_path = assets / asset_id / "norm_stats.json"
        policy = create_trained_policy(
            train_config, config["policy"]["checkpoint"],
            norm_stats=checkpoints.load_norm_stats(assets, asset_id),
            sample_kwargs={"num_steps": int(config["rldt"]["ode_steps"])},
        )
        policy_metadata = {
            "backend": "openpi-native-policy-infer", "model_config": "pi05_base",
            "horizon": 50, "action_dim": 32, "discrete_state_input": True,
            "observation_action_adapter": "libero", "use_quantile_norm": True,
            "normalization_path": str(stats_path.resolve()),
            "normalization_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "weight_origin": "Unchanged pi05_base; LIBERO calibration only; no task SFT weights",
        }

        def predict(raw, key):
            noise = np.asarray(jax.random.normal(key, (50, 32)))
            return np.asarray(policy.infer(raw["openpi"], noise=noise)["actions"])
    else:
        flow = make_flow(config["policy"])
        sample = jax.jit(lambda params, context, key, frozen: flow.sample(
            params, context, key, int(config["rldt"]["ode_steps"]), frozen=frozen
        )[0])
        policy_metadata = flow.metadata

        def predict(raw, key):
            normalized = sample(flow.params, flow.prepare([raw]), key, flow.frozen)
            return np.asarray(flow.decode(normalized, [raw]))[0]
    model_seconds = time.monotonic() - started
    env = None
    encoder = None
    actions_path = args.output / "actions.jsonl"
    steps = decisions = clipped = 0
    success = terminated = truncated = False
    inference_seconds = simulation_seconds = total_return = 0.0
    try:
        env = make_primitive_env(config["env"])
        raw, reset_info = env.reset(seed=args.seed)
        first = frame(raw, step=0, decision=0, policy_label=args.policy)
        Image.fromarray(first).save(args.output / "initial.png")
        video_partial = args.output / "baseline.partial.mp4"
        encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
            f"{first.shape[1]}x{first.shape[0]}", "-framerate", "20", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(video_partial),
        ], stdin=subprocess.PIPE)
        encoder.stdin.write(first.tobytes())
        key = jax.random.PRNGKey(args.seed)
        with actions_path.open("x") as log:
            while not (terminated or truncated):
                tick = time.monotonic()
                key, subkey = jax.random.split(key)
                decoded = predict(raw, subkey)
                inference_seconds += time.monotonic() - tick
                motor = decoded[:int(config["rldt"]["execute_steps"])]
                if motor.shape != (int(config["rldt"]["execute_steps"]), len(env.low)):
                    raise ValueError(f"Invalid action shape {motor.shape}")
                if not np.isfinite(motor).all():
                    raise ValueError("Nonfinite policy actions")
                decisions += 1
                for original in motor:
                    action = np.clip(original, env.low, env.high)
                    clipped += int(np.any(action != original))
                    tick = time.monotonic()
                    raw, reward, terminated, truncated, info = env.step(action)
                    simulation_seconds += time.monotonic() - tick
                    steps += 1
                    success |= bool(info.get("is_success", False))
                    total_return += float(reward)
                    outcome = "SUCCESS" if success else "timeout" if truncated else "running"
                    pixels = frame(raw, step=steps, decision=decisions,
                                   policy_label=args.policy, outcome=outcome)
                    encoder.stdin.write(pixels.tobytes())
                    log.write(json.dumps({"step": steps, "decision": decisions,
                        "decoded_action": original.tolist(), "applied_action": action.tolist(),
                        "reward": float(reward), "success": success,
                        "terminated": bool(terminated), "truncated": bool(truncated)}) + "\n")
                    log.flush()
                    if terminated or truncated:
                        break
                print(json.dumps({"steps": steps, "decisions": decisions,
                    "success": success, "inference_seconds": inference_seconds,
                    "simulation_seconds": simulation_seconds}), flush=True)
        encoder.stdin.close()
        if encoder.wait(timeout=60) != 0:
            raise RuntimeError("Video encoder failed")
        video_probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,width,height,avg_frame_rate:format=duration",
            "-of", "json", str(video_partial),
        ], text=True))
        stream = video_probe["streams"][0]
        if (int(stream["nb_read_frames"]) != steps + 1
                or (stream["width"], stream["height"]) != (first.shape[1], first.shape[0])
                or abs(float(video_probe["format"]["duration"]) - (steps + 1) / 20) > 0.1):
            raise RuntimeError(f"Encoded video does not match the executed trajectory: {video_probe}")
        video_partial.rename(args.output / "baseline.mp4")
        Image.fromarray(pixels).save(args.output / "final.png")
        result = {"baseline": True, "weights_updated": False,
            "checkpoint": str(Path(config["policy"]["checkpoint"]).resolve()),
            "policy_metadata": policy_metadata, "policy_variant": args.policy,
            "task": config["env"]["kwargs"],
            "prompt": raw["openpi"]["prompt"], "seed": args.seed, "reset": reset_info,
            "success": success, "terminated": bool(terminated), "truncated": bool(truncated),
            "controller_steps": steps, "policy_decisions": decisions,
            "clipped_action_steps": clipped, "return": total_return,
            "model_load_seconds": model_seconds, "inference_seconds": inference_seconds,
            "simulation_seconds": simulation_seconds, "elapsed_seconds": time.monotonic()-started,
            "devices": [str(device) for device in jax.devices()],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "renderer": "OSMesa/llvmpipe", "video": "baseline.mp4", "fps": 20,
            "video_frames": steps + 1, "video_probe": video_probe,
            "scope": "One diagnostic episode; not a benchmark success rate"}
        (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        try:
            if encoder is not None and encoder.poll() is None:
                if not encoder.stdin.closed:
                    try:
                        encoder.stdin.close()
                    except BrokenPipeError:
                        pass
                try:
                    encoder.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
