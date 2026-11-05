"""Record one RoboCasa365 episode using frozen pretrained native pi05_base weights.

Run in the pinned OpenPI environment with integrations/multibench/core and
integrations on PYTHONPATH. The isolated simulator uses CPU OSMesa rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
KEYS = ("observation/image", "observation/wrist_image", "observation/right_wrist_image")


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frame(raw, step, success):
    images = [np.asarray(raw["openpi"][key]) for key in KEYS]
    assert all(im.shape == (224, 224, 3) and im.dtype == np.uint8 for im in images)
    canvas = Image.new("RGB", (672, 288), (20, 24, 31))
    for i, im in enumerate(images):
        canvas.paste(Image.fromarray(im), (224 * i, 64))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), "RoboCasa365 | frozen pi05_base | CPU OSMesa | 20 Hz", fill="white")
    draw.text((8, 22), raw["openpi"]["prompt"], fill="white")
    draw.text((8, 41), f"Controller step {step}/750 | success={success}", fill="white")
    for i, name in enumerate(("left agent view", "wrist", "right agent view")):
        draw.text((224 * i + 6, 70), name, fill=(255, 255, 0))
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/openpi/pi05_base")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=100000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    video_path = args.output / "baseline.mp4"
    report_path = args.output / "rollout.json"
    if video_path.exists() or report_path.exists():
        raise FileExistsError("Use a new output directory for every rollout")
    from multibench.evaluate import NativePolicy

    from rl_vla.rldt_env import ProcessEnv

    config = {
        "policy": {
            "kind": "pi05",
            "benchmark": "robocasa",
            "checkpoint": str(args.checkpoint.resolve()),
            "projection_only": False,
        },
        "rldt": {"execute_steps": 5, "ode_steps": 10},
    }
    t0 = time.monotonic()
    policy = NativePolicy(config)
    model_seconds = time.monotonic() - t0
    policy.reset(args.policy_seed)
    env_paths = [
        ROOT / "integrations/multibench/core",
        ROOT / "integrations",
        ROOT / "third_party/openpi/packages/openpi-client/src",
    ]
    env = ProcessEnv(
        {
            "kind": "external",
            "factory": "multibench.robocasa.environment:make_env",
            "python": str(ROOT / ".venv-robocasa365/bin/python"),
            "gpu_rendering": False,
            "process_env": {
                "PYTHONPATH": os.pathsep.join(map(str, env_paths)),
                "MUJOCO_GL": "osmesa",
                "PYOPENGL_PLATFORM": "osmesa",
                "LP_NUM_THREADS": os.getenv("ROBOCASA_RENDER_THREADS", "4"),
                "NUMBA_DISABLE_JIT": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMBA_NUM_THREADS": "1",
            },
            "kwargs": {"split": "target", "seed": args.seed},
        }
    )
    writer = None
    steps = decisions = clipped_values = 0
    inference_seconds = simulation_seconds = 0.0
    success = terminated = truncated = False
    actions, states = [], []
    try:
        t = time.monotonic()
        raw, reset_info = env.reset(seed=args.seed)
        reset_seconds = time.monotonic() - t
        if reset_info["is_success"]:
            raise RuntimeError("Already successful at reset")
        for key in KEYS:
            if np.asarray(raw["openpi"][key]).std() < 1:
                raise RuntimeError(f"Blank camera {key}")
        first_frame = frame(raw, 0, False)
        Image.fromarray(first_frame).save(args.output / "initial.png")
        log = (args.output / "ffmpeg.log").open("w")
        writer = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "672x288",
                "-r",
                "20",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-preset",
                "fast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_path),
            ],
            stdin=subprocess.PIPE,
            stderr=log,
        )
        writer.stdin.write(first_frame.tobytes())
        with (args.output / "decisions.jsonl").open("x") as decisions_log:
            while not (terminated or truncated):
                t = time.monotonic()
                normalized, _decoded, motor, sampling = policy.predict(raw)
                infer_time = time.monotonic() - t
                inference_seconds += infer_time
                if np.asarray(motor).shape != (5, 12) or not np.isfinite(motor).all():
                    raise RuntimeError("Nonfinite or incorrectly shaped native actions")
                for action in motor:
                    clipped = np.clip(action, env.low, env.high).astype(np.float32)
                    clipped_values += int(np.count_nonzero(clipped != action))
                    t = time.monotonic()
                    raw, reward, terminated, truncated, info = env.step(clipped)
                    simulation_seconds += time.monotonic() - t
                    if float(reward) != float(info["is_success"]):
                        raise RuntimeError("Reward does not match native success")
                    success |= bool(info["is_success"])
                    steps += 1
                    actions.append(clipped)
                    states.append(raw["state"])
                    writer.stdin.write(frame(raw, steps, success).tobytes())
                    if terminated or truncated:
                        break
                    if steps >= 750:
                        raise RuntimeError("Native horizon was not enforced")
                decisions += 1
                record = {
                    "decision": decisions,
                    "steps": steps,
                    "inference_seconds": infer_time,
                    "success": success,
                    "sampling": sampling,
                    "normalized_min": float(np.min(normalized)),
                    "normalized_max": float(np.max(normalized)),
                    "decoded_motor_min": float(np.min(motor)),
                    "decoded_motor_max": float(np.max(motor)),
                }
                decisions_log.write(json.dumps(record) + "\n")
                decisions_log.flush()
                if decisions % 10 == 0:
                    print(json.dumps(record), flush=True)
        writer.stdin.close()
        if writer.wait(timeout=60):
            raise RuntimeError("ffmpeg failed; inspect ffmpeg.log")
        log.close()
        Image.fromarray(frame(raw, steps, success)).save(args.output / "final.png")
        np.savez_compressed(
            args.output / "actions_states.npz",
            actions=np.asarray(actions),
            states=np.asarray(states),
        )
        report = {
            "purpose": "Single baseline rollout; not a benchmark success-rate estimate",
            "baseline": True,
            "benchmark": "robocasa365",
            "task": "PickPlaceCounterToCabinet",
            "checkpoint": str(args.checkpoint.resolve()),
            "model_metadata": policy.metadata,
            "normalization": "Official RoboCasa mean/std metadata; no task SFT weights",
            "training_updates": 0,
            "config": config,
            "environment_seed": args.seed,
            "policy_seed": args.policy_seed,
            "reset": reset_info,
            "final_info": info,
            "success": success,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "controller_steps": steps,
            "policy_decisions": decisions,
            "action_values_clipped": clipped_values,
            "total_action_values": steps * 12,
            "action_clipped_fraction": clipped_values / (steps * 12),
            "source_revisions": {
                name: subprocess.check_output(
                    ["git", "-C", str(ROOT / path), "rev-parse", "HEAD"], text=True
                ).strip()
                for name, path in {
                    "openpi": "third_party/openpi",
                    "robocasa": "third_party/robocasa365",
                    "robosuite": "third_party/robocasa365/dependencies/robosuite",
                    "robocasa_openpi_reference": "third_party/robocasa365/dependencies/openpi_reference",
                }.items()
            },
            "jax_devices": [str(device) for device in policy.jax.devices()],
            "simulator_python": str(ROOT / ".venv-robocasa365/bin/python"),
            "ipc_client": "Pinned OpenPI client source on PYTHONPATH, tested against simulator NumPy 2",
            "renderer": "CPU OSMesa/llvmpipe",
            "render_threads": int(os.getenv("ROBOCASA_RENDER_THREADS", "4")),
            "fps": 20,
            "frames": steps + 1,
            "video": str(video_path.resolve()),
            "video_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            "model_load_seconds": model_seconds,
            "reset_seconds": reset_seconds,
            "inference_seconds": inference_seconds,
            "simulation_seconds": simulation_seconds,
            "elapsed_seconds": time.monotonic() - t0,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "complete": True,
        }
        json_write(report_path, report)
        print(
            json.dumps({"report": str(report_path), "steps": steps, "success": success}), flush=True
        )
    except BaseException as exc:
        json_write(
            args.output / "failure.json",
            {
                "error": repr(exc),
                "steps": steps,
                "decisions": decisions,
                "elapsed_seconds": time.monotonic() - t0,
                "complete": False,
            },
        )
        raise
    finally:
        if writer is not None and writer.poll() is None:
            writer.stdin.close()
            writer.wait(timeout=60)
        env.close()


if __name__ == "__main__":
    main()
