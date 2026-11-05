"""Record one real frozen native pi05_base episode at every controller step.

Run with the OpenPI interpreter and multibench/core before src on PYTHONPATH.
The separate ManiSkill interpreter owns simulation and Vulkan rendering.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def json_value(value):
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/openpi/pi05_base")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--policy-seed", type=int, default=100000)
    parser.add_argument("--render-backend", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "baseline": True,
        "adaptation_updates": 0,
        "task": "PutOnPlateInScene25Main-v3",
        "benchmark": "ManiSkill3",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_format": "native OpenPI JAX",
        "checkpoint_role": "Original pi05_base, with task calibration statistics only; no task SFT weights",
        "seed": args.seed,
        "policy_seed": args.policy_seed,
        "sim_backend": "cpu",
        "render_backend": args.render_backend,
        "control_frequency": 5,
        "native_horizon": 80,
        "prediction_horizon": 50,
        "execute_steps": 5,
        "ode_steps": 10,
        "video": "baseline.mp4",
        "video_fps": 5,
        "video_sampling": "reset and every controller step",
        "started_utc": datetime.now(UTC).isoformat(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "evaluation_note": "One diagnostic episode, not a success-rate benchmark; seed chosen before observing policy outcome.",
    }
    write_json(args.output / "report.json", report)
    env, encoder, encoder_log = None, None, None
    frames, steps, decisions, success = 0, 0, 0, False
    inference_seconds, simulator_seconds = 0.0, 0.0
    clip_count, action_coordinates = 0, 0
    executed_actions, unclipped_actions, states = [], [], []
    started = time.monotonic()
    try:
        from multibench.evaluate import NativePolicy

        from rl_vla.rldt_env import ProcessEnv

        config = {
            "policy": {
                "kind": "pi05",
                "benchmark": "maniskill",
                "checkpoint": str(args.checkpoint.resolve()),
            },
            "rldt": {"execute_steps": 5, "ode_steps": 10},
        }
        policy_start = time.monotonic()
        policy = NativePolicy(config)
        report["model_load_seconds"] = time.monotonic() - policy_start
        report["model_metadata"] = policy.metadata
        policy.reset(args.policy_seed)
        simulator_start = time.monotonic()
        worker_env = {
            "PYTHONPATH": os.pathsep.join(
                map(
                    str,
                    [
                        ROOT / "integrations/multibench/core",
                        ROOT / "integrations",
                        ROOT / "third_party/rlinf",
                        ROOT / "third_party/openpi/packages/openpi-client/src",
                    ],
                )
            ),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
            "LP_NUM_THREADS": "1",
        }
        if args.render_backend == "cpu":
            icd = ROOT / ".cache/native-maniskill/lvp_icd.json"
            if not icd.is_file():
                raise FileNotFoundError(
                    "Run integrations/multibench/maniskill/prepare_software_vulkan.py first"
                )
            worker_env.update(
                VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd), LP_NUM_THREADS="8"
            )
        else:
            # SAPIEN 3.0.3 only discovers /usr/share automatically; this Docker
            # image exposes its real NVIDIA ICD under /etc/vulkan instead.
            for icd in [
                Path("/etc/vulkan/icd.d/nvidia_icd.json"),
                Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
            ]:
                if icd.is_file():
                    worker_env.update(VK_DRIVER_FILES=str(icd), VK_ICD_FILENAMES=str(icd))
                    break
        env = ProcessEnv(
            {
                "kind": "external",
                "factory": "multibench.maniskill.environment:make_env",
                "python": str(ROOT / ".venv-maniskill3/bin/python"),
                "gpu_rendering": args.render_backend == "gpu",
                "process_env": worker_env,
                "kwargs": {
                    "sim_backend": "cpu",
                    "render_backend": args.render_backend,
                    "max_episode_steps": 80,
                },
            }
        )
        raw, reset_info = env.reset(seed=args.seed)
        report["simulator_init_and_reset_seconds"] = time.monotonic() - simulator_start
        report["reset"] = reset_info
        report["instruction"] = raw["openpi"]["prompt"]
        states.append(raw["state"].copy())
        image = np.asarray(raw["openpi"]["observation/image"])
        assert image.shape == (480, 640, 3) and image.dtype == np.uint8 and image.std() > 5
        from PIL import Image

        Image.fromarray(image).save(args.output / "initial.png")
        encoder_log = (args.output / "ffmpeg.log").open("w")
        encoder = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "640x480",
                "-r",
                "5",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(args.output / "baseline.mp4"),
            ],
            stdin=subprocess.PIPE,
            stderr=encoder_log,
        )
        encoder.stdin.write(np.ascontiguousarray(image).tobytes())
        frames += 1
        terminated = truncated = False
        with (args.output / "steps.jsonl").open("w") as trace:
            while not (terminated or truncated):
                t = time.monotonic()
                normalized, decoded, motor, sampling = policy.predict(raw)
                elapsed = time.monotonic() - t
                inference_seconds += elapsed
                assert motor.shape == (5, 7)
                assert all(np.isfinite(x).all() for x in [normalized, decoded, motor])
                for action in motor:
                    proposed = action.copy()
                    clipped = np.clip(proposed, env.low, env.high)
                    changed = int(np.count_nonzero(clipped != proposed))
                    clip_count += changed
                    action_coordinates += len(clipped)
                    t = time.monotonic()
                    raw, reward, terminated, truncated, info = env.step(clipped)
                    step_time = time.monotonic() - t
                    simulator_seconds += step_time
                    steps += 1
                    success |= bool(info.get("is_success", False))
                    assert np.isfinite(raw["state"]).all() and np.isfinite(reward)
                    image = np.asarray(raw["openpi"]["observation/image"])
                    encoder.stdin.write(np.ascontiguousarray(image).tobytes())
                    frames += 1
                    executed_actions.append(np.asarray(info.get("executed_action", clipped)).copy())
                    unclipped_actions.append(proposed)
                    states.append(raw["state"].copy())
                    row = {
                        "step": steps,
                        "decision": decisions,
                        "reward": reward,
                        "success": info.get("is_success", False),
                        "terminated": terminated,
                        "truncated": truncated,
                        "simulation_seconds": step_time,
                        "clipped_coordinates": changed,
                        "proposed_action": proposed,
                        "executed_action": executed_actions[-1],
                        "sampling": sampling,
                    }
                    trace.write(json.dumps(json_value(row), allow_nan=False) + "\n")
                    trace.flush()
                    if terminated or truncated:
                        break
                    if steps >= 80:
                        raise RuntimeError(
                            "Simulator did not truncate at its native 80-step horizon"
                        )
                decisions += 1
                print(
                    json.dumps(
                        {
                            "steps": steps,
                            "decisions": decisions,
                            "success": success,
                            "inference_seconds": elapsed,
                            "simulator_seconds_total": simulator_seconds,
                        }
                    ),
                    flush=True,
                )
        Image.fromarray(image).save(args.output / "final.png")
        encoder.stdin.close()
        if encoder.wait(timeout=60) != 0:
            raise RuntimeError("FFmpeg failed; inspect ffmpeg.log")
        encoder = None
        encoder_log.close()
        np.savez_compressed(
            args.output / "actions_states.npz",
            proposed=np.asarray(unclipped_actions),
            executed=np.asarray(executed_actions),
            state=np.asarray(states),
        )
        video_probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,nb_read_frames,r_frame_rate:format=duration",
                    "-of",
                    "json",
                    str(args.output / "baseline.mp4"),
                ],
                text=True,
            )
        )
        assert int(video_probe["streams"][0]["nb_read_frames"]) == frames
        report.update(
            complete=True,
            success=success,
            terminated=terminated,
            truncated=truncated,
            controller_steps=steps,
            policy_decisions=decisions,
            video_frames=frames,
            inference_seconds=inference_seconds,
            simulation_seconds=simulator_seconds,
            clipped_action_coordinates=clip_count,
            total_action_coordinates=action_coordinates,
            elapsed_seconds=time.monotonic() - started,
            final_info=info,
            video_probe=video_probe,
        )
        write_json(args.output / "report.json", report)
        print(json.dumps(json_value(report), indent=2), flush=True)
    except Exception:
        report.update(
            error=traceback.format_exc(),
            controller_steps=steps,
            policy_decisions=decisions,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(args.output / "report.json", report)
        raise
    finally:
        try:
            if encoder is not None:
                try:
                    if not encoder.stdin.closed:
                        encoder.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    encoder.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    encoder.terminate()
                    try:
                        encoder.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        encoder.kill()
                        encoder.wait(timeout=5)
        finally:
            if encoder_log is not None:
                encoder_log.close()
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
