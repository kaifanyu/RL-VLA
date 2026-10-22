"""Repeatable, incremental LIBERO evaluation of a served frozen VLA checkpoint."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from rl_vla.base_policy import OpenPIBasePolicy
from rl_vla.config import load_config
from rl_vla.env_adapter import ResidualChunkEnv
from rl_vla.envs.libero import make_env
from rl_vla.evaluate import wilson_interval
from rl_vla.runtime import seed_everything

HORIZONS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
            "libero_10": 520, "libero_90": 400}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def noise_seed(episode_key, call):
    text = f"rl-vla-evaluation-v1:{episode_key}:{call}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def summarize(rows, planned):
    successes = sum(row["success"] for row in rows)
    count = len(rows)
    latencies = [x for row in rows for x in row["inference_latency_ms"]]
    groups = {}
    for row in rows:
        groups.setdefault(f"{row['suite']}/{row['task_id']}", []).append(row)
    per_task = {}
    for key, group in groups.items():
        wins = sum(row["success"] for row in group)
        per_task[key] = {"episodes": len(group), "successes": wins,
                         "success_rate": wins / len(group),
                         "success_95pct_wilson": wilson_interval(wins, len(group))}
    return {
        "status": "complete" if count == planned else "partial",
        "episodes": count, "planned_episodes": planned, "successes": successes,
        "success_rate": successes / count if count else None,
        "success_95pct_wilson": wilson_interval(successes, count) if count else None,
        "per_task": per_task,
        "mean_return": float(np.mean([r["return"] for r in rows])) if rows else None,
        "mean_primitive_steps": float(np.mean([r["primitive_steps"] for r in rows])) if rows else None,
        "mean_steps_on_success": float(np.mean([r["primitive_steps"] for r in rows if r["success"]]))
        if successes else None,
        "inference_latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "inference_latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "total_inference_calls": len(latencies),
        "total_episode_seconds": sum(row["elapsed_seconds"] for row in rows),
    }


class ResourceGuard:
    """Pause between requests under memory/thermal pressure; never alter host settings."""
    def __init__(self, path):
        self.path = path
        self.checked_at = 0.0

    def check(self, force=False):
        if not force and time.monotonic() - self.checked_at < 5:
            return
        deadline = time.monotonic() + 300
        while True:
            output = subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"], text=True, timeout=10)
            values = [float(x.strip()) for x in output.strip().splitlines()[0].split(",")]
            meminfo = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, rest = line.split(":", 1)
                meminfo[key] = int(rest.split()[0])
            record = {"unix_time": time.time(), "gpu_memory_mib": values[0],
                      "gpu_util_percent": values[1], "gpu_temperature_c": values[2],
                      "gpu_power_w": values[3], "ram_available_gib": meminfo["MemAvailable"] / 2**20}
            with self.path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            self.checked_at = time.monotonic()
            if values[0] < 48 * 1024 and values[2] < 78 and record["ram_available_gib"] > 8:
                return
            if time.monotonic() > deadline:
                raise RuntimeError("Resource guard remained above limits for five minutes")
            print("Resource guard: cooling/waiting before the next inference request", flush=True)
            time.sleep(5)


class EvaluationClient:
    def __init__(self, host, port, guard, request_pause):
        # OpenPI's synchronous first JAX compilation blocks its websocket event
        # loop. Use the explicit receive deadline instead of a 20-second ping.
        self.connection = connect(f"ws://{host}:{port}", compression=None, max_size=None,
                                  open_timeout=10, close_timeout=2, ping_interval=None)
        self.packer = msgpack_numpy.Packer()
        self.metadata = msgpack_numpy.unpackb(self.connection.recv(timeout=15))
        if self.metadata.get("noise_protocol") != "numpy-pcg64-float32-standard-normal-v1":
            raise ValueError("Server must be started with scripts/serve_baseline.py")
        self.guard, self.request_pause = guard, request_pause
        self.episode_key = "unset"
        self.reset()

    def get_server_metadata(self):
        return self.metadata

    def reset(self):
        self.calls = 0
        self.latencies = []

    def infer(self, observation):
        self.guard.check()
        payload = dict(observation)
        payload["evaluation_noise_seed"] = noise_seed(self.episode_key, self.calls)
        started = time.perf_counter()
        self.connection.send(self.packer.pack(payload))
        response = self.connection.recv(timeout=1200)
        self.latencies.append((time.perf_counter() - started) * 1000)
        if isinstance(response, str):
            raise RuntimeError(response)  # noqa: TRY004 -- server errors use text frames
        self.calls += 1
        time.sleep(self.request_pause)
        return msgpack_numpy.unpackb(response)

    def close(self):
        self.connection.close()


class RecordingEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.frames = None
        self.last_observation = None

    def _record(self, observation):
        self.last_observation = observation
        if self.frames is not None:
            self.frames.append(observation["openpi"]["observation/image"].copy())

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._record(observation)
        return observation, info

    def step(self, action):
        result = self.env.step(action)
        self._record(result[0])
        return result


def run(args):
    from libero.libero import benchmark

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    guard = ResourceGuard(output / "resources.jsonl")
    guard.check(force=True)
    client = EvaluationClient(args.host, args.port, guard, args.request_pause)
    env = None
    rows = []
    try:
        if client.metadata["model_config"] != args.model_config:
            raise ValueError("The served model config does not match the requested evaluation")
        template = load_config(args.config)
        plan = []
        for suite_index, suite in enumerate(args.suites):
            task_count = benchmark.get_benchmark_dict()[suite]().n_tasks
            tasks = list(range(task_count)) if args.task_ids is None else args.task_ids
            if len(set(tasks)) != len(tasks) or any(t < 0 or t >= task_count for t in tasks):
                raise ValueError("Invalid or repeated task IDs")
            for task_id in tasks:
                for initial_state_id in range(args.episodes_per_task):
                    key = f"{suite}_task{task_id:02d}_init{initial_state_id:03d}"
                    plan.append({"episode_id": key, "suite": suite, "task_id": task_id,
                                 "initial_state_id": initial_state_id,
                                 "seed": args.seed + suite_index * 100_000 + task_id * 1000 + initial_state_id})
        contract = {"version": 1, "episodes": plan, "chunk": template["chunk"],
                    "checkpoint_sha256": client.metadata["checkpoint_sha256"],
                    "model_config": args.model_config, "noise_protocol": client.metadata["noise_protocol"],
                    "horizons": HORIZONS, "camera_resolution": 256, "resize_size": 224,
                    "settle_steps": 10, "request_pause_seconds": args.request_pause}
        contract["adapter_sha256"] = {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("envs/libero.py", "base_policy.py", "env_adapter.py")}
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            if not args.resume or json.loads(protocol_path.read_text()) != contract:
                raise ValueError("Output already has a protocol; use --resume with identical settings")
        else:
            atomic_json(protocol_path, contract)
            atomic_json(output / "manifest.json", {
                "server": client.metadata, "controller": "frozen_base_zero_residual",
                "packages": {p: version(p) for p in ("rl-vla", "torch", "numpy", "gymnasium", "mujoco", "robosuite")},
                "cpu_affinity": sorted(os.sched_getaffinity(0)), "created_unix_time": time.time(),
                "config": template, "uses_demonstration_dataset": False,
                "environment": {name: os.environ.get(name) for name in (
                    "MUJOCO_GL", "PYOPENGL_PLATFORM", "NUMBA_DISABLE_JIT", "LP_NUM_THREADS",
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "evaluation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            })
        current_task = None
        for episode in plan:
            episode_path = output / "episodes" / f"{episode['episode_id']}.json"
            if episode_path.exists():
                saved = json.loads(episode_path.read_text())
                if any(saved[key] != episode[key] for key in episode):
                    raise ValueError("Saved episode differs from evaluation protocol")
                rows.append(saved)
                continue
            task_key = (episode["suite"], episode["task_id"])
            if task_key != current_task:
                print(f"Preparing simulator for {task_key}", flush=True)
                if env is not None:
                    env.close()
                settings = copy.deepcopy(template["env"].get("kwargs", {}))
                settings.update(task_suite=episode["suite"], task_id=episode["task_id"],
                                max_steps=HORIZONS[episode["suite"]])
                recording = RecordingEnv(make_env(**settings))
                if args.episodes_per_task > len(recording.env.initial_states):
                    raise ValueError("Requested more initial states than the task provides")
                base = OpenPIBasePolicy(7, template["chunk"]["horizon"], client=client,
                                        host=args.host, port=args.port)
                env = ResidualChunkEnv(recording, base, template["chunk"])
                current_task = task_key
            recording.frames = [] if episode["initial_state_id"] in args.video_initial_states else None
            seed_everything(episode["seed"], 1)
            client.episode_key = f"{episode['episode_id']}:{episode['seed']}"
            print(f"Starting {episode['episode_id']}", flush=True)
            started = time.perf_counter()
            _, reset_info = env.reset(seed=episode["seed"],
                                      options={"initial_state_id": episode["initial_state_id"]})
            if episode["initial_state_id"] == 0:
                from PIL import Image
                image_dir = output / "observations" / f"{episode['suite']}_task{episode['task_id']:02d}"
                image_dir.mkdir(parents=True, exist_ok=True)
                for key, name in (("observation/image", "front"), ("observation/wrist_image", "wrist")):
                    Image.fromarray(recording.last_observation["openpi"][key]).save(image_dir / f"{name}.png")
            steps, decisions, total_return, projections = 0, 0, 0.0, 0.0
            while True:
                _, _, terminated, truncated, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
                decisions += 1
                steps += info["executed_steps"]
                total_return += info["raw_return"]
                projections += info["projection_fraction"] * info["executed_steps"]
                if terminated or truncated:
                    break
                if steps > HORIZONS[episode["suite"]]:
                    raise RuntimeError("Simulator did not enforce its episode horizon")
            row = {**episode, "task_description": recording.env.task_description,
                   "success": bool(info["is_success"]), "return": total_return,
                   "primitive_steps": steps, "decisions": decisions,
                   "terminated": terminated, "truncated": truncated,
                   "projection_fraction": projections / steps,
                   "inference_latency_ms": client.latencies.copy(),
                   "elapsed_seconds": time.perf_counter() - started, "reset_info": reset_info}
            if row["success"] != bool(terminated) or total_return != float(row["success"]):
                raise RuntimeError("LIBERO success/termination/sparse reward are inconsistent")
            atomic_json(episode_path, row)
            rows.append(row)
            atomic_json(output / "summary.json", summarize(rows, len(plan)))
            if recording.frames:
                import imageio.v2 as imageio
                video_dir = output / "videos"
                video_dir.mkdir(exist_ok=True)
                imageio.mimwrite(video_dir / f"{episode['episode_id']}.mp4", recording.frames,
                                fps=10, codec="libx264", quality=6,
                                ffmpeg_params=["-threads", "1"])
                recording.frames = None
            print(f"{len(rows)}/{len(plan)} {episode['episode_id']} "
                  f"success={row['success']} steps={steps} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
        summary = summarize(rows, len(plan))
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except Exception as error:
        atomic_json(output / "error.json", {"type": type(error).__name__, "message": str(error),
                                            "completed_episodes": len(rows), "unix_time": time.time()})
        raise
    finally:
        if env is not None:
            env.close()
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/libero_sac.toml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-config", default="pi05_libero")
    parser.add_argument("--suites", nargs="+", choices=list(HORIZONS), default=["libero_spatial"])
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100_000)
    parser.add_argument("--video-initial-states", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--request-pause", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.episodes_per_task < 1 or args.request_pause < 0:
        parser.error("Episode count must be positive and request pause nonnegative")
    run(args)


if __name__ == "__main__":
    main()
