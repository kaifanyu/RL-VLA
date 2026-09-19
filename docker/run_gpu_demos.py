"""Sequential, bounded local demos using the project's existing SAC/PPO learners.

Run in the prepared CUDA container; outputs and progress persist in runs/.
These single-seed diagnostics are not tuned benchmark or convergence claims.
"""

import argparse
import copy
import csv
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from rl_vla.config import load_config
from rl_vla.evaluate import evaluate
from rl_vla.runtime import (
    build_env,
    load_checkpoint,
    make_learner,
    seed_everything,
    tensor,
    write_json,
)
from rl_vla.train import train

TASKS = {
    "toy": (None, 2048),
    "weak_toy": (None, 8192),
    "pendulum": ("Pendulum-v1", 16384),
    "reacher": ("Reacher-v5", 8192),
    "panda_reach": ("PandaReachDense-v3", 8192),
    "panda_push": ("PandaPushDense-v3", 8192),
    "panda_pick_place": ("PandaPickAndPlaceDense-v3", 8192),
}


def configuration(task, algorithm, seed=0, steps=None):
    config = load_config(f"configs/{algorithm}_toy.toml")
    env_id, budget = TASKS[task]
    config["train"].update(device="cuda", seed=seed, total_decisions=steps or budget,
                           log_every=512, save_every=steps or budget)
    if task == "weak_toy":
        config["env"]["kwargs"]["base_gain"] = 0.3
    elif env_id:
        config["env"] = {"kind": "external", "factory": "rl_vla.envs.benchmarks:make_env",
                         "kwargs": {"env_id": env_id}}
        config["chunk"].update(horizon=1, execute_steps=1,
                                residual_scale=2.0 if task == "pendulum" else 1.0)
        config["train"].update(batch_size=128, learning_starts=256,
                               replay_capacity=50000, rollout_steps=512)
    return config


def telemetry(path, stop):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["unix_time", "gpu_util_percent", "memory_used_mib",
                         "memory_total_mib", "temperature_c"])
        while not stop.is_set():
            try:
                data = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits"], text=True, timeout=10)
                for line in data.splitlines():
                    writer.writerow([time.time(), *[part.strip() for part in line.split(",")]])
                stream.flush()
            except (OSError, subprocess.SubprocessError):
                pass
            stop.wait(2)


def toy_frame(env, label):
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (512, 512), "#101926")
    draw = ImageDraw.Draw(frame)
    draw.text((16, 16), label, fill="white")
    draw.text((16, 36), "blue: controller   green: goal", fill="#d0dae6")
    def point(position):
        return (256 + float(position[0]) * 175, 280 - float(position[1]) * 175)
    x, y = point(env.goal)
    radius = env.success_radius * 175
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="#258656")
    x, y = point(env.position)
    draw.ellipse((x-7, y-7, x+7, y+7), fill="#6bb9ff")
    return np.asarray(frame)


def video(config, path, checkpoint=None, seed=100000):
    import imageio.v2 as imageio

    cfg = copy.deepcopy(config)
    if cfg["env"]["kind"] == "external":
        cfg["env"]["kwargs"]["render_mode"] = "rgb_array"
    seed_everything(seed, 1)
    env = build_env(cfg)
    label = f"{path.stem} | seed {seed} | mean actions"
    try:
        obs, _ = env.reset(seed=seed)
        learner = None
        if checkpoint:
            learner = make_learner(cfg, env)
            learner.load_state_dict(load_checkpoint(checkpoint, "cuda")["learner"])
            learner.actor.eval()
        fps = 8 if cfg["env"]["kind"] == "toy" else 25
        with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=2) as writer:
            for _ in range(1000):
                frame = (toy_frame(env.env, label) if cfg["env"]["kind"] == "toy"
                         else env.env.render())
                writer.append_data(frame)
                if learner is None:
                    action = np.zeros(env.action_dim, dtype=np.float32)
                else:
                    with torch.no_grad():
                        action = learner.actor.sample(tensor(obs["obs"], "cuda").unsqueeze(0),
                                                      deterministic=True)[0][0].cpu().numpy()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    frame = (toy_frame(env.env, label) if cfg["env"]["kind"] == "toy"
                             else env.env.render())
                    writer.append_data(frame)
                    break
    finally:
        env.close()


def annotate_evaluation(path, result, task):
    # Generic evaluator's false success flags are not benchmark success metrics.
    result["success_metric_available"] = task not in {"pendulum", "reacher"}
    if not result["success_metric_available"]:
        result["success_metric_note"] = "Environment has no native success metric; compare return."
    write_json(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--algorithms", nargs="+", choices=["sac", "ppo"], default=["sac", "ppo"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment; no silent CPU fallback")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    run_info = {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(), "tasks": args.tasks,
                "algorithms": args.algorithms, "episodes": args.episodes,
                "seed": args.seed, "started_unix": time.time(),
                "notes": ["One training seed; bounded diagnostics, not convergence or algorithm ranking.",
                          "Existing learners and conservative near-zero actor initialization retained.",
                          "Physics/rendering use CPU; SAC/PPO tensors and optimization use CUDA.",
                          "GPU telemetry includes other host GPU activity."]}
    write_json(root / "experiment.json", run_info)
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(root / "gpu_telemetry.csv", stop), daemon=True)
    monitor.start()
    errors = []
    def status(task, stage, **extra):
        entry = {"task": task, "stage": stage, "updated_unix": time.time(), **extra}
        write_json(root / "status.json", entry)
        print(json.dumps(entry), flush=True)
    try:
        for task in args.tasks:
            task_root = root / task
            task_root.mkdir(exist_ok=True)
            base_config = configuration(task, args.algorithms[0], args.seed, args.steps)
            base_path = task_root / "base_eval.json"
            try:
                if not base_path.exists():
                    status(task, "base_evaluation")
                    result = evaluate(base_config, str(base_path), episodes=args.episodes)
                    annotate_evaluation(base_path, result, task)
                    print(f"BASE {task}: return={result['mean_return']:.4f} "
                          f"success={result['success_rate']:.1%}", flush=True)
            except Exception as exc:  # noqa: BLE001 - record task failure and continue the suite
                errors.append({"task": task, "stage": "base", "error": str(exc)})
                traceback.print_exc()
                continue
            if not args.no_video and not (task_root / "base.mp4").exists():
                try:
                    video(base_config, task_root / "base.mp4")
                except Exception as exc:  # noqa: BLE001 - video failure must preserve training results
                    errors.append({"task": task, "stage": "base_video", "error": str(exc)})
                    traceback.print_exc()
            for algorithm in args.algorithms:
                config = configuration(task, algorithm, args.seed, args.steps)
                out = task_root / f"{algorithm}_seed{args.seed}"
                try:
                    if not (out / "summary.json").exists():
                        if out.exists():
                            raise RuntimeError(f"Incomplete run exists at {out}; preserve it and use a new output root")
                        status(task, "training", algorithm=algorithm,
                               decisions=config["train"]["total_decisions"])
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        summary = train(config, str(out))
                        summary.update(cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                                       cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                        write_json(out / "summary.json", summary)
                    checkpoint = str(out / "final.pt")
                    for deterministic, suffix in [(False, ""), (True, "_deterministic")]:
                        eval_path = task_root / f"{algorithm}{suffix}_eval.json"
                        if not eval_path.exists():
                            status(task, "evaluation", algorithm=algorithm,
                                   deterministic=deterministic)
                            result = evaluate(config, str(eval_path), checkpoint, args.episodes,
                                              deterministic=deterministic)
                            annotate_evaluation(eval_path, result, task)
                            print(f"EVAL {task}/{algorithm}{suffix}: return={result['mean_return']:.4f} "
                                  f"success={result['success_rate']:.1%}", flush=True)
                    if not args.no_video and not (task_root / f"{algorithm}.mp4").exists():
                        status(task, "video", algorithm=algorithm)
                        try:
                            video(config, task_root / f"{algorithm}.mp4", checkpoint)
                        except Exception as exc:  # noqa: BLE001 - retain successful training/evaluation
                            errors.append({"task": task, "stage": algorithm + "_video", "error": str(exc)})
                            traceback.print_exc()
                except Exception as exc:  # noqa: BLE001 - independent demos can continue after a failure
                    errors.append({"task": task, "stage": algorithm, "error": str(exc)})
                    traceback.print_exc()
                finally:
                    torch.cuda.empty_cache()
            write_json(root / "errors.json", {"errors": errors})
        status("all", "complete", errors=len(errors))
    finally:
        stop.set()
        monitor.join(timeout=12)
        write_json(root / "errors.json", {"errors": errors})
        run_info["finished_unix"] = time.time()
        write_json(root / "experiment.json", run_info)


if __name__ == "__main__":
    main()
