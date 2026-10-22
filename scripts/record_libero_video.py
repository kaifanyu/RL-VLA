"""Record LIBERO rollouts of the frozen base or a trained residual checkpoint as MP4 videos.

Each video shows the agent-view and wrist cameras side by side, one frame per
controller step, with a status bar (step, decision, residual size, outcome).
Needs the same environment variables as training and a running OpenPI server.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from rl_vla.config import load_config
from rl_vla.runtime import build_env, load_checkpoint, make_learner, seed_everything, tensor

SCALE = 2
BAR = 44
GREEN, RED, GREY = (46, 160, 67), (200, 55, 45), (40, 46, 56)


class Recorder:
    """Collects one annotated frame per LIBERO controller step."""

    def __init__(self, label):
        self.label = label
        self.frames = []
        self.decision = 0
        self.residual_rms = 0.0

    def capture(self, raw, step, success=False):
        images = raw["openpi"]
        view = np.concatenate((images["observation/image"], images["observation/wrist_image"]), axis=1)
        image = Image.fromarray(view).resize((view.shape[1] * SCALE, view.shape[0] * SCALE),
                                             Image.BILINEAR)
        frame = Image.new("RGB", (image.width, image.height + BAR), GREEN if success else GREY)
        frame.paste(image, (0, BAR))
        draw = ImageDraw.Draw(frame)
        draw.text((10, 6), self.label, fill="white")
        draw.text((10, 24), f"step {step:3d}   decision {self.decision:3d}   "
                            f"residual RMS {self.residual_rms:.3f}" + ("   SUCCESS" if success else ""),
                  fill="white")
        draw.text((image.width // 2 + 10, BAR + 6), "wrist", fill="white")
        draw.text((10, BAR + 6), "agent view", fill="white")
        self.frames.append(np.asarray(frame))

    def finish(self, success, hold=20):
        last = Image.fromarray(self.frames[-1]).copy()
        draw = ImageDraw.Draw(last)
        draw.rectangle((0, 0, last.width, BAR), fill=GREEN if success else RED)
        draw.text((10, 6), self.label, fill="white")
        draw.text((10, 24), f"episode finished: {'SUCCESS' if success else 'FAILURE (timeout)'}",
                  fill="white")
        self.frames.extend([np.asarray(last)] * hold)


def attach(env, recorder):
    """Hook the raw LIBERO env under the residual chunk wrapper to see every motor step."""
    inner = env.env
    original_reset, original_step = inner.reset, inner.step

    def reset(**kwargs):
        raw, info = original_reset(**kwargs)
        recorder.capture(raw, 0)
        return raw, info

    def step(action):
        raw, reward, terminated, truncated, info = original_step(action)
        recorder.capture(raw, inner.steps, bool(info.get("is_success", False)))
        return raw, reward, terminated, truncated, info

    inner.reset, inner.step = reset, step


def write_video(frames, path, fps):
    import imageio.v2 as imageio

    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    with imageio.get_writer(str(temporary), fps=fps, codec="libx264", quality=7,
                            macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Residual .pt file; omit to record the frozen base")
    parser.add_argument("--output", type=Path, help="Directory for videos (default: next to checkpoint)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=100_000, help="First held-out episode seed")
    parser.add_argument("--deterministic", action="store_true", help="Use the residual mean action")
    parser.add_argument("--fps", type=int, default=20, help="LIBERO controls at 20 Hz")
    args = parser.parse_args()

    config = load_config(str(args.config))
    device = config["train"].get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = config["train"]["device"] = "cpu"  # Inference only; CPU is fast enough.
    name = args.checkpoint.stem if args.checkpoint else "base"
    output = args.output or ((args.checkpoint.parent if args.checkpoint else Path("runs")) / f"videos_{name}")
    output.mkdir(parents=True, exist_ok=True)
    mode = ("frozen base" if args.checkpoint is None else
            f"{config['algorithm']} {args.checkpoint.parent.name}/{args.checkpoint.name} "
            f"({'mean' if args.deterministic else 'sampled'} residual)")
    task = config["env"].get("kwargs", {})

    seed_everything(args.seed, int(config["train"].get("torch_threads", 1)))
    env = build_env(config)
    learner, results = None, []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            label = f"{mode} | {task.get('task_suite', '')} task {task.get('task_id', '')} | seed {seed}"
            recorder = Recorder(label)
            attach(env, recorder)
            seed_everything(seed, int(config["train"].get("torch_threads", 1)))
            obs, _ = env.reset(seed=seed)
            if args.checkpoint and learner is None:  # Dimensions are known after the first reset.
                learner = make_learner(config, env)
                saved = load_checkpoint(str(args.checkpoint), device)
                for key in ("algorithm", "base", "chunk"):
                    if saved["config"][key] != config[key]:
                        raise ValueError(f"Config {key} differs from the checkpoint's training config")
                learner.load_state_dict(saved["learner"])
                learner.actor.eval()
            success = False
            while True:
                recorder.decision += 1
                if learner is None:
                    action = np.zeros(env.action_dim, dtype=np.float32)
                else:
                    with torch.no_grad():
                        action = learner.actor.sample(tensor(obs["obs"], device).unsqueeze(0),
                                                      deterministic=args.deterministic)[0][0].cpu().numpy()
                recorder.residual_rms = float(np.sqrt(np.mean(action ** 2)))
                obs, _, terminated, truncated, info = env.step(action)
                success |= bool(info.get("is_success", False))
                if terminated or truncated:
                    break
            recorder.finish(success)
            path = output / f"seed{seed}_{'success' if success else 'fail'}.mp4"
            write_video(recorder.frames, path, args.fps)
            results.append({"seed": seed, "success": success, "motor_steps": env.env.steps,
                            "decisions": recorder.decision, "video": str(path)})
            print(f"seed {seed}: {'SUCCESS' if success else 'fail'} after {env.env.steps} steps -> {path}",
                  flush=True)
    finally:
        env.close()
    summary = {"controller": mode, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "config": str(args.config), "successes": sum(r["success"] for r in results),
               "episodes": len(results), "results": results}
    (output / "videos.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['successes']}/{summary['episodes']} successful; videos in {output}")


if __name__ == "__main__":
    main()
