"""Run `python -m rl_vla --help` from the repository root."""

import argparse
import json

from rl_vla.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen OpenPI + residual SAC/PPO")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train a residual controller")
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True, help="New run directory")
    training.add_argument("--steps", type=int, help="Override number of RL decisions")
    training.add_argument("--seed", type=int)
    training.add_argument("--init-from", help="Warm-start weights/optimizers; new replay and environment")
    testing = commands.add_parser("evaluate", help="Evaluate frozen base or saved residual")
    testing.add_argument("--config", required=True)
    testing.add_argument("--output", required=True, help="New JSON result file")
    testing.add_argument("--checkpoint", help="Omit for frozen-base evaluation")
    testing.add_argument("--episodes", type=int, default=20)
    testing.add_argument("--seed", type=int, default=100_000)
    testing.add_argument("--deterministic", action="store_true", help="Use residual mean (ablation)")
    testing.add_argument("--max-decisions", type=int, default=10_000)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "train":
        from rl_vla.train import train
        if args.steps is not None:
            if args.steps < 1:
                parser.error("--steps must be positive")
            config["train"]["total_decisions"] = args.steps
        if args.seed is not None:
            config["train"]["seed"] = args.seed
        result = train(config, args.output, args.init_from)
    else:
        from rl_vla.evaluate import evaluate
        result = evaluate(config, args.output, args.checkpoint, args.episodes, args.seed,
                          args.deterministic, args.max_decisions)
        result = {k: v for k, v in result.items() if k not in {"results", "config"}}
    print(json.dumps(result, indent=2, allow_nan=False))

