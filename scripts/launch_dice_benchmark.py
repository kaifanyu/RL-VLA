#!/usr/bin/env python
"""Launch a new audited DICE seed, validate exact milestones, then test once.

No held-out test result enters checkpoint selection. K>1 is a separate command
and output tree. Uses the current interpreter and scripts/run_dice.sh.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(
        ["bash", str(ROOT / "scripts/run_dice.sh"), *map(str, args)], check=True, cwd=ROOT
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/dice_audit_pi05_object0.toml"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, default=0, help="Local benchmark choice, not a paper requirement"
    )
    parser.add_argument(
        "--best-of-n", type=int, help="Separately costed evaluation after K1 selection"
    )
    args = parser.parse_args()
    if args.best_of_n is not None and args.best_of_n < 2:
        parser.error("Separate best-of-N requires N >= 2")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Set CUDA_VISIBLE_DEVICES to an allocated GPU")
    config = tomllib.loads(args.config.read_text())
    if config["evaluation"]["k"] != 1 or config["train"].get("max_steps") != 16000:
        parser.error("This orchestrator requires the audit K1/16K profile")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "launch.json").write_text(
        json.dumps(
            {
                "config": str(args.config.resolve()),
                "seed": args.seed,
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "selection": "validation successes; ties earliest command; initialization eligible",
                "test_scope": "reused audited states, not blind confirmation",
                "best_of_n": args.best_of_n,
            },
            indent=2,
        )
        + "\n"
    )
    training = args.output / "training"
    run("train", "--config", args.config, "--output", training, "--train-seed", args.seed)
    candidates = []
    for step in [0, *config["train"]["milestones"]]:
        checkpoint = training / f"step_{step:06d}.pt"
        output = args.output / f"validation_{step:06d}"
        run(
            "evaluate",
            "--config",
            args.config,
            "--output",
            output,
            "--checkpoint",
            checkpoint,
            "--split",
            "validation",
            "--evaluation-k",
            1,
        )
        summary = json.loads((output / "summary.json").read_text())
        if not summary["selection_eligible"]:
            raise RuntimeError("Validation result is not selection eligible")
        candidates.append(
            {
                "controller_steps": step,
                "checkpoint": str(checkpoint),
                "successes": summary["successes"],
                "episodes": summary["episodes"],
            }
        )
    winner = max(candidates, key=lambda r: (r["successes"], -r["controller_steps"]))
    (args.output / "selection.json").write_text(
        json.dumps(
            {
                "candidates": candidates,
                "selected": winner,
                "test_results_used_for_selection": False,
            },
            indent=2,
        )
        + "\n"
    )
    for split in ("test", "source"):
        run(
            "evaluate",
            "--config",
            args.config,
            "--output",
            args.output / f"selected_{split}_k1",
            "--checkpoint",
            winner["checkpoint"],
            "--split",
            split,
            "--evaluation-k",
            1,
        )
        run(
            "evaluate",
            "--config",
            args.config,
            "--output",
            args.output / f"prior_{split}_k1",
            "--baseline",
            "--split",
            split,
            "--evaluation-k",
            1,
        )
    if args.best_of_n:
        for split in ("validation", "test", "source"):
            run(
                "evaluate",
                "--config",
                args.config,
                "--output",
                args.output / f"selected_{split}_k{args.best_of_n}",
                "--checkpoint",
                winner["checkpoint"],
                "--split",
                split,
                "--evaluation-k",
                args.best_of_n,
            )


if __name__ == "__main__":
    main()
