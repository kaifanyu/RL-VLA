"""Collect a deterministic simulator observation bank without any policy rollouts."""

import argparse
import json
from pathlib import Path

from rl_vla.diagnostics.observation_bank import collect_bank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test", "source"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume",action="store_true",help="Verify and append to an interrupted immutable bank")
    args = parser.parse_args()
    print(json.dumps(collect_bank(args.protocol, args.task, args.split, args.output, limit=args.limit,resume=args.resume), indent=2))
