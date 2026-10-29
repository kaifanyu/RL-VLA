"""Save one untouched held-out simulator observation for inference parity."""

import argparse
import json
from pathlib import Path

import numpy as np

from rl_vla.envs.libero_pro import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=100000)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    env = make_env(protocol=args.protocol, task_key=args.task_key, split="validation")
    try:
        raw, info = env.reset(seed=args.seed)
        payload = raw["openpi"]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            image=payload["observation/image"],
            wrist_image=payload["observation/wrist_image"],
            state=payload["observation/state"],
            prompt=np.asarray(payload["prompt"]),
            noise=np.random.default_rng(args.seed).normal(size=(1, 10, 32)).astype(np.float32),
        )
        output.with_suffix(".json").write_text(json.dumps({
            "protocol": args.protocol, "task_key": args.task_key,
            "split": "validation", "seed": args.seed, "reset": info,
            "purpose": "Cross-framework inference parity only; no action executed, no training data",
        }, indent=2) + "\n")
        print(json.dumps({"fixture": str(output), "reset": info}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
