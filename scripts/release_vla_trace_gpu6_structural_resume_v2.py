"""Wait for accepted resumed causal coverage, then release its physical GPU."""

import argparse
import time
from pathlib import Path

from rl_vla.diagnostics.causal_release_bridge_v2 import publish, read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bridge-acceptance", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("poll interval must be positive")
    approval = read(args.bridge_acceptance)
    if approval.get("bridge_sources", {}).get(str(Path(__file__).resolve())) != sha(__file__):
        raise ValueError("Bridge CLI is not in the accepted source contract")
    while not publish(args.run, args.bridge_acceptance):
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
