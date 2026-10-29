"""Run one process under the same writable Python flock as the active queues."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--wait-seconds", type=int, default=24 * 3600 - 600)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    allocation = os.environ["SLURM_JOB_ID"]
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A child command is required")
    lock_path = ROOT / "runs" / f"gpu_allocation_{allocation}" / "experiment.lock"

    def status(state, **extra):
        value = {"state": state, "allocation": allocation,
                 "step_id": os.environ.get("SLURM_STEP_ID"), "command": command,
                 "updated": datetime.now(UTC).isoformat(), **extra}
        temporary = args.status.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(args.status)

    status("waiting_for_gpu", lock=str(lock_path))

    def timed_out(signum, frame):
        raise TimeoutError("GPU lock wait budget exhausted")

    try:
        with lock_path.open("a") as lock:
            # Join the kernel's blocking waiters instead of racing the active
            # queues through repeated nonblocking acquisition attempts.
            previous = signal.signal(signal.SIGALRM, timed_out)
            signal.alarm(args.wait_seconds)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            status("running")
            subprocess.run(command, cwd=ROOT, check=True)
        status("completed")
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
