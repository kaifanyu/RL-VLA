"""Separate original LIBERO-90 queue, following the frozen PRO/Plus screen."""

import argparse
import json
import os
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from benchmark_bridge import screen as common

base_make_config = common.make_config


def make_config(snapshot, screen, case, ready_file):
    if (case["benchmark"], case["suite"], case["max_steps"]) != ("libero90", "libero_90", 400):
        raise ValueError("The separate queue requires original LIBERO-90 with its 400-step cap")
    config = base_make_config(snapshot, screen, case, ready_file)
    config["env"]["factory"] = "benchmark_bridge.libero90_environment:make_env"
    return config


def wait_for_predecessor(screen, predecessor, allocation, timeout):
    start = time.monotonic()
    while True:
        status_path = predecessor / "screen_status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        state = previous.get("state", "not_started")
        if state in {"complete", "complete_with_errors"}:
            return
        record = {"state": "waiting_for_predecessor", "predecessor": str(predecessor),
                  "predecessor_state": state, "allocation": allocation,
                  "step_id": os.environ.get("SLURM_STEP_ID"), "host": socket.gethostname(),
                  "updated": datetime.now(ZoneInfo("America/New_York")).isoformat()}
        if state == "failed" or time.monotonic() - start > timeout:
            record.update(state="failed", error="Predecessor screen failed or wait timed out")
            common.write_json(screen / "screen_status.json", record)
            raise RuntimeError(record["error"])
        common.write_json(screen / "screen_status.json", record)
        time.sleep(30)


def run_queue(repo, screen, allocation, predecessor, predecessor_timeout,
              ready_timeout=3600, case_timeout=3600):
    common.verify_execution(screen / "execution", screen)
    wait_for_predecessor(screen, predecessor, allocation, predecessor_timeout)
    # The shared queue implementation remains unchanged in the active 46-cell
    # snapshot. This process swaps only its own factory builder for LIBERO-90.
    original = common.make_config
    try:
        common.make_config = make_config
        common.run_queue(repo, screen, allocation, ready_timeout, case_timeout)
    finally:
        common.make_config = original


def launch(repo, screen, allocation, predecessor):
    launch_path = screen / "screen_launch.json"
    if launch_path.exists():
        raise FileExistsError(launch_path)
    snapshot = common.freeze_execution(repo, screen)
    command = ["srun", f"--jobid={allocation}", "--overlap", "--nodes=1", "--ntasks=1",
               "--gres=gpu:1", "--cpus-per-task=8", "--time=1-00:00:00",
               "--job-name=libero90-screen", "/usr/bin/env",
               f"PYTHONPATH={snapshot / 'integrations'}:{snapshot / 'src'}",
               str(repo / ".venv/bin/python"), "-m", "benchmark_bridge.screen90", "queue",
               "--repo", str(repo), "--screen", str(screen), "--allocation", str(allocation),
               "--predecessor", str(predecessor)]
    with (screen / "screen_launcher.log").open("x") as log:
        process = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"allocation": allocation, "launcher_pid": process.pid, "command": command,
              "snapshot": str(snapshot), "plan_sha256": common.digest(screen / "screen_plan.json"),
              "predecessor": str(predecessor),
              "gpu_lock_scope": "One case; policy process exits before unlocking",
              "episodes_per_cell": 5, "total_cells": 4}
    common.write_json(launch_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["launch", "queue"])
    parser.add_argument("--repo", type=Path, default=common.REPO)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--predecessor", required=True, type=Path)
    parser.add_argument("--allocation", type=int, default=1470624)
    parser.add_argument("--predecessor-timeout", type=int, default=23 * 3600)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    parser.add_argument("--case-timeout", type=int, default=3600)
    args = parser.parse_args()
    repo, screen, predecessor = args.repo.resolve(), args.screen.resolve(), args.predecessor.resolve()
    if args.command == "launch":
        print(json.dumps(launch(repo, screen, args.allocation, predecessor), indent=2), flush=True)
    else:
        run_queue(repo, screen, args.allocation, predecessor, args.predecessor_timeout,
                  args.ready_timeout, args.case_timeout)


if __name__ == "__main__":
    main()
