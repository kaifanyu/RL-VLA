"""Wait for shared-node capacity and retry only recorded resource interruptions."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rl_vla.baseline import atomic_json
from rl_vla.robustness import ROOT


def wait_for_capacity(output):
    stable_since = None
    (output / "launcher.pid").write_text(str(os.getpid()) + "\n")
    while True:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=10)
        memory, temperature = map(float, raw.splitlines()[0].split(","))
        ram = next(int(line.split()[1]) / 2**20 for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith("MemAvailable:"))
        ready = memory < 32 * 1024 and temperature < 70 and ram > 24
        if not ready:
            stable_since = None
        elif stable_since is None:
            stable_since = time.monotonic()
        stable_seconds = time.monotonic() - stable_since if stable_since is not None else 0
        record = {"status": "waiting_for_capacity", "unix_time": time.time(),
                  "gpu_memory_mib": memory, "gpu_temperature_c": temperature,
                  "ram_available_gib": ram, "stable_seconds": stable_seconds,
                  "required_stable_seconds": 30, "pid": os.getpid()}
        atomic_json(output / "resource_wait.json", record)
        path = output / "queue.json"
        queue = json.loads(path.read_text()) if path.exists() else {}
        queue.update(status="waiting_for_capacity", active=None, pid=os.getpid(), updated_unix_time=time.time())
        atomic_json(path, queue)
        path = output / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            summary["status"] = "waiting_for_capacity"
            atomic_json(path, summary)
        if stable_seconds >= 30:
            record["status"] = "capacity_available"
            atomic_json(output / "resource_wait.json", record)
            return
        time.sleep(10)


def main(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT.parent / ".cache/rl-vla-resource-queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        print("Waiting for 30 seconds of spare GPU/RAM capacity", flush=True)
        wait_for_capacity(output)
        # Retain past stop events without presenting them as current failures.
        events = output / "startup_events" / str(time.time_ns())
        for name in ("resource_stop.json", "interruption.json", "error.json", "exit_code"):
            source = output / name
            if source.exists():
                events.mkdir(parents=True, exist_ok=True)
                source.replace(events / name)
        result = subprocess.run([sys.executable, str(ROOT / "scripts/robustness_queue.py"),
                                 "--output", str(output)], cwd=ROOT, check=False)
        if not (output / "resource_stop.json").exists():
            return result.returncode
        print("Resource monitor stopped this attempt; waiting before resuming saved episodes", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(main(parser.parse_args().output))
