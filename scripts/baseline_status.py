"""Show saved evaluation progress and measured resource peaks without touching jobs."""
import argparse
import json
import time
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def status(root):
    root = root.resolve()
    summary = read_json(root / "summary.json")
    protocol = read_json(root / "protocol.json")
    records = []
    path = root / "telemetry.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A live monitor may be writing the last line.
    active = []
    path = root / "pids.txt"
    if path.exists():
        for pid in map(int, path.read_text().split()):
            command = Path(f"/proc/{pid}/cmdline")
            try:
                argv = command.read_bytes().replace(b"\0", b" ").decode()
                if any(name in argv for name in ("serve_baseline.py", "rl_vla.baseline",
                                                 "rl_vla.robustness", "robustness_queue.py",
                                                 "resource_queue.py")):
                    active.append(pid)
            except FileNotFoundError:
                pass
    peaks = None
    if records:
        peaks = {
            "gpu_memory_gib": max(r["gpu_memory_mib"] for r in records) / 1024,
            "gpu_temperature_c": max(r["gpu_temperature_c"] for r in records),
            "gpu_util_percent": max(r["gpu_util_percent"] for r in records),
            "owned_cpu_cores": max(sum(p["cpu_cores"] for p in r["processes"]) for r in records),
            "owned_combined_rss_gib": max(sum(p["rss_gib"] for p in r["processes"]) for r in records),
        }
    exit_path = root / "exit_code"
    return {
        "run": str(root), "checked_unix_time": time.time(),
        "active_owned_pids": active,
        "exit_code": int(exit_path.read_text()) if exit_path.exists() else None,
        "completed_episodes": summary["episodes"] if summary else 0,
        "planned_episodes": (len(protocol["episodes"]) if "episodes" in protocol else
                             sum(c["episodes"] for c in protocol["cases"])) if protocol else None,
        "queue": read_json(root / "queue.json"),
        "heartbeat": read_json(root / "heartbeat.json"),
        "summary": summary, "sampled_resource_peaks": peaks,
        "last_telemetry_unix_time": records[-1]["unix_time"] if records else None,
        "error": read_json(root / "error.json"),
        "resource_stop": read_json(root / "resource_stop.json"),
        "resource_wait": read_json(root / "resource_wait.json"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(status(args.run), indent=2))
