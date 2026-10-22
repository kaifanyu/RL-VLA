"""Monitor this run; stop its owned processes if resource guardrails are exceeded."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    previous = {}
    with args.output.open("a", buffering=1) as log:
        while True:
            pids = [int(x) for x in args.pids.read_text().split()]
            now = time.monotonic()
            processes = []
            for pid in pids:
                try:
                    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                    ticks = int(stat[11]) + int(stat[12])
                    old = previous.get(pid)
                    cores = ((ticks - old[1]) / os.sysconf("SC_CLK_TCK") / (now - old[0])) if old else 0
                    previous[pid] = (now, ticks)
                    processes.append({"pid": pid, "cpu_cores": cores,
                                      "rss_gib": int(stat[21]) * os.sysconf("SC_PAGE_SIZE") / 2**30})
                except (FileNotFoundError, ProcessLookupError):
                    pass
            data = subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"], text=True, timeout=10)
            values = [float(x.strip()) for x in data.strip().splitlines()[0].split(",")]
            available = next(int(line.split()[1]) / 2**20 for line in
                             Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))
            record = {"unix_time": time.time(), "gpu_memory_mib": values[0],
                      "gpu_util_percent": values[1], "gpu_temperature_c": values[2],
                      "gpu_power_w": values[3], "ram_available_gib": available,
                      "processes": processes}
            log.write(json.dumps(record) + "\n")
            if values[0] >= 56 * 1024 or values[2] >= 83 or available < 4:
                reason = {"reason": "GPU memory, temperature or available RAM emergency threshold", **record}
                args.output.with_name("resource_stop.json").write_text(json.dumps(reason, indent=2))
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                return
            time.sleep(5)


if __name__ == "__main__":
    main()
