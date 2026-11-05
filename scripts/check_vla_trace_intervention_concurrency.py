"""Three simultaneous native diagnostic policies; validation-only, no success score."""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from queue_vla_trace_diagnostics import ROOT, cpu_lanes, file_hash, read, run_jobs, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--cpuset", default="72-79")
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    lanes = cpu_lanes(args.cpuset, 3)
    conditions = ["none", "no_image", "target:mosaic"]
    jobs = []
    for index in range(3):
        destination = args.output/f"worker_{index}"
        command = [str(ROOT/"third_party/openpi/.venv/bin/python"),
                   str(ROOT/"scripts/run_vla_trace_interventions.py"),
                   "--config", str(args.run/"tasks/object_0/validation.toml"),
                   "--token-metadata", str(args.run/"gate_b_actual/metadata.json"),
                   "--output", str(destination), "--episodes", "1", "--record-episodes", "1",
                   "--seed", "100000", "--technical-smoke", "--smoke-decisions", "2",
                   "--conditions", ",".join(conditions)]
        jobs.append({"name": f"worker_{index}", "destination": str(destination), "command": command,
                     "environment": {"CUDA_VISIBLE_DEVICES": args.gpu, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.25",
                                     "RL_VLA_OBJECT_TELEMETRY": "1"}})
    write(args.output/"plan.json", {"technical_only": True, "jobs": jobs, "lanes": lanes,
          "source_sha256": {str(p): file_hash(p) for p in [Path(__file__), ROOT/"scripts/queue_vla_trace_diagnostics.py",
                             ROOT/"scripts/run_vla_trace_interventions.py"]}})
    telemetry, stopped = [], threading.Event()
    def monitor():
        while not stopped.is_set():
            result = subprocess.run(["nvidia-smi", "-i", args.gpu, "--query-gpu=memory.used,utilization.gpu",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
            processes = subprocess.run(["nvidia-smi", "-i", args.gpu,
                         "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
            telemetry.append({"monotonic": time.monotonic(), "gpu": result.stdout.strip(),
                              "processes": processes.stdout.strip()})
            stopped.wait(2)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        run_jobs(jobs, lanes, args.output)
    finally:
        stopped.set()
        thread.join()
        write(args.output/"gpu_telemetry.json", telemetry)
    rows = []
    for condition in conditions:
        actions = []
        for index in range(3):
            directory = args.output/f"worker_{index}"/condition.replace(":", "_")
            summary = read(directory/"summary.json")
            episode = summary["episodes"][0]
            assert summary["completed"] and summary["technical_smoke"]
            assert episode["initial_state_id"] == 100 and episode["seed"] == 100000
            assert episode["success"] is None and episode["primitive_steps"] == 10
            assert read(directory/"no_op.json")["passed"]
            assert Path(episode["video"]).is_file()
            with np.load(directory/"episode_000_actions.npz", allow_pickle=False) as data:
                actions.append(data["normalized"].copy())
            assert np.isfinite(actions[-1]).all() and len(actions[-1]) == 2
            assert len(read(directory/"episode_000_telemetry.json")) == 10
        exact = all(np.array_equal(actions[0], a) for a in actions[1:])
        if not exact:
            raise ValueError("Concurrent identical policy/case/noise produced different actions")
        rows.append({"condition": condition, "three_worker_actions_bit_exact": exact})
    count = max(len(row["processes"].splitlines()) if row["processes"] else 0 for row in telemetry)
    if count != 3:
        raise RuntimeError(f"Expected three simultaneous GPU processes, saw {count}")
    write(args.output/"summary.json", {"complete": True, "passed": True, "technical_only": True,
          "gpu": args.gpu, "peak_gpu_processes": count,
          "peak_gpu_memory_mib": max(int(row["gpu"].split(",")[0]) for row in telemetry),
          "max_workers": 3, "memory_fraction_per_worker": .25, "cpu_lanes": lanes,
          "conditions": rows, "elapsed_seconds": time.monotonic()-started,
          "scope": "Three native baseline processes, one frozen validation state, two decisions per condition; no performance or success outcome."})
    print(json.dumps(read(args.output/"summary.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
