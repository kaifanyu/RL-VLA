"""Run both real PPO smoke tests sequentially in one short GPU allocation."""

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def select_healthy_gpu(suite):
    """Probe only assigned GPUs in isolated processes before loading models."""
    assigned = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    probe = ("import json,torch; "
             "x=torch.ones((1024,1024),device='cuda'); y=x@x; "
             "torch.cuda.synchronize(); assert torch.isfinite(y).all(); "
             "print(json.dumps({'device':torch.cuda.get_device_name(0),"
             "'memory_bytes':torch.cuda.get_device_properties(0).total_memory}))")
    attempts = []
    for device in assigned:
        result = subprocess.run([str(ROOT / ".venv-rlinf/bin/python"), "-c", probe],
                                env=os.environ | {"CUDA_VISIBLE_DEVICES": device},
                                capture_output=True, text=True, timeout=60, check=False)
        attempts.append({"assigned_device": device, "returncode": result.returncode,
                         "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]})
        if result.returncode == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            break
    record = {"allocation": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
              "attempts": attempts, "selected": device if result.returncode == 0 else None}
    with (suite / "smoke/gpu_preflight.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    if record["selected"] is None:
        raise RuntimeError("No assigned GPU passed the CUDA preflight")
    return record["selected"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    suite = args.suite.resolve()
    allocation = os.environ["SLURM_JOB_ID"]
    (ROOT / "runs" / f"gpu_allocation_{allocation}").mkdir(exist_ok=True)
    selected = select_healthy_gpu(suite)
    for method in ("flow_sde", "flow_noise"):
        directory = suite / "smoke" / method
        # RLinf detects all Slurm-accessible devices through NVML, independently
        # of CUDA_VISIBLE_DEVICES. Explicit placement keeps one worker per role
        # and prevents its scheduler from selecting the failed device.
        config_path = directory / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        config["cluster"]["component_placement"] = {"actor,env,rollout": selected}
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, str(ROOT / "scripts/run_locked_gpu_grasp.py"),
                   "--status", str(directory / "lock_status.json"), "--",
                   "bash", str(ROOT / "scripts/run_pi_rl.sh"),
                   str(ROOT / "integrations/rlinf_bridge/train.py"),
                   "--config-path", str(directory), "--config-name", "config"]
        record = {"command": command, "allocation": allocation,
                  "step_id": os.environ.get("SLURM_STEP_ID", "batch"),
                  "host": socket.gethostname(),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "started_at": datetime.now(UTC).isoformat(),
                  "purpose": "Functional smoke only; full comparisons use the original GPU allocation"}
        with (directory / "launch.json").open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        with (directory / "launch.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
