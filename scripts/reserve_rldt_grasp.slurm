#!/usr/bin/env bash
#SBATCH --job-name=rl-vla-development
#SBATCH --account=gu-account
#SBATCH --partition=gu-compute
#SBATCH --qos=gu-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue
#SBATCH --output=runs/gpu_allocation_%j.slurm.log

# Keep one allocation alive while independent srun steps use its GPU.
# Submit from the RL-VLA repository root. No training starts automatically.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the RL-VLA repository root}"
ALLOCATION_DIR="$PWD/runs/gpu_allocation_${SLURM_JOB_ID}"
mkdir -p "$ALLOCATION_DIR"

.venv/bin/python - "$ALLOCATION_DIR/allocation.json" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

record = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "host": socket.gethostname(),
    "workspace": str(Path.cwd()),
    "started_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
    "requested_duration": "4 days",
    "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
    "memory_mib": os.environ.get("SLURM_MEM_PER_NODE"),
    "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "purpose": "Reusable allocation for LIBERO-PRO RLDT development",
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
PY

printf 'Reusable GPU allocation %s on %s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME"
printf 'Metadata: %s/allocation.json\n' "$ALLOCATION_DIR"
printf 'Attach: srun --jobid=%s --overlap --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --pty bash\n' "$SLURM_JOB_ID"

HOLD_PID=
cleanup() {
    if [[ -n "$HOLD_PID" ]]; then
        kill "$HOLD_PID" 2>/dev/null || true
        wait "$HOLD_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 0' TERM INT

while true; do
    sleep 3600 &
    HOLD_PID=$!
    wait "$HOLD_PID"
done
