#!/usr/bin/env bash
#SBATCH --job-name=rl-vla-rldt-smoke
#SBATCH --account=gu-account
#SBATCH --partition=gu-compute
#SBATCH --qos=gu-low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=runs/rldt_pi05_smoke_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the RL-VLA repository root}"
source scripts/env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
RLDT_OUTPUT="$RL_VLA_ROOT/runs/rldt_pi05_smoke_${SLURM_JOB_ID}"
printf 'RLDT smoke: job=%s host=%s output=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$RLDT_OUTPUT"
nvidia-smi
nvidia-smi --query-gpu=timestamp,name,uuid,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv --loop=5 > "${RLDT_OUTPUT}.gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true' EXIT
bash scripts/run_rldt.sh train --config configs/rldt_pi05_libero.toml \
  --smoke --particles 8 --output "$RLDT_OUTPUT"
