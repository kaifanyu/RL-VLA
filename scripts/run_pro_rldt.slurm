#!/usr/bin/env bash
#SBATCH --job-name=pi05-pro-rldt
#SBATCH --account=jgu32-lab
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%1
#SBATCH --no-requeue
#SBATCH --output=runs/pro_rldt_%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the RL-VLA repository root}"
source scripts/env.sh
export LIBERO_CONFIG_PATH="$RL_VLA_ROOT/.cache/libero-pro"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export PYTHONPATH="$RL_VLA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PRO_EXPERIMENT="${1:?Provide experiment directory}"
PRO_PHASE="${2:?Provide baseline or train}"
PRO_TASKS=(spatial_0 object_0)
PRO_TASK="${PRO_TASKS[${SLURM_ARRAY_TASK_ID:?}]}"
PRO_LOG="$PRO_EXPERIMENT/$PRO_TASK/${PRO_PHASE}_gpu_${SLURM_JOB_ID}.csv"
printf 'phase=%s task=%s job=%s host=%s\n' "$PRO_PHASE" "$PRO_TASK" "$SLURM_JOB_ID" "$(hostname)"
nvidia-smi
# Use a job-specific file; on MIG, this reports physical-device totals as well.
nvidia-smi --query-gpu=timestamp,name,uuid,utilization.gpu,memory.used,memory.total \
  --format=csv --loop=5 > "$PRO_LOG" &
PRO_MONITOR_PID=$!
trap 'kill "$PRO_MONITOR_PID" 2>/dev/null || true; wait "$PRO_MONITOR_PID" 2>/dev/null || true' EXIT
python scripts/run_pro_rldt.py --experiment "$PRO_EXPERIMENT" --task "$PRO_TASK" --phase "$PRO_PHASE"
