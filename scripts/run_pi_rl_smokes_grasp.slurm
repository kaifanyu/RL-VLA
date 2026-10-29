#!/usr/bin/env bash
#SBATCH --job-name=pirl-functional-smokes
#SBATCH --account=gu-account
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=runs/setup/pirl_smokes_%j.slurm.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the RL-VLA repository root}"
source scripts/env.sh
export SLURM_STEP_ID="${SLURM_STEP_ID:-batch}"
exec "$RL_VLA_ROOT/.venv/bin/python" scripts/run_pi_rl_smokes_grasp.py --suite "${1:?Suite path required}"
