#!/usr/bin/env bash
#SBATCH --job-name=rl-vla-install
#SBATCH --account=jgu32-lab
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=00:15:00
#SBATCH --output=runs/setup/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the RL-VLA repository root}"
source scripts/env.sh
export PYTHONPATH="$RL_VLA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi
third_party/openpi/.venv/bin/python scripts/check_local_install.py \
  --output "runs/setup/gpu-${SLURM_JOB_ID}"
third_party/openpi/.venv/bin/python scripts/check_rldt_pi05.py \
  --skip-gradient --output "runs/setup/rldt-parity-${SLURM_JOB_ID}.json"
