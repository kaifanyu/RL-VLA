#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
source scripts/env.sh
export LIBERO_CONFIG_PATH="$RL_VLA_ROOT/.cache/libero-pro"
export PYTHONPATH="$RL_VLA_ROOT/integrations:$RL_VLA_ROOT/src:$RL_VLA_ROOT/third_party/rlinf${PYTHONPATH:+:$PYTHONPATH}"
export ROBOT_PLATFORM=LIBERO
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu
export USE_TF=0 USE_TORCH=1
export WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
# RLinf shares CUDA tensors between its actor and rollout processes. Expandable
# allocations require pidfd_getfd for IPC, which this cluster disallows.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export RAY_TMPDIR="/tmp/rlvla-${USER}-${SLURM_JOB_ID:-local}-$$"
exec "$RL_VLA_ROOT/.venv-rlinf/bin/python" "$@"
