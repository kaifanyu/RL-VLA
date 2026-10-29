#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/../.."
MULTIBENCH_XLA_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}"
source scripts/env.sh
export PYTHONPATH="$RL_VLA_ROOT/integrations/multibench/core:$RL_VLA_ROOT/integrations:$RL_VLA_ROOT/third_party/rlinf${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MULTIBENCH_XLA_FRACTION"
export JAX_COMPILATION_CACHE_DIR="$RL_VLA_ROOT/runs/.jax-multibench-cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_MODE=disabled HF_HUB_DISABLE_TELEMETRY=1
export RLINF_EXT_MODULE=multibench.plugin
export HYDRA_FULL_ERROR=1 NCCL_DEBUG=WARN USE_TF=0 USE_TORCH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export RAY_TMPDIR="/tmp/multibench-${USER}-${SLURM_JOB_ID:-local}-$$"
backend="$1"
shift
case "$backend" in
  native) unset JAX_PLATFORMS; interpreter="$RL_VLA_ROOT/third_party/openpi/.venv/bin/python" ;;
  torch) export JAX_PLATFORMS=cpu; interpreter="$RL_VLA_ROOT/.venv-rlinf/bin/python" ;;
  *) echo 'Expected native or torch backend' >&2; exit 2 ;;
esac
ulimit -c 0
exec "$interpreter" "$@"
