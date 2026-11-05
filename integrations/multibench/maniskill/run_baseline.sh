#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="$ROOT/integrations/multibench/core:$ROOT/integrations:$ROOT/third_party/openpi/packages/openpi-client/src:$ROOT/third_party/rlinf"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ROOT/.cache/openpi}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
export JAX_COMPILATION_CACHE_DIR="$ROOT/.cache/jax-maniskill-baseline"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
unset JAX_PLATFORMS
ulimit -c 0
exec taskset -c "${MANISKILL_CPUSET:-24-31}" "$ROOT/third_party/openpi/.venv/bin/python" \
  integrations/multibench/maniskill/record_baseline.py "$@"
