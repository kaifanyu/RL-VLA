#!/usr/bin/env bash
# One frozen baseline episode. Supply a new --output directory each time.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/integrations/multibench/core:$ROOT/integrations"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ROOT/.cache/openpi}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$ROOT/.cache/jax-robocasa}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export ROBOCASA_RENDER_THREADS="${ROBOCASA_RENDER_THREADS:-4}"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS="$ROBOCASA_RENDER_THREADS"
export NUMBA_DISABLE_JIT=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
unset JAX_PLATFORMS
exec taskset -c "${ROBOCASA_CPUSET:-16-23}" "$ROOT/third_party/openpi/.venv/bin/python" \
  -m multibench.robocasa.record_baseline "$@"
