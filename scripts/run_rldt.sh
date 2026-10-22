#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
CACHE_ROOT="$(dirname -- "$ROOT_DIR")/.cache"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$CACHE_ROOT/openpi}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$CACHE_ROOT/rl-vla-libero}"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1
export NUMBA_DISABLE_JIT=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$ROOT_DIR/runs/.jax-rldt-cache}"
ulimit -c 0
PYTHON_BIN="${RLDT_PYTHON:-$ROOT_DIR/third_party/openpi/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "OpenPI Python missing: $PYTHON_BIN. See docs/rldt.md." >&2
  exit 1
fi
# Bound compiler/BLAS CPU fan-out. Does not change any existing process affinity.
if command -v taskset >/dev/null 2>&1; then
  if [[ -z "${RLDT_CPUSET:-}" ]]; then
    RLDT_CPUSET="$("$PYTHON_BIN" -c 'import os; allowed=sorted(os.sched_getaffinity(0)); preferred=[4,5,6,7]; chosen=preferred if set(preferred)<=set(allowed) else allowed[:4]; print(",".join(map(str,chosen)))')"
  fi
  exec taskset -c "$RLDT_CPUSET" "$PYTHON_BIN" -m rl_vla.rldt "$@"
else
  exec "$PYTHON_BIN" -m rl_vla.rldt "$@"
fi
