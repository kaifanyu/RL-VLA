#!/usr/bin/env bash
# Source only for the isolated simulator, never into an active learner process.
ROBOCASA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROBOCASA_ROOT/.venv-robocasa365/bin/activate"
export PYTHONPATH="$ROBOCASA_ROOT:$ROBOCASA_ROOT/third_party/robocasa365${PYTHONPATH:+:$PYTHONPATH}"
ROBOCASA_NATIVE_LIBDIR="$ROBOCASA_ROOT/.cache/native/usr/lib/x86_64-linux-gnu"
if [[ -f /etc/redhat-release ]]; then
  ROBOCASA_NATIVE_LIBDIR="$ROBOCASA_ROOT/.cache/native/usr/lib64"
fi
export LD_LIBRARY_PATH="$ROBOCASA_NATIVE_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR="$ROBOCASA_ROOT/integrations/multibench/robocasa/assets/numba_cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
