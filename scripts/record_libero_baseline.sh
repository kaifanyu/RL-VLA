#!/usr/bin/env bash
# Frozen native pi0.5, isolated OSMesa simulator, complete episode video.
# Default: pi05_libero; --policy pi05_base records the general base checkpoint.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export LP_NUM_THREADS="${LP_NUM_THREADS:-4}"
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export PYTHONPATH="$ROOT/src:$ROOT/third_party/openpi/packages/openpi-client/src"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
unset JAX_PLATFORMS
exec taskset -c "${LIBERO_CPUSET:-32-39}" "$ROOT/third_party/openpi/.venv/bin/python" \
  scripts/record_native_libero.py "$@"
