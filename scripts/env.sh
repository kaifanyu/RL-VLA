# Source this file from any directory to use this repository's local environment.
export RL_VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RL_VLA_ROOT/.venv/bin/activate"
export PATH="$RL_VLA_ROOT/.venv/bin:$RL_VLA_ROOT/.cache/bootstrap/bin:$RL_VLA_ROOT/.cache/native/usr/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RL_VLA_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$RL_VLA_ROOT/.cache/python}"
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RL_VLA_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$RL_VLA_ROOT/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$RL_VLA_ROOT/datasets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$RL_VLA_ROOT/.cache/openpi}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$RL_VLA_ROOT/.cache/libero}"
export LD_LIBRARY_PATH="$RL_VLA_ROOT/.cache/native/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export LP_NUM_THREADS=1 NUMBA_CACHE_DIR="$RL_VLA_ROOT/.cache/numba"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# Retain the repository's Robosuite/Mesa workaround using Numba's Python fallbacks.
# MuJoCo physics and Mesa rendering remain native.
export NUMBA_DISABLE_JIT=1
ulimit -c 0
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export RL_VLA_OPENPI_HOST="${RL_VLA_OPENPI_HOST:-127.0.0.1}"
export RL_VLA_OPENPI_PORT="${RL_VLA_OPENPI_PORT:-8001}"
# Preserve the GPU assignment supplied by Slurm (including MIG UUIDs).
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$RL_VLA_ROOT/.cache/jax-rl-vla}"
