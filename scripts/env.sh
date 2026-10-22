# Source this file from any directory to use this repository's local environment.
export RL_VLA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$RL_VLA_ROOT/.venv/bin/activate"
export PATH="$RL_VLA_ROOT/.venv/bin:/workspace/upenn/.bootstrap/bin:$PATH"
export UV_CACHE_DIR=/workspace/upenn/.uv-cache
export UV_PYTHON_INSTALL_DIR=/workspace/upenn/.uv-python
export HF_HOME=/workspace/upenn/.cache/huggingface
export HF_LEROBOT_HOME=/workspace/upenn/datasets
export OPENPI_DATA_HOME=/workspace/upenn/.cache/openpi
export LIBERO_CONFIG_PATH=/workspace/upenn/.cache/rl-vla-libero
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# The pinned Numba compiler crashes during Robosuite initialization on this image.
# Its Python fallbacks work; MuJoCo physics and Mesa rendering remain native.
export NUMBA_DISABLE_JIT=1
ulimit -c 0
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export RL_VLA_OPENPI_HOST=127.0.0.1 RL_VLA_OPENPI_PORT=8001
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR=/workspace/upenn/.cache/jax-rl-vla
