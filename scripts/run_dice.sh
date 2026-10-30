#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$XDG_CACHE_HOME/openpi}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
  LIBERO_CONFIG_PATH="$XDG_CACHE_HOME/libero"
  ARGS=("$@")
  for ((INDEX = 0; INDEX < ${#ARGS[@]}; INDEX++)); do
    if [[ "${ARGS[$INDEX]}" == "--config" && $((INDEX + 1)) -lt ${#ARGS[@]} ]]; then
      CONFIG_PATH="${ARGS[$((INDEX + 1))]}"
      if [[ -f "$CONFIG_PATH" ]] && grep -q 'factory = "rl_vla.envs.libero_pro:make_env"' "$CONFIG_PATH"; then
        LIBERO_CONFIG_PATH="$XDG_CACHE_HOME/libero-pro"
      fi
      break
    fi
  done
fi
export LIBERO_CONFIG_PATH
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$XDG_CACHE_HOME/jax-dice}"
NATIVE_LIBDIR="$PWD/.cache/native/usr/lib/x86_64-linux-gnu"
if [[ -f /etc/redhat-release ]]; then
  NATIVE_LIBDIR="$PWD/.cache/native/usr/lib64"
fi
export LD_LIBRARY_PATH="$NATIVE_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -z "${SSL_CERT_FILE:-}" || ! -f "$SSL_CERT_FILE" ]]; then
  for CA_BUNDLE in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt; do
    if [[ -f "$CA_BUNDLE" ]]; then
      export SSL_CERT_FILE="$CA_BUNDLE"
      break
    fi
  done
fi
if [[ -n "${SSL_CERT_FILE:-}" && ( -z "${REQUESTS_CA_BUNDLE:-}" || ! -f "$REQUESTS_CA_BUNDLE" ) ]]; then
  export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
fi
export MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMBA_DISABLE_JIT=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}"
# Preserve scheduler assignments, including an explicitly empty assignment.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
PYTHON_BIN="${DICE_PYTHON:-$PWD/third_party/openpi/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "DICE Python missing: $PYTHON_BIN. See docs/dice-prismax.md." >&2
  exit 1
fi
if [[ -n "${DICE_CPUSET:-}" ]]; then
  exec taskset -c "$DICE_CPUSET" "$PYTHON_BIN" -m rl_vla.dice "$@"
fi
exec "$PYTHON_BIN" -m rl_vla.dice "$@"
