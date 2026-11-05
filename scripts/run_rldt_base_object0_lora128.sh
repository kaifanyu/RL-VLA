#!/usr/bin/env bash
# Native JAX RLDT from general pi05_base, with rank-128 action-expert adapters.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-3}"
export RLDT_CPUSET="${RLDT_CPUSET:-80-87}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$ROOT_DIR/runs/.jax-rldt-base-object0-lora128}"
COMMAND="${1:-preflight}"
if [[ $# -gt 0 ]]; then shift; fi
exec bash scripts/run_rldt.sh "$COMMAND" \
  --config configs/rldt_pi05_base_object0_lora128.toml "$@"
