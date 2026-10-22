#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR=/workspace/upenn/.cache/jax-dice
# Default is read-only setup checking. No persistent server is started.
exec nice -n 10 taskset -c 0-3 third_party/openpi/.venv/bin/python -m rl_vla.dice "$@"
