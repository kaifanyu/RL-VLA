#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$RL_VLA_ROOT"
if [ "$#" -ne 1 ]; then
  echo 'Usage: bash scripts/run_robustness.sh OUTPUT_DIRECTORY'
  exit 2
fi
exec nice -n 10 ionice -c 2 -n 7 taskset -c 0-3 \
  .venv-robustness/bin/python scripts/resource_queue.py --output "$1"
