#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
source scripts/env.sh
export LIBERO_CONFIG_PATH="$RL_VLA_ROOT/.cache/libero-pro"
exec python scripts/run_pi_rl_suite_grasp.py --suite "${1:?suite directory required}" --gates "${2:?readiness JSON required}"
