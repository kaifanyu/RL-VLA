#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
source scripts/env.sh
export LIBERO_CONFIG_PATH="$RL_VLA_ROOT/.cache/libero-pro"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
exec python scripts/run_object_campaign_grasp.py --suite "${1:?campaign directory required}" --gates "${2:?readiness file required}" "${@:3}"
