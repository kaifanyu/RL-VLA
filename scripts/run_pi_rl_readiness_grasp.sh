#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
source scripts/env.sh
exec "$RL_VLA_ROOT/.venv/bin/python" scripts/run_pi_rl_readiness_grasp.py "$@"
