#!/usr/bin/env bash
# FlashSAC residual on LIBERO-PRO: OpenPI/JAX + Torch learner in third_party/openpi/.venv,
# simulator children in .venv-robustness (PRO) or .venv (original LIBERO).
# CPUs default to 4-7, away from the robustness queue's 0-3. Override with RL_VLA_CPUSET.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$RL_VLA_ROOT"
export PYTHONPATH="$RL_VLA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CPUSET="${RL_VLA_CPUSET:-4-7}"
exec nice -n 10 ionice -c 2 -n 7 taskset -c "$CPUSET" \
  third_party/openpi/.venv/bin/python -m rl_vla.pro_residual "$@"
