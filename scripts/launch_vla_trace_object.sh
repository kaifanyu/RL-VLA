#!/usr/bin/env bash
# One durable task worker; preparation and technical gates must already be running.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/env.sh
CAMPAIGN="${1:?campaign root}"
TASK_KEY="${2:?task key}"
export CUDA_VISIBLE_DEVICES="${3:?training GPU}"
export RL_VLA_VALIDATION_GPU="${4:?dedicated validation GPU}"
export RLDT_CPUSET="${5:?CPU set}"
export RL_VLA_EVALUATION_CPUSET="${6:?evaluation CPU set}"
export RL_VLA_VALIDATION_CPUSET="$RL_VLA_EVALUATION_CPUSET"
export RL_VLA_EVAL_SHARDS=3
export LIBERO_CONFIG_PATH="$ROOT_DIR/.cache/libero-pro"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export LP_NUM_THREADS=4
TASK_DIR="$CAMPAIGN/tasks/$TASK_KEY"
GATE="$CAMPAIGN/gate_b_actual/gate_summary.json"
DEADLINE=$((SECONDS + 7200))
while [[ ! -f "$TASK_DIR/preparation_complete.json" || ! -f "$TASK_DIR/setup_check.json" || ! -f "$TASK_DIR/train.toml" ]]; do
  if (( SECONDS > DEADLINE )); then
    echo "Preparation did not finish within the two-hour startup window" >&2
    exit 1
  fi
  sleep 10
done
echo "$(date --iso-8601=seconds): starting real-model smoke for $TASK_KEY"
taskset -c "$RLDT_CPUSET" third_party/openpi/.venv/bin/python \
  scripts/smoke_vla_trace_object.py --config "$TASK_DIR/train.toml" \
  --output "$CAMPAIGN/smoke/$TASK_KEY" > "$CAMPAIGN/smoke_${TASK_KEY}.log" 2>&1
while [[ ! -f "$GATE" || ! -f "$CAMPAIGN/launch_ready.json" ]]; do
  if (( SECONDS > DEADLINE )); then
    echo "Real-observation attention gate did not finish within startup window" >&2
    exit 1
  fi
  sleep 10
done
echo "$(date --iso-8601=seconds): starting independent campaign for $TASK_KEY"
exec .venv/bin/python scripts/run_vla_trace_objects.py \
  --task "$TASK_DIR" --output "$CAMPAIGN/methods/rldt/$TASK_KEY" --gate-b "$GATE"
