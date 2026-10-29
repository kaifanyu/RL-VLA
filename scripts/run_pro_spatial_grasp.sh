#!/usr/bin/env bash
# Run as an srun step inside the existing GRASP GPU allocation.
set -euo pipefail
cd "$(dirname -- "$0")/.."
source scripts/env.sh
export LIBERO_CONFIG_PATH="$RL_VLA_ROOT/.cache/libero-pro"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
EXPERIMENT="$(realpath "${1:?experiment directory required}")"
# Prevent simultaneous steps from competing for this allocation's GPU.
exec 9>"$RL_VLA_ROOT/runs/gpu_allocation_${SLURM_JOB_ID}/experiment.lock"
flock -n 9
nvidia-smi > "$EXPERIMENT/gpu.txt"
nvidia-smi --query-gpu=timestamp,name,uuid,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv --loop=10 > "$EXPERIMENT/gpu.csv" &
MONITOR_PID=$!
cleanup() {
  result=$?
  trap - EXIT
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  if [[ "$result" -ne 0 ]]; then
    python - "$EXPERIMENT/status.json" "$result" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text()) if path.exists() else {}
data.update(state="failed", exit_code=int(sys.argv[2]))
tmp = path.with_suffix('.tmp')
tmp.write_text(json.dumps(data, indent=2) + '\n')
tmp.replace(path)
PY
  fi
  exit "$result"
}
trap cleanup EXIT
python scripts/run_pro_spatial_grasp.py --experiment "$EXPERIMENT"
