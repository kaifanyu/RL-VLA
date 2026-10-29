#!/usr/bin/env bash
#SBATCH --job-name=rl-vla-rldt-train
#SBATCH --account=gu-account
#SBATCH --partition=gu-compute
#SBATCH --qos=gu-low
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=runs/rldt_pi05_libero_%j.slurm.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the RL-VLA repository root}"
source scripts/env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
EXPERIMENT="$RL_VLA_ROOT/runs/rldt_pi05_libero_${SLURM_JOB_ID}"
mkdir "$EXPERIMENT"
cp configs/rldt_pi05_libero.toml "$EXPERIMENT/config.toml"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$EXPERIMENT/source_snapshot.tar.gz" \
  src scripts configs pyproject.toml
git rev-parse HEAD > "$EXPERIMENT/source_commit.txt"
git diff --binary > "$EXPERIMENT/source_changes.patch"
nvidia-smi > "$EXPERIMENT/gpu.txt"
nvidia-smi --query-gpu=timestamp,name,uuid,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv --loop=10 > "$EXPERIMENT/gpu.csv" &
MONITOR_PID=$!
STAGE=starting
write_status() {
  python - "$EXPERIMENT/status.json" "$STAGE" "$1" "${2:-0}" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
path = Path(sys.argv[1])
record = dict(job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
              stage=sys.argv[2], state=sys.argv[3], exit_code=int(sys.argv[4]),
              updated=datetime.now(ZoneInfo('America/New_York')).isoformat())
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(record, indent=2) + '\n')
temporary.replace(path)
PY
}
cleanup() {
  result=$?
  trap - EXIT
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  if [[ "$result" -ne 0 ]]; then write_status failed "$result"; fi
  exit "$result"
}
trap cleanup EXIT
printf 'RLDT experiment: %s\n' "$EXPERIMENT"
STAGE=training
write_status running
bash scripts/run_rldt.sh train --config "$EXPERIMENT/config.toml" \
  --output "$EXPERIMENT/training" > "$EXPERIMENT/train.log" 2>&1
python scripts/analyze_rldt.py "$EXPERIMENT" > "$EXPERIMENT/analysis.log" 2>&1
STAGE=evaluation_frozen
write_status running
bash scripts/run_rldt.sh evaluate --config "$EXPERIMENT/config.toml" \
  --episodes 50 --seed 100000 --output "$EXPERIMENT/evaluation_frozen.json" \
  > "$EXPERIMENT/evaluation_frozen.log" 2>&1
STAGE=evaluation_trained
write_status running
bash scripts/run_rldt.sh evaluate --config "$EXPERIMENT/config.toml" \
  --checkpoint "$EXPERIMENT/training/final" --episodes 50 --seed 100000 \
  --output "$EXPERIMENT/evaluation_trained.json" > "$EXPERIMENT/evaluation_trained.log" 2>&1
STAGE=analysis
write_status running
python scripts/analyze_rldt.py "$EXPERIMENT" > "$EXPERIMENT/analysis.log" 2>&1
STAGE=complete
write_status completed
