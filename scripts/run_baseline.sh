#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$RL_VLA_ROOT"
if [ "$#" -lt 1 ]; then
  echo 'Usage: bash scripts/run_baseline.sh OUTPUT_DIRECTORY [evaluation options]'
  exit 2
fi
RL_VLA_RUN_DIR="$(realpath -m -- "$1")"
shift
mkdir -p "$RL_VLA_RUN_DIR"
printf '%s\n' "$$" > "$RL_VLA_RUN_DIR/launcher.pid"
RL_VLA_MODEL_DIR="${RL_VLA_MODEL_DIR:-$RL_VLA_ROOT/checkpoints/openpi/pi05_libero}"
RL_VLA_MODEL_CONFIG="${RL_VLA_MODEL_CONFIG:-pi05_libero}"
RL_VLA_SERVER_PID=''
RL_VLA_MONITOR_PID=''
RL_VLA_EVAL_PID=''
cleanup() {
  local RL_VLA_EXIT_CODE=$?
  for RL_VLA_CHILD in "$RL_VLA_EVAL_PID" "$RL_VLA_SERVER_PID" "$RL_VLA_MONITOR_PID"; do
    if [ -n "$RL_VLA_CHILD" ] && kill -0 "$RL_VLA_CHILD" 2>/dev/null; then
      kill -TERM "$RL_VLA_CHILD" 2>/dev/null || true
      wait "$RL_VLA_CHILD" 2>/dev/null || true
    fi
  done
  printf '%s\n' "$RL_VLA_EXIT_CODE" > "$RL_VLA_RUN_DIR/exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
.venv/bin/python - <<'PY'
import socket, subprocess
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 8001))
free = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()[0])
if free < 32768:
    raise SystemExit('Need at least 32 GiB free GPU memory before starting this baseline')
PY
nice -n 10 ionice -c 2 -n 7 taskset -c 0-3 \
  third_party/openpi/.venv/bin/python scripts/serve_baseline.py \
  --checkpoint "$RL_VLA_MODEL_DIR" --model-config "$RL_VLA_MODEL_CONFIG" \
  --port 8001 --manifest "$RL_VLA_RUN_DIR/server_manifest.json" \
  > "$RL_VLA_RUN_DIR/server.log" 2>&1 &
RL_VLA_SERVER_PID=$!
printf '%s\n' "$RL_VLA_SERVER_PID" > "$RL_VLA_RUN_DIR/pids.txt"
nice -n 10 taskset -c 0-3 .venv/bin/python scripts/monitor_baseline.py \
  --pids "$RL_VLA_RUN_DIR/pids.txt" --output "$RL_VLA_RUN_DIR/telemetry.jsonl" \
  > "$RL_VLA_RUN_DIR/monitor.log" 2>&1 &
RL_VLA_MONITOR_PID=$!
RL_VLA_READY=0
for ((RL_VLA_WAIT=0; RL_VLA_WAIT<240; RL_VLA_WAIT++)); do
  if ! kill -0 "$RL_VLA_SERVER_PID" 2>/dev/null; then
    tail -60 "$RL_VLA_RUN_DIR/server.log"
    exit 1
  fi
  if curl -fsS --max-time 2 http://127.0.0.1:8001/healthz >/dev/null 2>&1; then
    RL_VLA_READY=1
    break
  fi
  sleep 5
done
if [ "$RL_VLA_READY" -ne 1 ]; then
  echo 'Server readiness timed out; see server.log'
  exit 1
fi
nice -n 10 ionice -c 2 -n 7 taskset -c 0-3 .venv/bin/python -m rl_vla.baseline \
  --output "$RL_VLA_RUN_DIR" --model-config "$RL_VLA_MODEL_CONFIG" "$@" \
  > "$RL_VLA_RUN_DIR/evaluation.log" 2>&1 &
RL_VLA_EVAL_PID=$!
printf '%s\n%s\n' "$RL_VLA_SERVER_PID" "$RL_VLA_EVAL_PID" > "$RL_VLA_RUN_DIR/pids.txt"
wait "$RL_VLA_EVAL_PID"
cat "$RL_VLA_RUN_DIR/summary.json"
