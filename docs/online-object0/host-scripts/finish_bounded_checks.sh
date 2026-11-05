#!/usr/bin/env bash
# Only bounded verification. NEVER starts a benchmark/production launcher.
set -u
cd /workspace/upenn/worktrees/online-object0-ready-20261105
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src"
export LIBERO_CONFIG_PATH=/workspace/upenn/RL-VLA/.cache/libero-pro
export LD_LIBRARY_PATH=/workspace/upenn/RL-VLA/.cache/native/usr/lib/x86_64-linux-gnu
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1 LP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export JAX_COMPILATION_CACHE_DIR=/workspace/smrithi/object0-launch-readiness-20261105/jax-cache
export OPENPI_DATA_HOME=/workspace/upenn/RL-VLA/.cache/openpi OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O=/workspace/smrithi/object0-launch-readiness-20261105
RETRY_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
P="$O/parity_retry_$RETRY_ID"
B="$O/dice_final_equivalence_restore_$RETRY_ID"
mkdir "$P"
cp "$O/common_real_parity/common_input.npz" "$P/common_input.npz"
printf '%s\n' "Bounded finisher started $(date -u --iso-8601=seconds)"
timeout 600 taskset -c 4-11 third_party/openpi/.venv/bin/python scripts/check_online_object0_parity.py --output "$P" --wrapper dice > "$P/dice_parity.log" 2>&1
parity_status=$?
printf '%s\n' "DICE batching parity status: $parity_status"
# Do not validate an optimization further if its required serial parity failed.
if [[ "$parity_status" == 0 ]]; then
  timeout 900 taskset -c 4-11 third_party/openpi/.venv/bin/python scripts/benchmark_dice_production.py --config configs/online_object0/dice_seed0.toml --observation /workspace/upenn/RL-VLA/runs/vla_trace_objects_20261031/methods/rldt/object_0/training/trajectories/episode_000008/chunk_000005.npz --output "$B" --updates 2 --optimized --microbatch-rows 16 --distinct-observations --restore-equivalence > "$P/dice_final_equivalence_restore.log" 2>&1
  benchmark_status=$?
else
  benchmark_status=125
fi
printf '{"parity_exit":%s,"benchmark_exit":%s,"production_launched":false,"parity_directory":"%s","benchmark_directory":"%s"}\n' "$parity_status" "$benchmark_status" "$P" "$B" > "$P/status.json"
cp "$P/status.json" "$O/evidence/detached_finisher_latest.json"
# Existing bounded checks were launched through persistent exec sessions. Monitor
# their original PIDs; do not kill/restart or claim they were reparented to tmux.
for check_pid in 1262815 1270032; do
  timeout 1200 bash -c 'while kill -0 "$1" 2>/dev/null; do sleep 10; done' _ "$check_pid" || true
done
python3 "$O/save_handoff_state.py"
printf '%s\n' "Bounded finisher ended $(date -u --iso-8601=seconds)"
