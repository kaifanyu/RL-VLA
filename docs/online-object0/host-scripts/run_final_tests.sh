#!/usr/bin/env bash
cd /workspace/upenn/worktrees/online-object0-ready-20261105
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O=/workspace/smrithi/object0-launch-readiness-20261105
timeout 180 taskset -c 12-15 third_party/openpi/.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp="$O/tests_final" tests/test_online_object0.py tests/test_dice.py tests/test_dice_fidelity.py tests/test_rldt_pipeline.py tests/test_rldt_pi05.py tests/test_rldt_math.py tests/test_pro_residual.py > "$O/evidence/tests_final.log" 2>&1
printf '{"exit_code":%s,"production_launched":false}\n' "$?" > "$O/evidence/tests_final_status.json"
third_party/openpi/.venv/bin/python "$O/save_handoff_state.py" > "$O/evidence/final_tests_snapshot.log" 2>&1
