#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TASK_ROOT"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Run setup inside an allocated compute step; installation must not compete on the login host.' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MANISKILL_REVISION=33967b9e3ead1f841eec57cc9f31d0d8b8cf0907
if [[ ! -d third_party/maniskill3/.git ]]; then
  git clone --depth 1 --branch v3.0.0b22 https://github.com/haosulab/ManiSkill.git third_party/maniskill3
fi
test "$(git -C third_party/maniskill3 rev-parse HEAD)" = "$MANISKILL_REVISION"
if [[ ! -f .venv-maniskill3/pyvenv.cfg ]]; then
  uv venv --python .cache/python/cpython-3.11.13-linux-x86_64-gnu/bin/python3.11 .venv-maniskill3
fi
uv pip install --python .venv-maniskill3/bin/python torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-maniskill3/bin/python -e third_party/maniskill3 numpy==1.26.4 huggingface-hub==0.36.2 omegaconf==2.3.0 msgpack==1.1.2
uv pip freeze --python .venv-maniskill3/bin/python > integrations/multibench/maniskill/requirements.lock
.venv-maniskill3/bin/python integrations/multibench/maniskill/download_assets.py
