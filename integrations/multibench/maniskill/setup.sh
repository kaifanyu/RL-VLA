#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TASK_ROOT"
if [[ -z "${SLURM_JOB_ID:-}" && "${1:-}" != "--local" ]]; then
  echo 'Run in an allocated compute step, or pass --local on a standalone workstation/container.' >&2
  exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TASK_ROOT/.cache/uv-maniskill}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$TASK_ROOT/.cache/python}"
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-1800}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export HF_HOME="${HF_HOME:-$TASK_ROOT/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MANISKILL_REVISION=33967b9e3ead1f841eec57cc9f31d0d8b8cf0907
if [[ ! -d third_party/maniskill3/.git ]]; then
  git init third_party/maniskill3
  git -C third_party/maniskill3 remote add origin https://github.com/haosulab/ManiSkill.git
  git -C third_party/maniskill3 config remote.origin.promisor true
  git -C third_party/maniskill3 config remote.origin.partialclonefilter blob:none
  git -C third_party/maniskill3 sparse-checkout init --cone
  git -C third_party/maniskill3 sparse-checkout set mani_skill
  git -C third_party/maniskill3 fetch --depth 1 --filter=blob:none origin "$MANISKILL_REVISION"
  git -C third_party/maniskill3 checkout --detach FETCH_HEAD
fi
test "$(git -C third_party/maniskill3 rev-parse HEAD)" = "$MANISKILL_REVISION"
RLINF_REVISION=0505431899574619da86f551bad70b71e0ea2177
if [[ ! -d third_party/rlinf/.git ]]; then
  git init third_party/rlinf
  git -C third_party/rlinf remote add origin https://github.com/RLinf/RLinf.git
  git -C third_party/rlinf fetch --depth 1 origin "$RLINF_REVISION"
  git -C third_party/rlinf checkout --detach FETCH_HEAD
  git -C third_party/rlinf apply ../patches/rlinf.patch
fi
test "$(git -C third_party/rlinf rev-parse HEAD)" = "$RLINF_REVISION"
if [[ ! -f .venv-maniskill3/pyvenv.cfg ]]; then
  uv python install 3.11.13
  uv venv --python 3.11.13 .venv-maniskill3
fi
# Rebase only the editable source path at runtime; preserve the committed lock.
mkdir -p runs/setup/maniskill
.venv-maniskill3/bin/python - <<'PYLOCK'
from pathlib import Path
root = Path.cwd()
source = root / "integrations/multibench/maniskill/requirements.lock"
lines = ["-e " + str(root / "third_party/maniskill3") if line.startswith("-e ") else line
         for line in source.read_text().splitlines()]
(root / "runs/setup/maniskill/requirements-local.txt").write_text("\n".join(lines) + "\n")
PYLOCK
wheel_args=()
for cache in .cache/maniskill-wheels .cache/wheelhouse-maniskill-small; do
  if [[ -d "$cache" ]]; then wheel_args+=(--find-links "$cache"); fi
done
uv pip install --python .venv-maniskill3/bin/python --torch-backend cu128 \
  "${wheel_args[@]}" -r runs/setup/maniskill/requirements-local.txt
uv pip freeze --python .venv-maniskill3/bin/python > runs/setup/maniskill/requirements.txt
uv pip check --python .venv-maniskill3/bin/python
.venv-maniskill3/bin/python integrations/multibench/maniskill/download_assets.py
.venv-maniskill3/bin/python integrations/multibench/maniskill/prepare_software_vulkan.py
