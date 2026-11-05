#!/usr/bin/env bash
set -euo pipefail
ROBOCASA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROBOCASA_ROOT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROBOCASA_ROOT/.cache/uv-robocasa}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROBOCASA_ROOT/.cache/python}"
export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-1800}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1200}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-8}"
export UV_LINK_MODE=copy
export GIT_LFS_SKIP_SMUDGE=1
export HF_HOME="${HF_HOME:-$ROBOCASA_ROOT/.cache/huggingface}"
mkdir -p runs/setup/robocasa365

clone_pinned() {
  local repo_url="$1" checkout_path="$2" commit="$3"
  if [[ ! -d "$checkout_path/.git" ]]; then
    git clone "$repo_url" "$checkout_path"
    git -C "$checkout_path" checkout --detach "$commit"
  fi
  [[ "$(git -C "$checkout_path" rev-parse HEAD)" == "$commit" ]]
}
clone_pinned https://github.com/robocasa/robocasa.git third_party/robocasa365 dc73a8ce18e4b8947e8df2ae2958245ca823f3b4
clone_pinned https://github.com/ARISE-Initiative/robosuite.git third_party/robocasa365/dependencies/robosuite 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
clone_pinned https://github.com/robocasa-benchmark/openpi.git third_party/robocasa365/dependencies/openpi_reference 5a6beda9ff99da30b4e1b59320f6a32971d7c397
if [[ ! -x .venv-robocasa365/bin/python ]]; then
  uv venv --python 3.11.13 .venv-robocasa365
fi

# The isolated environment serves MuJoCo simulation. LeRobot dataset conversion
# and Tianshou's vector benchmark are unused. The runtime shim also avoids
# recursively packaging the simulator asset directory. Policy runtimes are separate.
uv pip install --python .venv-robocasa365/bin/python --link-mode=copy \
  -e third_party/robocasa365/dependencies/robosuite \
  numpy==2.2.5 numba==0.61.2 scipy==1.15.3 mujoco==3.3.1 \
  pygame Pillow opencv-python pyyaml pynput tqdm termcolor imageio h5py lxml hidapi gymnasium
uv pip install --python .venv-robocasa365/bin/python --no-deps -e integrations/multibench/robocasa/runtime_package
# Load the pinned source bridge via env.sh / ProcessEnv's PYTHONPATH. Installing
# its distribution would declare NumPy<2, conflicting with RoboCasa's NumPy2.
# The camera/IPC helpers are verified in the real simulator smoke checks.
uv pip install --python .venv-robocasa365/bin/python msgpack websockets dm-tree huggingface-hub==0.36.2
source integrations/multibench/robocasa/env.sh
if [[ ! -f third_party/robocasa365/dependencies/robosuite/robosuite/macros_private.py ]]; then
  python third_party/robocasa365/dependencies/robosuite/robosuite/scripts/setup_macros.py
fi
if [[ ! -f third_party/robocasa365/robocasa/macros_private.py ]]; then
  python -m robocasa.scripts.setup_macros
fi
if [[ "${ROBOCASA_ASSET_SOURCE:-huggingface}" == "box" ]]; then
  python integrations/multibench/robocasa/download_assets.py
else
  asset_options=()
  if [[ "${ROBOCASA_OPTIONAL_ASSETS:-0}" == "1" ]]; then asset_options+=(--include-optional); fi
  python integrations/multibench/robocasa/download_assets_hf.py "${asset_options[@]}" \
    --transport "${ROBOCASA_ASSET_TRANSPORT:-ranges}" \
    --workers "${ROBOCASA_ASSET_WORKERS:-16}" --file-workers "${ROBOCASA_ASSET_FILES:-3}"
fi
# Preserve the historical lock; record the actual host installation separately.
uv pip freeze --python .venv-robocasa365/bin/python > runs/setup/robocasa365/requirements-installed.txt
