#!/usr/bin/env bash
set -euo pipefail
ROBOCASA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROBOCASA_ROOT"

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
  uv venv --python 3.11 .venv-robocasa365
fi

# The isolated environment serves MuJoCo simulation. LeRobot dataset conversion
# and Tianshou's vector benchmark are unused; LeRobot 0.3.3 requires rerun-sdk
# wheels incompatible with this host's glibc 2.28. Policy runtimes are separate.
uv pip install --python .venv-robocasa365/bin/python --link-mode=copy \
  -e third_party/robocasa365/dependencies/robosuite \
  numpy==2.2.5 numba==0.61.2 scipy==1.15.3 mujoco==3.3.1 \
  pygame Pillow opencv-python pyyaml pynput tqdm termcolor imageio h5py lxml hidapi gymnasium
uv pip install --python .venv-robocasa365/bin/python --no-deps -e integrations/multibench/robocasa/runtime_package
uv pip install --python .venv-robocasa365/bin/python --no-deps -e third_party/openpi/packages/openpi-client
uv pip install --python .venv-robocasa365/bin/python msgpack websockets dm-tree
source integrations/multibench/robocasa/env.sh
if [[ ! -f third_party/robocasa365/robocasa/macros_private.py ]]; then
  python -m robocasa.scripts.setup_macros
fi
python integrations/multibench/robocasa/download_assets.py
uv pip freeze --python .venv-robocasa365/bin/python > integrations/multibench/robocasa/requirements.lock
