#!/usr/bin/env bash
# Continue the official installer on the Ubuntu GPU node: its glibc supports
# LeRobot's pinned rerun wheel, whereas the RHEL login node cannot resolve it.
set -euo pipefail
cd "$(dirname -- "$0")/.."
source scripts/env.sh
export PYTHONPATH="${PYTHONPATH:-}"
source "$RL_VLA_ROOT/.venv-rlinf/bin/activate"
uv pip install "$RL_VLA_ROOT/third_party/rlinf_openpi" 'rerun-sdk==0.23.1'
uv pip install -r "$RL_VLA_ROOT/third_party/rlinf/requirements/embodied/models/openpi.txt"
uv pip install 'pyarrow==20.0.0'
# The pinned RLinf entrypoint requests Hydra's 1.1 compatibility behavior,
# which Hydra 1.4 no longer accepts. Keep its supported stable runtime.
uv pip install 'hydra-core==1.3.2' 'omegaconf==2.3.0'
uv pip install --torch-backend cu128 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' 'torchaudio==2.7.1+cu128'
uv pip install --no-deps -e "$RL_VLA_ROOT/third_party/rlinf"
cp -r "$RL_VLA_ROOT/third_party/rlinf_openpi/src/openpi/models_pytorch/transformers_replace/"* \
  "$RL_VLA_ROOT/.venv-rlinf/lib/python3.11/site-packages/transformers/"
bash "$RL_VLA_ROOT/third_party/rlinf/requirements/embodied/download_assets.sh" --dir "$RL_VLA_ROOT" --assets openpi
uv pip uninstall pynvml || true
# TensorFlow Addons is an unused common-installer extra for other models; its
# obsolete typeguard<3 requirement conflicts with upstream OpenPI's Tyro pin.
uv pip uninstall tensorflow-addons tensorflow-graphics
# TensorFlow is optional in RLinf and unused by this OpenPI/PRO runner. Its
# prerelease wheel requires NumPy 2's C API, while OpenPI requires NumPy <2.
uv pip uninstall tensorflow keras-nightly
uv pip freeze > "$RL_VLA_ROOT/runs/setup/rlinf_freeze.txt"
uv pip check > "$RL_VLA_ROOT/runs/setup/rlinf_dependency_check.txt" 2>&1
python -c 'import torch,openpi,rlinf,transformers; print(torch.__version__, torch.version.cuda, openpi.__file__, rlinf.__file__, transformers.__version__)'
