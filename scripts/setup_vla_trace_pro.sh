#!/usr/bin/env bash
# Separate pinned simulator environment for official PRO task preparation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
PRO_SETUP_UV="${PRO_SETUP_UV:-/workspace/upenn/bin/uv}"
PRO_SETUP_COMMIT=eafdb809426b13153aa1e4c42d6601844217dfec
if [[ ! -d third_party/libero_pro/.git ]]; then
  git init third_party/libero_pro
  git -C third_party/libero_pro remote add origin https://github.com/Zxy-MLlab/LIBERO-PRO.git
  git -C third_party/libero_pro fetch --depth 1 origin "$PRO_SETUP_COMMIT"
  git -C third_party/libero_pro checkout --detach FETCH_HEAD
fi
[[ "$(git -C third_party/libero_pro rev-parse HEAD)" == "$PRO_SETUP_COMMIT" ]] || {
  echo 'Existing PRO checkout has another revision; preserving it.' >&2
  exit 1
}
[[ -z "$(git -C third_party/libero_pro status --porcelain --untracked-files=no)" ]] || {
  echo 'Existing PRO checkout has tracked changes; preserving it.' >&2
  exit 1
}
if [[ ! -e .venv-pro ]]; then
  "$PRO_SETUP_UV" venv --python .venv/bin/python .venv-pro
fi
.venv-pro/bin/python -c 'import sys; assert sys.version_info[:2] == (3,11)'
"$PRO_SETUP_UV" pip install --python .venv-pro/bin/python --require-hashes --torch-backend cpu \
  -r docker/requirements-libero.lock -r docker/requirements-test.lock
"$PRO_SETUP_UV" pip install --python .venv-pro/bin/python \
  editables==0.5 setuptools==80.9.0 imageio==2.37.0 imageio-ffmpeg==0.6.0
"$PRO_SETUP_UV" pip install --python .venv-pro/bin/python --no-deps --no-build-isolation \
  -e . -e third_party/openpi/packages/openpi-client
"$PRO_SETUP_UV" pip install --python .venv-pro/bin/python --no-deps --no-build-isolation \
  --config-setting editable_mode=compat -e third_party/libero_pro
"$PRO_SETUP_UV" pip check --python .venv-pro/bin/python
printf '%s\n' 'PRO simulator ready. Set LIBERO_CONFIG_PATH=$PWD/.cache/libero-pro for task jobs.'
