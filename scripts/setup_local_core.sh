#!/usr/bin/env bash
# Recreate the persistent OpenPI/JAX and LIBERO environments on a standalone host.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_LOCK_TIMEOUT="${UV_LOCK_TIMEOUT:-1800}"
export PATH="/workspace/upenn/bin:$PATH"
command -v uv >/dev/null || { echo 'Install uv first; see docs/dice-prismax.md' >&2; exit 1; }
fetch_pin() {
  local destination="$1" source_url="$2" revision="$3"
  if [[ -d "$destination" ]]; then
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || {
      echo "Existing $destination has another revision; inspect it before changing it." >&2
      exit 1
    }
    [[ -z "$(git -C "$destination" status --porcelain --untracked-files=no)" ]] || {
      echo "Existing $destination has tracked edits; inspect them first." >&2
      exit 1
    }
  else
    git init "$destination"
    git -C "$destination" remote add origin "$source_url"
    git -C "$destination" fetch --depth 1 origin "$revision"
    git -C "$destination" checkout --detach FETCH_HEAD
  fi
}
mkdir -p third_party
fetch_pin third_party/openpi https://github.com/Physical-Intelligence/openpi.git 215abfb217dbac7d5f1273282331b9b1866c0479
git -C third_party/openpi submodule update --init --recursive --depth 1
check_python311() {
  local interpreter="$1"
  if [[ ! -x "$interpreter" ]] || ! "$interpreter" -c \
      'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)'; then
    echo "Existing environment must use Python 3.11; preserving it unchanged: $interpreter" >&2
    return 1
  fi
}
if [[ ! -e .venv ]]; then
  uv venv --python 3.11.13 .venv
else
  check_python311 .venv/bin/python
fi
if [[ -e third_party/openpi/.venv ]]; then
  check_python311 third_party/openpi/.venv/bin/python
fi
uv pip install --python .venv/bin/python --require-hashes --torch-backend cpu \
  -r docker/requirements-libero.lock -r docker/requirements-test.lock
uv pip install --python .venv/bin/python editables==0.5 setuptools==80.9.0 google-crc32c==1.9.0 imageio==2.37.0 imageio-ffmpeg==0.6.0
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  -e . -e third_party/openpi/packages/openpi-client
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  --config-setting editable_mode=compat -e third_party/openpi/third_party/libero
uv sync --project third_party/openpi --frozen --python 3.11.13
uv pip install --python third_party/openpi/.venv/bin/python --no-deps -e third_party/openpi
third_party/openpi/.venv/bin/python - <<'PY'
import importlib.util
import shutil
from pathlib import Path
source = Path('third_party/openpi/src/openpi/models_pytorch/transformers_replace')
target = Path(importlib.util.find_spec('transformers').origin).parent
shutil.copytree(source, target, dirs_exist_ok=True)
import torch
print('OpenPI Torch:', torch.__version__)
PY
# LIBERO's config is host-dependent. Never overwrite an existing configuration.
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import yaml
folder = Path(os.environ.get('LIBERO_CONFIG_PATH', '.cache/libero'))
folder.mkdir(parents=True, exist_ok=True)
path = folder / 'config.yaml'
if not path.exists():
    root = Path('third_party/openpi/third_party/libero/libero/libero').resolve()
    payload = {'benchmark_root': str(root), 'bddl_files': str(root / 'bddl_files'),
               'init_states': str(root / 'init_files'), 'datasets': str(Path('datasets').resolve()),
               'assets': str(root / 'assets')}
    with path.open('x') as stream:
        yaml.safe_dump(payload, stream)
PY
uv pip check --python .venv/bin/python
uv pip check --python third_party/openpi/.venv/bin/python
printf '%s\n' 'Core environments ready. Next: download_pi05.py, then real simulator and native policy checks.'

