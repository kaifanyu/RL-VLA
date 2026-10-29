#!/usr/bin/env bash
# Reproduce this task-specific simulator without changing any existing runtime.
set -euo pipefail
ROBOTWIN_INTEGRATION=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROBOTWIN_PROJECT=$(cd -- "$ROBOTWIN_INTEGRATION/../../.." && pwd)
cd "$ROBOTWIN_PROJECT"
ROBOTWIN_PIN=dca9ec682688821c42944eacd7f5dd9bfb15396f
if [[ ! -d third_party/robotwin/.git ]]; then
  git clone --depth 1 --branch RLinf_support https://github.com/RoboTwin-Platform/RoboTwin.git third_party/robotwin
fi
if [[ "$(git -C third_party/robotwin rev-parse HEAD)" != "$ROBOTWIN_PIN" ]]; then
  git -C third_party/robotwin fetch --depth 1 origin "$ROBOTWIN_PIN"
  git -C third_party/robotwin checkout --detach "$ROBOTWIN_PIN"
fi
if [[ ! -x .venv-robotwin/bin/python ]]; then
  uv venv --python 3.10 .venv-robotwin
fi
uv pip install --python .venv-robotwin/bin/python -r "$ROBOTWIN_INTEGRATION/requirements.lock"
# This one-line fix is explicitly required by RoboTwin's installation guide.
.venv-robotwin/bin/python - <<'PY'
from pathlib import Path
import mplib
p = Path(mplib.__file__).parent / 'planner.py'
s = p.read_text()
old = 'if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:'
new = 'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:'
assert old in s or new in s
p.write_text(s.replace(old, new))
PY
if [[ ! -f third_party/robotwin/assets/embodiments.zip ]]; then
  curl --fail --location --retry 3 --output third_party/robotwin/assets/embodiments.zip \
    https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/3dc3b798668feb99ac61cc9086d84cbcc3d79186/embodiments.zip
fi
.venv-robotwin/bin/python - <<'PY'
from pathlib import Path
import hashlib, zipfile
p = Path('third_party/robotwin/assets/embodiments.zip')
assert hashlib.sha256(p.read_bytes()).hexdigest() == '85ffeff55a5066def5931224a85cfa3f8abaa1fbf779bd17789a7fb3f85bc789'
with zipfile.ZipFile(p) as z:
    for name in z.namelist():
        if '/aloha-agilex/' in '/' + name and not name.startswith('__MACOSX/'):
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            z.extract(name, p.parent)
PY
if [[ ! -f third_party/robotwin/assets/objects.zip ]]; then
  curl --fail --location --retry 3 --output third_party/robotwin/assets/objects.zip \
    https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/3dc3b798668feb99ac61cc9086d84cbcc3d79186/objects.zip
fi
.venv-robotwin/bin/python - <<'PY'
from pathlib import Path
import hashlib, zipfile
p = Path('third_party/robotwin/assets/objects.zip')
h = hashlib.sha256()
with p.open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024**2), b''):
        h.update(block)
assert h.hexdigest() == '6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27'
with zipfile.ZipFile(p) as z:
    for name in ('objects/objaverse/list.json', 'objects/same.json'):
        z.extract(name, p.parent)
PY
if [[ ! -f "$ROBOTWIN_INTEGRATION/handover_block_demo_clean.zip" ]]; then
  curl --fail --location --retry 3 --output "$ROBOTWIN_INTEGRATION/handover_block_demo_clean.zip" \
    https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/3dc3b798668feb99ac61cc9086d84cbcc3d79186/dataset/handover_block/demo_clean.zip
fi
.venv-robotwin/bin/python "$ROBOTWIN_INTEGRATION/compute_stats.py"
echo 'Install and calibration complete. Run smoke.py inside a GPU allocation before training.'
