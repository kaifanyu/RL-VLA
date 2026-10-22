#!/usr/bin/env bash
# Clone FlashSAC (MIT) into third_party/FlashSAC at the commit rl_vla.flashsac was tested with.
# No FlashSAC dependencies are installed: RL-VLA loads only its pure-PyTorch layers.
set -euo pipefail
COMMIT="87edc9061150ae9e962dd84e6544e27a1554b3ab"
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/FlashSAC"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/Holiday-Robot/FlashSAC.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"
echo "FlashSAC at $(git -C "$DEST" rev-parse HEAD) in $DEST"
