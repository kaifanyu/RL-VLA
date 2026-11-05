"""Index existing scientific measurements without new fitting or statistics."""

import argparse
import fcntl
import json
from pathlib import Path

from rl_vla.diagnostics.evidence_digest import build_digest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    root = parser.parse_args().run.resolve()
    with (root / ".analysis.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(json.dumps(build_digest(root), indent=2))
