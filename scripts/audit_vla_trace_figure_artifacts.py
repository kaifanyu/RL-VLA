"""Wait for required decoded figures/data, then publish bound readiness manifests."""
import argparse
import json
import time
from pathlib import Path

from rl_vla.diagnostics.figure_artifacts import refresh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    while True:
        result = refresh(args.run)
        print(json.dumps({key: result[key] for key in ("complete", "passed", "completed_cells", "pending_cells", "failed_cells")}), flush=True)
        if not args.watch or result["complete"] or result["passed"] is False:
            break
        time.sleep(args.interval)
