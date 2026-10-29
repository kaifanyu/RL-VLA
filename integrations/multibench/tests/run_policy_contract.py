"""Run contract regressions and capture a fixture in one policy interpreter."""
import argparse
from pathlib import Path

import pytest

from multibench.tests.capture_policy_contract import capture


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = pytest.main([str(Path(__file__).with_name("test_policy_contract.py")), "-q"])
    if status:
        raise SystemExit(status)
    capture(args.output)
