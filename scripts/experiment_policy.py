"""Enforce the user's current method/suite preferences in full-run launchers."""

import json
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1] / "configs/experiment_policy.json"


def enforce(method, suite, policy_path=POLICY):
    policy = json.loads(policy_path.read_text())
    if method not in policy["allowed_methods"]:
        raise RuntimeError(f"Method {method} disabled by current user experiment policy: {policy_path}")
    if suite in policy["paused_suites"]:
        raise RuntimeError(f"Suite {suite} paused by current user experiment policy: {policy_path}")


def require_active_queue(queue):
    marker = queue / "CANCELLED.json"
    if marker.exists():
        raise RuntimeError(f"This queue was cancelled by the user: {marker}")
