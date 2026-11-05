"""Baseline/trained profile separation without importing numerical backends."""

import json
from pathlib import Path

import pytest

from rl_vla.diagnostics import compiler_dispatch as dispatch

RUN = Path(__file__).parents[1] / "runs/vla_trace_objects_20261031"


@pytest.mark.parametrize("profile_id", ["native_trained_v2", "native_baseline_v1"])
def test_profile_routes_to_immutable_matching_helper(profile_id):
    manifest = RUN / "compiler_profiles" / profile_id / "manifest.json"
    profile = dispatch.load_profile(manifest)
    cache = RUN / "compiler_caches" / profile_id / "test_dispatch"
    environment, contract = dispatch.child_environment(profile, cache, base_env={})
    assert contract["profile_id"] == profile_id
    assert dispatch.verify_effective_environment(manifest, contract, environment=environment)["passed"]
    other = "native_baseline_v1" if profile_id == "native_trained_v2" else "native_trained_v2"
    with pytest.raises(ValueError, match="isolated cache"):
        dispatch.child_environment(profile, RUN / "compiler_caches" / other / "wrong", base_env={})


def test_run_root_preserves_trained_default():
    assert dispatch.load_profile(RUN)["manifest"]["profile_id"] == "native_trained_v2"


def test_unknown_profile_fails_closed(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"profile_id": "unknown"}))
    with pytest.raises(ValueError, match="Unknown"):
        dispatch.load_profile(manifest)
