"""No backend/model is needed to enforce the diagnostic startup contract."""

import json
from pathlib import Path
import sys

import pytest

from rl_vla.diagnostics import compiler_profile as cp


def fixture_profile(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"campaign": "synthetic fixture"}))
    directory = tmp_path / "compiler_profiles" / cp.PROFILE_VERSION
    directory.mkdir(parents=True)
    tuning = directory / "autotune.textproto"
    tuning.write_text("version: 3\n")
    helper = Path(cp.__file__).resolve()
    manifest = {"schema_version": 1, "profile_id": cp.PROFILE_VERSION,
                "autotune_file": {"name": tuning.name, "sha256": cp.sha256(tuning)},
                "environment_helper": {"path": str(helper), "sha256": cp.sha256(helper)},
                "isolated_cache_root": str(tmp_path / "compiler_caches" / cp.PROFILE_VERSION),
                "versions": {"jax": "0.5.3", "jaxlib": "0.5.3"}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return cp.load_profile(tmp_path)


def test_child_environment_is_pure_and_bound_to_unique_private_cache(tmp_path):
    imported_before = set(sys.modules)
    profile = fixture_profile(tmp_path)
    original = {"CUDA_VISIBLE_DEVICES": "5", "JAX_COMPILATION_CACHE_DIR": "/shared"}
    cache = Path(profile["manifest"]["isolated_cache_root"]) / "worker_1"
    env, contract = cp.child_environment(profile, cache, original)
    assert original["JAX_COMPILATION_CACHE_DIR"] == "/shared"
    assert env["CUDA_VISIBLE_DEVICES"] == "5" and not cache.exists()
    assert env["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] == ""
    assert cp.verify_effective_environment(profile, contract, environment=env)["passed"]
    assert not ({"jax", "jaxlib"} & (set(sys.modules) - imported_before))
    with pytest.raises(ValueError, match="child-specific"):
        cp.child_environment(profile, tmp_path / "shared", original)
    with pytest.raises(ValueError, match="child-specific"):
        cp.child_environment(profile, cache.parent, original)


def test_profile_and_environment_mutation_fail_closed(tmp_path):
    profile = fixture_profile(tmp_path)
    cache = Path(profile["manifest"]["isolated_cache_root"]) / "worker"
    with pytest.raises(ValueError, match="Existing XLA_FLAGS"):
        cp.child_environment(profile, cache, {"XLA_FLAGS": "--xla_gpu_autotune_level=0"})
    env, contract = cp.child_environment(profile, cache, {})
    with pytest.raises(ValueError, match="Effective compiler environment"):
        cp.verify_effective_environment(profile, contract, environment={**env, "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all"})
    with pytest.raises(ValueError, match="contract changed"):
        cp.verify_effective_environment(profile, {**contract, "profile_sha256": "wrong"}, environment=env)
    Path(profile["profile_path"]).write_text("changed")
    with pytest.raises(ValueError, match="profile bytes"):
        cp.load_profile(profile["manifest_path"])


def test_versions_are_lazy_and_verified_without_jax_import(tmp_path, monkeypatch):
    profile = fixture_profile(tmp_path)
    env, contract = cp.child_environment(profile, Path(profile["manifest"]["isolated_cache_root"]) / "worker", {})
    monkeypatch.setattr(cp.importlib.metadata, "version", lambda package: "0.5.3")
    assert cp.verify_effective_environment(profile, contract, check_versions=True, environment=env)["observed_versions"] == {"jax": "0.5.3", "jaxlib": "0.5.3"}
    monkeypatch.setattr(cp.importlib.metadata, "version", lambda package: "other")
    with pytest.raises(ValueError, match="versions differ"):
        cp.verify_effective_environment(profile, contract, check_versions=True, environment=env)
