"""Environment-only, startup compiler profile for diagnostic child processes.

This module imports no numerical libraries and never changes the current process
environment. Set the returned environment when spawning a fresh native child;
changing XLA flags after backend initialization cannot establish this contract.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex


PROFILE_VERSION = "native_trained_v2"
ENV_KEYS = ("XLA_FLAGS", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "JAX_COMPILATION_CACHE_DIR")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            result.update(block)
    return result.hexdigest()


def load_profile(path):
    """Load a run root, profile directory, or manifest and verify its small contract."""
    path = Path(path).resolve()
    if path.is_dir():
        candidate = path / "manifest.json"
        # Campaign roots also contain manifest.json; only a profile manifest has
        # this schema discriminator. Never silently interpret the run manifest.
        is_profile = candidate.exists() and "autotune_file" in json.loads(candidate.read_text())
        path = candidate if is_profile else path / "compiler_profiles" / PROFILE_VERSION / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("profile_id") != PROFILE_VERSION:
        raise ValueError("Unsupported compiler profile")
    profile = (path.parent / manifest["autotune_file"]["name"]).resolve()
    if profile.parent != path.parent or sha256(profile) != manifest["autotune_file"]["sha256"]:
        raise ValueError("Compiler profile bytes changed or escaped its immutable directory")
    helper = Path(__file__).resolve()
    if sha256(helper) != manifest["environment_helper"]["sha256"]:
        raise ValueError("Compiler environment helper changed since profile freeze")
    return {"manifest_path": str(path), "manifest_sha256": sha256(path),
            "profile_path": str(profile), "profile_sha256": sha256(profile),
            "helper_path": str(helper), "helper_sha256": sha256(helper), "manifest": manifest}


def _cache_path(profile, cache_dir):
    cache = Path(cache_dir).resolve()
    allowed = Path(profile["manifest"]["isolated_cache_root"]).resolve()
    if cache == allowed or not cache.is_relative_to(allowed):
        raise ValueError(f"Use a child-specific isolated cache below {allowed}")
    return str(cache)


def child_environment(profile, cache_dir, base_env=None):
    """Return (copied child environment, JSON source/environment contract).

    The caller chooses a distinct cache subdirectory per worker or immutable
    restart. No directories, shared caches, or global environment are modified.
    """
    profile = load_profile(profile["manifest_path"] if isinstance(profile, dict) else profile)
    env = dict(os.environ if base_env is None else base_env)
    flag = "--xla_gpu_load_autotune_results_from=" + profile["profile_path"]
    # XLA's flag parser is not a shell; whitespace in this path is unsupported.
    if any(char.isspace() for char in profile["profile_path"]):
        raise ValueError("Compiler profile path cannot contain whitespace")
    existing = shlex.split(env.get("XLA_FLAGS", ""))
    if existing not in ([], [flag]):
        raise ValueError("Existing XLA_FLAGS differ from the verified compiler profile")
    env.update(XLA_FLAGS=flag, JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="",
               JAX_COMPILATION_CACHE_DIR=_cache_path(profile, cache_dir))
    contract = {key: profile[key] for key in ("manifest_path", "manifest_sha256", "profile_path",
                                            "profile_sha256", "helper_path", "helper_sha256")}
    contract.update(profile_id=PROFILE_VERSION, environment={key: env[key] for key in ENV_KEYS},
                    expected_versions=profile["manifest"]["versions"],
                    startup_required=True, shared_cache_used=False,
                    coverage="Environment/profile verification only; requested-checkpoint numerical proof is a separate gate.")
    return env, contract


def verify_effective_environment(profile, contract=None, *, check_versions=False, environment=None):
    """Verify profile bytes and effective child settings; optionally check package versions.

    Call before constructing/loading a model. Package-version checks read installed
    metadata and do not import JAX or initialize any numerical backend.
    """
    profile = load_profile(profile["manifest_path"] if isinstance(profile, dict) else profile)
    effective = dict(os.environ if environment is None else environment)
    cache = effective.get("JAX_COMPILATION_CACHE_DIR")
    if cache is None:
        raise ValueError("Missing isolated compiler cache")
    expected, derived = child_environment(profile, cache, base_env={})
    if any(effective.get(key) != expected[key] for key in ENV_KEYS):
        raise ValueError("Effective compiler environment differs from the startup profile")
    if contract is not None:
        for key in ("manifest_path", "manifest_sha256", "profile_path", "profile_sha256", "helper_path",
                    "helper_sha256", "profile_id", "environment", "expected_versions", "startup_required"):
            if contract.get(key) != derived[key]:
                raise ValueError(f"Compiler source/environment contract changed: {key}")
    versions = None
    if check_versions:
        versions = {name: importlib.metadata.version(name) for name in profile["manifest"]["versions"]}
        if versions != profile["manifest"]["versions"]:
            raise ValueError(f"Native compiler package versions differ: {versions}")
    return {"passed": True, **derived, "versions_checked": check_versions, "observed_versions": versions,
            "verification_scope": "Profile/helper bytes, environment and optional installed package versions. Caller must spawn with this environment before backend initialization; no claim that inspecting strings can retroactively change an initialized backend."}
