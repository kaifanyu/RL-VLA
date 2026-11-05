"""Dispatch immutable baseline/trained compiler profiles without importing JAX.

Each profile is loaded at fresh child startup. Their colliding device/HLO tuning
keys cannot share different kernel choices in one XLA process.
"""

import json
from pathlib import Path

from rl_vla.diagnostics import compiler_profile, compiler_profile_baseline

MODULES = {"native_trained_v2": compiler_profile, "native_baseline_v1": compiler_profile_baseline}


def _resolve(profile):
    path = Path(profile["manifest_path"] if isinstance(profile, dict) else profile).resolve()
    if path.is_dir():
        candidate = path / "manifest.json"
        is_profile = candidate.exists() and "profile_id" in json.loads(candidate.read_text())
        path = candidate if is_profile else path / "compiler_profiles/native_trained_v2/manifest.json"
    manifest = json.loads(path.read_text())
    try:
        module = MODULES[manifest["profile_id"]]
    except KeyError as exception:
        raise ValueError("Unknown immutable compiler profile identity") from exception
    return module, path


def load_profile(profile):
    module, path = _resolve(profile)
    return module.load_profile(path)


def child_environment(profile, cache_dir, base_env=None):
    module, path = _resolve(profile)
    return module.child_environment(path, cache_dir, base_env)


def verify_effective_environment(profile, contract=None, *, check_versions=False, environment=None):
    module, path = _resolve(profile)
    return module.verify_effective_environment(path, contract, check_versions=check_versions, environment=environment)
