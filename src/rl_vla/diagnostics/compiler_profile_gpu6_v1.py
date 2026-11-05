"""Explicit GPU6 descriptor-derived startup profiles; no numerical imports.

The parent profiles remain immutable. Only the exact XLA device descriptor is
changed in the derived textproto; kernel choices and all other bytes are fixed.
"""

import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess

from rl_vla.diagnostics.compiler_dispatch import load_profile as load_parent
from rl_vla.diagnostics.compiler_profile import sha256

PROFILE_PARENTS = {"native_baseline_gpu6_v1": "native_baseline_v1",
                   "native_trained_gpu6_v1": "native_trained_v2"}
GPU_UUID = "GPU-ffd52ae9-e4f5-4012-abc6-9dd9f5200e8d"
OLD_DEVICE = "CUDA: 8.0, Cores: 70, GPU clock: 1.41 GHz, Memory bandwidth: 1769 GB/s, L2 cache: 32 MB"
NEW_DEVICE = OLD_DEVICE.replace("1769 GB/s", "1742 GB/s")
ENV_KEYS = ("XLA_FLAGS", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
            "JAX_COMPILATION_CACHE_DIR", "CUDA_VISIBLE_DEVICES")


def load_profile(path):
    path = Path(path["manifest_path"] if isinstance(path, dict) else path).resolve()
    if path.is_dir():
        path = path / "manifest.json"
    manifest = json.loads(path.read_text())
    identifier = manifest.get("profile_id")
    if manifest.get("schema_version") != 1 or identifier not in PROFILE_PARENTS:
        raise ValueError("Require an explicit versioned GPU6 profile manifest")
    helper = Path(__file__).resolve()
    if manifest["environment_helper"]["sha256"] != sha256(helper):
        raise ValueError("GPU6 startup helper differs from frozen manifest")
    profile_path = (path.parent / manifest["autotune_file"]["name"]).resolve()
    if profile_path.parent != path.parent or sha256(profile_path) != manifest["autotune_file"]["sha256"]:
        raise ValueError("Derived autotune profile escaped or changed")
    parent_record = manifest["parent_profile"]
    parent = load_parent(parent_record["manifest_path"])
    if (parent_record.get("profile_id") != PROFILE_PARENTS[identifier]
            or parent["manifest"]["profile_id"] != PROFILE_PARENTS[identifier]
            or parent_record.get("manifest_sha256") != parent["manifest_sha256"]
            or parent_record.get("autotune_sha256") != parent["profile_sha256"]
            or Path(parent_record["autotune_path"]).resolve() != Path(parent["profile_path"])
            or manifest["versions"] != parent["manifest"]["versions"]):
        raise ValueError("Derived profile parent identity changed")
    derivation = manifest["descriptor_derivation"]
    original = Path(parent["profile_path"]).read_bytes()
    derived = profile_path.read_bytes()
    count = original.count(OLD_DEVICE.encode())
    if (derivation.get("old_device") != OLD_DEVICE or derivation.get("new_device") != NEW_DEVICE
            or derivation.get("only_descriptor_changed") is not True
            or count < 1 or derivation.get("replaced_entries") != count
            or derived != original.replace(OLD_DEVICE.encode(), NEW_DEVICE.encode())
            or derived.replace(NEW_DEVICE.encode(), OLD_DEVICE.encode()) != original
            or sha256(derivation["audit_path"]) != derivation.get("audit_sha256")):
        raise ValueError("Derived profile changes more than the exact GPU6 descriptor")
    if manifest.get("device") != {"index": 6, "uuid": GPU_UUID, "max_memory_clock_mhz": 1701}:
        raise ValueError("Derived profile is not bound to the measured physical GPU6")
    cache_root = Path(manifest["isolated_cache_root"]).resolve()
    if cache_root == Path(parent["manifest"]["isolated_cache_root"]).resolve():
        raise ValueError("Derived profile must use a distinct private cache root")
    return {"manifest_path": str(path), "manifest_sha256": sha256(path),
            "profile_path": str(profile_path), "profile_sha256": sha256(profile_path),
            "helper_path": str(helper), "helper_sha256": sha256(helper),
            "manifest": manifest, "parent_profile": parent}


def child_environment(profile, cache_dir, base_env=None):
    profile = load_profile(profile)
    env = dict(os.environ if base_env is None else base_env)
    cache = Path(cache_dir).resolve()
    allowed = Path(profile["manifest"]["isolated_cache_root"]).resolve()
    if cache == allowed or not cache.is_relative_to(allowed):
        raise ValueError("Require a distinct child cache below the GPU6 profile cache root")
    if any(character.isspace() for character in profile["profile_path"]):
        raise ValueError("Autotune path cannot contain whitespace")
    flag = "--xla_gpu_load_autotune_results_from=" + profile["profile_path"]
    if shlex.split(env.get("XLA_FLAGS", "")) not in ([], [flag]):
        raise ValueError("Conflicting compiler flags for GPU6 child")
    if env.get("CUDA_VISIBLE_DEVICES", GPU_UUID) not in ("", "6", GPU_UUID):
        raise ValueError("GPU6 derived profile cannot run on another physical device")
    env.update(XLA_FLAGS=flag, JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="",
               JAX_COMPILATION_CACHE_DIR=str(cache), CUDA_VISIBLE_DEVICES=GPU_UUID)
    contract = {key: profile[key] for key in ("manifest_path", "manifest_sha256", "profile_path",
                                            "profile_sha256", "helper_path", "helper_sha256")}
    contract.update(profile_id=profile["manifest"]["profile_id"],
                    environment={key: env[key] for key in ENV_KEYS},
                    expected_versions=profile["manifest"]["versions"],
                    parent_profile=profile["manifest"]["parent_profile"],
                    descriptor_derivation=profile["manifest"]["descriptor_derivation"],
                    device=profile["manifest"]["device"], startup_required=True,
                    shared_cache_used=False,
                    coverage="Startup/device/descriptor lineage only; exact numerical proof is separately mandatory")
    return env, contract


def observed_device():
    output = subprocess.check_output(["nvidia-smi", "-i", GPU_UUID,
               "--query-gpu=index,uuid,clocks.max.memory", "--format=csv,noheader,nounits"], text=True)
    fields = [part.strip() for part in output.strip().split(",")]
    if len(fields) != 3:
        raise ValueError("Cannot identify GPU6 physical device")
    result = {"index": int(fields[0]), "uuid": fields[1], "max_memory_clock_mhz": int(fields[2])}
    if result != {"index": 6, "uuid": GPU_UUID, "max_memory_clock_mhz": 1701}:
        raise ValueError("Physical GPU6 identity or maximum memory clock changed")
    return result


def verify_effective_environment(profile, contract=None, *, check_versions=False, environment=None):
    profile = load_profile(profile)
    effective = dict(os.environ if environment is None else environment)
    if not effective.get("JAX_COMPILATION_CACHE_DIR"):
        raise ValueError("Missing child-specific GPU6 compilation cache")
    expected, derived = child_environment(profile, effective["JAX_COMPILATION_CACHE_DIR"], base_env={})
    if any(effective.get(key) != expected[key] for key in ENV_KEYS):
        raise ValueError("Effective GPU6 compiler environment differs from startup contract")
    if contract is not None and contract != derived:
        raise ValueError("GPU6 parent environment/source contract changed")
    versions, device = None, None
    if check_versions:
        versions = {name: importlib.metadata.version(name) for name in profile["manifest"]["versions"]}
        if versions != profile["manifest"]["versions"]:
            raise ValueError("GPU6 native package versions changed")
        device = observed_device()
    return {"passed": True, **derived, "versions_checked": check_versions,
            "observed_versions": versions, "observed_device": device}
