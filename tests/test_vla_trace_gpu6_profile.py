import copy
import json

import pytest

from rl_vla.diagnostics import compiler_profile_gpu6_v1 as profile


@pytest.fixture
def derived(tmp_path, monkeypatch):
    parent_file = tmp_path / "parent.textproto"
    parent_file.write_text(f'device: "{profile.OLD_DEVICE}"\ntriton: 8\n')
    parent_manifest = tmp_path / "parent.json"
    parent_manifest.write_text("{}")
    parent = {"manifest_path": str(parent_manifest), "manifest_sha256": profile.sha256(parent_manifest),
              "profile_path": str(parent_file), "profile_sha256": profile.sha256(parent_file),
              "manifest": {"profile_id": "native_trained_v2", "versions": {},
                           "isolated_cache_root": str(tmp_path / "parent_cache")}}
    monkeypatch.setattr(profile, "load_parent", lambda path: parent)
    audit = tmp_path / "audit.json"
    audit.write_text("{}")
    derived_file = tmp_path / "autotune.textproto"
    derived_file.write_text(parent_file.read_text().replace(profile.OLD_DEVICE, profile.NEW_DEVICE))
    manifest = {"schema_version": 1, "profile_id": "native_trained_gpu6_v1", "versions": {},
                "environment_helper": {"sha256": profile.sha256(profile.__file__)},
                "autotune_file": {"name": derived_file.name, "sha256": profile.sha256(derived_file)},
                "parent_profile": {"profile_id": "native_trained_v2", "manifest_path": str(parent_manifest),
                                   "manifest_sha256": parent["manifest_sha256"], "autotune_path": str(parent_file),
                                   "autotune_sha256": parent["profile_sha256"]},
                "descriptor_derivation": {"old_device": profile.OLD_DEVICE, "new_device": profile.NEW_DEVICE,
                                          "only_descriptor_changed": True, "replaced_entries": 1,
                                          "audit_path": str(audit), "audit_sha256": profile.sha256(audit)},
                "device": {"index": 6, "uuid": profile.GPU_UUID, "max_memory_clock_mhz": 1701},
                "isolated_cache_root": str(tmp_path / "gpu6_cache")}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, derived_file


def test_derived_profile_roundtrip_does_not_mutate_parent_environment(derived, tmp_path):
    path, _manifest, _ = derived
    original = {"CUDA_VISIBLE_DEVICES": "6", "OTHER": "keep"}
    env, contract = profile.child_environment(path, tmp_path / "gpu6_cache/worker", original)
    assert original == {"CUDA_VISIBLE_DEVICES": "6", "OTHER": "keep"}
    assert env["CUDA_VISIBLE_DEVICES"] == profile.GPU_UUID
    assert env["OTHER"] == "keep"
    assert profile.verify_effective_environment(path, contract, environment=env)["passed"]


def test_kernel_change_rejected_even_if_derived_digest_is_updated(derived):
    path, manifest, textproto = derived
    textproto.write_text(textproto.read_text().replace("triton: 8", "triton: 16"))
    manifest["autotune_file"]["sha256"] = profile.sha256(textproto)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="more than the exact"):
        profile.load_profile(path)


@pytest.mark.parametrize("mutation", ["parent", "device", "audit", "helper", "count"])
def test_lineage_corruption_rejected(derived, mutation):
    path, manifest, _ = derived
    if mutation == "parent":
        manifest["parent_profile"]["autotune_sha256"] = "wrong"
    elif mutation == "device":
        manifest["device"]["index"] = 5
    elif mutation == "audit":
        manifest["descriptor_derivation"]["audit_sha256"] = "wrong"
    elif mutation == "helper":
        manifest["environment_helper"]["sha256"] = "wrong"
    else:
        manifest["descriptor_derivation"]["replaced_entries"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        profile.load_profile(path)


def test_wrong_device_and_shared_cache_rejected(derived, tmp_path):
    path, _, _ = derived
    with pytest.raises(ValueError, match="another physical"):
        profile.child_environment(path, tmp_path / "gpu6_cache/worker", {"CUDA_VISIBLE_DEVICES": "5"})
    with pytest.raises(ValueError, match="distinct child cache"):
        profile.child_environment(path, tmp_path / "gpu6_cache", {})


def test_effective_settings_and_contract_must_match(derived, tmp_path):
    path, _, _ = derived
    env, contract = profile.child_environment(path, tmp_path / "gpu6_cache/worker", {})
    changed = copy.deepcopy(contract)
    changed["descriptor_derivation"]["replaced_entries"] = 100
    with pytest.raises(ValueError, match="parent environment/source"):
        profile.verify_effective_environment(path, changed, environment=env)
    env["CUDA_VISIBLE_DEVICES"] = "5"
    with pytest.raises(ValueError, match="Effective GPU6"):
        profile.verify_effective_environment(path, contract, environment=env)


def test_observed_physical_clock_cannot_silently_change(monkeypatch):
    monkeypatch.setattr(profile.subprocess, "check_output", lambda *a, **k: f"6, {profile.GPU_UUID}, 1728\n")
    with pytest.raises(ValueError, match="maximum memory clock changed"):
        profile.observed_device()
