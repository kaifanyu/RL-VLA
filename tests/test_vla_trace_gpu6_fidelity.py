import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import queue_vla_trace_gpu6_interventions_v1 as queue

from rl_vla.diagnostics import device_fidelity_gpu6_v1 as gate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def proof(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "verify_effective_environment", lambda *a, **k: {"passed": True})
    base = {"input_sha256": {}, "global_failure_path": str(tmp_path / "parent_failure.json"), "autotune_sha256": "parent"}
    monkeypatch.setattr(gate, "parent_contract", lambda *a: base)
    profile = {"manifest": {"profile_id": "native_baseline_gpu6_v1", "device": {"index": 6},
                           "descriptor_derivation": {}}, "parent_profile": {}}
    for key in ("manifest", "profile", "helper"):
        p = tmp_path / (key + ".txt")
        p.write_text(key)
        profile[key + "_path"] = str(p)
        profile[key + "_sha256"] = gate.sha256(p)
    audit_derivation = tmp_path / "derivation.json"
    write(audit_derivation, {})
    profile["manifest"]["descriptor_derivation"] = {"audit_path": str(audit_derivation), "audit_sha256": gate.sha256(audit_derivation)}
    monkeypatch.setattr(gate, "load_profile", lambda p: profile)
    method = tmp_path / "methods/rldt/object_1"
    audit_path = method / "checkpoint_audit.json"
    write(audit_path, {"passed": True})
    selection_path = method / "selection.json"
    write(selection_path, {"iteration": 25})
    report = method / "baseline_test.json"
    episodes = [{"reset": {"initial_state_id": state}, "seed": 100000 + state - 150,
                 "trajectory": str(tmp_path / f"trajectory_{state}")} for state in range(150, 200)]
    write(report, {"checkpoint": None, "episodes": episodes})
    sources = {}
    for source in gate.SOURCES:
        p = tmp_path / source
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("source")
        sources[str(p)] = gate.sha256(p)
    for p in (report, selection_path, audit_path):
        sources[str(p)] = gate.sha256(p)
    directory = tmp_path / "compiler_profile_integration/gpu6_fidelity_v1"
    rows = []
    exact = {"exact": True, "finite": True, "max_abs_difference": 0.0, "different_elements": 0}
    for state, decision in gate.CASES:
        path = directory / "pi05_libero" / f"state_{state}" / f"decision_{decision}" / "summary.json"
        path.parent.mkdir(parents=True)
        arrays = path.with_name("arrays.npz")
        arrays.write_bytes(b"array evidence")
        chunk = Path(episodes[state - 150]["trajectory"]) / f"chunk_{decision + 1:06d}.npz"
        chunk.parent.mkdir(exist_ok=True)
        chunk.write_bytes(b"recorded decision")
        case = {"complete": True, "passed": True, "task": "object_1", "policy": "pi05_libero", "iteration": None,
                "state_id": state, "decision_index": decision, "seed": 100000 + state - 150,
                "recorded_action_replay": exact, "checks": {name: exact for name in gate.EXPECTED_CHECKS},
                "compiler_profile": {}, "arrays": {"path": str(arrays), "sha256": gate.sha256(arrays)},
                "recorded_chunk": str(chunk), "recorded_chunk_sha256": gate.sha256(chunk),
                "recorded_report": str(report), "recorded_report_sha256": gate.sha256(report),
                "source_input_sha256": {**sources, str(chunk): gate.sha256(chunk)},
                "raw_input_arrays": {"state": "hash"}, "context_leaf_sha256": ["hash"]}
        write(path, case)
        rows.append({"path": str(path), "sha256": gate.sha256(path)})
    path = directory / "pi05_libero/summary.json"
    policy = {"passed": True, "complete": True, "task": "object_1", "policy": "pi05_libero", "iteration": None,
              "checkpoint": None, "cases_expected": 4, "cases": rows, "checks_per_case": 20,
              "compiler_profile_manifest_sha256": profile["manifest_sha256"], "autotune_sha256": profile["profile_sha256"],
              "compiler_profile": {}, "compiler_audit": {"passed": True, "versions_checked": True, "observed_device": {"index": 6}},
              "source_input_sha256": sources}
    write(path, policy)
    aggregate = {"passed": True, "complete": True, "task": "object_1", "policies": list(gate.POLICIES),
                 "proofs": {name: {"path": str(directory / name / "summary.json"), "sha256": gate.sha256(path)} for name in gate.POLICIES},
                 "total_cases": 16, "total_exact_checks": 320}
    write(directory / "summary.json", aggregate)
    return tmp_path, profile, path, policy, aggregate


def republish(proof):
    _root, _profile, path, policy, aggregate = proof
    for row in policy["cases"]:
        row["sha256"] = gate.sha256(row["path"])
    write(path, policy)
    aggregate["proofs"]["pi05_libero"]["sha256"] = gate.sha256(path)
    write(path.parent.parent / "summary.json", aggregate)


def test_complete_baseline_proof_and_changed_arrays_fail(proof):
    root, profile, _path, policy, _aggregate = proof
    contract = gate.policy_fidelity_contract(root, "object_1", profile)
    gate.verify_task_fidelity(contract, full_hash=True)
    arrays = Path(policy["cases"][0]["path"]).with_name("arrays.npz")
    arrays.write_bytes(b"changed")
    with pytest.raises(ValueError, match="prerequisite changed"):
        gate.verify_task_fidelity(contract)


@pytest.mark.parametrize("mutation", ["nonzero", "nan", "missing_check", "wrong_case", "wrong_chunk", "device", "count"])
def test_false_exact_or_identity_proofs_fail_even_when_rehashed(proof, mutation):
    root, profile, _path, policy, aggregate = proof
    case_path = Path(policy["cases"][0]["path"])
    case = json.loads(case_path.read_text())
    if mutation == "nonzero":
        case["recorded_action_replay"]["max_abs_difference"] = 0.001
    elif mutation == "nan":
        case["checks"]["native_actions_only"]["max_abs_difference"] = float("nan")
    elif mutation == "missing_check":
        case["checks"].pop("native_actions_only")
    elif mutation == "wrong_case":
        case["state_id"] = 152
    elif mutation == "wrong_chunk":
        case["recorded_chunk"] = str(Path(case["recorded_chunk"]).with_name("chunk_000002.npz"))
        case["recorded_chunk_sha256"] = gate.sha256(case["recorded_chunk"])
    elif mutation == "device":
        policy["compiler_audit"]["observed_device"] = {"index": 5}
    else:
        aggregate["total_exact_checks"] = 319
    write(case_path, case)
    republish(proof)
    with pytest.raises(ValueError):
        gate.policy_fidelity_contract(root, "object_1", profile)


def test_pending_and_failed_numeric_parent_are_distinct(proof):
    root, profile, path, _policy, _aggregate = proof
    aggregate_path = path.parent.parent / "summary.json"
    aggregate_path.unlink()
    assert gate.policy_fidelity_contract(root, "object_1", profile) is None
    write(aggregate_path.with_name("launch.json"), {"jobs": [{"returncode": 1}]})
    with pytest.raises(ValueError, match="child failed"):
        gate.policy_fidelity_contract(root, "object_1", profile)


def test_original_native_numerical_functions_unchanged():
    assert queue.source_contract()


def test_concurrency_summary_rejects_missing_actual_recorded_replay(tmp_path):
    noop = {"passed": True, "max_abs_difference": 0.0, "persisted_noise_recurrence_max_abs_difference": 0.0}
    summary = {"complete": True, "passed": True, "workers": 4, "memory_fraction_per_worker": .2,
               "peak_gpu_processes": 4, "physical_gpu_lock_released": True,
               "conditions": [{"condition": name, "four_workers_finite": True, "native_no_op": [noop] * 4,
                               "action_artifact_sha256": ["sha"] * 4, "none_recorded_native_replay": []}
                              for name in queue.native.CONDITIONS]}
    with pytest.raises(ValueError, match="all four actual"):
        queue.validate_concurrency_summary(summary)


def test_versioned_jobs_preserve_every_native_case_seed_condition_and_budget(tmp_path, monkeypatch):
    args = SimpleNamespace(run=tmp_path, task="object_1", gpu="6", workers=4)
    policies = {"pi05_libero": None, **{f"rldt_{i}": tmp_path / f"iteration_{i:06d}" for i in (25, 50, 100)}}
    selection, output = tmp_path / "selection.json", tmp_path / "output"
    profiles = {role: {"manifest_path": str(tmp_path / (role + ".json")),
                       "manifest": {"isolated_cache_root": str(tmp_path / role)}} for role in ("baseline", "trained")}
    def child(profile, cache, base_env):
        return {**base_env, "CUDA_VISIBLE_DEVICES": "GPU6"}, {"environment": {"CUDA_VISIBLE_DEVICES": "GPU6"}}
    monkeypatch.setattr(queue, "child_environment", child)
    before = queue.native.policy_jobs(args, policies, selection, output)
    after = queue.policy_jobs(args, policies, selection, output, profiles)
    assert len(after) == 4
    for native, repaired in zip(before, after, strict=True):
        expected = list(native["command"])
        expected[1] = str(queue.CHILD)
        assert repaired["command"][:-4] == expected
        assert repaired["checkpoint"] == native["checkpoint"]
        assert repaired["destination"] == native["destination"]
        assert repaired["environment"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.20"
    assert queue.native.cpu_lanes("8-15", 4) == ["8,9", "10,11", "12,13", "14,15"]


def test_native_ood_cache_watch_rejects_hash_or_environment_change(tmp_path):
    import watch_vla_trace_gpu6_native_cache as watcher
    path = tmp_path / "binary-cache"
    path.write_bytes(b"original")
    records = {str(path): watcher.sha(path)}
    watcher.verify_files(records)
    watcher.verify_cache_environment({"JAX_COMPILATION_CACHE_DIR": str(tmp_path)}, tmp_path)
    with pytest.raises(ValueError, match="cache/profile environment"):
        watcher.verify_cache_environment({"JAX_COMPILATION_CACHE_DIR": str(tmp_path), "XLA_FLAGS": "--other"}, tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="executable or proof changed"):
        watcher.verify_files(records)
