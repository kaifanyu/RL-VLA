"""Acceptance guards for immutable numerical evidence, without loading a model.

Fixtures use clearly synthetic byte payloads. Numerical equality is established
by the separate real-checkpoint auditor, never by these schema/integrity tests.
"""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_vla.diagnostics import trained_fidelity as gate

QUEUE_PATH = Path(__file__).resolve().parents[1] / "scripts/queue_vla_trace_diagnostics.py"
QUEUE_SPEC = importlib.util.spec_from_file_location("fidelity_queue_fixture", QUEUE_PATH)
queue = importlib.util.module_from_spec(QUEUE_SPEC)
QUEUE_SPEC.loader.exec_module(queue)


CHECKS = {
    "native_actions_only", "explicit_native_actions", "intervention_noop_actions",
    "intervention_noop_path", "captured_actions", "captured_path", "layer_noop_actions", "layer_noop_path",
    *(f"step{step:02d}_{kind}" for step in (0, 5, 9)
      for kind in ("capture_velocity", "projection", "layer_velocity", "euler_path")),
}
CASES = [(100, 0), (100, 1), (101, 0), (101, 1)]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def proof_tree(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    run = repository / "runs/synthetic_gate_fixture"
    monkeypatch.setattr(gate, "ROOT", repository)
    source_hashes = {}
    for relative in gate.SOURCES:
        source = repository / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"synthetic source identity: {relative}\n")
        source_hashes[str(source)] = gate.sha256(source)
    profile = {"profile_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    amendment_path = run / "compiler_profile_integration/fidelity_protocol_amendment_v1.json"
    write(amendment_path, {"synthetic_test_fixture": True, "rule": "Independent cached-native oracle; historical replay separate"})
    amendment = {"path": str(amendment_path), "sha256": gate.sha256(amendment_path)}
    oracle_source_sha = source_hashes[str(repository / "scripts/check_pi05_native_sampler_oracle.py")]
    cached_executables = []
    for kind in ("prefix", "sampler"):
        executable = run / "private_native_cache" / f"{kind}_executable"
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(f"synthetic cached {kind} executable".encode())
        cached_executables.append({"cache_key": f"synthetic_{kind}_key", "path": str(executable),
                                   "sha256": gate.sha256(executable)})
    method = run / "methods/rldt/object_0"
    audit = {"passed": True, "checkpoints": {}}
    ready = {"complete": True, "passed": True, "task": "object_0", "required_iterations": [25, 50, 100],
             "proofs": {}, "autotune_sha256": profile["profile_sha256"],
             "amendment": amendment, "native_oracle_source_sha256": oracle_source_sha,
             "compiler_profile_manifest_sha256": profile["manifest_sha256"],
             "auditor_source_sha256": source_hashes[str(repository / gate.SOURCES[-1])]}
    proof_paths, case_paths = {}, {}
    for iteration in (25, 50, 100):
        checkpoint = method / "training" / f"iteration_{iteration:06d}"
        files, checkpoint_sources, checkpoint_files = {}, {}, {}
        for name in ("expert.npz", "rldt_pi05.json"):
            path = checkpoint / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic checkpoint {iteration}/{name}".encode())
            digest = gate.sha256(path)
            files[name] = {"sha256": digest, "bytes": path.stat().st_size}
            checkpoint_sources[str(path)] = digest
            checkpoint_files[name] = {"path": str(path), "sha256": digest}
        audit["checkpoints"][str(iteration)] = {"files": files}
        proof_path = run / "trained_sampler_fidelity/object_0" / f"iteration_{iteration:06d}/summary.json"
        case_rows = []
        case_paths[iteration] = []
        for state, decision in CASES:
            case_path = proof_path.parent / f"state_{state:06d}/decision_{decision:06d}/summary.json"
            arrays = case_path.parent / "arrays.npz"
            arrays.parent.mkdir(parents=True, exist_ok=True)
            arrays.write_bytes(f"synthetic evidence arrays {iteration}/{state}/{decision}".encode())
            oracle_path = run / "trained_sampler_fidelity/native_oracle/object_0" / f"iteration_{iteration:06d}/state_{state:06d}/decision_{decision:06d}/summary.json"
            oracle_arrays = oracle_path.parent / "arrays.npz"
            oracle_arrays.parent.mkdir(parents=True, exist_ok=True)
            oracle_arrays.write_bytes(f"synthetic independent native arrays {iteration}/{state}/{decision}".encode())
            write(oracle_path, {"complete": True, "passed": True, "cache_hits_verified": True,
                "task": "object_0", "iteration": iteration, "state_id": state, "decision_index": decision,
                "arrays_sha256": gate.sha256(oracle_arrays), "cached_executables": cached_executables,
                "native_oracle_source_sha256": oracle_source_sha,
                "amendment": amendment, "source_input_sha256": {**source_hashes, **checkpoint_sources},
                "historical_recorded_replay": {"exact": False, "finite": True,
                                                "max_abs_difference": .0007, "different_elements": 320},
                "context_leaf_sha256": {"synthetic_leaf": "e" * 64}})
            write(case_path, {"complete": True, "passed": True, "task": "object_0", "iteration": iteration,
                "state_id": state, "decision_index": decision,
                "arrays_sha256": gate.sha256(arrays),
                "amendment": amendment, "context_matches_independent_oracle": True,
                "noise_matches_independent_oracle": True,
                "oracle": {"summary_path": str(oracle_path), "summary_sha256": gate.sha256(oracle_path),
                           "arrays_path": str(oracle_arrays), "arrays_sha256": gate.sha256(oracle_arrays),
                           "native_oracle_source_sha256": oracle_source_sha, "cache_hits_verified": True},
                "historical_recorded_replay": {"exact": False, "finite": True,
                                                "max_abs_difference": .0007, "different_elements": 320},
                "checks": {name: {"exact": True, "finite": True, "max_abs_difference": 0,
                                   "different_elements": 0} for name in CHECKS}})
            case_rows.append({"path": str(case_path), "sha256": gate.sha256(case_path), "passed": True})
            case_paths[iteration].append(case_path)
        write(proof_path, {"complete": True, "passed": True, "technical_only": True,
            "task": "object_0", "iteration": iteration, "cases_expected": 4, "cases": case_rows,
            "amendment": amendment, "native_oracle_source_sha256": oracle_source_sha,
            "autotune_sha256": profile["profile_sha256"],
            "source_input_sha256": {**source_hashes, **checkpoint_sources}})
        ready["proofs"][str(iteration)] = {"path": str(proof_path), "sha256": gate.sha256(proof_path),
                                          "passed": True, "checkpoint_files": checkpoint_files}
        proof_paths[iteration] = proof_path
    ready_path = run / "trained_sampler_fidelity/task_readiness/object_0.json"
    audit_path = method / "checkpoint_audit.json"
    write(ready_path, ready)
    write(audit_path, audit)
    return SimpleNamespace(run=run, repository=repository, profile=profile, ready_path=ready_path,
                           audit_path=audit_path, proof_paths=proof_paths, case_paths=case_paths,
                           amendment_path=amendment_path, cached_executables=cached_executables)


def contract(tree):
    return gate.task_fidelity_contract(tree.run, "object_0", tree.profile)


def reseal_proof(tree, iteration=25):
    """Refresh outer hashes so semantic tests cannot pass on a hash-only rejection."""
    path = tree.proof_paths[iteration]
    proof = read(path)
    for row in proof["cases"]:
        row["sha256"] = gate.sha256(row["path"])
    write(path, proof)
    ready = read(tree.ready_path)
    ready["proofs"][str(iteration)]["sha256"] = gate.sha256(path)
    write(tree.ready_path, ready)


def mutate_native_oracle(tree, mutate, *, synchronize_historical=False):
    case_path = tree.case_paths[25][0]
    case = read(case_path)
    native_path = Path(case["oracle"]["summary_path"])
    native = read(native_path)
    mutate(native)
    write(native_path, native)
    case["oracle"]["summary_sha256"] = gate.sha256(native_path)
    if synchronize_historical:
        case["historical_recorded_replay"] = native["historical_recorded_replay"]
    write(case_path, case)
    reseal_proof(tree)


def test_complete_three_checkpoint_contract_covers_raw_evidence(proof_tree):
    result = contract(proof_tree)
    assert result["task"] == "object_0"
    assert result["autotune_sha256"] == proof_tree.profile["profile_sha256"]
    for paths in proof_tree.case_paths.values():
        for path in paths:
            assert result["input_sha256"][str(path)] == gate.sha256(path)
            assert result["input_sha256"][str(path.parent / "arrays.npz")] == gate.sha256(path.parent / "arrays.npz")
            oracle = read(path)["oracle"]
            for key, hash_key in (("summary_path", "summary_sha256"), ("arrays_path", "arrays_sha256")):
                assert result["input_sha256"][oracle[key]] == oracle[hash_key]
    for executable in proof_tree.cached_executables:
        assert result["input_sha256"][executable["path"]] == executable["sha256"]
    gate.verify_task_fidelity(result)


@pytest.mark.parametrize("stage", ["no_marker", "partial_marker", "no_native_audit"])
def test_missing_or_partial_prerequisites_stay_pending(proof_tree, stage):
    if stage == "no_marker":
        proof_tree.ready_path.unlink()
    elif stage == "partial_marker":
        ready = read(proof_tree.ready_path)
        ready.update(complete=False, passed=None)
        ready["proofs"].pop("100")
        write(proof_tree.ready_path, ready)
    else:
        proof_tree.audit_path.unlink()
    assert contract(proof_tree) is None


def test_matching_global_numerical_failure_rejects_even_without_task_marker(proof_tree):
    proof_tree.ready_path.unlink()
    write(proof_tree.run / "trained_checkpoint_sampler_fidelity.json", {
        "complete": False, "passed": False, "autotune_sha256": proof_tree.profile["profile_sha256"],
        "failure": {"returncode": 1, "reason": "recorded-action mismatch"}})
    with pytest.raises(ValueError):
        contract(proof_tree)


@pytest.mark.parametrize("passed,profile_sha", [(False, "e" * 64), (None, "a" * 64)])
def test_stale_profile_failure_or_pending_aggregate_does_not_invalidate_completed_task(proof_tree, passed, profile_sha):
    write(proof_tree.run / "trained_checkpoint_sampler_fidelity.json", {
        "complete": False, "passed": passed, "autotune_sha256": profile_sha})
    result = contract(proof_tree)
    assert result is not None
    # The growing cross-task aggregate is deliberately not frozen into a task contract.
    assert str(proof_tree.run / "trained_checkpoint_sampler_fidelity.json") not in result["input_sha256"]


def test_new_matching_global_failure_stops_existing_contract_at_next_guard(proof_tree):
    result = contract(proof_tree)
    write(proof_tree.run / "trained_checkpoint_sampler_fidelity.json", {
        "complete": False, "passed": False, "autotune_sha256": proof_tree.profile["profile_sha256"]})
    with pytest.raises(ValueError):
        gate.verify_task_fidelity(result)


@pytest.mark.parametrize("field,value", [("passed", False), ("task", "object_2"),
    ("required_iterations", [25, 50]), ("autotune_sha256", "c" * 64),
    ("compiler_profile_manifest_sha256", "d" * 64)])
def test_false_or_misattributed_readiness_is_rejected(proof_tree, field, value):
    ready = read(proof_tree.ready_path)
    ready[field] = value
    write(proof_tree.ready_path, ready)
    with pytest.raises(ValueError):
        contract(proof_tree)


@pytest.mark.parametrize("mutation", ["wrong_state", "wrong_decision", "wrong_task", "wrong_iteration",
    "invented_check", "missing_check", "nonexact", "nonfinite", "nonzero_error", "nonzero_elements"])
def test_resealed_invalid_case_cannot_satisfy_gate(proof_tree, mutation):
    path = proof_tree.case_paths[25][0]
    case = read(path)
    if mutation == "wrong_state":
        case["state_id"] = 102
    elif mutation == "wrong_decision":
        case["decision_index"] = 9
    elif mutation == "wrong_task":
        case["task"] = "object_1"
    elif mutation == "wrong_iteration":
        case["iteration"] = 50
    elif mutation == "invented_check":
        case["checks"]["invented_check"] = case["checks"].pop("captured_path")
    elif mutation == "missing_check":
        case["checks"].pop("captured_path")
    else:
        field, value = {"nonexact": ("exact", False), "nonfinite": ("finite", False),
                        "nonzero_error": ("max_abs_difference", .001),
                        "nonzero_elements": ("different_elements", 1)}[mutation]
        case["checks"]["captured_actions"][field] = value
    write(path, case)
    reseal_proof(proof_tree)
    with pytest.raises(ValueError):
        contract(proof_tree)


def test_duplicate_case_does_not_count_as_four_predetermined_decisions(proof_tree):
    path = proof_tree.proof_paths[25]
    proof = read(path)
    proof["cases"][1] = dict(proof["cases"][0])
    write(path, proof)
    reseal_proof(proof_tree)
    with pytest.raises(ValueError, match="predetermined"):
        contract(proof_tree)


@pytest.mark.parametrize("mutation", ["source", "raw_arrays", "audit_checkpoint", "checkpoint_alias"])
def test_source_and_native_checkpoint_binding_is_enforced(proof_tree, mutation):
    if mutation == "source":
        (proof_tree.repository / gate.SOURCES[1]).write_text("changed numerical source")
    elif mutation == "raw_arrays":
        (proof_tree.case_paths[25][0].parent / "arrays.npz").write_bytes(b"changed arrays")
    elif mutation == "audit_checkpoint":
        audit = read(proof_tree.audit_path)
        audit["checkpoints"]["25"]["files"]["expert.npz"]["sha256"] = "f" * 64
        write(proof_tree.audit_path, audit)
    else:
        ready = read(proof_tree.ready_path)
        ready["proofs"]["25"]["checkpoint_files"]["expert.npz"]["path"] = str(proof_tree.run / "foreign/expert.npz")
        write(proof_tree.ready_path, ready)
    with pytest.raises(ValueError):
        contract(proof_tree)


@pytest.mark.parametrize("changed", ["raw_arrays", "case", "readiness", "source"])
def test_frozen_prerequisite_mutation_is_detected_between_phases(proof_tree, changed):
    result = contract(proof_tree)
    path = {"raw_arrays": proof_tree.case_paths[25][0].parent / "arrays.npz",
            "case": proof_tree.case_paths[25][0], "readiness": proof_tree.ready_path,
            "source": proof_tree.repository / gate.SOURCES[0]}[changed]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="prerequisite changed"):
        gate.verify_task_fidelity(result)


def test_exact_instrumentation_can_pass_without_relabeling_historical_variance(proof_tree):
    result = contract(proof_tree)
    assert result is not None
    case = read(proof_tree.case_paths[25][0])
    assert case["historical_recorded_replay"] == {
        "exact": False, "finite": True, "max_abs_difference": .0007, "different_elements": 320}
    assert all(check["exact"] for check in case["checks"].values())


@pytest.mark.parametrize("field", ["context_matches_independent_oracle", "noise_matches_independent_oracle"])
def test_actual_context_and_noise_must_match_independent_oracle(proof_tree, field):
    path = proof_tree.case_paths[25][0]
    case = read(path)
    case[field] = False
    write(path, case)
    reseal_proof(proof_tree)
    with pytest.raises(ValueError):
        contract(proof_tree)


@pytest.mark.parametrize("field,value", [("cache_hits_verified", False), ("passed", False),
    ("state_id", 101), ("decision_index", 1), ("task", "object_2"), ("iteration", 50),
    ("native_oracle_source_sha256", "f" * 64)])
def test_resealed_native_oracle_identity_and_cache_hit_claims_are_required(proof_tree, field, value):
    mutate_native_oracle(proof_tree, lambda native: native.update({field: value}))
    with pytest.raises(ValueError):
        contract(proof_tree)


@pytest.mark.parametrize("mutation", ["one_executable", "duplicate_key", "duplicate_path", "changed_bytes"])
def test_two_distinct_unchanged_native_cached_executables_are_required(proof_tree, mutation):
    if mutation == "changed_bytes":
        Path(proof_tree.cached_executables[0]["path"]).write_bytes(b"changed cached executable")
    else:
        def mutate(native):
            if mutation == "one_executable":
                native["cached_executables"] = native["cached_executables"][:1]
            else:
                key = "cache_key" if mutation == "duplicate_key" else "path"
                native["cached_executables"][1][key] = native["cached_executables"][0][key]
        mutate_native_oracle(proof_tree, mutate)
    with pytest.raises(ValueError):
        contract(proof_tree)


@pytest.mark.parametrize("changed", ["native_arrays", "native_summary", "cached_executable", "amendment"])
def test_frozen_native_oracle_and_amendment_cannot_change_after_acceptance(proof_tree, changed):
    result = contract(proof_tree)
    oracle = read(proof_tree.case_paths[25][0])["oracle"]
    path = {"native_arrays": Path(oracle["arrays_path"]), "native_summary": Path(oracle["summary_path"]),
            "cached_executable": Path(proof_tree.cached_executables[0]["path"]),
            "amendment": proof_tree.amendment_path}[changed]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        gate.verify_task_fidelity(result)


@pytest.mark.parametrize("source", ["oracle", "native_runtime", "expert", "metadata"])
def test_independent_oracle_uses_the_same_source_and_checkpoint(proof_tree, source):
    path = {"oracle": proof_tree.repository / "scripts/check_pi05_native_sampler_oracle.py",
            "native_runtime": proof_tree.repository / "src/rl_vla/rldt_pi05.py",
            "expert": proof_tree.run / "methods/rldt/object_0/training/iteration_000025/expert.npz",
            "metadata": proof_tree.run / "methods/rldt/object_0/training/iteration_000025/rldt_pi05.json"}[source]
    mutate_native_oracle(proof_tree, lambda native: native["source_input_sha256"].update({str(path): "0" * 64}))
    with pytest.raises(ValueError):
        contract(proof_tree)


@pytest.mark.parametrize("field,value", [("max_abs_difference", float("nan")), ("max_abs_difference", -1),
    ("max_abs_difference", 0), ("different_elements", 0), ("different_elements", -1),
    ("exact", True), ("finite", False)])
def test_historical_variance_must_remain_finite_and_internally_consistent(proof_tree, field, value):
    mutate_native_oracle(proof_tree, lambda native: native["historical_recorded_replay"].update({field: value}),
                         synchronize_historical=True)
    with pytest.raises(ValueError):
        contract(proof_tree)


def test_historical_comparison_cannot_differ_between_profile_case_and_native_oracle(proof_tree):
    mutate_native_oracle(proof_tree, lambda native: native["historical_recorded_replay"].update(max_abs_difference=.001))
    with pytest.raises(ValueError, match="Historical"):
        contract(proof_tree)


@pytest.mark.parametrize("layer", ["ready", "proof", "case", "native"])
def test_protocol_amendment_is_bound_at_each_provenance_layer(proof_tree, layer):
    if layer == "native":
        mutate_native_oracle(proof_tree, lambda native: native.pop("amendment"))
    else:
        path = {"ready": proof_tree.ready_path, "proof": proof_tree.proof_paths[25],
                "case": proof_tree.case_paths[25][0]}[layer]
        value = read(path)
        value.pop("amendment")
        write(path, value)
        if layer != "ready":
            reseal_proof(proof_tree)
    with pytest.raises(ValueError):
        contract(proof_tree)


@pytest.fixture
def rollout_launcher(tmp_path, monkeypatch):
    from contextlib import contextmanager
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    profiles, contracts = {}, {}
    for stage, identifier in (("baseline", "native_baseline_v1"), ("trained", "native_trained_v2")):
        profiles[stage] = {"manifest_path": str(tmp_path / identifier / "manifest.json"),
            "manifest": {"profile_id": identifier, "isolated_cache_root": str(campaign / "compiler_caches" / identifier)}}
        contracts[stage] = {"environment": {"XLA_FLAGS": f"--xla_gpu_load_autotune_results_from=/immutable/{identifier}",
            "JAX_COMPILATION_CACHE_DIR": profiles[stage]["manifest"]["isolated_cache_root"] + f"/rollout/{stage}",
            "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": ""}, "profile_id": identifier}
    monkeypatch.setenv("XLA_FLAGS", "parent-environment-must-not-change")
    monkeypatch.setattr(queue, "profiles_for", lambda args: profiles)
    monkeypatch.setattr(queue, "policy_fidelity_contract", lambda run, task, profile: {"task": task, "profile": profile["manifest"]["profile_id"]})
    monkeypatch.setattr(queue, "verify_task_fidelity", lambda contract: None)
    lock_events = []
    @contextmanager
    def exclusive(gpu):
        assert gpu == "5"
        lock_events.append("locked")
        yield "GPU-test"
        lock_events.append("released")
    monkeypatch.setattr(queue, "exclusive_gpu", exclusive)
    def child_environment(actual_profile, cache):
        stage = next(stage for stage, profile in profiles.items() if actual_profile is profile)
        child_contract = contracts[stage]
        assert cache == Path(child_contract["environment"]["JAX_COMPILATION_CACHE_DIR"])
        return {**os.environ, **child_contract["environment"]}, child_contract
    monkeypatch.setattr(queue, "child_environment", child_environment)
    return SimpleNamespace(campaign=campaign, profiles=profiles, contracts=contracts, lock_events=lock_events,
        args=SimpleNamespace(run=campaign, compiler_profile=tmp_path / "trained/manifest.json",
                             baseline_compiler_profile=tmp_path / "baseline/manifest.json", gpu="5", cpuset="76-77"))


@pytest.mark.parametrize("result", ["complete", "missing", "pending", "failed", "wrong_contract"])
def test_rollout_launcher_sets_startup_environment_and_requires_complete_matching_child(rollout_launcher, monkeypatch, result):
    fixture = rollout_launcher
    calls = []
    def fake_child(command, environment, log_path):
        calls.append(command)
        stage = command[command.index("--stage") + 1]
        assert fixture.lock_events == ["locked"]
        assert command[:3] == ["taskset", "-c", "76-77"]
        assert command[4].endswith("/scripts/trace_pi05_rollout_bank.py")
        assert command[command.index("--compiler-profile") + 1] == fixture.profiles[stage]["manifest_path"]
        contract_path = Path(command[command.index("--compiler-contract") + 1])
        assert read(contract_path) == fixture.contracts[stage]
        assert {key: environment[key] for key in fixture.contracts[stage]["environment"]} == fixture.contracts[stage]["environment"]
        assert environment["CUDA_VISIBLE_DEVICES"] == "5"
        assert environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.85"
        assert not (fixture.campaign / "rollout_trace_queue_complete.json").exists()
        outcome = result if stage == "trained" else "complete"
        if outcome != "missing":
            write(fixture.campaign / "rollout_traces/stages" / stage / "summary.json", {
                "complete": outcome != "pending", "passed": outcome != "failed",
                "compiler_profile": fixture.contracts[stage] if outcome != "wrong_contract" else {"wrong": True}})
        return {"pid": 100 + len(calls), "returncode": 0, "proc_start_ticks": 500 + len(calls)}
    monkeypatch.setattr(queue, "launch_profile_child", fake_child)
    if result == "complete":
        queue.run_rollout(fixture.args)
        completion = read(fixture.campaign / "rollout_trace_queue_complete.json")
        assert completion["complete"] is True
        assert [job["stage"] for job in completion["jobs"]] == ["baseline", "trained"]
        assert [job["compiler_profile"] for job in completion["jobs"]] == list(fixture.contracts.values())
        assert completion["child_exit"] == {"pid": 102, "returncode": 0, "proc_start_ticks": 502}
        assert fixture.lock_events == ["locked", "released"]
    else:
        with pytest.raises(ValueError):
            queue.run_rollout(fixture.args)
        assert not (fixture.campaign / "rollout_trace_queue_complete.json").exists()
    assert len(calls) == 2
    assert os.environ["XLA_FLAGS"] == "parent-environment-must-not-change"


@pytest.fixture
def baseline_tree(proof_tree):
    import copy
    import hashlib
    tree = proof_tree
    source_hashes = {}
    for relative in gate.BASELINE_SOURCES:
        path = tree.repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"synthetic baseline source: {relative}\n")
        source_hashes[str(path)] = gate.sha256(path)
    checkpoint = tree.repository / "checkpoints/openpi/pi05_libero"
    basefiles = [checkpoint / "params/_METADATA", checkpoint / "params/manifest.ocdbt",
                 checkpoint / "assets/libero/norm_stats.json"]
    for path in basefiles:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic baseline asset: {path.name}".encode())
        source_hashes[str(path)] = gate.sha256(path)
    model = {"checkpoint": str(checkpoint), "model_config": "pi05_libero",
             "checkpoint_metadata_sha256": gate.sha256(basefiles[0]),
             "checkpoint_manifest_sha256": gate.sha256(basefiles[1]),
             "normalization_assets_sha256": hashlib.sha256(json.dumps([
                 (str(basefiles[2].relative_to(checkpoint / "assets")), gate.sha256(basefiles[2]))
             ]).encode()).hexdigest()}
    profile = {**tree.profile, "manifest": {"profile_id": "native_baseline_v1"}}
    amendment_path = tree.run / "compiler_profile_integration/baseline_dispatch_protocol_amendment_v1.json"
    write(amendment_path, {"synthetic_test_fixture": True, "rule": "Separate original baseline compiler profile"})
    amendment = {"path": str(amendment_path), "sha256": gate.sha256(amendment_path)}
    oracle_sha = source_hashes[str(tree.repository / "scripts/check_pi05_native_baseline_oracle.py")]
    proof_path = tree.run / "baseline_sampler_fidelity/object_0/summary.json"
    case_rows, case_paths, native_paths = [], [], []
    for trained_path in tree.case_paths[25]:
        case = copy.deepcopy(read(trained_path))
        case.pop("iteration")
        case.update(policy="pi05_libero", amendment=amendment)
        case_path = proof_path.parent / f"state_{case['state_id']:06d}/decision_{case['decision_index']:06d}/summary.json"
        arrays = case_path.parent / "arrays.npz"
        arrays.parent.mkdir(parents=True, exist_ok=True)
        arrays.write_bytes(b"synthetic baseline evidence")
        native_path = tree.run / "baseline_sampler_fidelity/native_oracle/object_0" / f"state_{case['state_id']:06d}/decision_{case['decision_index']:06d}/summary.json"
        native_arrays = native_path.parent / "arrays.npz"
        native_arrays.parent.mkdir(parents=True, exist_ok=True)
        native_arrays.write_bytes(b"synthetic baseline independent native oracle")
        native = copy.deepcopy(read(Path(case["oracle"]["summary_path"])))
        native.pop("iteration")
        native.update(policy="pi05_libero", amendment=amendment, arrays_sha256=gate.sha256(native_arrays),
                      native_oracle_source_sha256=oracle_sha, source_input_sha256=source_hashes,
                      baseline_model_contract=model)
        write(native_path, native)
        case.update(arrays_sha256=gate.sha256(arrays), oracle={"summary_path": str(native_path),
                    "summary_sha256": gate.sha256(native_path), "arrays_path": str(native_arrays),
                    "arrays_sha256": gate.sha256(native_arrays), "native_oracle_source_sha256": oracle_sha,
                    "cache_hits_verified": True})
        write(case_path, case)
        case_rows.append({"path": str(case_path), "sha256": gate.sha256(case_path)})
        case_paths.append(case_path)
        native_paths.append(native_path)
    write(proof_path, {"complete": True, "passed": True, "task": "object_0", "policy": "pi05_libero",
        "cases_expected": 4, "cases": case_rows, "source_input_sha256": source_hashes,
        "baseline_model_contract": model, "amendment": amendment, "native_oracle_source_sha256": oracle_sha,
        "autotune_sha256": profile["profile_sha256"], "compiler_profile_manifest_sha256": profile["manifest_sha256"]})
    ready_path = tree.run / "baseline_sampler_fidelity/task_readiness/object_0.json"
    write(ready_path, {"complete": True, "passed": True, "task": "object_0", "policy": "pi05_libero",
        "proof": {"path": str(proof_path), "sha256": gate.sha256(proof_path)},
        "amendment": amendment, "native_oracle_source_sha256": oracle_sha,
        "autotune_sha256": profile["profile_sha256"], "compiler_profile_manifest_sha256": profile["manifest_sha256"],
        "auditor_source_sha256": source_hashes[str(tree.repository / gate.BASELINE_SOURCES[-1])]})
    return SimpleNamespace(**{**vars(tree), "profile": profile, "ready_path": ready_path,
        "proof_path": proof_path, "baseline_case_paths": case_paths, "native_paths": native_paths,
        "basefiles": basefiles, "baseline_model": model, "baseline_amendment": amendment_path})


def reseal_baseline(tree):
    proof = read(tree.proof_path)
    for item in proof["cases"]:
        item["sha256"] = gate.sha256(item["path"])
    write(tree.proof_path, proof)
    ready = read(tree.ready_path)
    ready["proof"]["sha256"] = gate.sha256(tree.proof_path)
    write(tree.ready_path, ready)


def baseline_contract(tree):
    return gate.policy_fidelity_contract(tree.run, "object_0", tree.profile)


def test_baseline_contract_is_separate_and_does_not_wait_for_trained_proofs(baseline_tree):
    tree = baseline_tree
    (tree.run / "trained_sampler_fidelity/task_readiness/object_0.json").unlink()
    result = baseline_contract(tree)
    assert result["policy"] == "pi05_libero"
    assert result["global_failure_path"] == str(tree.run / "baseline_sampler_fidelity.json")
    for path in [*tree.basefiles, *tree.native_paths, tree.baseline_amendment]:
        assert result["input_sha256"][str(path)] == gate.sha256(path)
    gate.verify_task_fidelity(result)


def test_policy_dispatch_preserves_trained_contract(proof_tree):
    profile = {**proof_tree.profile, "manifest": {"profile_id": "native_trained_v2"}}
    assert gate.policy_fidelity_contract(proof_tree.run, "object_0", profile) == contract(proof_tree)
    with pytest.raises(ValueError, match="Unsupported"):
        gate.policy_fidelity_contract(proof_tree.run, "object_0", {**profile, "manifest": {"profile_id": "unknown"}})


@pytest.mark.parametrize("change", ["missing", "incomplete", "failed", "wrong_policy", "wrong_profile"])
def test_baseline_readiness_stays_pending_or_rejects_invalid_proof(baseline_tree, change):
    tree = baseline_tree
    value = read(tree.ready_path)
    if change == "missing":
        tree.ready_path.unlink()
    else:
        if change == "incomplete": value["complete"] = False
        elif change == "failed": value["passed"] = False
        elif change == "wrong_policy": value["policy"] = "rldt_25"
        else: value["autotune_sha256"] = "0" * 64
        write(tree.ready_path, value)
    if change in ("missing", "incomplete"):
        assert baseline_contract(tree) is None
    else:
        with pytest.raises(ValueError):
            baseline_contract(tree)


@pytest.mark.parametrize("change", ["source", "asset", "raw_arrays", "oracle_arrays", "amendment", "cached_executable"])
def test_baseline_immutable_inputs_cannot_change_after_acceptance(baseline_tree, change):
    tree = baseline_tree
    result = baseline_contract(tree)
    choices = {"source": tree.repository / gate.BASELINE_SOURCES[1], "asset": tree.basefiles[-1],
        "raw_arrays": tree.baseline_case_paths[0].parent / "arrays.npz",
        "oracle_arrays": tree.native_paths[0].parent / "arrays.npz", "amendment": tree.baseline_amendment,
        "cached_executable": Path(tree.cached_executables[0]["path"])}
    choices[change].write_bytes(b"changed immutable baseline input")
    with pytest.raises(ValueError):
        gate.verify_task_fidelity(result)


@pytest.mark.parametrize("change", ["different_state", "different_policy", "invented_check", "nonexact",
    "nonzero_elements", "context", "noise", "duplicate_case"])
def test_resealed_baseline_cases_cannot_change_fixed_identities_or_exact_checks(baseline_tree, change):
    tree = baseline_tree
    path = tree.baseline_case_paths[0]
    case = read(path)
    if change == "different_state": case["state_id"] = 102
    elif change == "different_policy": case["policy"] = "rldt_25"
    elif change == "invented_check": case["checks"]["other"] = case["checks"].pop("captured_actions")
    elif change == "nonexact": case["checks"]["captured_actions"]["exact"] = False
    elif change == "nonzero_elements": case["checks"]["captured_actions"]["different_elements"] = 1
    elif change == "context": case["context_matches_independent_oracle"] = False
    elif change == "noise": case["noise_matches_independent_oracle"] = False
    elif change == "duplicate_case":
        proof = read(tree.proof_path)
        proof["cases"][1] = dict(proof["cases"][0])
        write(tree.proof_path, proof)
    write(path, case)
    reseal_baseline(tree)
    with pytest.raises(ValueError):
        baseline_contract(tree)


@pytest.mark.parametrize("change", ["model_contract", "native_core_source", "cache_hit", "historical_delta"])
def test_baseline_native_oracle_model_and_provenance_must_match(baseline_tree, change):
    tree = baseline_tree
    path = tree.native_paths[0]
    native = read(path)
    if change == "model_contract": native["baseline_model_contract"]["model_config"] = "pi05_base"
    elif change == "native_core_source": native["source_input_sha256"][str(tree.repository / "src/rl_vla/rldt_pi05.py")] = "0" * 64
    elif change == "cache_hit": native["cache_hits_verified"] = False
    else: native["historical_recorded_replay"]["max_abs_difference"] = 0
    write(path, native)
    case_path = tree.baseline_case_paths[0]
    case = read(case_path)
    case["oracle"]["summary_sha256"] = gate.sha256(path)
    if change == "historical_delta": case["historical_recorded_replay"] = native["historical_recorded_replay"]
    write(case_path, case)
    reseal_baseline(tree)
    with pytest.raises(ValueError):
        baseline_contract(tree)


def test_baseline_model_contract_recomputed_from_actual_asset_files(baseline_tree):
    tree = baseline_tree
    proof = read(tree.proof_path)
    proof["baseline_model_contract"]["normalization_assets_sha256"] = "0" * 64
    write(tree.proof_path, proof)
    reseal_baseline(tree)
    with pytest.raises(ValueError, match="model contract"):
        baseline_contract(tree)


def test_new_matching_baseline_failure_rejects_existing_baseline_contract(baseline_tree):
    tree = baseline_tree
    result = baseline_contract(tree)
    write(tree.run / "baseline_sampler_fidelity.json", {"passed": False, "autotune_sha256": tree.profile["profile_sha256"]})
    with pytest.raises(ValueError):
        gate.verify_task_fidelity(result)
