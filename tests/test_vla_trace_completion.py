"""Acceptance cannot be inferred from missing, inconsistent, or changed evidence."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_vla_trace_completion import check, file_hash


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_queued_or_empty_campaign_is_not_completed(tmp_path):
    value = check(tmp_path)
    assert not value["complete"]
    assert all(not gate["passed"] for gate in value["gates"].values())
    assert value["gates"]["F_report"]["pending"]


def test_early_or_forged_figure_completion_cannot_close_e_or_f(tmp_path):
    name="All27 required figure/control/vector artifact cells validated"
    value=check(tmp_path)
    for gate in ("E_five_tests","F_report"):
        assert next(row for row in value["gates"][gate]["checks"] if row["name"]==name)["passed"] is None
    put(tmp_path/"figure_artifacts/summary.json",{"complete":True,"passed":True,"cells":[]})
    value=check(tmp_path)
    for gate in ("E_five_tests","F_report"):
        assert next(row for row in value["gates"][gate]["checks"] if row["name"]==name)["passed"] is False


def test_readable_causal_figures_require_each_bound_cell_not_global_marker(tmp_path):
    name = "All12 readable causal figure derivatives validated"
    put(tmp_path / "figures/causal/summary.json", {"complete": True, "passed": True, "cells": []})
    value = check(tmp_path)
    for gate in ("E_five_tests", "F_report"):
        assert next(row for row in value["gates"][gate]["checks"] if row["name"] == name)["passed"] is None
    put(tmp_path / "figures/causal/object_0/pi05_libero/summary.json", {"complete": True, "passed": True})
    value = check(tmp_path)
    for gate in ("E_five_tests", "F_report"):
        assert next(row for row in value["gates"][gate]["checks"] if row["name"] == name)["passed"] is False


def test_validation_tie_must_select_earliest_and_not_test(tmp_path):
    method = tmp_path / "methods/rldt/object_0"
    for iteration in (25, 50, 100):
        put(method / f"validation_{iteration:06d}.json", {"episodes": [{"success": True}] * 50})
    record = {"locked": "timestamp", "iteration": 50, "test_results_seen": False,
              "validation_counts": {"25": 50, "50": 50, "100": 50}}
    put(method / "selection.json", record)
    name = "object_0 validation-only selection"
    assert name in check(tmp_path)["gates"]["C_training"]["failed"]
    record["iteration"] = 25
    put(method / "selection.json", record)
    assert name not in check(tmp_path)["gates"]["C_training"]["failed"]
    record["test_results_seen"] = True
    put(method / "selection.json", record)
    assert name in check(tmp_path)["gates"]["C_training"]["failed"]


def test_changed_runtime_and_reviewed_evidence_fail(tmp_path):
    source = tmp_path / "runtime.py"
    source.write_text("original")
    expected = file_hash(source)
    put(tmp_path / "methods/rldt/object_0/input_contract.json", {str(source): expected})
    put(tmp_path / "scientific_review.json", {"passed": True, "reviewed_artifacts": {"runtime.py": expected}})
    source.write_text("changed")
    value = check(tmp_path)
    assert "object_0 runtime and frozen inputs unchanged" in value["gates"]["A_readiness"]["failed"]
    assert "Reviewed scientific evidence unchanged" in value["gates"]["F_report"]["failed"]
    assert not value["complete"]


def test_in_progress_artifact_is_pending_but_explicit_failure_is_failed(tmp_path):
    path = tmp_path / "rollout_traces/status.json"
    name = "Actual policy rollout traces"
    put(path, {"complete": False, "pending": ["object_0"]})
    gate = check(tmp_path)["gates"]["E_five_tests"]
    assert name in gate["pending"] and name not in gate["failed"]
    put(path, {"complete": False, "state": "failed"})
    assert name in check(tmp_path)["gates"]["E_five_tests"]["failed"]


def test_completed_training_requires_retained_final_snapshot(tmp_path):
    training = tmp_path / "methods/rldt/object_0/training"
    put(training / "summary.json", {})
    (training / "metrics.jsonl").write_text("".join(json.dumps({
        "iterations": i, "decisions": 32 * i, "actor_updates": 4 * max(i - 2, 0),
        "critic_updates": 64 * i}) + "\n" for i in range(1, 101)))
    name = "object_0 final snapshot retained"
    assert name in check(tmp_path)["gates"]["C_training"]["failed"]
    for filename in ("expert.npz", "learner.npz", "rldt_pi05.json"):
        put(training / "final" / filename, {"fixture": True})
    assert name not in check(tmp_path)["gates"]["C_training"]["failed"]


def test_restore_completion_claim_requires_all_nine_bound_results(tmp_path):
    path = tmp_path / "learner_restore_audit/summary.json"
    put(path, {"complete": True, "passed": True, "checkpoints": []})
    name = "All nine restore artifacts match audited checkpoints"
    assert name in check(tmp_path)["gates"]["C_training"]["failed"]
    put(path, {"complete": False, "passed": None, "checkpoints": []})
    gate = check(tmp_path)["gates"]["C_training"]
    assert "Native learner serialization restored" in gate["pending"]
    assert not gate["failed"]


def test_digest_uses_evidence_completeness_field(tmp_path):
    path = tmp_path / "evidence_digest.json"
    name = "Complete evidence digest"
    put(path, {"evidence_complete": False})
    gate = check(tmp_path)["gates"]["F_report"]
    assert name in gate["pending"] and name not in gate["failed"]
    put(path, {"evidence_complete": True})
    gate = check(tmp_path)["gates"]["F_report"]
    assert name not in gate["pending"] and name not in gate["failed"]


def test_numerical_completion_claim_requires_independent_oracle_bindings(tmp_path):
    path = tmp_path / "trained_checkpoint_sampler_fidelity.json"
    name = "All nine numerical proofs bind independent native oracles and unchanged sources"
    put(path, {"complete": True, "passed": True, "completed_checkpoints": 9,
               "expected_checkpoints": 9, "checkpoints": {}})
    assert name in check(tmp_path)["gates"]["B_fidelity"]["failed"]
    put(path, {"complete": False, "passed": None, "completed_checkpoints": 2})
    gate = check(tmp_path)["gates"]["B_fidelity"]
    assert name in gate["pending"] and name not in gate["failed"]


def test_baseline_requires_its_own_profile_and_all_three_oracles(tmp_path):
    path = tmp_path / "baseline_sampler_fidelity.json"
    name = "All three baseline proofs bind independent native oracles and unchanged sources"
    put(path, {"complete": True, "passed": True, "completed_tasks": 3,
               "expected_tasks": 3, "tasks": {}})
    assert name in check(tmp_path)["gates"]["B_fidelity"]["failed"]
    put(path, {"complete": False, "passed": None, "completed_tasks": 1})
    gate = check(tmp_path)["gates"]["B_fidelity"]
    assert name in gate["pending"] and name not in gate["failed"]
