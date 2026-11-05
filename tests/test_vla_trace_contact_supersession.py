import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("contact_supersession", Path(__file__).parents[1] / "scripts/supersede_vla_trace_contact_cache.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root):
    write, sha = module.write, module.sha
    folder = root / "contact_analysis"
    folder.mkdir()
    source = root / "unchanged_worker.py"
    source.write_text("# unchanged extraction\n")
    write(folder / "source_manifest.json", {"code_hash": "same", "sources": {str(source): sha(source)}, "rule": {"minimum_lift_m": .03}})
    write(root / "technical_attempts/gpu6_descriptor_profile_miss_v1/failure_adjudication.json",
          {"native_main_evaluations_changed": False, "retry_cases_seeds_conditions_change": False})
    cache, cells, errors = {"reports": {}, "retry": {}, "files": {}}, [], []
    for policy in module.POLICIES:
        current = root / "interventions/object_1" / policy / "none/summary.json"
        archived = root / "technical_attempts/gpu6_descriptor_profile_miss_v1/object_1" / policy / "none/summary.json"
        write(current, {"policy": policy, "attempt": "repaired"})
        write(archived, {"policy": policy, "attempt": "invalid"})
        telemetry = archived.parent / "episode_000_telemetry.json"
        write(telemetry, {"old": True})
        artifact = folder / "reports" / sha(archived)
        artifact.mkdir(parents=True)
        (artifact / "report.json").write_bytes(archived.read_bytes())
        (artifact / "episodes.jsonl").write_text('{"reviewed":false}\n')
        (artifact / "source_inputs.jsonl").write_text(json.dumps({"path": str(current.parent / telemetry.name), "sha256": sha(telemetry)}) + "\n")
        manifest = {"identity": {"domain": "causal", "task": "object_1", "policy": policy, "variant": "none"},
                    "artifact": str(artifact), "report_sha256": sha(archived),
                    "evidence_sha256": sha(artifact / "episodes.jsonl"), "source_inputs_sha256": sha(artifact / "source_inputs.jsonl")}
        write(artifact / "summary.json", manifest)
        cache["reports"][str(current)] = manifest
        error = {"report": str(current), "error": "Previously completed source report changed; prior immutable evidence preserved"}
        errors.append(error)
        cache["retry"][str(current)] = error
        proof = root / "proofs" / f"{policy}.json"
        write(proof, {"task": "object_1", "policy": policy, "passed": True,
                      **dict.fromkeys(("paired_episodes", "outcomes_equal", "full_normalized_action_sequences_exact",
                                      "full_executed_action_sequences_exact", "full_simulator_state_sequences_exact", "full_step_metadata_sequences_exact"), 10),
                      "sources": [{"path": str(current), "sha256": sha(current)}]})
        cells.append({"result": {"path": str(proof), "sha256": sha(proof)}})
    write(root / "compiler_profile_integration/object1_repaired_none_rollouts_v1/summary.json",
          {"complete": True, "passed": True, "expected_cells": 4, "cells": cells})
    write(folder / "cache.json", cache)
    write(folder / "summary.json", {"persistent_errors": errors})
    return cache


def test_old_lineage_is_preserved_and_only_explicit_entries_rebound(tmp_path):
    fixture(tmp_path)
    old_cache = (tmp_path / "contact_analysis/cache.json").read_bytes()
    result = module.apply(tmp_path)
    archive = tmp_path / "technical_attempts/contact_sidecar_gpu6_supersession_v1"
    assert result["reextract_cached_reports"] == 4
    assert (archive / "contact_analysis/cache.json").read_bytes() == old_cache
    assert len(list((archive / "contact_analysis/reports").iterdir())) == 4
    new = module.read(tmp_path / "contact_analysis/cache.json")
    assert new["reports"] == {} and new["retry"] == {}
    assert new["supersession"]["sha256"] == module.sha(archive / "supersession.json")
    with pytest.raises(FileExistsError):
        module.apply(tmp_path)


def test_stale_repaired_proof_rejects_supersession(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "interventions/object_1/rldt_25/none/summary.json"
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="bind current"):
        module.build_plan(tmp_path)


def test_unverifiable_archived_telemetry_rejects_supersession(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "technical_attempts/gpu6_descriptor_profile_miss_v1/object_1/pi05_libero/none/episode_000_telemetry.json"
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="telemetry"):
        module.build_plan(tmp_path)


def test_unrelated_error_is_not_silently_cleared(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "contact_analysis/summary.json"
    value = module.read(path)
    value["persistent_errors"].append({"report": "unrelated", "error": "unknown"})
    module.write(path, value)
    with pytest.raises(ValueError, match="Additional contact errors"):
        module.build_plan(tmp_path)
