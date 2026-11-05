import copy
import json
from pathlib import Path

import numpy as np
import pytest

from rl_vla.diagnostics import causal_structural_validation_v1 as v


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return v.binding(path)


def structural_case(tmp_path, *, decision=1):
    folder = tmp_path / "episode_008_structural_na"
    folder.mkdir(parents=True)
    reset = {"initial_state_id": 158, "task_id": 1, "experiment_split": "test",
             "object_telemetry": {"simulator_state": [0., 1.]}}
    row = {"episode": 8, "seed": 100008, "initial_state_id": 158, "condition": "random:background_fill",
           "status": "structural_NA", "success": None, "primitive_steps": decision * 5, "reset": reset,
           "availability_reason": "insufficient_unowned_background_for_exact_target_area"}
    masks = {}
    for camera in ("image", "wrist_image"):
        target = np.zeros((224, 224), bool)
        target[:120 if camera == "image" else 10] = True
        masks[f"{camera}__cream_cheese_1"] = target
        masks[f"{camera}__alphabet_soup_1"] = np.zeros_like(target)
        masks[f"{camera}__unowned_background"] = ~target
    np.savez(folder / "masks.npz", **masks)
    np.savez(folder / "observation.npz", image=np.zeros((224, 224, 3), np.uint8),
             wrist_image=np.zeros((224, 224, 3), np.uint8), state=np.zeros(8), prompt=np.asarray("pick cream cheese"),
             sampling_key=np.asarray([1, 2], np.uint32), key_after_split=np.asarray([3, 4], np.uint32),
             normalized_action_prefix=np.zeros((decision, 10, 32)), initial_noise=np.zeros((1, 10, 32)),
             simulator_state=np.asarray([0., 1.]))
    write(folder / "partial_telemetry.json", [{"primitive_steps": i + 1, "simulator_state": [0., 1.]} for i in range(decision * 5)])
    write(folder / "partial_intervention_severity.json", [{} for _ in range(decision)])
    amendment = write(tmp_path / "amendment.json", {})
    support = [{"camera": camera, "requested_pixels": int(masks[f"{camera}__cream_cheese_1"].sum()),
                "available_pixels": int(masks[f"{camera}__unowned_background"].sum()),
                "exact_match_possible": camera != "image"} for camera in ("image", "wrist_image")]
    evidence = {"complete": True, "status": "structural_NA", "task": "object_1", "policy": "rldt_100",
                "condition": row["condition"], "episode": 8, "initial_state_id": 158, "seed": 100008,
                "target": "cream_cheese_1", "distractor": "alphabet_soup_1", "success": None, "outcome_available": False,
                "decision": decision, "random_mask_seed": 180000 + decision,
                "primitive_steps_before_undefined_input": decision * 5, "normalized_chunks_before_undefined_input": decision,
                "support": support, "first_failing_camera": "image", "reset": reset,
                "mask_metadata": {"instances": {"cream_cheese_1": {}, "alphabet_soup_1": {}}},
                "amendment": amendment, "artifacts": [v.binding(folder / name) for name in
                    ("observation.npz", "masks.npz", "partial_telemetry.json", "partial_intervention_severity.json")]}
    row["structural_evidence"] = write(folder / "summary.json", evidence)
    return row, amendment, evidence


def validate(row, amendment):
    return v.validate_structural_case(row, policy="rldt_100", condition="random:background_fill", index=8,
                                      amendment=amendment, expected_prompt="pick cream cheese")


def test_actual_mask_witness_is_not_an_outcome(tmp_path):
    row, amendment, _ = structural_case(tmp_path)
    result = validate(row, amendment)
    assert result["support"][0]["available_pixels"] < result["support"][0]["requested_pixels"]
    assert "success" not in result


@pytest.mark.parametrize("field,value", [("decision", 2), ("episode", 7), ("seed", 100007),
                                          ("first_failing_camera", "wrist_image"), ("success", False)])
def test_rehashed_false_witness_rejected(tmp_path, field, value):
    row, amendment, evidence = structural_case(tmp_path)
    evidence[field] = value
    row["structural_evidence"] = write(Path(row["structural_evidence"]["path"]), evidence)
    with pytest.raises(ValueError):
        validate(row, amendment)


def test_rehashed_eligible_mask_cannot_exclude_arbitrary_pixels(tmp_path):
    row, amendment, evidence = structural_case(tmp_path)
    path = Path(evidence["artifacts"][1]["path"])
    with np.load(path) as data:
        masks = {k: data[k] for k in data.files}
    masks["image__unowned_background"][200] = False
    np.savez(path, **masks)
    evidence["artifacts"][1] = v.binding(path)
    row["structural_evidence"] = write(Path(row["structural_evidence"]["path"]), evidence)
    with pytest.raises(ValueError, match="complement"):
        validate(row, amendment)


def test_pending_never_produces_complete_certificate(tmp_path):
    assert v.validate_execution(tmp_path) is None


def fixture_campaign(root):
    """360 resolved synthetic identities; original256 are retained unchanged."""
    selection = write(root / "methods/rldt/object_1/selection.json", {"selected_iteration": 25})
    approval_dir = root / "compiler_profile_integration/causal_structural_na_v1"
    controls = {k: True for k in ("strict_matched_area_definition_unchanged", "no_area_capping", "no_replacement_sampling",
                                  "no_semantic_pixel_fallback", "no_silent_camera_skip", "normal_cases_use_original_control_implementation")}
    aggregation = {k: True for k in ("any_structural_na_makes_condition_primary_success_unavailable",
                                    "any_structural_na_makes_condition_primary_paired_comparison_unavailable",
                                    "no_complete_case_mean_as_full_predeclared_effect", "no_imputation",
                                    "partial_trajectory_not_full_episode_outcome", "structural_na_disables_condition_proxy_rates")}
    amendment = write(approval_dir / "amendment_v1.json", {"schema": "causal-strict-support-amendment-v1", "approved_for_execution": True,
                        "control_semantics": controls, "primary_aggregation": aggregation,
                        "coverage": {"main_and_ood_native_cases_required": 4200, "causal_planned_cases": 1080, "all_planned_cases": 5280},
                        "preservation": {"completed_episodes_expected": 256, "whole_conditions_expected": 23}})
    checkpoints = {p: "pi05_libero" if p == "pi05_libero" else str(root / "methods/rldt/object_1/training" / f"iteration_{int(p.split('_')[1]):06d}") for p in v.POLICIES}
    reuse = {"task": "object_1", "completed_episodes": 256, "whole_conditions_complete": 23, "inputs": {}, "policies": {}}
    originals = {}
    def native(folder, p, c, i):
        folder.mkdir(parents=True, exist_ok=True)
        row = {"episode": i, "initial_state_id": 150 + i, "seed": 100000 + i, "condition": c,
               "checkpoint": checkpoints[p], "selection_sha256": selection["sha256"], "target": "cream_cheese_1",
               "distractor": "alphabet_soup_1", "technical_smoke": False, "success": i % 2 == 0, "primitive_steps": 5,
               "reset": {"initial_state_id": 150 + i, "task_id": 1, "experiment_split": "test", "object_telemetry": {"simulator_state": [0., 1.]}}}
        np.savez(folder / f"episode_{i:03d}_actions.npz", normalized=np.zeros((1, 10, 32)))
        write(folder / f"episode_{i:03d}_telemetry.json", [{"primitive_steps": s + 1, "action": [0.] * 7,
              "simulator_state": [0., 1.], "objects": {"cream_cheese_1": {}, "alphabet_soup_1": {}}} for s in range(5)])
        video = folder / f"episode_{i:03d}.mp4"
        video.write_bytes(b"test-only-video-content")
        row["video"] = str(video)
        return row
    for p in v.POLICIES:
        cells = {}
        for ci, c in enumerate(v.CONDITIONS):
            n = 10 if ci < 5 or (ci == 5 and p != "rldt_100") else 8 if ci == 5 else {"pi05_libero": 6, "rldt_25": 9, "rldt_50": 3, "rldt_100": 0}[p] if ci == 6 else 0
            old = root / "interventions/object_1" / p / c.replace(":", "_")
            old.mkdir(parents=True, exist_ok=True)
            rows = [native(old, p, c, i) for i in range(n)]
            lines = [json.dumps(row) for row in rows]
            (old / "episodes.jsonl").write_text("\n".join(lines) + ("\n" if n else ""))
            write(old / "no_op.json", {"passed": True, "max_abs_difference": 0., "persisted_noise_recurrence_max_abs_difference": 0.})
            artifacts = [{"episode": i, **{kind: v.binding(old / f"episode_{i:03d}{suffix}") for kind, suffix in
                         (("actions", "_actions.npz"), ("telemetry", "_telemetry.json"), ("video", ".mp4"))}} for i in range(n)]
            cell = {"original_directory": str(old), "next_episode": n, "whole_condition_complete": n == 10,
                    "rows": rows, "raw_lines": lines, "episode_artifacts": artifacts}
            if n == 10:
                write(old / "summary.json", {"complete": True, "technical_smoke": False, "condition": c, "n": 10,
                                              "episodes": rows, "successes": sum(r["success"] for r in rows)})
            for file in old.iterdir():
                reuse["inputs"][str(file)] = v.binding(file)
            cells[c] = cell
            originals[p, c] = cell
        reuse["policies"][p] = {"conditions": cells}
    reuse_record = write(approval_dir / "reuse_manifest.json", reuse)
    proof = write(approval_dir / "resume_proof.json", {"complete": True, "passed": True})
    review = write(approval_dir / "independent_review.json", {"passed": True})
    resume_plan = write(approval_dir / "resume_plan.json", {})
    acceptance = write(root / v.APPROVAL, {"approved": True, "passed": True, "amendment": amendment,
              "reuse_manifest": reuse_record, "resume_proof": proof, "independent_review": review, "resume_plan": resume_plan,
              "source_input_sha256": {}, "execution_validator_source": v.binding(v.__file__)})
    raw = root / v.RAW
    queue_policies, jobs = [], []
    for p in v.POLICIES:
        folder = raw / p
        plan = {"task": "object_1", "policy": p, "checkpoint": checkpoints[p], "technical_smoke": False,
                "conditions": list(v.CONDITIONS), "state_ids": list(range(150, 160)), "seeds": list(range(100000, 100010)), "source_sha256": {}}
        write(folder / "case_plan.json", plan)
        summaries = []
        for c in v.CONDITIONS:
            old = originals[p, c]
            cell = folder / c.replace(":", "_")
            if old["whole_condition_complete"]:
                cell.symlink_to(old["original_directory"], target_is_directory=True)
                report = v.read(cell / "summary.json")
            else:
                cell.mkdir()
                rows = copy.deepcopy(old["rows"])
                for artifact in old["episode_artifacts"]:
                    for kind in ("actions", "telemetry", "video"):
                        source = Path(artifact[kind]["path"])
                        (cell / source.name).symlink_to(source)
                for i in range(old["next_episode"], 10):
                    rows.append(native(cell, p, c, i))
                (cell / "episodes.jsonl").write_text("\n".join(old["raw_lines"] + [json.dumps(r) for r in rows[old["next_episode"]:]]) + "\n")
                write(cell / "no_op.json", {"passed": True, "max_abs_difference": 0., "persisted_noise_recurrence_max_abs_difference": 0.})
                report = {"complete": True, "technical_smoke": False, "condition": c, "n": 10, "episodes": rows,
                          "attempted_cases": 10, "completed_outcome_cases": 10, "structural_na_cases": 0,
                          "primary_outcome_available": True, "successes": sum(r["success"] for r in rows),
                          "primary_success_rate": sum(r["success"] for r in rows) / 10,
                          "primary_paired_comparison": None, "amendment": amendment, "reuse_manifest": reuse_record}
                write(cell / "summary.json", report)
            summaries.append({k: value for k, value in report.items() if k != "episodes"})
        top = write(folder / "summary.json", {"complete": True, "execution_coverage_complete": True, "case_plan": plan,
                    "conditions": summaries, "primary_outcomes_all_available": True})
        queue_policies.append({"policy": p, "summary": top, "native_completed": 90, "structural_NA": 0, "resolved": 90})
        jobs.append({"name": p, "checkpoint": checkpoints[p], "destination": str(folder)})
    plan = {"task": "object_1", "states": list(range(150, 160)), "seeds": list(range(100000, 100010)),
            "conditions": list(v.CONDITIONS), "planned_cases": 360, "original_parent": 363111, "parent": {"pid": 1},
            "source_input_sha256": {acceptance["path"]: acceptance["sha256"]}, "amendment": amendment,
            "reuse_manifest": reuse_record, "resume_proof": proof, "jobs": jobs}
    plan_record = write(raw / "queue_plan.json", plan)
    write(raw / "queue_complete.json", {"complete": True, "passed": True, "execution_coverage_complete": True,
          "canonical_completion_published": False, "parent": plan["parent"], "source_input_sha256": plan["source_input_sha256"],
          "amendment": amendment, "reuse_manifest": reuse_record, "queue_plan": plan_record,
          "policy_summaries": queue_policies, "coverage": {"planned_cases": 360, "native_completed": 360, "structural_NA": 0, "total_resolved": 360}})
    return raw


def test_complete_original_case_plan_and_preserved_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "key_recurrence", lambda *_: ({}, {}))
    fixture_campaign(tmp_path)
    certificate = v.validate_execution(tmp_path)
    assert certificate["coverage"]["native_completed"] == 360
    assert certificate["preserved_native_cases"] == 256
    assert certificate["preserved_whole_conditions"] == 23
    assert len(certificate["cases"]) == 360


def test_changed_original_outcome_bytes_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "key_recurrence", lambda *_: ({}, {}))
    fixture_campaign(tmp_path)
    path = tmp_path / "interventions/object_1/pi05_libero/none/episodes.jsonl"
    path.write_text(path.read_text().replace('"success": true', '"success": false', 1))
    with pytest.raises(ValueError, match="Changed execution evidence"):
        v.validate_execution(tmp_path)


def test_wrong_retained_native_key_rejected(tmp_path):
    row, amendment, _ = structural_case(tmp_path)
    table = {name: np.zeros((10, 56, 2), np.uint32) for name in ("sampling_key", "key_after_split")}
    with pytest.raises(ValueError, match="seed/split"):
        v.validate_structural_case(row, policy="rldt_100", condition="random:background_fill", index=8,
                                   amendment=amendment, key_table=table)


def test_changed_native_key_audit_rejected_before_loading_arrays(tmp_path):
    path = tmp_path / "compiler_profile_integration/causal_structural_na_v1/native_key_recurrence_v1.json"
    record = write(path, {"complete": True, "passed": True})
    with pytest.raises(ValueError, match="recurrence proof"):
        v.key_recurrence(tmp_path, {"native_key_recurrence": record})


def test_full_na_certificate_counts_and_publish_without_imputation(tmp_path,monkeypatch):
    raw=fixture_campaign(tmp_path)
    cell=raw/'rldt_100/random_background_fill'
    original=v.read(cell/'summary.json')
    row,_,evidence=structural_case(cell)
    amendment=v.read(tmp_path/v.APPROVAL)['amendment']
    evidence['amendment']=amendment
    row['structural_evidence']=write(Path(row['structural_evidence']['path']),evidence)
    original['episodes'][8].update(row)
    original.update(completed_outcome_cases=9,structural_na_cases=1,primary_outcome_available=False,
                    successes=None,primary_success_rate=None,primary_paired_comparison=None)
    write(cell/'summary.json',original)
    (cell/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in original['episodes'])+'\n')
    with np.load(Path(row['structural_evidence']['path']).parent/'observation.npz') as data:
        np.savez(cell/'episode_008_actions.npz',normalized=data['normalized_action_prefix'])
    write(cell/'episode_008_telemetry.json',v.read(Path(row['structural_evidence']['path']).parent/'partial_telemetry.json'))
    write(tmp_path/'observation_bank/object_1/test/state_000158/metadata.json',{'prompt':'pick cream cheese'})
    top=v.read(raw/'rldt_100/summary.json');top['conditions'][5]={k:value for k,value in original.items() if k!='episodes'}
    top['primary_outcomes_all_available']=False
    top_record=write(raw/'rldt_100/summary.json',top)
    queue=v.read(raw/'queue_complete.json');item=queue['policy_summaries'][3]
    item.update(summary=top_record,native_completed=89,structural_NA=1)
    queue['coverage'].update(native_completed=359,structural_NA=1)
    write(raw/'queue_complete.json',queue)
    table={name:np.zeros((10,56,2),np.uint32) for name in ('sampling_key','key_after_split')}
    table['sampling_key'][8,1]=[1,2];table['key_after_split'][8,1]=[3,4]
    monkeypatch.setattr(v,'key_recurrence',lambda *_:(table,{}))
    result=v.validate_execution(tmp_path)
    assert result['coverage']=={'planned_cases':360,'native_completed':359,'structural_NA':1,'total_resolved':360}
    assert result['preserved_native_cases']==256 and len(result['structural_cases'])==1
    published=v.publish_execution(tmp_path,result)
    assert 'cases' not in published and len(v.read(published['case_records']['path'])['cases'])==360
    assert v.publish_execution(tmp_path,result)==published
    assert v.read(cell/'summary.json')['successes'] is None
