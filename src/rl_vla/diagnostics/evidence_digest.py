"""A deterministic index of existing measurements, never a new analysis fit."""

import csv
import hashlib
import io
import json
from pathlib import Path

TASKS = ("object_0", "object_1", "object_2")
POLICIES = ("pi05_base", "pi05_libero", "rldt_25", "rldt_50", "rldt_100")
TESTS = ("localization_probe", "text_image_attention", "expert_image_attention",
         "causal_interventions", "unseen_layouts")
TABLES = {
    "training_coverage": "training_state_coverage.csv",
    "success": "success.csv", "paired_success": "paired_success.csv",
    "diagnostics": "diagnostic_status.csv", "wrong_object": "wrong_object.csv",
    "failure_taxonomy": "failure_taxonomy.csv", "probe": "probe_metrics.csv",
    "failure_label_codebook": "failure_label_codebook.csv",
    "failure_transition_summary": "failure_transition_summary.csv",
    "failure_transition_counts": "failure_transition_counts.csv",
    "counterfactual_probe": "counterfactual_probe_metrics.csv",
    "counterfactual_same_class": "counterfactual_same_class.csv",
    "probe_prompt_change": "probe_prompt_changes.csv", "paired_attention": "paired_attention.csv",
    "causal_success": "causal.csv", "causal_grasp_proxy": "causal_grasp_proxies.csv",
    "ood_success": "ood_success.csv", "ood_paired": "ood_paired.csv",
    "ood_wrong_object": "ood_wrong_object.csv", "ood_failure_taxonomy": "ood_failure_taxonomy.csv",
    "ood_visibility_support": "ood_visibility_support.csv",
    "drift": "checkpoint_drift.csv", "projection_ablation": "projection_ablation.csv",
    "compiler_controls": "compiler_precision_controls.csv",
    "independent_expert_controls": "independent_expert_control_summary.csv",
    "layer_knockouts": "layer_knockouts.csv", "grounding_index": "grounding_artifact_index.csv",
    "qualitative_figures": "qualitative_figures.csv", "vla_trace_exports": "vla_trace_exports.csv",
    "supplemental_artifacts": "supplemental_artifacts.csv",
    "contact_candidate_rates": "contact_candidate_rates.csv",
    "contact_manual_calibration": "contact_manual_calibration.csv",
}


def decode_cell(value):
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
    # Preserve object/model/state labels that happen to be JSON strings.
    return parsed


def canonical_policy(value):
    if isinstance(value, str) and value.startswith("rldt_"):
        return f"rldt_{int(value.split('_')[1])}"
    return value


class Sources:
    def __init__(self, root):
        self.root = Path(root)
        self.records = {}

    def index(self, relative):
        """Hash a potentially longer case table without loading or parsing it."""
        path = self.root / relative
        if not path.exists():
            self.records[str(relative)] = {"status": "missing", "sha256": None}
            return
        if path.suffix not in {".json", ".csv", ".md"}:
            raise ValueError("Digest index accepts source JSON/CSV tables and Markdown documentation only")
        checksum = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
                size += len(block)
        self.records[str(relative)] = {"status": "present", "sha256": checksum.hexdigest(), "bytes": size}

    def read(self, relative):
        path = self.root / relative
        key = str(Path(relative))
        if not path.exists():
            self.records[key] = {"status": "missing", "sha256": None}
            return None
        # Only compact summaries/tables belong here; never load NPZ arrays or
        # multi-hundred-megabyte per-head/per-patch files into this digest.
        if path.suffix not in {".json", ".csv"} or path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError(f"Digest source is not a compact JSON/CSV summary: {relative}")
        payload = path.read_bytes()
        self.records[key] = {"status": "present", "sha256": hashlib.sha256(payload).hexdigest(),
                             "bytes": len(payload)}
        return payload.decode()

    def table(self, filename, *, relative=None):
        relative = relative or f"tables/{filename}"
        payload = self.read(relative)
        if payload is None:
            return []
        rows = []
        for index, row in enumerate(csv.DictReader(io.StringIO(payload)), start=2):
            record = {key: decode_cell(value) for key, value in row.items()}
            if "policy" in record:
                record["policy"] = canonical_policy(record["policy"])
            if "source" in record:
                record["reported_source_field"] = record["source"]
            record["source"] = {"path": relative, "csv_line": index}
            rows.append(record)
        return rows

    def document(self, relative):
        payload = self.read(relative)
        return json.loads(payload) if payload is not None else None


def full_inventory(rows, keys, expected):
    indexed = {}
    for row in rows:
        identity = tuple(row[key] for key in keys)
        if identity in indexed:
            raise ValueError(f"Duplicate digest evidence cell: {identity}")
        indexed[identity] = row
    result = []
    for identity in expected:
        result.append(indexed.get(identity, {**dict(zip(keys, identity, strict=True)),
                                              "status": "pending", "reason": "Evidence table row missing"}))
    unknown = set(indexed) - set(expected)
    if unknown:
        raise ValueError(f"Unexpected digest evidence cells: {unknown}")
    return result


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def build_digest(root):
    root = Path(root).resolve()
    sources = Sources(root)
    tables = {name: sources.table(filename) for name, filename in TABLES.items()}
    from rl_vla.diagnostics.causal_support_overlay_v1 import operational_snapshot
    operational_health = operational_snapshot(root)
    for row in operational_health["rows"]:
        sources.index(str(Path(row["status_path"]).relative_to(root)))
    training_coverage = full_inventory(tables["training_coverage"], ("task",), [(task,) for task in TASKS])
    sources.index("tables/training_state_visit_counts.csv")
    success = full_inventory(tables["success"], ("task", "split", "policy"),
        [(task, split, policy) for task in TASKS for split in ("validation", "test", "source") for policy in POLICIES])
    diagnostics = full_inventory(tables["diagnostics"], ("task", "policy", "test"),
        [(task, policy, test) for task in TASKS for policy in POLICIES for test in TESTS])
    selections = {}
    for task in TASKS:
        path = f"methods/rldt/{task}/selection.json"
        selection = sources.document(path)
        selections[task] = {"status": "locked" if selection and selection.get("locked") else "pending",
                            "source": path, "record": selection}
    causal_open_loop = []
    from rl_vla.diagnostics.causal_support_overlay_v1 import analysis_paths
    from rl_vla.diagnostics.causal_support_overlay_v1 import execution as support_execution
    support_status = support_execution(root)
    for path in analysis_paths(root):
        relative = str(path.relative_to(root))
        summary = sources.document(relative)
        if not summary.get("complete"):
            continue
        for row in summary.get("open_loop_aggregates", []):
            causal_open_loop.append({"task": summary["task"], "policy": canonical_policy(summary["policy"]),
                **row, "source": {"path": relative, "json_field": "open_loop_aggregates"}})
    panels = sources.document("rollout_trace_analysis/summary.json")
    rollout_audit = sources.document("compiler_profile_integration/rollout_completion_independent_review_v1.json")
    rollout_visual_review = sources.document("compiler_profile_integration/rollout_phase_selected_visual_review_v1.json")
    rollout_audit_digest = None
    if rollout_audit:
        rollout_audit_digest = {key: rollout_audit.get(key) for key in
            ("complete", "passed", "episode_count", "phase_decisions", "saved_native_zero_checks_per_decision",
             "independent_arrays_checked", "absent_phase_counts", "episodes", "visual_review_scope", "limitations")}
        rollout_audit_digest["bound_files"] = len(rollout_audit.get("input_sha256", {}))
        rollout_audit_digest["figures_decoded"] = len(rollout_audit.get("figures_decoded_and_bound", []))
        rollout_audit_digest["scope"] = "Sampled own-policy phase decisions only, not every intermediate trajectory decision. Absent basket-approach phases are omitted without substitution. File decoding does not imply manual inspection of every figure."
        rollout_audit_digest["selected_visual_review"] = rollout_visual_review
        rollout_audit_digest["color_scale_scope"] = "A single phase-panel figure shares its color scale across phases and flow times, but maxima differ between policy/camera panels. Cross-panel brightness is not a quantitative comparison; use recorded color limits and raw metrics."
    contact_summary = sources.document("contact_analysis/summary.json")
    contact_lineage = "contact_analysis"
    if support_status["active"]:
        contact_lineage = "contact_analysis_structural_na_v1"
        contact_summary = sources.document(contact_lineage+"/summary.json")
        if contact_summary:
            from rl_vla.diagnostics.contact_support_overlay_v1 import verify_overlay
            contact_summary = verify_overlay(root)
        for field, filename in (("contact_candidate_rates","contact_candidate_rates.csv"),("contact_manual_calibration","contact_manual_calibration.csv")):
            tables[field] = sources.table(filename, relative=str(Path(contact_summary["tables"][filename]["path"]).relative_to(root))) if contact_summary else []
        for filename in ("contact_cases.csv","support_unavailable_cases.csv"):
            sources.index(str(Path(contact_summary["tables"][filename]["path"]).relative_to(root)) if contact_summary else contact_lineage+"/tables/"+filename)
    sources.index("contact_analysis/supersession.json")
    sources.index("technical_attempts/contact_sidecar_gpu6_supersession_v1/supersession.json")
    sources.index("compiler_profile_integration/contact_sidecar_supersession_acceptance_v1/acceptance.json")
    sources.index("tables/contact_cases.csv")
    sources.document("analysis_source_manifest.json")
    sources.index("README.md")
    sources.document("native_controller_contract.json")
    sources.index("compiler_profile_integration/native_state_path_clarification_v2/summary.json")
    prefix_mask_audit = sources.document("prefix_mask_interpretation_audit.json")
    learner_restore_audit = sources.document("learner_restore_audit/summary.json")
    compiler_amendment = sources.document("compiler_profile_integration/fidelity_protocol_amendment_v1.json")
    compiler_control_declaration = sources.document("compiler_profile_integration/compiler_control_declaration.json")
    compiler_control_declaration_v2 = sources.document("compiler_profile_integration/compiler_control_declaration_v2.json")
    sources.index("compiler_profile_integration/grid_staged_dispatch_acceptance.json")
    sources.index("diagnostic_policy_dispatch_launch/attention_acceptance_binding_v2.json")
    sources.index("technical_attempts/gpu6_descriptor_profile_miss_v1/failure_adjudication.json")
    sources.index("compiler_review/gpu6_private_cache_v1/kernel_comparison.json")
    healthy_none = sources.document("compiler_profile_integration/healthy_device_none_rollouts_v1/summary.json")
    sources.index("compiler_review/gpu6_derived_hlo_v1/kernel_audit_complete.json")
    sources.index("compiler_profile_integration/gpu6_fidelity_v1/summary.json")
    sources.index("compiler_profile_integration/gpu6_native_cache_oracle_v1/summary.json")
    sources.index("compiler_profile_integration/gpu6_ood_native_runtime_v1.json")
    sources.index("compiler_profile_integration/gpu6_descriptor_probe_v1/summary.json")
    sources.index("compiler_profile_integration/gpu6_concurrency_v1/summary.json")
    sources.index("compiler_profile_integration/gpu6_independent_review_v1/acceptance.json")
    repaired_none = sources.document("compiler_profile_integration/object1_repaired_none_rollouts_v1/summary.json")
    none_cells = [row for audit in (healthy_none, repaired_none) if audit for row in audit.get("cells", [])]
    none_complete = (all(audit and audit.get("complete") is True and audit.get("passed") is True for audit in (healthy_none, repaired_none))
                     and len(none_cells) == 12
                     and {(row["task"], canonical_policy(row["policy"])) for row in none_cells} == {(task, policy) for task in TASKS for policy in POLICIES[1:]}
                     and all(row.get("passed") is True and row.get("paired_episodes") == 10
                             and all(row.get(field) == 10 for field in ("outcomes_equal", "full_normalized_action_sequences_exact", "full_executed_action_sequences_exact", "full_simulator_state_sequences_exact"))
                             for row in none_cells))
    for row in none_cells:
        relative = str(Path(row["result"]["path"]).relative_to(root))
        sources.index(relative)
        if sources.records[relative]["sha256"] != row["result"]["sha256"]:
            raise ValueError("Full-rollout none-control audit artifact changed")
    sources.index("diagnostic_policy_dispatch_launch/object_1_gpu6_v1.json")
    sources.index("compiler_profile_integration/object0_noun_mapping_independent_review_v1.json")
    sources.index("compiler_profile_integration/test2_prefill_interpretation_v1.json")
    sources.index("compiler_profile_integration/main_result_figure_visual_review_v1.json")
    sources.index("compiler_profile_integration/vlm_input_contract_correction_v2.json")
    sources.index("compiler_profile_integration/representation_input_contract_audit_v1.json")
    sources.index("compiler_profile_integration/counterfactual_probe_identity_audit_v1.json")
    sources.index("compiler_profile_integration/figure_artifact_guard_independent_review_v1.json")
    sources.index("compiler_profile_integration/object0_supplement_independent_review_v1.json")
    sources.index("compiler_profile_integration/supplemental_object0_numeric_review_v1.json")
    sources.index("scientific_protocol_review_v1.json")
    sources.index("compiler_profile_integration/independent_trace_review_wait_complete_v1.json")
    for relative in ("ood_per_lane_release_review_v1/launch_manifest_v1.json",
                     "ood_per_lane_release_review_v1/scheduling_amendment_v1.json",
                     "ood_per_lane_release_review_v1/root_acceptance_v1.json",
                     "ood_per_lane_release_review_v1/independent_attention_review.json",
                     "ood_per_lane_release_review_v1/completion_consumer_review.json",
                     "ood_per_lane_release_review_v1/first_native_shards_audit_v1.json"):
        sources.index(relative)
    from rl_vla.diagnostics.ood_completion_support_v2 import completion as ood_completion
    from rl_vla.diagnostics.ood_figures import verify as verify_ood_figures
    ood_dispatch = {task: ood_completion(root, task) for task in TASKS}
    ood_figures = {task: verify_ood_figures(root, task) for task in TASKS}
    sources.index("compiler_profile_integration/ood_figure_presentation_v1.json")
    sources.index("compiler_profile_integration/ood_figure_implementation_v2.json")
    sources.index("compiler_profile_integration/object_1_ood_figures_candidate_review_v1.json")
    sources.index("compiler_profile_integration/object_1_ood_figures_candidate_review_v2.json")
    for task in TASKS:
        sources.index(f"figures/ood/{task}/summary.json")
        sources.index(f"ood/{task}/queue_complete.json")
        sources.index(f"ood/{task}/input_manifest.json")
        sources.index(f"ood_per_lane_release_review_v1/release_plans/{task}.json")
        sources.index(f"ood_per_lane_release_review_v1/{task}_final_consumer_acceptance_v1.json")
    sources.index("ood/object_1/gpu6_native_cache_binding.json")
    sources.index("ood/object_1/gpu6_native_cache_completion.json")
    sources.index("tables/failure_transition_cases.csv")
    supplemental = {}
    for task in TASKS:
        value = sources.document(f"supplemental_analysis/{task}/summary.json")
        supplemental[task] = value or {"complete": False, "status": "pending"}
        if value and value.get("complete"):
            for artifact in value["artifacts"]:
                path = Path(artifact["path"])
                if path.suffix in {".json", ".csv"}:
                    sources.index(str(path.relative_to(root)))
    sources.index("compiler_profile_integration/supplemental_analysis_amendment_v1.json")
    for task in TASKS:
        sources.index(f"trace_analysis/{task}/noun_span_resolution.json")
        sources.index(f"compiler_profile_integration/{task}_counterfactual_same_class_review_v1.json")
        sources.index(f"compiler_profile_integration/{task}_counterfactual_same_class_comparison_v1.csv")
        for review in ("supplement_independent_review", "figures_visual_review", "expert_compiler_control_review"):
            sources.index(f"compiler_profile_integration/{task}_{review}_v1.json")
    review_plan = sources.document("video_reviews/review_plan.json")
    for path in sorted((root / "video_reviews/reviewed_labels").glob("*.json")):
        sources.index(str(path.relative_to(root)))
    sources.index("compiler_profile_integration/object_0_review_marker_schema_audit_v1.json")
    sources.index("compiler_profile_integration/review_inventory_marker_compatibility_v2/acceptance.json")
    for path in sorted((root / "compiler_profile_integration").glob("object_*_ood_*_review_v1.json")):
        sources.index(str(path.relative_to(root)))
    if review_plan and (review_plan.get("expected_main_reports"), review_plan.get("expected_ood_reports")) != (36, 48):
        raise ValueError("Declared failure-review coverage differs from the fixed main/OOD matrix")
    sources.index("tables/ood_failure_reviews.csv")
    main_reviews = full_inventory(tables["wrong_object"], ("task", "split", "policy"),
        [(task, split, policy) for task in TASKS for split in ("validation", "test", "source") for policy in POLICIES[1:]])
    variants = ("both_new_regions", "continuous_new_offsets", "target_moved_distractor_fixed", "distractor_moved_target_fixed")
    ood_reviews = full_inventory(tables["ood_wrong_object"], ("task", "policy", "variant"),
        [(task, policy, variant) for task in TASKS for policy in POLICIES[1:] for variant in variants])
    pending_reviews = [row for row in main_reviews + ood_reviews
                       if row["status"] != "complete_review" or row.get("episodes") != 50
                       or row.get("unreviewed_failures") != 0 or row.get("review_coverage_complete") is not True]
    attention = []
    for row in tables["paired_attention"]:
        if row["step"] not in (0, 5, 9):
            raise ValueError("Attention digest accepts only the predeclared flow steps0,5,9")
        attention.append(row)
    required = ("paired_success", "probe", "counterfactual_probe", "paired_attention", "causal_success", "ood_success", "ood_paired", "drift",
                "contact_candidate_rates", "contact_manual_calibration", "ood_wrong_object", "ood_failure_taxonomy")
    pending = [row for row in success + diagnostics if row["status"] not in {"complete", "not_applicable"}]
    missing_sections = [name for name in required if not tables[name]]
    if operational_health["failures"]:
        missing_sections.append("failed_execution_queues")
    if not all(value["complete"] for value in ood_dispatch.values()):
        missing_sections.append("ood_dispatch_provenance")
    if not all(value["complete"] for value in ood_figures.values()):
        missing_sections.append("ood_figures_and_review_coverage")
    if not none_complete:
        missing_sections.append("paired_none_controls")
    from rl_vla.diagnostics.causal_support_overlay_v1 import figure_coverage
    from rl_vla.diagnostics.figure_artifacts import cells, manifest_path
    figure_artifacts=sources.document("figure_artifacts/summary.json")
    figure_overlay={}
    try:
        figure_overlay=figure_coverage(root)
        figures_ready=figure_overlay["complete"]
    except (OSError,ValueError,KeyError,TypeError):
        figures_ready=False
    for domain,task,policy in cells():
        sources.index(str(manifest_path(root,domain,task,policy).relative_to(root)))
    if not figures_ready:
        missing_sections.append("validated_figures_controls_vectors")
    from rl_vla.diagnostics.causal_support_overlay_v1 import (
        readable_coverage as causal_figure_coverage,
    )
    causal_figures = causal_figure_coverage(root)
    sources.index("causal_figure_renderer_launch_v1.json")
    sources.index("compiler_profile_integration/causal_figure_integration_review_v1.json")
    sources.index("compiler_profile_integration/tests123_provisional_scientific_notes_v2.json")
    sources.index("compiler_profile_integration/causal_open_loop_cross_checkpoint_notes_v1.json")
    sources.index("compiler_profile_integration/object_0_selected_ood_failure_transitions_v1.json")
    sources.index("compiler_profile_integration/object_0_selected_ood_failure_transitions_independent_review_v1.json")
    sources.index("compiler_profile_integration/object_0_selected_ood_transition_review_v1/source_snapshot.json")
    sources.index("compiler_profile_integration/object_0_selected_ood_transition_review_v1/ood_paired_at_review.csv")
    sources.index("compiler_profile_integration/object_2_ood_rldt_25_distractor_moved_review_v1.json")
    sources.index("compiler_profile_integration/causal_structural_na_v1/analysis_consumer_validation_v1.json")
    sources.index("compiler_profile_integration/causal_structural_na_v1/contact_overlay_independent_review_v1.json")
    sources.index("compiler_profile_integration/causal_structural_na_v1/independent_ood_bridge_consumer_review_v2/review.json")
    sources.index("compiler_profile_integration/causal_structural_na_v1/unaffected_figure_coverage_review_v1.json")
    for name in ("validation.json", "contact_overlay_adoption.json", "bridge_v2_acceptance.json"):
        sources.index("compiler_profile_integration/causal_structural_na_v1/independent_actual_execution_review_v1/" + name)
    sources.index("compiler_profile_integration/causal_structural_na_v1/final_numeric_review_v1/summary.json")
    sources.index("compiler_profile_integration/object_1_all_causal_support_review_v1.json")
    sources.index("technical_attempts/superseded_causal_cpu_waiters_v1/receipt.json")
    sources.index("contact_analysis_structural_na_v1/worker_launch_v1.json")
    sources.index("contact_analysis_structural_na_v1/acceptance_v1.json")
    sources.index("contact_analysis_structural_na_v1/final_acceptance_v1.json")
    sources.index("resource_cost_final_v1.json")
    sources.index("compiler_profile_integration/final_writer_freeze_v1/receipt.json")
    sources.index("compiler_profile_integration/final_scientific_narrative_independent_review_v1/summary.json")
    sources.index("compiler_profile_integration/causal_structural_na_v1/first_witness_independent_launch_v1.json")
    sources.index("compiler_profile_integration/causal_structural_na_v1/first_witness_independent_v1/summary.json")
    sources.index("compiler_profile_integration/causal_structural_na_v1/first_witness_explicit_visual_review_v1.json")
    sources.index("compiler_profile_integration/causal_structural_na_v1/first_witness_visual_v1/summary.json")
    sources.index("compiler_profile_integration/causal_structural_na_v1/independent_execution_validator_review_v1/review.json")
    sources.index("figures/causal/summary.json")
    for task in TASKS:
        sources.index(f"compiler_profile_integration/{task}_all_causal_review_v1.json")
    for name in ("amendment_v1.json","root_acceptance.json","execution_coverage.json","execution_cases.json","native_key_recurrence_v1.json",
                 "research_launch_v1.json","research_launch_candidate_v1.json","bridge_acceptance.json","bridge_launch_v1.json",
                 "bridge_acceptance_v2.json","bridge_launch_v2.json","analysis_consumer_launch_v2.json"):
        sources.index("compiler_profile_integration/causal_structural_na_v1/"+name)
    sources.index("causal_analysis_structural_na_v1/summary.json")
    for cell in causal_figures["cells"]:
        if cell["status"] == "complete":
            relative = str(Path(cell["summary"]["path"]).relative_to(root))
            manifest = sources.document(relative)
            sources.index(str(Path(manifest["manifest"]["path"]).relative_to(root)))
            for item in cell["artifacts"]:
                if Path(item["path"]).suffix in {".json", ".csv"}:
                    sources.index(str(Path(item["path"]).relative_to(root)))
    if not causal_figures["complete"]:
        missing_sections.append("readable_causal_figures")
    if support_status["active"] and not support_status["complete"]:
        missing_sections.append("accepted_strict_support_execution_coverage")
    digest = {
        "schema_version": 1,
        "purpose": "Compact evidence index for human scientific review; no conclusions are inferred",
        "evidence_complete": not pending and not missing_sections and bool(panels and panels.get("complete"))
                             and bool(contact_summary and contact_summary.get("complete"))
                             and figures_ready
                             and not pending_reviews and bool(review_plan)
                             and all(row["status"] == "complete" for row in training_coverage)
                             and all(value["status"] == "locked" for value in selections.values()),
        "completion_scope": "Existing result cells and required summary sections only; this is not the final acceptance/scientific-review gate",
        "operational_health": operational_health,
        "native_learner_restore_coverage": learner_restore_audit,
        "figure_artifact_coverage": {"complete":figures_ready,"summary":figure_artifacts,"accepted_lineage_coverage":figure_overlay,
            "scope":"All27 original-probe/counterfactual-probe/causal/supplemental cells require decoded figures and bound source tables/maps; early producer metrics alone are insufficient."},
        "causal_presentation_coverage": causal_figures,
        "causal_success_floor_caveat": "All-zero task-success observations and paired bootstrap[0,0] do not establish pathway irrelevance or equivalence; open-loop responses and grasp proxies can differ at a success floor.",
        "structural_support": {**{key:value for key,value in support_status.items() if key!="value"},
            "timing": "Availability amendment adopted after strict-control failure and partial outcomes were known; support can disappear dynamically along a policy trajectory, so the completed subset does not estimate the original planned effect. Original masks/cases retained, no fallback/imputation or complete-case primary estimate."},
        "rollout_phase_independent_review": rollout_audit_digest,
        "independent_expert_control_means": {"rows": tables["independent_expert_controls"],
            "scope": "Copied independent first-five-state means at steps0/5/9; no new aggregation. Changes exceeding these measured compiler controls do not establish task improvement, successful grounding, explained behavioral variance, or a universal numerical-error bound."},
        "final_training_reset_coverage": training_coverage,
        "failure_review_coverage": {"plan": review_plan, "expected_main_reports": 36, "expected_ood_reports": 48,
            "complete": not pending_reviews and bool(review_plan), "pending_reports": pending_reviews,
            "scope": "All declared main and OOD failed episodes require explicit review. Native successes require no failure review; causal proxies remain outside this coverage."},
        "pending_cells": pending, "missing_summary_sections": missing_sections,
        "selection": selections, "success": success, "paired_success": tables["paired_success"],
        "diagnostic_inventory": diagnostics, "wrong_object": tables["wrong_object"], "failure_taxonomy": tables["failure_taxonomy"],
        "failure_label_codebook": tables["failure_label_codebook"],
        "paired_failure_transitions": {"summary": tables["failure_transition_summary"], "counts": tables["failure_transition_counts"],
            "case_table": "tables/failure_transition_cases.csv", "scope": "Descriptive verified state/seed pairs; a wrong-object failure changing to a target-grasp placement failure is not recovery to task success. Failure labels do not exhaust first-choice behavior among successes."},
        "probes": {"original_prompt_and_cross_layout": tables["probe"], "counterfactual_requested_object": tables["counterfactual_probe"],
                   "same_physical_class_prompt_transfer": tables["counterfactual_same_class"],
                   "generalization_scope": "Narrow native XY/fixed-swap support permits position-correlated decoding. Held-out localization establishes distributional decodability, not object-identity generalization to custom OOD layouts. Same-class prompt-transfer changes concern the fixed original-prompt readout, not a proof of absent information or causal policy use.",
                   "paired_prompt_changes": tables["probe_prompt_change"],
                   "requested_role_scope": "Original-requested versus counterfactual-requested scores switch physical class1 to class2 as well as the instruction. Class difficulty therefore confounds that contrast; same-identity paired prediction/localization changes and exact image-only controls are separate comparisons.",
                   "raw_base_comparison_scope": "Raw base is an image/text-only checkpoint reference with no validated LIBERO action adapter. The pinned pi05_libero config also disables discrete state input, so both observed VLM prefixes contain instruction tokens without state tokens. Numeric state is retained in native context and critic features; this pinned actor has no active numeric-state embedding or decoder state offset. Physical simulator/controller dynamics remain state-dependent. The direct RLDT contrast is against pi05_libero."},
        "attention": {"flow_steps": [0, 5, 9], "nominal_flow_times": [1., .5, .1],
            "prior_location_scope": "Projected prior-location geometry and visible instance masks are distinct and can overlap because the official swap puts the distractor at the former target region. Attention/ablation there alone cannot separate identity from a spatial shortcut; target-fixed, distractor-fixed and new-region conditions address that confound.",
            "aggregation": "Existing paired_attention.csv means: first average all expert layers/heads/action queries within each observation, then average observations; cameras remain separate",
            "uncertainty": "Copied existing observation-paired bootstrap intervals for checkpoint minus LIBERO. No new resampling, independent-head replicates or baseline mean CIs are introduced.",
            "all_predeclared_step_comparisons": attention, "raw_artifact_index": tables["grounding_index"]},
        "causal": {"native_success_pairs": tables["causal_success"], "existing_open_loop_means_and_intervals": causal_open_loop,
                   "full_rollout_none_controls": {"complete": none_complete, "paired_episodes": sum(row["paired_episodes"] for row in none_cells if row.get("passed") is True),
                       "cells": none_cells, "healthy_device_scope": healthy_none.get("scope") if healthy_none else None,
                       "repaired_object1_scope": repaired_none.get("scope") if repaired_none else None,
                       "scope": "All120 paired controls compare normalized/executed actions, controlled simulator states, reset/outcomes. The repaired40 Object1 episodes additionally compare every shared retained object-telemetry field. Causal raw images/keys and unretained per-step reward/done fields were not directly compared."},
                   "compiler_scope": "Separate baseline and trained diagnostic processes use their verified compiler profiles and immutable shared baseline contexts/latent paths. Each must match independent uninstrumented native outputs obtained via verified original-cache hits exactly. Historical recorded-action variance is separate reproducibility evidence, never relabeled exact or tolerance-passed. Actual held-out rollout traces still require exact saved-action replay. Original training/evaluation runtime, outcomes and shared caches remain untouched.",
                   "compiler_protocol_amendment": compiler_amendment,
                   "compiler_control_declaration": compiler_control_declaration_v2 or compiler_control_declaration,
                   "first_grasp_proxies_unreviewed": tables["causal_grasp_proxy"],
                   "pathway_scope": "The bidirectional image/text prefix can mix both modalities. No-image key masking can retain visual information through contextualized text keys; no-text key masking can retain instruction information through contextualized image keys. Neither intervention removes all information from a modality.",
                   "native_prefix_mask_audit": prefix_mask_audit,
                   "units": "Decoded command norms are not metres; nominal physical request metrics remain in per-policy open_loop_actions.csv after native clipping/scaling, not achieved motion"},
        "unseen_layouts": {"success": tables["ood_success"], "paired": tables["ood_paired"],
                           "figure_readiness": ood_figures,
                           "dispatch_provenance": ood_dispatch,
                           "scheduling_scope": "Approved resource-order amendment only; released main resources may start before causal completion. Every resource retains its own exact release gate. Scientific cases, seeds, policies, layouts and three native shards are unchanged; superseded empty waiters are archived.",
                           "camera_visibility_support": tables["ood_visibility_support"],
                           "spatial_scope": "Variants reuse50 official base states and two fixed novel centers; continuous offsets widen overlapping neighborhoods. Keep variants separate, not200 independent novel-region samples. Each object passes20 visible pixels in at least one camera; Object0 target_moved_distractor_fixed distractor has fewer than20 external pixels throughout, so camera-specific evidence differs.",
                           "wrong_object_failures": tables["ood_wrong_object"], "failure_taxonomy": tables["ood_failure_taxonomy"],
                           "per_episode_review_table": "tables/ood_failure_reviews.csv",
                           "scope": "Custom diagnostic layouts, separate from official PRO and original-layout retention. Difficulty can vary in either direction; performance at two novel centers/offset families does not rule out coarse regional priors. Policy comparisons remain within each variant."},
        "parameter_drift": tables["drift"],
        "native_contact_proxies": {"coverage": contact_summary, "candidate_rates": tables["contact_candidate_rates"],
            "manual_calibration": tables["contact_manual_calibration"], "case_table": (str(Path(contact_summary["tables"]["contact_cases.csv"]["path"]).relative_to(root)) if support_status["active"] and contact_summary else "tables/contact_cases.csv"),
            "lineage": contact_lineage, "support_unavailable_cases": str(Path(contact_summary["tables"]["support_unavailable_cases.csv"]["path"]).relative_to(root)) if support_status["active"] and contact_summary else None,
            "scope": "Automated sustained-contact/lift candidates remain separate from immutable reviewed semantic-failure labels. Per-case rows are linked/hashed, not copied into this compact digest. Explicit GPU6 supersession preserves the prior sidecar lineage, excludes four archived invalid controls, and re-extracts repaired Object1 causal records with unchanged code/rules."},
        "secondary_ablation_index": {
            "compiler_precision": {"path": "tables/compiler_precision_controls.csv", "rows": tables["compiler_controls"],
                                   "scope": "Unchanged baseline weights/context/latent/time under both verified compiler profiles; first5 fixed cases at0/5/9 only. Full attention/hidden/velocity arrays retained. No behavioral scores and no compiler-independence claim for the other45 cases."},
            "projection": {"path": "tables/projection_ablation.csv", "rows": len(tables["projection_ablation"]),
                           "scope": "First five fixed cases; native hidden/output projection swaps; per-case normalized velocity effects, no new aggregate. Baseline-weight and trained-weight combinations retain their verified native compiler profiles; same-hidden baseline-weight controls quantify compiler arithmetic separately from parameter effects."},
            "layer_knockout": {"path": "tables/layer_knockouts.csv", "rows": len(tables["layer_knockouts"]),
                               "scope": "First five fixed cases; early/middle/late and individual-layer image/text-key masks; no new aggregate"}},
        "qualitative_artifacts": tables["qualitative_figures"], "vla_trace_exports": tables["vla_trace_exports"],
        "supplemental_exploratory_analysis": {"tasks": supplemental, "artifact_index": tables["supplemental_artifacts"],
            "scope": "Declared after held-out task outcomes, before supplemental-map inspection. Modality routing, token controls, layer-response figures and action vectors are exploratory. Nominal controller requests and directional alignment are not achieved motion or predicted first-choice accuracy."},
        "rollout_panel_completion": {"complete": bool(panels and panels.get("complete")), "source": "rollout_trace_analysis/summary.json"},
        "sources": sources.records,
        "provenance_note": "SHA256 values bind the exact compact source bytes read. Live telemetry, acceptance timestamps, generated report text and scientific-review files are excluded to avoid unstable hashes or circular review dependencies.",
    }
    atomic_json(root / "evidence_digest.json", digest)
    markdown = render_markdown(digest)
    temporary = root / "evidence_digest.md.tmp"
    temporary.write_text(markdown)
    temporary.replace(root / "evidence_digest.md")
    return {"evidence_complete": digest["evidence_complete"], "pending_cells": len(pending),
            "missing_summary_sections": missing_sections, "sources": len(sources.records)}


def number(value):
    return "—" if value is None else f"{value:.4g}"


def interval(value):
    return "—" if not value else f"[{number(value[0])}, {number(value[1])}]"


def render_markdown(digest):
    status = "all indexed evidence is present; final scientific review is still separate" if digest["evidence_complete"] else "incomplete; absent cells remain pending"
    lines = ["# Scientific evidence digest", "", f"Status: **{status}**.", "",
        "This file copies existing measurements and intervals without fitting, resampling or inferring conclusions. All three tasks, five policy conditions and predeclared flow steps remain in [the structured digest](evidence_digest.json), whose `sources` map records exact paths and SHA256 hashes.", "",
        f"Pending action/diagnostic cells: **{len(digest['pending_cells'])}**. Missing summary sections: {', '.join(digest['missing_summary_sections']) or 'none'}. Raw-base action incompatibilities are retained as N/A, never zero success.", "",
        f"Pending failure-review reports: **{len(digest['failure_review_coverage']['pending_reports'])}/84** across 36 main and 48 custom OOD reports. The [predeclared review plan](video_reviews/review_plan.json) requires every failed case to be reviewed; native successes need no failure label. Causal contact proxies remain separate.", "",
        "The 100 prescribed training states are a sampling bank, not a claim of 100 distinct visits. [Final completed-episode reset coverage](tables/training_state_coverage.csv) and [per-state counts](tables/training_state_visit_counts.csv) count sampling with replacement; final unfinished episodes are excluded.", "",
        "Compiler replay is diagnostic-only and uses separate baseline/trained processes with isolated child caches; exact baseline contexts and latent paths are exchanged as immutable artifacts. The [pre-diagnostic amendment](compiler_profile_integration/fidelity_protocol_amendment_v1.json) requires exact agreement with independent original-native outputs obtained through verified cache hits for all requested checkpoints. Historical recorded-action variance stays separate and is not relabeled tolerance-passed; actual held-out rollout traces still require exact saved-action replay. Original training/evaluation runtime, outcomes and shared caches remain untouched.", "",
        "## Action results and paired checkpoints", "",
        "Validation is reused checkpoint-selection data; its gains and descriptive intervals do not establish held-out improvement. `source` is original-layout retention. Raw base supports image/text representation diagnostics but has no validated LIBERO action adapter. The pinned native LIBERO configuration also disables discrete state input: both captured prefixes contain instruction tokens without proprioceptive state tokens; numeric state is retained in native context and critic features, with no active actor numeric-state embedding or decoder state offset in this pinned configuration. Physical execution remains state-dependent. The direct RLDT comparison is against pi05_libero. Locked selections: " + "; ".join(
            f"{task}: iteration {value['record']['iteration']}" if value["status"] == "locked" else f"{task}: pending"
            for task, value in digest["selection"].items()) + ". Nonselected checkpoint test comparisons remain exploratory.", "",
        "| Task | Split | Policy | Status / successes | Wilson95% |", "|---|---|---|---|---|"]
    for row in digest["success"]:
        outcome = f"{row['successes']}/{row['episodes']}" if row["status"] == "complete" else row["status"]
        ci = interval([row.get("wilson95_low"), row.get("wilson95_high")]) if row["status"] == "complete" else "—"
        lines.append(f"| {row['task']} | {row['split']} | {row['policy']} | {outcome} | {ci} |")
    lines += ["", "| Task | Split | Checkpoint | Paired difference | Bootstrap95% | Recovered / regressed | Exact McNemar p |", "|---|---|---|---:|---|---:|---:|"]
    for row in digest["paired_success"]:
        lines.append(f"| {row['task']} | {row['split']} | {row['policy']} | {number(row['difference'])} | {interval(row['paired_bootstrap95'])} | {row['recovered']} / {row['regressed']} | {number(row['mcnemar_exact_two_sided_p'])} |")
    if not digest["paired_success"]:
        lines.append("| — | — | — | pending | — | — | — |")
    lines += ["", "Fractions and differences use units of1 (not percentage points). [Success table](tables/success.csv), [paired table](tables/paired_success.csv), [reviewed failure taxonomy](tables/failure_taxonomy.csv), [confirmed wrong-object fractions and review coverage](tables/wrong_object.csv). Unreviewed or ambiguous failures are not automatically semantic errors. Wrong-object labels use native evidence and selected-frame review by one AI reviewer per failed episode, with work distributed across agents and no independent second visual rater; episode-count intervals do not cover reviewer error or uncertain classification.", "",
        "## Probe and prompt-transfer measurements", "",
        "| Task | Model | Features | Split | Balanced accuracy | Target AUC | Target IoU |", "|---|---|---|---|---:|---:|---:|"]
    for row in digest["probes"]["original_prompt_and_cross_layout"]:
        lines.append(f"| {row['task']} | {row['policy']} | {row['features']} | {row['split']} | {number(row['balanced_accuracy'])} | {number(row['target_auc'])} | {number(row['target_iou'])} |")
    if not digest["probes"]["original_prompt_and_cross_layout"]:
        lines.append("| — | — | — | pending | — | — | — |")
    lines += ["", "Counterfactual requested-object scores and identity-prediction changes are retained in the structured digest and [counterfactual table](tables/counterfactual_probe_metrics.csv), [prompt-change table](tables/probe_prompt_changes.csv). The original-prompt train-only probe is reused without refitting; semantic identity labels stay fixed while the requested role switches. Original-requested versus counterfactual-requested scores also switch class/difficulty; they do not alone measure a same-class prompt effect. Same-identity paired changes and exact image-only controls remain separate. Task-specific three-class probes trained on fixed or nearly fixed swap positions measure distributional decodability, not position-invariant recognition. Same-layout success can reflect spatial or appearance regularities. Source-layout transfer is the existing location-shift control; image-only/prefix comparisons and counterfactual prompts alone do not remove location confounding.", "",
        "## Matched expert attention: all layers, heads and action queries", "",
        "The following values are existing region-density means and checkpoint-minus-LIBERO paired intervals. Cameras stay separate. Rows cover predeclared denoising steps0,5,9 (nominal t=1,.5,.1). Full-denominator mass, camera-conditional fraction and old-location region measurements are also copied into the JSON digest; baseline mean uncertainty is not invented.", "",
        "| Task | Policy | Camera | Step | Region | States | LIBERO mean | Policy mean | Difference | Paired95% |", "|---|---|---|---:|---|---:|---:|---:|---:|---|"]
    attention = [row for row in digest["attention"]["all_predeclared_step_comparisons"] if row["metric"] == "region_density" and row["region"] in {"target", "distractor"}]
    for row in attention:
        lines.append(f"| {row['task']} | {row['policy']} | {row['camera']} | {row['step']} | {row['region']} | {row['states']} | {number(row['baseline_mean'])} | {number(row['checkpoint_mean'])} | {number(row['difference'])} | {interval([row['paired_bootstrap95_low'], row['paired_bootstrap95_high']])} |")
    if not attention:
        lines.append("| — | — | — | — | pending | — | — | — | — | — |")
    lines += ["", "[Paired attention](tables/paired_attention.csv), [per-observation grounding](tables/grounding.csv), [full layer/head/token and routing table index](tables/grounding_artifact_index.csv). Attention averages and changes do not establish causal use. Projected prior-location geometry differs from visible instance masks, and the official swap places the distractor in that former target region. Overlapping masks mean attention or ablation there alone cannot separate object identity from a position shortcut; the target-fixed, distractor-fixed and new-region conditions address that confound.", "",
        "## Causal and unseen-layout results", "",
        "| Task | Policy | Condition | Successes | Difference vs no-op | Paired95% |", "|---|---|---|---:|---:|---|"]
    for row in digest["causal"]["native_success_pairs"]:
        lines.append(f"| {row['task']} | {row['policy']} | {row['condition']} | {row['successes']}/{row['episodes']} | {number(row['difference'])} | {interval(row['paired_bootstrap95'])} |")
    if not digest["causal"]["native_success_pairs"]:
        lines.append("| — | — | pending | — | — | — |")
    lines += ["", "Open-loop mean command deltas and their existing intervals are copied from each completed causal summary into the JSON digest. [Causal success](tables/causal.csv), [unreviewed native grasp-choice proxies](tables/causal_grasp_proxies.csv). The native prefix permits bidirectional image/text mixing: blocking image keys can leave visual information in contextualized text keys, and blocking text keys can leave instruction information in contextualized image keys. These direct key-pathway masks do not erase either modality's information; pixel and prefill interventions are distinct. [Native mask audit](prefix_mask_interpretation_audit.json). Decoded command norms are not metres; separately labeled nominal OSC request deltas are not achieved motion.", "",
        "| Task | Policy | Custom layout | Successes | Wilson95% |", "|---|---|---|---:|---|"]
    for row in digest["unseen_layouts"]["success"]:
        lines.append(f"| {row['task']} | {row['policy']} | {row['variant']} | {row['successes']}/{row['episodes']} | {interval([row['wilson95_low'], row['wilson95_high']])} |")
    if not digest["unseen_layouts"]["success"]:
        lines.append("| — | — | pending | — | — |")
    lines += ["", "[OOD paired recovered/regressed cases and uncertainty](tables/ood_paired.csv) remain separate from official PRO and original-layout retention. [OOD confirmed wrong-object failures](tables/ood_wrong_object.csv), [failure taxonomy](tables/ood_failure_taxonomy.csv) and [per-episode review evidence](tables/ood_failure_reviews.csv) retain review coverage and ambiguity. Fractions use all50episodes and remain pending until all failures are reviewed; they are not exhaustive first-choice errors among eventual successes. The kinematic audits establish local pose feasibility, not continuous-path or grasp guarantees. [Camera visibility/support](tables/ood_visibility_support.csv) records the external-camera distractor visibility limitation for Object0 target_moved_distractor_fixed. Variants reuse the same50 base states and overlapping neighborhoods around two new centers; they must not be pooled as200 independent novel-region samples.", "",
        "[Paired failure transitions](tables/failure_transition_summary.csv) and [full category matrices](tables/failure_transition_counts.csv) distinguish wrong-object failures recovered to native success from failures that become target-grasp placement failures. Ambiguous/unreviewed labels remain distinct; these are descriptive transitions, not new statistical tests.", "",
        "[Supplemental figures and raw exports](tables/supplemental_artifacts.csv) add modality curves, token controls, layer-response plots and action vectors. Their amendment followed held-out outcome availability and preceded supplemental-map inspection; these are exploratory additions. Nominal directional cosines are not achieved motion or predicted first-choice accuracy. Each task remains pending until the independent artifact guard verifies every required figure and raw export.", "",
        "## Parameter changes and source artifacts", "",
        "All recorded expert-layer/projection drift rows are copied into the JSON digest. [Parameter drift](tables/checkpoint_drift.csv), [first-five-case output projection swaps](tables/projection_ablation.csv), [first-five-case layer knockouts](tables/layer_knockouts.csv), [qualitative figure index](tables/qualitative_figures.csv), [VLA-Trace invocations](tables/vla_trace_exports.csv). Secondary per-case ablations are indexed without adding new means or intervals.", "",
        "[Native contact candidate rates](tables/contact_candidate_rates.csv), [manual calibration](tables/contact_manual_calibration.csv) and the [hashed case table](tables/contact_cases.csv) are a separate provisional evidence stream. Their coverage summary is retained in the structured digest; automated candidates never replace reviewed wrong-object labels.", "",
        "The structured digest's `sources` hashes bind compact source tables and summaries. Live telemetry, changing acceptance timestamps and generated reports are excluded. The final review can bind `evidence_digest.json`, `evidence_digest.md`, and the stable evidence tables it cites. [Diagnostic pending/N/A cells](tables/diagnostic_status.csv) remain authoritative for coverage. Single RL seed and exploratory, unadjusted comparisons limit interpretation.", ""]
    return "\n".join(lines)
