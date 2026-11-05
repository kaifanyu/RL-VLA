"""Check handoff acceptance gates without equating queued jobs with results."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

TASKS = ("object_0", "object_1", "object_2")
ITERATIONS = (25, 50, 100)
POLICIES = ("pi05_base", "pi05_libero", "rldt_25", "rldt_50", "rldt_100")


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def check(root):
    root = Path(root).resolve()
    gates = {name: {"passed": False, "checks": [], "pending": [], "failed": []}
             for name in ("A_readiness", "B_fidelity", "C_training", "D_policy_comparison", "E_five_tests", "F_report")}

    def record(gate, name, value, path=None):
        # None means absent/incomplete; False means supplied evidence failed.
        row = {"name": name, "passed": value, "artifact": str(path) if path else None}
        gates[gate]["checks"].append(row)
        if value is not True:
            gates[gate]["pending" if value is None else "failed"].append(name)

    def artifact(gate, name, relative, field="passed"):
        path = root / relative
        value = read(path)
        result = None if value is None else value.get(field)
        if value and (value.get("state") == "failed" or value.get("passed") is False):
            result = False
        elif field in ("complete", "evidence_complete") and result is False:
            result = None
        record(gate, name, result, path)
        return value

    bases_path = root / "base_checkpoint_verification.json"
    bases = read(bases_path)
    for model in ("pi05_base", "pi05_libero"):
        record("A_readiness", f"{model} verified original checkpoint", None if bases is None else
               bases.get(model, {}).get("verified") is True, bases_path)
    for task in TASKS:
        artifact("A_readiness", f"{task} official preparation", f"tasks/{task}/preparation_complete.json")
        plan = read(root / "tasks" / task / "protocol.json")
        record("A_readiness", f"{task} validation schedule", None if plan is None else
               plan.get("validation_iterations") == [25, 50, 100] and plan.get("online_validation") is True)
        contract_path = root / "methods/rldt" / task / "input_contract.json"
        contract = read(contract_path)
        record("A_readiness", f"{task} runtime and frozen inputs unchanged", None if contract is None else
               all(Path(path).is_file() and file_hash(path) == expected for path, expected in contract.items()), contract_path)
        artifact("B_fidelity", f"{task} real optimizer and frozen backbone", f"smoke/{task}/verification.json")
    for relative in ("gate_b_actual/gate_summary.json", "moving_mask_alignment_gate.json",
                     "grid_adapter_15condition_technical_smoke/gate_summary.json",
                     "layer_grid_technical_smoke/gate_summary.json", "projection_swap_technical_smoke/gate_summary.json"):
        artifact("B_fidelity", relative, relative)
    fidelity = artifact("B_fidelity", "Saved RLDT checkpoints canonical sampler fidelity",
                        "trained_checkpoint_sampler_fidelity.json")
    fidelity_valid = None
    fidelity_error = None
    if fidelity and fidelity.get("passed") is True:
        try:
            from rl_vla.diagnostics.compiler_profile import load_profile
            from rl_vla.diagnostics.trained_fidelity import (
                task_fidelity_contract,
                verify_task_fidelity,
            )
            profile = load_profile(root / "compiler_profiles/native_trained_v2/manifest.json")
            if (fidelity.get("complete") is not True or fidelity.get("completed_checkpoints") != 9
                    or fidelity.get("expected_checkpoints") != 9
                    or set(fidelity.get("checkpoints", {})) != set(TASKS)
                    or fidelity.get("autotune_sha256") != profile["profile_sha256"]):
                raise ValueError("Numerical summary does not cover all nine requested checkpoints")
            for task in TASKS:
                contract = task_fidelity_contract(root, task, profile)
                if contract is None:
                    raise ValueError(f"Numerical task proof remains incomplete: {task}")
                verify_task_fidelity(contract)
                ready = read(root / "trained_sampler_fidelity/task_readiness" / f"{task}.json")
                rows = fidelity["checkpoints"][task]
                if set(rows) != {str(i) for i in ITERATIONS}:
                    raise ValueError("Missing requested numerical checkpoint")
                for iteration, row in rows.items():
                    if row.get("passed") is not True or any(row.get(key) != ready["proofs"][iteration][key] for key in ("path", "sha256")):
                        raise ValueError("Aggregate numerical proof differs from immutable task readiness")
            fidelity_valid = True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            fidelity_valid = False
            fidelity_error = str(exc)
    record("B_fidelity", "All nine numerical proofs bind independent native oracles and unchanged sources",
           fidelity_valid, root / "trained_sampler_fidelity/task_readiness")
    if fidelity_error:
        gates["B_fidelity"]["checks"][-1]["error"] = fidelity_error
    baseline_fidelity = artifact("B_fidelity", "Untouched baseline native compiler fidelity",
                                 "baseline_sampler_fidelity.json")
    baseline_valid = None
    baseline_error = None
    if baseline_fidelity and baseline_fidelity.get("passed") is True:
        try:
            from rl_vla.diagnostics.compiler_dispatch import load_profile as load_policy_profile
            from rl_vla.diagnostics.trained_fidelity import (
                policy_fidelity_contract,
                verify_task_fidelity,
            )
            profile = load_policy_profile(root / "compiler_profiles/native_baseline_v1/manifest.json")
            if (baseline_fidelity.get("complete") is not True or baseline_fidelity.get("completed_tasks") != 3
                    or baseline_fidelity.get("expected_tasks") != 3
                    or set(baseline_fidelity.get("tasks", {})) != set(TASKS)
                    or baseline_fidelity.get("autotune_sha256") != profile["profile_sha256"]):
                raise ValueError("Baseline numerical summary does not cover all three tasks")
            for task in TASKS:
                contract = policy_fidelity_contract(root, task, profile)
                if contract is None:
                    raise ValueError(f"Baseline numerical proof remains incomplete: {task}")
                verify_task_fidelity(contract)
                ready = read(root / "baseline_sampler_fidelity/task_readiness" / f"{task}.json")
                row = baseline_fidelity["tasks"][task]
                if row.get("passed") is not True or any(row.get(key) != ready["proof"][key] for key in ("path", "sha256")):
                    raise ValueError("Baseline aggregate differs from immutable task readiness")
            baseline_valid = True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            baseline_valid = False
            baseline_error = str(exc)
    record("B_fidelity", "All three baseline proofs bind independent native oracles and unchanged sources",
           baseline_valid, root / "baseline_sampler_fidelity/task_readiness")
    if baseline_error:
        gates["B_fidelity"]["checks"][-1]["error"] = baseline_error

    analysis = read(root / "analysis.json")
    from rl_vla.diagnostics.causal_support_overlay_v1 import adoption
    support_adopted=adoption(root)["active"]
    for task in TASKS:
        method = root / "methods/rldt" / task
        summary = read(method / "training/summary.json")
        metrics_path = method / "training/metrics.jsonl"
        metrics = []
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines(keepends=True):
                if line.endswith("\n"):
                    metrics.append(json.loads(line))
        complete = len(metrics) == 100 and summary is not None
        record("C_training", f"{task} uninterrupted100 iterations", None if not complete else
               [r["iterations"] for r in metrics] == list(range(1, 101)))
        if complete:
            count = metrics[-1]
            record("C_training", f"{task} native counters", (count["decisions"], count["actor_updates"], count["critic_updates"]) == (3200, 392, 6400))
        final = method / "training/final"
        record("C_training", f"{task} final snapshot retained", None if not complete else
               all((final / name).is_file() and (final / name).stat().st_size > 0
                   for name in ("expert.npz", "learner.npz", "rldt_pi05.json")), final)
        artifact("C_training", f"{task} checkpoint and initialization audit", f"methods/rldt/{task}/checkpoint_audit.json")
        selection = read(method / "selection.json")
        if selection is None:
            record("C_training", f"{task} validation-only selection", None)
        else:
            reports = {i: read(method / f"validation_{i:06d}.json") for i in ITERATIONS}
            valid = all(x is not None and len(x.get("episodes", [])) == 50 for x in reports.values())
            if valid:
                counts = {str(i): sum(e["success"] for e in x["episodes"]) for i, x in reports.items()}
                expected = min(ITERATIONS, key=lambda i: (-counts[str(i)], i))
                valid = bool(selection.get("locked")) and selection.get("test_results_seen") is False and selection.get("iteration") == expected and selection.get("validation_counts") == counts
            record("C_training", f"{task} validation-only selection", valid, method / "selection.json")
        status = read(method / "status.json")
        done = status is not None and status.get("stage") == "action_evaluations_complete" and status.get("state") == "completed"
        record("D_policy_comparison", f"{task} four native policies test and retention", True if done else False if status and status.get("state") == "failed" else None, method / "status.json")
        for policy in POLICIES:
            for test in ("localization_probe", "text_image_attention", "expert_image_attention", "causal_interventions", "unseen_layouts"):
                rows = [] if analysis is None else [row for row in analysis.get("diagnostics", [])
                    if row["task"] == task and row["policy"] == policy and row["test"] == test]
                row = rows[0] if len(rows) == 1 else None
                expected_na = policy == "pi05_base" and test in ("expert_image_attention", "causal_interventions", "unseen_layouts")
                value = None if row is None or row.get("status") == "pending" else row.get("status") == ("not_applicable" if expected_na else "complete")
                record("E_five_tests", f"{task}/{policy}/{test}", value, root / row["artifact"] if row and row.get("artifact") else None)
        artifact("E_five_tests", f"{task} unseen-layout kinematics", f"diagnostic_layouts/{task}/kinematic_audit.json")
        if task=="object_1" and support_adopted:
            artifact("E_five_tests", f"{task} resolved original cases (native and structural N/A)", "compiler_profile_integration/causal_structural_na_v1/execution_coverage.json")
        else:
            artifact("E_five_tests", f"{task} closed-loop pairing and counts", f"interventions/{task}/queue_complete.json")
        artifact("E_five_tests", f"{task} matched trace analyses", f"trace_analysis/{task}/summary.json", field="complete")
    restores = artifact("C_training", "Native learner serialization restored", "learner_restore_audit/summary.json")
    if restores and restores.get("passed"):
        rows = restores.get("checkpoints", [])
        expected = {(task, iteration) for task in TASKS for iteration in ITERATIONS}
        valid = restores.get("complete") is True and len(rows) == 9 and {
            (row.get("task"), row.get("iteration")) for row in rows} == expected
        for row in rows:
            path = Path(row["path"])
            result = read(path)
            valid &= bool(result and row.get("passed") is True and result.get("passed") is True
                          and result.get("complete") is True and file_hash(path) == row.get("sha256")
                          and (result.get("task"), result.get("iteration")) == (row["task"], row["iteration"])
                          and result.get("native_restore_invoked") is True
                          and result.get("files") == row.get("expected_checkpoint_files"))
            audit = read(root / "methods/rldt" / row["task"] / "checkpoint_audit.json")
            checkpoint = root / "methods/rldt" / row["task"] / "training" / f"iteration_{row['iteration']:06d}"
            valid &= bool(audit and all(row.get("expected_checkpoint_files", {}).get(str(checkpoint / name)) ==
                          audit.get("checkpoints", {}).get(str(row["iteration"]), {}).get("files", {}).get(name)
                          for name in ("expert.npz", "learner.npz", "rldt_pi05.json")))
        record("C_training", "All nine restore artifacts match audited checkpoints", bool(valid))
    for name, relative in (
        ("Full matched-input trace grid", "attention_grid_queue_complete.json"),
        ("Actual policy rollout traces", "rollout_traces/status.json"),
        ("Rollout and paired qualitative panels", "rollout_trace_analysis/summary.json"),
    ):
        artifact("E_five_tests", name, relative, field="complete")
    if support_adopted:
        contact_path=root/"contact_analysis_structural_na_v1/summary.json"
        contacts=read(contact_path)
        verified=None;error=None
        if contacts:
            try:
                from rl_vla.diagnostics.contact_support_overlay_v1 import verify_overlay
                contacts=verify_overlay(root)
                verified=True if contacts and contacts.get("complete") else None
            except (OSError,ValueError,KeyError,TypeError) as exc:
                verified=False;error=str(exc)
        record("E_five_tests","Exact contact coverage:4200 native main/OOD plus1080 resolved causal cases",verified,contact_path)
        if error:gates["E_five_tests"]["checks"][-1]["error"]=error
    else:
        contacts=artifact("E_five_tests","Full native contact evidence coverage","contact_analysis/summary.json",field="complete")
        if contacts and contacts.get("complete"):
            record("E_five_tests","Exact contact coverage of all5,280 episodes",contacts.get("episodes_processed")==5280 and contacts.get("reports_processed")==192 and contacts.get("automatic_labels_reviewed") is False)
    from rl_vla.diagnostics.causal_support_overlay_v1 import operational_snapshot
    from rl_vla.diagnostics.ood_completion_support_v2 import completion as ood_completion
    for row in operational_snapshot(root)["rows"]:
        record("E_five_tests", f"{row['task']} {row['domain']} queue operational health",
               False if row["state"]=="failed" else (True if row["state"]=="complete" else None),
               Path(row["status_path"]))
        if row["state"]=="failed":
            gates["E_five_tests"]["checks"][-1]["error"]=row["reason"]
    from rl_vla.diagnostics.ood_figures import verify as verify_ood_figures
    for task in TASKS:
        dispatch = ood_completion(root, task)
        record("E_five_tests", f"{task}: exact16 OOD results and dispatch/cache provenance", dispatch["passed"],
               root / "ood" / task / "queue_complete.json")
        if dispatch["passed"] is False:
            gates["E_five_tests"]["checks"][-1]["error"] = dispatch["reason"]
        plots = verify_ood_figures(root, task, full_hash=True)
        for gate in ("E_five_tests", "F_report"):
            record(gate, f"{task}: OOD figures for16 native cells and full reviewed failures", plots["passed"],
                   root / "figures/ood" / task / "summary.json")
            if plots["passed"] is False:
                gates[gate]["checks"][-1]["error"] = plots["reason"]
    from rl_vla.diagnostics.causal_support_overlay_v1 import execution as support_execution
    from rl_vla.diagnostics.causal_support_overlay_v1 import figure_coverage, readable_coverage
    figures=figure_coverage(root,full_hash=True)
    for gate in ("E_five_tests","F_report"):
        record(gate,"All27 required figure/control/vector artifact cells validated",figures["passed"],root/"figure_artifacts/summary.json")
        if figures["passed"] is False:gates[gate]["checks"][-1]["errors"]=[row for row in figures["cells"] if row["status"]=="failed"]
    causal_figures=readable_coverage(root,full_hash=True)
    for gate in ("E_five_tests","F_report"):
        record(gate,"All12 readable causal figure derivatives validated",causal_figures["passed"],root/"figures/causal/summary.json")
        if causal_figures["passed"] is False:gates[gate]["checks"][-1]["errors"]=[row for row in causal_figures["cells"] if row["status"]=="failed"]
    support_status=support_execution(root,full_hash=True)
    if support_status["active"]:
        for gate in ("E_five_tests","F_report"):
            record(gate,"Accepted exact360-case structural-support execution coverage",support_status["passed"],root/"compiler_profile_integration/causal_structural_na_v1/execution_coverage.json")
            if support_status["passed"] is False:gates[gate]["checks"][-1]["error"]=support_status.get("reason")
    for name in ("report.md", "runbook.md", "tables/success.csv", "tables/paired_success.csv", "tables/diagnostic_status.csv", "resource_cost.json"):
        path = root / name
        record("F_report", name, True if path.exists() and path.stat().st_size > 0 else None, path)
    review = artifact("F_report", "Evidence-based scientific interpretation reviewed", "scientific_review.json")
    digest = artifact("F_report", "Complete evidence digest", "evidence_digest.json", field="evidence_complete")
    coverage = digest.get("failure_review_coverage", {}) if digest else {}
    record("F_report", "All main and OOD failures explicitly reviewed", None if not coverage.get("complete") else
           coverage.get("expected_main_reports") == 36 and coverage.get("expected_ood_reports") == 48
           and not coverage.get("pending_reports"), root / "evidence_digest.json")
    if review and review.get("passed"):
        hashes = review.get("reviewed_artifacts", {})
        valid = bool(hashes) and all((root / path).is_file() and file_hash(root / path) == expected for path, expected in hashes.items())
        record("F_report", "Reviewed scientific evidence unchanged", valid)
    for gate in gates.values():
        gate["passed"] = not gate["pending"] and not gate["failed"]
    # A scientific report cannot be completed while its preceding evidence gates are open.
    gates["F_report"]["passed"] &= all(gates[g]["passed"] for g in gates if g != "F_report")
    return {"updated_utc": datetime.now(UTC).isoformat(), "complete": all(g["passed"] for g in gates.values()),
            "gates": gates, "scope": "Handoff acceptance audit. Pending evidence is never counted as a result; raw-base action incompatibility is explicit."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    value = check(args.run)
    output = args.run.resolve() / "acceptance_gates.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"complete": value["complete"], "gates": {k: {"passed": v["passed"], "pending": len(v["pending"]), "failed": v["failed"]} for k, v in value["gates"].items()}}, indent=2))
