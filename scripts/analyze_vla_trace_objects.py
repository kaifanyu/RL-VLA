"""Build reproducible tables, figures and an honest report from campaign artifacts."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from rl_vla.diagnostics.causal_support_overlay_v1 import cell_ready
from rl_vla.diagnostics.causal_support_overlay_v1 import figure_coverage as verify_coverage
from rl_vla.diagnostics.metrics import paired_success, wilson

TASKS = ("object_0", "object_1", "object_2")
POLICIES = ("pi05_base", "pi05_libero", "rldt_25", "rldt_50", "rldt_100")
ITERATIONS = (25, 50, 100)
BASE_NA = "N/A — incompatible action contract; raw pi05_base has no validated LIBERO action adapter"
FAILURE_LABEL_RULES = {
    "wrong object": "Reviewed non-target grasp/selection corroborated by native object-ID contact/lift evidence; mixed target/non-target sequences remain ambiguous.",
    "correct target but failed grasp": "Reviewed target-directed interaction with sufficiently clear target identity/choice, but acquisition fails; insufficient corroboration remains ambiguous rather than forcing this label.",
    "correct grasp but placement failure": "Reviewed, corroborated target grasp followed by a false final native containment goal, including drops, rim placements or retained grasps.",
    "timeout": "Explicitly reviewed timeout as the primary failure description when supported; reaching the280-step cap alone never automatically assigns this label.",
    "other/ambiguous": "Mixed sequences, strict-contact misses, unclear identity or insufficient evidence; absence of a strict candidate is not proof that no grasp occurred.",
}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def acceptance_status(gate):
    """Distinguish an unfinished producer from an explicit failed check."""
    if gate.get("failed") or any(check.get("passed") is False for check in gate.get("checks", [])):
        return "failed"
    return "passed" if gate.get("passed") is True else "pending"


def read_jsonl(path):
    result = []
    if not path.exists():
        return result
    for line in path.read_text().splitlines(keepends=True):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if line.endswith("\n"):
                raise
            break
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix(".tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    path.with_suffix(".tmp").replace(path)


def reviewed_labels(root):
    """Join only immutable reviewed evidence whose report/video/sheet hashes match."""
    labels={};verified={}
    for path in sorted((root/"video_reviews/reviewed_labels").glob("*.json")):
        for row in json.loads(path.read_text())["records"]:
            if not row.get("reviewed"):continue
            for name in ("report","video","sheet","evidence_manifest"):
                source=Path(row[name]).resolve();expected=row[f"{name}_sha256"]
                if source not in verified:
                    checksum=hashlib.sha256()
                    with source.open("rb") as stream:
                        for block in iter(lambda:stream.read(1024*1024),b""):checksum.update(block)
                    verified[source]=checksum.hexdigest()
                if verified[source]!=expected:raise ValueError(f"Reviewed evidence source hash changed: {source}")
            key=(row["task"],str(Path(row["report"]).resolve()),row["state_id"],row["seed"])
            if key in labels and labels[key]["failure_type"]!=row["failure_type"]:
                raise ValueError("Conflicting immutable review labels require explicit resolution")
            labels[key]=dict(row,label_source=str(path.resolve()))
    return labels


def evaluation_path(root, task, policy, split):
    standard = root / "evaluations" / task / policy / split / "summary.json"
    if standard.exists():
        return standard
    method = root / "methods/rldt" / task
    if policy == "pi05_libero":
        return method / f"baseline_{split}.json"
    if policy.startswith("rldt_"):
        iteration = int(policy.split("_")[1])
        if split == "validation":
            return method / f"validation_{iteration:06d}.json"
        full = method / f"checkpoint_{iteration:06d}_{split}.json"
        if full.exists():
            return full
        selected = read_json(method / "selection.json")
        if selected and selected["iteration"] == iteration:
            return method / f"selected_{split}.json"
        return full
    return standard


def checked_episodes(report, path, definition, split):
    episodes = report["episodes"]
    if not isinstance(episodes, list):
        episode_path = path.parent / "episodes.jsonl"
        episodes = read_jsonl(episode_path)
    expected = definition.get(f"{split}_state_ids") if definition else None
    ids = [e["reset"]["initial_state_id"] for e in episodes]
    if not episodes or len(set(ids)) != len(ids):
        raise ValueError(f"Missing or duplicated evaluation states in {path}")
    if expected is not None and ids != expected:
        raise ValueError(f"Evaluation states differ from frozen protocol: {path}")
    if [e["seed"] for e in episodes] != list(range(100000, 100000 + len(episodes))):
        raise ValueError(f"Evaluation noise seeds differ from predeclared protocol: {path}")
    if any(type(e["success"]) is not bool for e in episodes):
        raise ValueError(f"Nonboolean success in {path}")
    return episodes


def audit_selection(method):
    selection = read_json(method / "selection.json")
    if selection is None:
        return {"status": "pending", "reason": "Checkpoint selection is not yet locked"}
    counts = {}
    for iteration in ITERATIONS:
        value = read_json(method / f"validation_{iteration:06d}.json")
        if value is None or len(value["episodes"]) != 50:
            raise ValueError(f"Selection exists without 50 validation episodes at {method}, iteration {iteration}")
        counts[iteration] = sum(e["success"] for e in value["episodes"])
    expected = min(counts, key=lambda i: (-counts[i], i))
    if selection["iteration"] != expected:
        raise ValueError(f"Selection does not follow count/earliest rule: {method}")
    return {"status": "locked", "iteration": expected, "validation_counts": counts,
            "rule": "Highest validation success count at 25/50/100; ties earliest"}


def collect(root, *, reviews=None):
    success, paired, episodes_flat, learning, training, selection_audits = [], [], [], [], {}, {}
    failures = []
    reviews=reviewed_labels(root) if reviews is None else reviews
    for task in TASKS:
        method = root / "methods/rldt" / task
        protocol = read_json(root / "tasks" / task / "protocol.json")
        definition = protocol["tasks"][task] if protocol else {}
        selection_audits[task] = audit_selection(method)
        metrics = read_jsonl(method / "training/metrics.jsonl")
        training[task] = {"completed_iterations": metrics[-1]["iterations"] if metrics else 0,
                          "summary": read_json(method / "training/summary.json"),
                          "status": read_json(method / "status.json")}
        by_iteration = {row["iterations"]: row for row in metrics}
        for row in metrics:
            if not all(np.isfinite(v) for v in row.get("metrics", {}).values()):
                raise ValueError(f"Nonfinite training metric: {task} iteration {row['iterations']}")
        reports = {}
        for split in ("validation", "test", "source"):
            for policy in POLICIES:
                row = {"task": task, "split": split, "policy": policy}
                path = evaluation_path(root, task, policy, split)
                report = read_json(path)
                if policy == "pi05_base":
                    row.update(status="not_applicable", reason=BASE_NA)
                elif report is None:
                    row.update(status="pending", reason="Evaluation has not produced a completed report")
                else:
                    if split != "validation" and selection_audits[task]["status"] != "locked":
                        raise ValueError(f"Final test/retention report exists before locked selection: {path}")
                    episodes = checked_episodes(report, path, definition, split)
                    reports[split, policy] = episodes
                    count = len(episodes)
                    successes = sum(e["success"] for e in episodes)
                    low, high = wilson(successes, count)
                    row.update(status="complete", successes=successes, episodes=count,
                               success_rate=successes / count, wilson95_low=low, wilson95_high=high,
                               artifact=str(path.relative_to(root)),
                               comparison_scope="validation" if split == "validation" else
                               "baseline" if policy == "pi05_libero" else
                               "selected_confirmatory" if int(policy.split("_")[1]) == selection_audits[task]["iteration"]
                               else "exploratory_after_selection")
                    for e in episodes:
                        reviewed=reviews.get((task,str(path.resolve()),e["reset"]["initial_state_id"],e["seed"]))
                        if reviewed and (reviewed["success"]!=e["success"] or reviewed["video"]!=e.get("video")):
                            raise ValueError("Reviewed episode does not match its evaluation identity")
                        flat = {"task": task, "split": split, "policy": policy,
                                "initial_state_id": e["reset"]["initial_state_id"], "seed": e["seed"],
                                "bddl_sha256": definition.get("source" if split == "source" else "pro", {}).get("bddl", {}).get("sha256"),
                                "success": e["success"], "primitive_steps": e.get("primitive_steps"),
                                "video": e.get("video"), "trajectory": e.get("trajectory"),
                                "grasped_object_id": e.get("grasped_object_id"),
                                "failure_type": e.get("failure_type", "success" if e["success"] else "other/ambiguous — not reviewed")}
                        flat.update(report=str(path.resolve()),reviewed=bool(reviewed),review_source=reviewed["label_source"] if reviewed else None)
                        if reviewed:
                            flat.update(failure_type=reviewed["failure_type"],grasped_object_id=reviewed.get("grasped_object_id"),
                                        reviewer=reviewed["reviewer"],review_notes=reviewed["notes"],review_sheet=reviewed["sheet"])
                        episodes_flat.append(flat)
                        if not e["success"]:
                            failures.append(flat)
                    if split == "validation":
                        iteration = 0 if policy == "pi05_libero" else int(policy.split("_")[1])
                        metric = by_iteration.get(iteration, {})
                        learning.append(dict(row, iteration=iteration,
                            primitive_steps=metric.get("primitive_steps", 0 if iteration == 0 else None),
                            actor_updates=metric.get("actor_updates", 0 if iteration == 0 else None),
                            critic_updates=metric.get("critic_updates", 0 if iteration == 0 else None)))
                success.append(row)
            before = reports.get((split, "pi05_libero"))
            if before:
                for policy in POLICIES[2:]:
                    after = reports.get((split, policy))
                    if after:
                        paired.append(dict(task=task, split=split, policy=policy,
                                           **paired_success(before, after)))
    return {"success": success, "paired": paired, "episodes": episodes_flat, "failures": failures,
            "learning": learning, "training": training, "selection": selection_audits}


def figures(root, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = root / "figures/task_comparison"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if data["learning"]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for task, ax in zip(TASKS, axes, strict=True):
            rows = sorted((r for r in data["learning"] if r["task"] == task and r["primitive_steps"] is not None), key=lambda r: r["iteration"])
            if rows:
                y = np.array([r["success_rate"] for r in rows])
                error = np.array([[r["wilson95_low"], r["wilson95_high"]] for r in rows]).T
                ax.errorbar([r["primitive_steps"] for r in rows], y, yerr=np.abs(error-y), fmt="o-", capsize=3)
                for row in rows:
                    ax.annotate(str(row["iteration"]), (row["primitive_steps"], row["success_rate"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
            ax.set(title=task, xlabel="Actual primitive training steps", ylabel="Validation success", ylim=(-.03, 1.03))
            ax.grid(alpha=.2)
        fig.suptitle("Validation only · Wilson 95% intervals · labels show iterations")
        fig.savefig(figure_dir / "validation_by_steps.png", dpi=170)
        plt.close(fig)
    complete = [r for r in data["success"] if r["status"] == "complete" and r["split"] in ("test", "source")]
    if complete:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
        for split, row_axes in zip(("test", "source"), axes, strict=True):
            for task, ax in zip(TASKS, row_axes, strict=True):
                rows = [r for r in complete if r["task"] == task and r["split"] == split]
                for row in rows:
                    x = POLICIES.index(row["policy"])-1
                    ax.errorbar(x, row["success_rate"], yerr=[[row["success_rate"]-row["wilson95_low"]], [row["wilson95_high"]-row["success_rate"]]], fmt="o", capsize=4)
                    ax.annotate(f"{row['successes']}/{row['episodes']}", (x, row["success_rate"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
                ax.set(title=f"{task} · {'PRO swap test' if split == 'test' else 'original retention'}", ylim=(-.03, 1.08), xticks=range(4), xticklabels=["LIBERO", "25", "50", "100"], ylabel="Success fraction")
                ax.grid(alpha=.2)
        fig.suptitle("One seed, paired frozen states · raw base action contract: N/A")
        fig.savefig(figure_dir / "success_and_retention.png", dpi=170)
        plt.close(fig)


def diagnostic_inventory(root):
    from rl_vla.diagnostics.causal_support_overlay_v1 import operational_snapshot
    from rl_vla.diagnostics.causal_support_overlay_v1 import (
        readable_coverage as causal_figure_coverage,
    )
    from rl_vla.diagnostics.ood_completion_support_v2 import completion as ood_completion
    from rl_vla.diagnostics.ood_figures import verify as ood_figures
    ood_dispatch = {task: ood_completion(root, task) for task in TASKS}
    ood_plot_status = {task: ood_figures(root, task) for task in TASKS}
    causal_plots = {(row["task"], row["policy"]): row for row in causal_figure_coverage(root)["cells"]}
    operational = {(row["domain"],row["task"]):row for row in operational_snapshot(root)["rows"]}
    rows = []
    for task in TASKS:
        for policy in POLICIES:
            for test, folder, filename in (("localization_probe", "probes", "metrics.json"),
                                           ("text_image_attention", "traces", "text_summary.json"),
                                           ("expert_image_attention", "traces", "expert_summary.json"),
                                           ("causal_interventions", "causal_analysis", "summary.json"),
                                           ("unseen_layouts", "ood", "summary.json")):
                path = root / folder / task / policy / filename
                if policy.startswith("rldt_") and not path.exists():
                    padded=f"rldt_{int(policy.split('_')[1]):06d}"
                    path=root/folder/task/padded/filename
                result = read_json(path)
                if test=="causal_interventions" and task=="object_1" and policy!="pi05_base":
                    from rl_vla.diagnostics.causal_support_overlay_v1 import adoption
                    if adoption(root)["active"]:
                        padded=f"rldt_{int(policy.split('_')[1]):06d}" if policy.startswith("rldt_") else policy
                        path=root/"causal_analysis_structural_na_v1"/task/padded/filename
                        result=read_json(path)
                shared_reason=""
                if test=="localization_probe" and policy.startswith("rldt_"):
                    equality=root/"traces"/task/f"rldt_{int(policy.split('_')[1]):06d}"/"prefix_equality.json"
                    audit=read_json(equality)
                    if audit and audit.get("passed") and audit.get("complete"):
                        path=root/"probes"/task/"pi05_libero"/"metrics.json"
                        result=read_json(path)
                        shared_reason="Shared frozen LIBERO features/probe; exact checkpoint prefix equality audited"
                status = "complete" if result is not None and result.get("complete",result.get("completed")) is True else "pending"
                reason = "No completed diagnostic summary" if status == "pending" else ""
                if status=="complete" and shared_reason:reason=shared_reason
                if status=="complete" and test=="causal_interventions" and result.get("structural_na_cases",0):
                    reason="All planned cases resolved; strict original controls with insufficient support are explicit structural N/A, so affected whole-condition primary effects are unavailable"
                if status=="complete" and test=="localization_probe":
                    model="pi05_libero" if policy.startswith("rldt_") else policy
                    counterfactual=read_json(root/"probes_counterfactual"/task/model/"metrics.json")
                    if not counterfactual or not counterfactual.get("complete"):
                        status="pending";reason="Original-prompt probe complete; required fixed-probe counterfactual prompt scoring pending"
                    elif not all(cell_ready(root,domain,task,model) for domain in ("probes","probes_counterfactual")):
                        status="pending";reason="Probe metrics exist; required confusion/ROC/prediction figures and source maps await independent artifact validation"
                if status=="complete" and test=="expert_image_attention":
                    padded=f"rldt_{int(policy.split('_')[1]):06d}" if policy.startswith("rldt_") else policy
                    rollout=read_json(root/"rollout_trace_analysis"/task/padded/"summary.json")
                    cases=read_json(root/"rollout_trace_analysis/fixed_input_qualitative"/task/"summary.json")
                    if not rollout or not rollout.get("complete") or not cases or not cases.get("complete"):
                        status="pending";reason="Fixed-input expert maps complete; actual rollout-phase and reviewed/outcome-labeled qualitative panels pending"
                if status=="complete" and test in ("text_image_attention","expert_image_attention","causal_interventions") and not cell_ready(root,"supplemental_analysis",task,"all_policies"):
                    status="pending";reason="Required token controls, modality/layer curves and action-vector artifacts await independent validation"
                if status=="complete" and test=="causal_interventions" and not cell_ready(root,"causal_analysis",task,policy):
                    status="pending";reason="Causal metrics exist; required sensitivity figures and compact source tables await independent validation"
                if status=="complete" and test=="causal_interventions" and causal_plots.get((task,policy),{}).get("status")!="complete":
                    status="pending";reason="Required readable causal figure derivative and exact plotting-data bindings pending"
                if status=="complete" and test=="unseen_layouts" and not ood_dispatch[task]["complete"]:
                    status="pending";reason=ood_dispatch[task]["reason"]
                if status=="complete" and test=="unseen_layouts" and not ood_plot_status[task]["complete"]:
                    status="pending";reason=ood_plot_status[task]["reason"]
                domain={"causal_interventions":"causal","unseen_layouts":"ood"}.get(test)
                if status!="complete" and domain and operational[domain,task]["state"]=="failed":
                    status="failed";reason=operational[domain,task]["reason"]
                if policy == "pi05_base" and test in ("expert_image_attention", "causal_interventions", "unseen_layouts"):
                    status, reason = "not_applicable", BASE_NA
                rows.append({"task": task, "policy": policy, "test": test, "status": status,
                             "reason": reason, "artifact": str(path.relative_to(root)) if result else ""})
    return rows


def resource_cost(root):
    logical_bytes=0;files=0
    for directory,_,names in os.walk(root):
        for name in names:
            path=Path(directory)/name
            try:
                if not path.is_symlink():logical_bytes+=path.stat().st_size;files+=1
            except FileNotFoundError:pass
    devices={}
    telemetry=root/"gpu_telemetry.csv"
    if telemetry.exists():
        for fields in csv.reader(telemetry.read_text().splitlines()):
            if len(fields)!=6:continue
            try:
                when=datetime.strptime(fields[0].strip(),"%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=ZoneInfo("UTC"))
                index=int(fields[1]);util=float(fields[3].split()[0])/100
                memory=float(fields[4].split()[0]);watts=float(fields[5].split()[0])
            except ValueError:continue
            row=devices.setdefault(index,{"peak_memory_mib":0.,"utilization_weighted_gpu_hours":0.,"device_energy_kwh":0.,"samples":0,"previous":None})
            if row["previous"]:
                previous_time,previous_util,previous_watts=row["previous"]
                dt=(when-previous_time).total_seconds()
                if 0<dt<120:
                    row["utilization_weighted_gpu_hours"]+=dt*(util+previous_util)/2/3600
                    row["device_energy_kwh"]+=dt*(watts+previous_watts)/2/3600000
            row["previous"]=(when,util,watts);row["peak_memory_mib"]=max(row["peak_memory_mib"],memory);row["samples"]+=1
    for row in devices.values():row.pop("previous")
    return {"run_local_file_bytes_excluding_symlink_targets":logical_bytes,"file_count":files,"devices":devices,
            "scope":"Telemetry estimates cover all visible GPU usage and idle power during monitoring, not training-only billing. Logical file sizes are not physical disk allocation."}


def ood_review_rows(task, policy, variant, path, value, reviews):
    """Join reviewed OOD failures; native success and automatic proxies stay separate."""
    if value.get("task") != task or value.get("variant") != variant:
        raise ValueError("OOD report task/layout differs from its declared cell")
    episodes = checked_episodes(value, path, {"ood_state_ids": list(range(50))}, "ood")
    report_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    identity = {"task": task, "policy": policy, "variant": variant}
    allowed = {"wrong object", "correct target but failed grasp", "correct grasp but placement failure",
               "timeout", "other/ambiguous"}
    for episode in episodes:
        state, seed = episode["reset"]["initial_state_id"], episode["seed"]
        reviewed = reviews.get((task, str(path.resolve()), state, seed))
        if reviewed and (reviewed["success"] != episode["success"] or reviewed["video"] != episode.get("video")
                         or reviewed["report_sha256"] != report_sha):
            raise ValueError("Reviewed OOD episode differs from native report identity")
        if reviewed and not episode["success"] and reviewed["failure_type"] not in allowed:
            raise ValueError("Reviewed OOD failure has an unsupported taxonomy label")
        rows.append({**identity, "report": str(path.resolve()), "report_sha256": report_sha,
                     "initial_state_id": state, "seed": seed, "success": episode["success"],
                     "primitive_steps": episode.get("primitive_steps"), "video": episode.get("video"),
                     "trajectory": episode.get("trajectory"), "reviewed": bool(reviewed),
                     "failure_type": "success" if episode["success"] else reviewed["failure_type"] if reviewed else "other/ambiguous — not reviewed",
                     "grasped_object_id": reviewed.get("grasped_object_id") if reviewed else None,
                     "review_source": reviewed["label_source"] if reviewed else None,
                     "reviewer": reviewed["reviewer"] if reviewed else None,
                     "review_notes": reviewed["notes"] if reviewed else None,
                     "review_sheet": reviewed["sheet"] if reviewed else None,
                     "review_sheet_sha256": reviewed["sheet_sha256"] if reviewed else None,
                     "evidence_manifest": reviewed["evidence_manifest"] if reviewed else None,
                     "evidence_manifest_sha256": reviewed["evidence_manifest_sha256"] if reviewed else None})
    confirmed = sum(not row["success"] and row["reviewed"] and row["failure_type"] == "wrong object" for row in rows)
    unreviewed = sum(not row["success"] and not row["reviewed"] for row in rows)
    ambiguous = sum(not row["success"] and row["reviewed"] and row["failure_type"] == "other/ambiguous" for row in rows)
    low, high = wilson(confirmed, 50)
    summary = {**identity, "report": str(path.resolve()), "report_sha256": report_sha,
               "episodes": 50, "native_successes": sum(row["success"] for row in rows),
               "confirmed_wrong_object_cases_so_far": confirmed,
               "status": "pending_review" if unreviewed else "complete_review",
               "confirmed_fraction_all_episodes": None if unreviewed else confirmed / 50,
               "confirmed_wilson95_low": None if unreviewed else low,
               "confirmed_wilson95_high": None if unreviewed else high,
               "reviewed_ambiguous_failures": ambiguous, "unreviewed_failures": unreviewed,
               "reviewed_failures": sum(not row["success"] and row["reviewed"] for row in rows),
               "review_coverage_complete": not unreviewed,
               "interpretation": "Confirmed wrong-object failures among all 50 episodes; not exhaustive first-choice errors among eventual successes. Selected-frame AI review and native evidence; Wilson intervals exclude reviewer error."}
    categories = ["success" if row["success"] else row["failure_type"] if row["reviewed"] else "not reviewed" for row in rows]
    taxonomy = []
    for kind in sorted(set(categories)):
        count = categories.count(kind)
        low, high = wilson(count, 50)
        taxonomy.append({**identity, "failure_type": kind, "count": count, "episodes": 50,
                         "rate_all_episodes": count / 50, "wilson95_low": low, "wilson95_high": high})
    return rows, summary, taxonomy


def scientific_interpretation_sections(root):
    """Only the final narrative is adopted; an explicit marker promotes its lead."""
    path = Path(root) / "scientific_interpretation.md"
    if not path.exists():
        return "", ""
    content = path.read_text()
    marker = "<!-- report-lead-end -->"
    if marker not in content:
        return "", content
    if content.count(marker) != 1:
        raise ValueError("Final scientific narrative has multiple report-lead-end markers")
    lead, detail = content.split(marker)
    return lead.strip(), detail.strip()


def report(root, data, diagnostics, *, reviews=None):
    reviews=reviewed_labels(root) if reviews is None else reviews
    measured = [r for r in data["success"] if r["status"] == "complete"]
    pending = sum(r["status"] == "pending" for r in data["success"] + diagnostics)
    acceptance=read_json(root/"acceptance_gates.json")
    panel_summary=read_json(root/"rollout_trace_analysis/summary.json")
    accepted=bool(acceptance and acceptance.get("complete") and not pending and panel_summary and panel_summary.get("complete"))
    try:
        figure_coverage=verify_coverage(root)
    except (OSError,ValueError,KeyError,TypeError):
        figure_coverage=None
    accepted=accepted and bool(figure_coverage and figure_coverage.get("complete"))
    scientific_lead, scientific_detail = scientific_interpretation_sections(root)
    now = datetime.now(ZoneInfo("America/New_York")).isoformat()
    text = ["# RLDT Object 0/1/2 and VLA-Trace report", "", f"Updated: {now}", "",
            "**Status: " + ("complete; all acceptance gates and final scientific review passed." if accepted else "incomplete; results below include completed stages only. Final acceptance and scientific review are required.") + "**", "",
            "The study tests whether independently trained RLDT action experts improve object-conditioned behavior while the vision–language backbone remains frozen. Object 0 requests alphabet soup, Object 1 cream cheese, and Object 2 salad dressing. Each primary condition uses its official LIBERO-PRO Object Position/swap task. Object 1-hard is a separate custom condition and is excluded here.", "",
            "## Experiment and safeguards", "",
            "Three trajectories start independently from the original pi05_libero checkpoint. The planned budget per task is 100 iterations, 32 new policy decisions per iteration, up to five executed actions per decision, 64 critic updates per iteration, and four actor updates after two warmup iterations. The frozen banks contain 100 generated training states, 50 generated validation states, 50 untouched published PRO test states, and 50 original-layout retention states. Actual controlled steps are reported separately from validation and reset settling.", "",
            "Checkpoint selection considers only iterations 25, 50 and 100, chooses the highest validation success count, and breaks ties by earliest iteration. Extra checkpoint-wise test comparisons are exploratory after selection. Test success must not cause reselection. Raw pi05_base is a representation reference with a validated image/text interface; this campaign has no validated raw-base LIBERO action adapter. Its closed-loop and action-attention scores are **N/A — incompatible action contract**. The pinned pi05_libero configuration sets discrete_state_input=False, and both captured image/text prefixes contain instruction tokens without proprioceptive state tokens. Numeric state is retained in the native context and critic features; this pinned actor configuration adds neither state tokens nor a decoder state offset. Physical simulator/controller dynamics remain state-dependent. The [state-path clarification](compiler_profile_integration/native_state_path_clarification_v2/summary.json) records the source dependencies and bounded CPU transform checks, without claiming a new GPU state-invariance experiment. An [actual preprocessing audit](compiler_profile_integration/representation_input_contract_audit_v1.json) verified exact image arrays, camera masks, prompt IDs and prompt masks on all 150 frozen test observations; the [versioned interpretation correction](compiler_profile_integration/vlm_input_contract_correction_v2.json) preserves the earlier mistaken state-token assumption. Raw-base versus LIBERO is a checkpoint-level representation comparison under this matched image/text reference interface; the direct RLDT comparison is against its pi05_libero initialization.", "",
            "Diagnostic compiler replay uses separate baseline and trained-policy processes, each with its verified versioned startup autotune profile and isolated cache. Exact baseline contexts, noise and latent paths are persisted for matched trained-policy comparisons. The pre-diagnostic [fidelity amendment](compiler_profile_integration/fidelity_protocol_amendment_v1.json) requires exact agreement with independent uninstrumented native outputs obtained through verified original-cache hits for all requested checkpoints. Historical recorded-action differences are retained separately as compiler-history reproducibility evidence; they are not relabeled exact or tolerance-passed. Actual held-out rollout traces still require exact replay of their saved actions. Original training/evaluation runtime, outcomes and shared cache remain untouched. [Baseline numerical gate](baseline_sampler_fidelity.json), [requested-checkpoint numerical gate](trained_checkpoint_sampler_fidelity.json) and [profile provenance](compiler_profiles/release_notes.json) include unsuccessful technical attempts.", "",
            "Exact launched commands and isolated compiler environments are recorded in the [runbook](runbook.md#verified-compiler-dispatch-launch--2026-10-31-1325-utc) and [attention queue binding](diagnostic_policy_dispatch_launch/attention_acceptance_binding_v2.json). The real [staged integration acceptance](compiler_profile_integration/grid_staged_dispatch_acceptance.json) is a software-fidelity check on one predetermined case, excluded from research tables.", "",
            "## Execution status", "", "| Task | Completed iterations | Training summary | Selection |", "|---|---:|---|---|"]
    if scientific_lead:
        index = text.index("## Experiment and safeguards")
        text[index:index] = [scientific_lead, ""]
    if (root/"technical_attempts/gpu6_descriptor_profile_miss_v1/failure_adjudication.json").exists():
        index=text.index("## Execution status")
        text[index:index]=["An initial Object1 causal attempt on GPU6 was retained and excluded after its device bandwidth descriptor failed to match the autotune profile; saved expert and prefix kernels differed from the verified device. A local native no-op check alone did not detect that applicability failure. [Adjudication](technical_attempts/gpu6_descriptor_profile_miss_v1/failure_adjudication.json) and [read-only kernel audit](compiler_review/gpu6_private_cache_v1/kernel_comparison.json) preserve the evidence. Native main outcomes and retry cases/seeds/conditions are unchanged; replacement causal evidence requires a device-specific numerical proof.", ""]
    repaired_gate=read_json(root/"compiler_profile_integration/gpu6_independent_review_v1/acceptance.json")
    if repaired_gate and repaired_gate.get("passed") is True:
        index=text.index("## Execution status")
        text[index:index]=["The isolated GPU6 repair passed [320 recorded-input component checks](compiler_profile_integration/gpu6_fidelity_v1/summary.json), [36 concurrent native no-op and 36 explicit-noise checks](compiler_profile_integration/gpu6_concurrency_v1/summary.json), and exact replay of the first two recorded decisions for all four policies, including executed commands and simulator states. An [independent consumer review](compiler_profile_integration/gpu6_independent_review_v1/acceptance.json) verified the source/profile/device bindings and saved evidence. These are technical fidelity checks; they do not score a new performance cohort or establish equality for later untested rollout decisions.", ""]
    if (root/"diagnostic_policy_dispatch_launch/object_1_gpu6_v1.json").exists():
        index=text.index("## Execution status")
        text[index:index]=["The [repaired Object1 launch](diagnostic_policy_dispatch_launch/object_1_gpu6_v1.json) preserves the original ten cases, seeds and nine conditions; its exact command is in the [runbook](runbook.md#gpu6-device-specific-repair-launch--2026-10-31-1408-utc). A separate [full-rollout control audit](compiler_profile_integration/object1_repaired_none_rollouts_v1/summary.json) checks all saved actions, controlled-step simulator states, shared telemetry, reset and final outcomes as each none cohort completes. Pending or failed audits remain explicit.", ""]
        repaired_none=read_json(root/"compiler_profile_integration/object1_repaired_none_rollouts_v1/summary.json")
        if repaired_none and repaired_none.get("complete") is True and repaired_none.get("passed") is True:
            text[index:index]=["The repaired Object1 full-rollout audit subsequently passed all 40 paired episodes: entire normalized action chunks, executed commands, every recorded controlled-step simulator state, shared object telemetry, reset, primitive counts and final outcomes were exact. This validates the complete retained none-control trajectories; unretained per-step reward/done flags, raw images and PRNG keys are not claimed to have been directly compared.", ""]
    none_audit=read_json(root/"compiler_profile_integration/healthy_device_none_rollouts_v1/summary.json")
    if none_audit:
        passed_cells=[row for row in none_audit.get("cells",[]) if row.get("passed") is True]
        paired=sum(row["paired_episodes"] for row in passed_cells)
        index=text.index("## Execution status")
        none_status = "All eight cells passed." if none_audit.get("complete") is True and none_audit.get("passed") is True and len(passed_cells)==8 and paired==80 else "Unfinished cells remain pending, and any failed cell is retained in the audit."
        text[index:index]=[f"The separate [whole-rollout none-control audit](compiler_profile_integration/healthy_device_none_rollouts_v1/summary.json) verifies exact normalized actions, executed commands, reset metadata and every recorded controlled-step simulator state for {paired} paired episodes across {len(passed_cells)}/8 Object0/Object2 policy cells on GPUs4/7. {none_status} This comparison covers the first ten predefined test cases per policy; causal raw images and PRNG keys were not retained and are not claimed to have been directly compared.", ""]
        repaired=read_json(root/"compiler_profile_integration/object1_repaired_none_rollouts_v1/summary.json")
        if none_audit.get("complete") is True and none_audit.get("passed") is True and paired==80 and repaired and repaired.get("complete") is True and repaired.get("passed") is True:
            text[index:index]=["Together the two audits cover all 120 paired none-control episodes across the three tasks and four native policies, with exact full normalized/executed action sequences, controlled simulator states, reset and final outcomes. The repaired 40 Object1 episodes additionally compare every shared retained object-telemetry field. This is a completed control-fidelity check for the predefined first ten cases per policy; it does not establish exactness on untested devices or intervention trajectories.", ""]
    for task in TASKS:
        training, selection = data["training"][task], data["selection"][task]
        text.append(f"| {task} | {training['completed_iterations']}/100 | {'present' if training['summary'] else 'not yet present'} | {selection.get('iteration', selection['status'])} |")
    text += ["", "The prescribed 100 training states form a sampling bank; training resets sample with replacement. The following final counts come from completed episodes in `training/episodes.jsonl`. A final unfinished episode is excluded, so a zero count does not prove that state was never entered.", "",
             "| Task | Final completed episode resets | Distinct states in completed episodes | Repeated draws | Status |", "|---|---:|---:|---:|---|"]
    for row in data.get("training_reset_coverage", []):
        text.append(f"| {row['task']} | {row.get('completed_episode_resets', '—')} | {row.get('distinct_states_in_completed_episodes', '—')}/100 | {row.get('repeated_reset_draws', '—')} | {row['status']} |")
    text += ["", "[Final reset coverage](tables/training_state_coverage.csv) and [all 100 per-state reset counts](tables/training_state_visit_counts.csv) retain source hashes.", ""]
    if acceptance:
        text += ["", "| Acceptance gate | Status |", "|---|---|"]
        for name,value in acceptance["gates"].items():text.append(f"| {name} | {acceptance_status(value)} |")
        text += ["", "[Acceptance checks and remaining requirements](acceptance_gates.json). The final scientific-review gate requires manually assessed evidence and source hashes; complete raw tables alone do not satisfy it.", ""]
    from rl_vla.diagnostics.causal_support_overlay_v1 import operational_snapshot
    health=operational_snapshot(root)
    if health["failures"]:
        text += ["", "**Execution failures require resolution.** Completed valid evidence is preserved; remaining conditions are incomplete and support no final scientific claim. [Live operational alerts](operational_alerts.json).", "",
                 "| Domain | Task | Queue state | Error |", "|---|---|---|---|"]
        for row in health["failures"]:
            text.append(f"| {row['domain']} | {row['task']} | failed | {row['reason']} |")
    artifact_status=verify_coverage(root)
    if artifact_status:
        text += ["", f"[Independent figure/control/vector validation](figure_artifacts/summary.json): {artifact_status.get('completed_cells',0)}/27 artifact cells validated. Required PNG/PDF figures and compact source tables/maps must decode and retain their recorded hashes; a producer's early metrics-complete marker does not establish figure readiness.", ""]
        if (root / "causal_analysis_structural_na_v1/summary.json").exists():
            text += ["This selected coverage combines the original per-cell manifests with the [accepted Object1 successor figures](causal_analysis_structural_na_v1/summary.json). The original global marker remains preserved at its historical incomplete state; it is not a claim that the superseded Object1 attempt completed.", ""]
    restore_audit = read_json(root / "learner_restore_audit/summary.json")
    if restore_audit:
        text += ["", f"The separate CPU-native learner serialization audit has verified {restore_audit['completed_checkpoints']}/{restore_audit['required_checkpoints']} requested snapshots through actual `Learner.restore()`. It checks actor, moving reference, critics, optimizer states/counters and PRNG arrays against the saved archives while preserving the original expert and frozen backbone. This does not establish replay-buffer or environment-state resume. [Restore coverage and immutable checkpoint evidence](learner_restore_audit/summary.json).", ""]
    review_path=root/"figures/first_validation_case/object0_review.json"
    review=read_json(review_path)
    if review and review.get("reviewed"):
        text += ["", "The first predetermined Object 0 validation case (state 100, seed 100000) provides a reviewed motivating example: the frozen LIBERO policy grasps butter and deposits it in the basket while alphabet soup remains at the swapped location. The native soup goal is false. Native object-ID evidence records 86 both-finger contact steps and 81 consecutive contact-plus-lift steps for butter; the maximum lift is 0.289 m. This is **one validation episode**, not an aggregate or held-out test result. [Review evidence](figures/first_validation_case/object0_review.json).", "",
                 "![Reviewed first Object0 validation episode](figures/first_validation_case/object0_frame_review.png)",""]
    text += ["", "## Measured action performance", "",
             "Success intervals are Wilson 95% intervals. Paired bootstrap intervals resample complete state/noise pairs (20,000 draws; seed 20261031). Exact two-sided McNemar tests use only discordant pairs. These are episode-level, single-training-seed results; exploratory comparisons are unadjusted. Validation episodes are reused for checkpoint selection: validation gains and their descriptive intervals do not establish held-out improvement. Final test, original-layout retention and new-region results must be assessed separately.", "",
             "| Task | Split | Policy | Successes/episodes | Wilson 95% | Scope |", "|---|---|---|---:|---|---|"]
    for row in measured:
        text.append(f"| {row['task']} | {row['split']} | {row['policy']} | {row['successes']}/{row['episodes']} | {100*row['wilson95_low']:.1f}–{100*row['wilson95_high']:.1f}% | {row['comparison_scope']} |")
    if not measured:
        text.append("| — | — | — | No completed evaluation yet | — | — |")
    text += ["", "| Task | Split | Checkpoint | Change (pp) | Paired bootstrap 95% (pp) | Recovered | Regressed | Exact p |", "|---|---|---|---:|---|---:|---:|---:|"]
    for row in data["paired"]:
        lo, hi = row["paired_bootstrap95"]
        text.append(f"| {row['task']} | {row['split']} | {row['policy']} | {100*row['difference']:+.1f} | {100*lo:+.1f}–{100*hi:+.1f} | {row['recovered']} | {row['regressed']} | {row['mcnemar_exact_two_sided_p']:.4g} |")
    text += ["", "The full matrix, including every pending and incompatible cell, is in [tables/success.csv](tables/success.csv). Individual measured episodes and videos are linked in [tables/episodes.csv](tables/episodes.csv).", "",
             "## Five diagnostic tests", "", "| Test | Completed task/policy cells | Pending | Incompatible raw-base cells |", "|---|---:|---:|---:|"]
    for name in dict.fromkeys(r["test"] for r in diagnostics):
        cells = [r for r in diagnostics if r["test"] == name]
        text.append(f"| {name} | {sum(r['status']=='complete' for r in cells)} | {sum(r['status']=='pending' for r in cells)} | {sum(r['status']=='not_applicable' for r in cells)} |")
    failure_rows=[]
    wrong_object_rows=[]
    for measured_row in measured:
        cases=[row for row in data["episodes"] if all(row[key]==measured_row[key] for key in ("task","split","policy"))]
        types={"success" if row["success"] else row["failure_type"] if row["reviewed"] else "not reviewed" for row in cases}
        for kind in sorted(types):
            count=sum(("success" if row["success"] else row["failure_type"] if row["reviewed"] else "not reviewed")==kind for row in cases)
            lo,hi=wilson(count,len(cases))
            failure_rows.append({**{key:measured_row[key] for key in ("task","split","policy")},"failure_type":kind,
                                 "count":count,"episodes":len(cases),"rate_all_episodes":count/len(cases),"wilson95_low":lo,"wilson95_high":hi})
        confirmed=sum(not row["success"] and row["reviewed"] and row["failure_type"]=="wrong object" for row in cases)
        unreviewed=sum(not row["success"] and not row["reviewed"] for row in cases)
        ambiguous=sum(not row["success"] and row["reviewed"] and row["failure_type"]=="other/ambiguous" for row in cases)
        lo,hi=wilson(confirmed,len(cases))
        wrong_object_rows.append({**{key:measured_row[key] for key in ("task","split","policy")},"episodes":len(cases),
            "confirmed_wrong_object_cases_so_far":confirmed,"status":"complete_review" if not unreviewed else "pending_review",
            "confirmed_fraction_all_episodes":confirmed/len(cases) if not unreviewed else None,
            "confirmed_wilson95_low":lo if not unreviewed else None,"confirmed_wilson95_high":hi if not unreviewed else None,"reviewed_ambiguous_failures":ambiguous,
            "unreviewed_failures":unreviewed,"review_coverage_complete":unreviewed==0,
            "interpretation":"Confirmed wrong-object failures among all evaluated episodes, not exhaustive first-choice errors among eventual successes; selected-frame AI review; unreviewed failures are not classified as correct"})
    write_csv(root/"tables/failure_taxonomy.csv",failure_rows,fields=None if failure_rows else ["task","split","policy","failure_type","status"])
    write_csv(root/"tables/wrong_object.csv",wrong_object_rows,fields=None if wrong_object_rows else ["task","split","policy","status"])
    from rl_vla.diagnostics.review_transitions import transitions
    transition_cases, transition_counts, transition_summary = transitions(data["episodes"])
    for name, rows in (("cases",transition_cases),("counts",transition_counts),("summary",transition_summary)):
        write_csv(root/f"tables/failure_transition_{name}.csv", rows,
                  fields=None if rows else ["task","split","before_policy","after_policy","status"])
    if transition_counts:
        text += ["", "[Paired failure transitions](tables/failure_transition_summary.csv), [full category matrices](tables/failure_transition_counts.csv), and [source-bound episode pairs](tables/failure_transition_cases.csv) distinguish recovery to native success from a change between failure types. These descriptive counts use identical state/seed pairs; ambiguous and unreviewed failures remain separate. A wrong-object failure becoming a correct-target grasp/placement failure is not a task-success recovery. Reviewed failure categories do not exhaust first-choice behavior in eventual successes.", ""]
    if any(row.get("reviewed") for row in data["episodes"]):
        text += ["", "Reviewed failure taxonomy uses native contact/lift evidence plus the saved video review sheets, joined to exact report/state/seed and verified source hashes. Confirmed wrong-object rates use **all evaluated episodes** as the denominator. Each failed episode was reviewed by one AI reviewer using selected video frames, with review work distributed across agents; no independent second visual rater is claimed. Strict contact criteria can miss real grasps, so ambiguous cases remain separate. These fractions count only confirmed cases; their Wilson intervals describe episode sampling, not reviewer error or uncertain classification. Native goal success is recorded independently of review.", "",
                 "| Task | Split | Policy | Outcome / reviewed failure type | Cases / all episodes |", "|---|---|---|---|---:|"]
        for row in failure_rows:text.append(f"| {row['task']} | {row['split']} | {row['policy']} | {row['failure_type']} | {row['count']}/{row['episodes']} |")
        text += ["", "[Failure counts and Wilson intervals](tables/failure_taxonomy.csv); [per-episode review labels, notes and sheets](tables/episodes.csv). Automated contact candidates without a reviewed label do not enter the confirmed wrong-object count.", ""]
    probe_rows=[]
    for path in sorted((root/"probes").glob("*/*/metrics.json")):
        probe=json.loads(path.read_text())
        for kind,split_results in probe.get("measurements",{}).items():
            for split,value in split_results.items():
                if value.get("status")=="complete":
                    probe_rows.append({"task":probe["task"],"policy":probe["model"],"features":kind,"split":split,
                                       "balanced_accuracy":value["balanced_accuracy"],"target_auc":value["one_vs_rest_auc"].get("1"),
                                       "distractor_auc":value["one_vs_rest_auc"].get("2"),"target_iou":value["target_mean_iou_visible"],
                                       "target_centroid_error_patches":value["target_mean_centroid_error_when_predicted"]})
    write_csv(root/"tables/probe_metrics.csv",probe_rows,fields=None if probe_rows else ["task","policy","features","split","status"])
    if probe_rows:
        text += ["", "Measured diagnostic probes use fixed linear models fitted only on the 100 training states; all settings were frozen before test scoring. The image-feature probe omits contextual language integration; the prefix-feature probe uses contextualized spatial tokens. Task-specific three-class probes are fitted on fixed or nearly fixed official-swap XY support. High same-layout scores can reflect spatial or appearance regularities; original-layout transfer is the existing location-shift control. Image-only versus prefix features and counterfactual prompts do not by themselves remove location confounding.","",
                 "| Task | Model | Features | Split | Balanced accuracy | Target AUC | Target IoU |", "|---|---|---|---|---:|---:|---:|"]
        for row in probe_rows:
            fmt=lambda value:"N/A" if value is None else f"{value:.3f}"
            text.append(f"| {row['task']} | {row['policy']} | {row['features']} | {row['split']} | {fmt(row['balanced_accuracy'])} | {fmt(row['target_auc'])} | {fmt(row['target_iou'])} |")
        transfers=[]
        for task in TASKS:
            cells={row["split"]:row for row in probe_rows if row["task"]==task and row["policy"]=="pi05_libero" and row["features"]=="prefix_features"}
            if "test" in cells and "source" in cells:
                transfers.append((task,cells["test"],cells["source"]))
        if transfers:
            text += ["", "Original/source-layout scoring uses the same swap-trained probe and unchanged physical classes, without refitting. The native prefix probe shows the following test-swap→original-layout transfer; these are existing summary metrics, with per-camera measurements retained in the probe artifacts.", "",
                     "| Task | Target AUC swap→source | Visible target IoU swap→source |", "|---|---:|---:|"]
            for task,swap,source in transfers:
                fmt4=lambda value:"N/A" if value is None else f"{value:.4f}"
                text.append(f"| {task} | {fmt4(swap['target_auc'])}→{fmt4(source['target_auc'])} | {fmt4(swap['target_iou'])}→{fmt4(source['target_iou'])} |")
            strong_source=[row["task"] for row in measured if row["split"]=="source" and row["policy"]=="pi05_libero" and row["successes"]==50]
            text += ["", "This transfer drop concerns a fixed readout fitted on narrow, position-correlated swap support; it is not evidence that object identity is absent from the representations or that the policy cannot act on the original layout. " + (f"The native baseline nevertheless succeeds50/50 on original-layout retention for {', '.join(strong_source)}. " if strong_source else "") + "High swap localization alone cannot separate semantics from position. These source-layout results do not test probe generalization to the custom new-region layouts.", ""]
    counterfactual_rows=[];prompt_changes=[]
    for path in sorted((root/"probes_counterfactual").glob("*/*/metrics.json")):
        value=read_json(path)
        if not value or not value.get("complete"):continue
        for kind,conditions in value["measurements"].items():
            for condition in ("original","counterfactual"):
                measurement=conditions[condition];requested=measurement["requested_object"]
                counterfactual_rows.append({"task":value["task"],"policy":value["model"],"features":kind,"prompt":condition,
                    "identity_balanced_accuracy":measurement["balanced_accuracy"],"requested_identity_class":requested["physical_identity_class"],
                    "requested_auc":requested["one_vs_rest_auc"],"requested_iou_visible":requested["mean_iou_visible"],
                    "requested_centroid_error_patches":requested["mean_centroid_error_when_predicted"]})
            prompt_changes.append({"task":value["task"],"policy":value["model"],"features":kind,**conditions["paired_prompt_change"]})
    write_csv(root/"tables/counterfactual_probe_metrics.csv",counterfactual_rows,fields=None if counterfactual_rows else ["task","policy","status"])
    write_csv(root/"tables/probe_prompt_changes.csv",prompt_changes,fields=None if prompt_changes else ["task","policy","status"])
    if counterfactual_rows:
        text += ["", "Counterfactual probe scoring transfers the unchanged original-prompt training probe to valid distractor instructions on the same50 test images/states. Physical identity labels stay fixed (class1 task target, class2 distractor); the requested role switches to class2. Comparing original-requested with counterfactual-requested scores therefore also changes object class and difficulty; that contrast alone is not a within-class instruction effect. [Paired same-identity prediction changes](tables/probe_prompt_changes.csv) provide a separate comparison. Image-only features and predictions must remain bit-exact under the prompt edit. No counterfactual source-layout transfer or mismatched-goal policy success is scored. [Requested-object metrics](tables/counterfactual_probe_metrics.csv) retain the role-specific scores; the [independent Object0 identity audit](compiler_profile_integration/counterfactual_probe_identity_audit_v1.json) verifies the fixed physical labels and original-prediction replay.", ""]
    same_class=[]
    for task in TASKS:
        path=root/"compiler_profile_integration"/f"{task}_counterfactual_same_class_review_v1.json"
        audit=read_json(path)
        if not audit or audit.get("complete") is not True or audit.get("passed") is not True:
            continue
        table=Path(audit["table"]["path"])
        if audit["task"]!=task or hashlib.sha256(table.read_bytes()).hexdigest()!=audit["table"]["sha256"]:
            raise ValueError("Same-class counterfactual review/table identity changed")
        for row in audit["same_class_comparisons"]:
            if row["task"]!=task or row["same_class_across_prompts"] is not True:
                raise ValueError("Counterfactual transfer comparison changes physical class")
            same_class.append({**row,"audit":str(path.relative_to(root)),"audit_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                               "review_scope":audit.get("scope",audit.get("interpretation"))})
    write_csv(root/"tables/counterfactual_same_class.csv",same_class,fields=None if same_class else ["task","model","feature","physical_class","status"])
    if same_class:
        text += ["", "[Same-class prompt transfer and per-task audit provenance](tables/counterfactual_same_class.csv) holds the physical instance/class fixed under both instructions and reuses the original-prompt probe without refitting. The linked audits distinguish independent array recomputation from exact copying of previously audited metrics. Changes measure transfer of that frozen readout to altered contextual features; they do not show that identity information is absent or causally used by the policy. High localization on held-out states with narrow native XY support can exploit position-correlated features. It establishes decodability on this distribution, not object-identity generalization to the separate custom OOD layouts.", "",
                 "| Task | Model | Features | Physical instance | AUC original→CF | Visible localization IoU original→CF |",
                 "|---|---|---|---|---:|---:|"]
        for row in same_class:
            text.append(f"| {row['task']} | {row['model']} | {row['feature']} | {row['physical_instance']} | {row['original_auc']:.4f}→{row['counterfactual_auc']:.4f} | {row['original_visible_mean_localization_iou']:.4f}→{row['counterfactual_visible_mean_localization_iou']:.4f} |")
        text.append("")
    causal_rows=[];causal_grasp_rows=[]
    from rl_vla.diagnostics.causal_support_overlay_v1 import analysis_paths
    from rl_vla.diagnostics.causal_support_overlay_v1 import execution as support_execution
    for path in analysis_paths(root):
        causal=json.loads(path.read_text())
        if not causal.get("complete"):continue
        for row in causal["paired_success"]:
            causal_rows.append({"task":causal["task"],"policy":causal["policy"],"condition":row["condition"],
                               "successes":row["candidate_successes"],"episodes":row["episodes"],"difference":row["difference"],
                               "paired_bootstrap95":row["paired_bootstrap95"],"recovered":row["recovered"],"regressed":row["regressed"],
                               "primary_outcome_available":row.get("primary_outcome_available",True),
                               "completed_outcome_cases":row.get("completed_outcome_cases",10),"structural_na_cases":row.get("structural_na_cases",0)})
        for row in causal.get("paired_target_first_grasp_proxy",[]):
            causal_grasp_rows.append({"task":causal["task"],"policy":causal["policy"],"condition":row["condition"],
                                     "target_first_grasp_candidates":row["candidate_successes"],"episodes":row["episodes"],
                                     "difference":row["difference"],"paired_bootstrap95":row["paired_bootstrap95"],
                                     "primary_outcome_available":row.get("primary_outcome_available",True),
                                     "completed_outcome_cases":row.get("completed_outcome_cases",10),"structural_na_cases":row.get("structural_na_cases",0),
                                     "semantics":"Automated native contact/lift proxy; any undefined original control makes whole-condition primary proxy unavailable"})
    write_csv(root/"tables/causal.csv",causal_rows,fields=None if causal_rows else ["task","policy","condition","status"])
    write_csv(root/"tables/causal_grasp_proxies.csv",causal_grasp_rows,fields=None if causal_grasp_rows else ["task","policy","condition","status"])
    if causal_rows:
        text += ["", "Closed-loop causal diagnostics below use all nine predeclared conditions on the same first 10 test states and per-episode noise seeds. Their small, exploratory sample has wide uncertainty. All 15 open-loop conditions use the same 50 matched observations; decoded changes are LIBERO OSC/gripper controller-command coordinates, **not metres**. Prior target-location masks project the source BDDL region as a vertical prism of the original target's native height; they are geometric region controls, distinct from visible instance masks.","",
                 "| Task | Policy | Intervention | Successes/episodes | Difference vs no-op (pp) | Recovered/regressed |", "|---|---|---|---:|---:|---:|"]
        for row in causal_rows:
            if row["primary_outcome_available"]:
                text.append(f"| {row['task']} | {row['policy']} | {row['condition']} | {row['successes']}/{row['episodes']} | {100*row['difference']:+.1f} | {row['recovered']}/{row['regressed']} |")
            else:
                text.append(f"| {row['task']} | {row['policy']} | {row['condition']} | Structural N/A ({row['completed_outcome_cases']} native + {row['structural_na_cases']} undefined /10 planned) | N/A | N/A |")
        object2_causal=[row for row in causal_rows if row["task"]=="object_2"]
        if len(object2_causal)==36 and all(row["successes"]==0 and row["episodes"]==10 for row in object2_causal):
            text += ["", "Object2 has zero successes in all 36 policy-by-intervention cells (10 cases each). This is a task-success floor: a paired bootstrap interval of [0,0] from all-zero observations does not establish pathway irrelevance or policy equivalence. Open-loop action responses and native grasp proxies can still differ while the task-success endpoint remains zero.", ""]
        text += ["", "[First sustained target-grasp proxy shifts](tables/causal_grasp_proxies.csv) use native both-finger contact plus 3 cm lift for three consecutive controlled steps, with paired intervals. They are automated contact measurements and do not become confirmed semantic-failure labels without video review. Per-policy causal tables retain target/distractor/random sensitivity ratios together with mask size and pixel RMS severity; different object-mask areas and replacement artifacts limit causal comparisons.", ""]
        text += ["", "The raw per-policy open-loop tables also apply the audited native OSC clipping and affine scaling to each saved seven-dimensional command before comparison: translation channels request up to ±0.05 m per control command and rotation channels up to ±0.5 rad. Additional translation-request norms are in metres, including the first five commands executed per policy decision; they are **nominal controller targets, not achieved displacement or predicted trajectories**. Rotation and gripper changes remain separate. [Native controller contract](native_controller_contract.json).", ""]
    support_status=support_execution(root)
    if support_status["active"]:
        text += ["", "Object1's original causal queue stopped when the exact target-area random-background control required more unique eligible pixels than existed. The [accepted strict-support amendment](compiler_profile_integration/causal_structural_na_v1/amendment_v1.json) preserves the original masks and all 256 completed native episodes; it permits only independently evidenced structural unavailability, with no capped/fallback mask or replacement case. An undefined input is neither success nor failure. If any case is unavailable, its entire original 10-case primary success, paired effect and target-choice/grasp-proxy rate are N/A; valid individual native outcomes remain in the raw tables. Partial trajectories are excluded from full-episode contact proxies. All 4200 main/OOD cases still require native outcomes. The original failed attempt remains preserved. This amendment was adopted after the failure and partial outcomes were known; it changes availability accounting while preserving the original intervention definition.", ""]
        if support_status["complete"]:
            coverage=support_status["coverage"]
            text += [f"[Independent Object1 execution coverage](compiler_profile_integration/causal_structural_na_v1/execution_coverage.json) resolves all 360 planned causal cases as {coverage['native_completed']} native completed outcomes plus {coverage['structural_NA']} structural N/A cases. Full planned scientific coverage is distinct from the number of completed native episodes.", ""]
            successor_review=read_json(root/"compiler_profile_integration/object_1_all_causal_support_review_v1.json")
            if successor_review and successor_review.get("passed"):
                text += ["The [independent successor analysis review](compiler_profile_integration/object_1_all_causal_support_review_v1.json) checked all 72 primary outcome/proxy rows, 3000 fixed-input open-loop rows and the unchanged bootstrap results. All 16 figure files decoded, and all eight PNG panels were explicitly viewed. Whole-condition N/A fields remain separate from the nine completed observations in the affected condition.", ""]
        else:text += ["The accepted support-aware resumption/validation remains pending; no missing condition is counted as a completed native outcome.", ""]
        first_witness=read_json(root/"compiler_profile_integration/causal_structural_na_v1/first_witness_independent_v1/summary.json")
        if first_witness and first_witness.get("passed"):
            witness=first_witness["witness"]
            unsupported=[row for row in witness["support"] if not row["exact_match_possible"]]
            detail="; ".join(f"{row['camera']}: {row['requested_pixels']} unique target-area pixels required, {row['available_pixels']} eligible background pixels available" for row in unsupported)
            text += [f"The [first independently validated support witness](compiler_profile_integration/causal_structural_na_v1/first_witness_independent_v1/summary.json) is {witness['policy']} / {witness['condition']}, original state {witness['state_id']}, decision {witness['decision']}: {detail}. Its original masks, input, native keys and preceding controlled prefix are retained. The control was initially defined and became unavailable along the policy trajectory. Availability is therefore trajectory-dependent: the completed subset cannot estimate the original 10-case primary effect. This witness documents one undefined intervention input, not a completed episode outcome; total coverage comes from the full execution audit.", ""]
        visual_review=read_json(root/"compiler_profile_integration/causal_structural_na_v1/first_witness_explicit_visual_review_v1.json")
        if visual_review and visual_review.get("passed"):
            render_path=Path(visual_review["render_summary"]["path"])
            if hashlib.sha256(render_path.read_bytes()).hexdigest()!=visual_review["render_summary"]["sha256"]:
                raise ValueError("First support-witness rendering changed after explicit review")
            for figure in read_json(render_path)["artifacts"]:
                path=Path(figure["path"])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=figure["sha256"]:
                    raise ValueError("First support-witness figure changed after review")
                if figure["kind"]=="png":
                    text += [f"![Actual strict-control support exhaustion: retained RGB and exact target/background masks]({path.relative_to(root)})", "The [explicit visual review](compiler_profile_integration/causal_structural_na_v1/first_witness_explicit_visual_review_v1.json) checks the saved mask silhouettes and support counts. The unfinished episode has no native success/failure outcome.", ""]
        contact_path=root/"contact_analysis_structural_na_v1/summary.json"
        if contact_path.exists():
            from rl_vla.diagnostics.contact_support_overlay_v1 import verify_overlay
            contacts=verify_overlay(root)
            text += [f"The [support-aware contact overlay](contact_analysis_structural_na_v1/summary.json) currently resolves {contacts['resolved_cases']}/5280 planned cases: {contacts['completed_native_cases']} completed native episodes and {contacts['structural_na_cases']} explicit unavailable controls. Its [native case table](contact_analysis_structural_na_v1/tables/contact_cases.csv), [availability events](contact_analysis_structural_na_v1/tables/support_unavailable_cases.csv) and [candidate rates](contact_analysis_structural_na_v1/tables/contact_candidate_rates.csv) are separate. Any condition containing N/A has no primary contact/outcome rate; automatic contact candidates remain unreviewed proxies. Main/OOD failure review remains a separate strict 4200-native-case requirement.", ""]
            if not contacts["support_execution_adopted"]:
                text += ["The contact overlay has not yet adopted the resumed Object1 execution certificate; its current unavailable-event count covers only validated inputs and is not a final count for the resumed experiment.", ""]
    drift_rows=[]
    for path in sorted((root/"traces").glob("*/rldt_*/weight_drift.json")):
        for group,value in json.loads(path.read_text()).items():
            drift_rows.append({"task":path.parent.parent.name,"policy":path.parent.name,"group":group,
                               "l2":value["l2"],"relative_l2":value["relative_l2"],"parameters":value["parameters"]})
    write_csv(root/"tables/checkpoint_drift.csv",drift_rows,fields=None if drift_rows else ["task","policy","group","status"])
    if drift_rows:
        text += ["", "Checkpoint parameter changes are measured against the untouched LIBERO expert in [tables/checkpoint_drift.csv](tables/checkpoint_drift.csv), separately for every expert layer and action/time projection. The final action output projection has no attention map. Exact frozen-prefix identity audits are retained beside each checkpoint trace.",""]
    projection=[]
    for path in sorted((root/"trace_analysis").glob("*/projection_ablation.json")):
        projection.extend(json.loads(path.read_text()))
    write_csv(root/"tables/projection_ablation.csv",projection,fields=None if projection else ["task","policy","state_id","step","status"])
    if projection:
        text += ["", "A secondary decomposition on the first five predetermined test states swaps the final normalized expert features and the actual native output projection, including its bias. [Projection swaps](tables/projection_ablation.csv) report hidden-state-only, projection-only, interaction and total changes in normalized flow velocity. Baseline-weight combinations use the verified baseline compiler profile; checkpoint-weight combinations use the verified trained profile. Same-hidden baseline-weight calculations under both profiles quantify the compiler arithmetic contribution separately, so these contrasts cannot be attributed entirely to weight changes. Native endpoint reconstruction must be exact. The signed alignment fractions sum to one when total change is nonzero; they can be negative or exceed one and are not variance-explained scores. These are matched-input open-loop diagnostics, not closed-loop parameter-ablation outcomes.", ""]
    compiler_controls=[]
    for path in sorted((root/"traces").glob("*/compiler_controls/state_*/step*.json")):
        value=json.loads(path.read_text())
        if value.get("complete") is not True:
            continue
        compiler_controls.extend({"task":value["task"], "state_id":value["state_id"], "step":value["step"],
                                  "quantity":quantity, **effects, "source":str(path), "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "units":{"attention":"native softmax probability", "hidden":"native normalized hidden feature coordinates", "velocity":"normalized flow velocity"}[quantity],
                                  "baseline_profile":value["baseline_profile"]["profile_id"],
                                  "control_profile":value["control_profile"]["profile_id"]}
                                 for quantity,effects in value["differences"].items())
    write_csv(root/"tables/compiler_precision_controls.csv",compiler_controls,
              fields=None if compiler_controls else ["task","state_id","step","quantity","status"])
    if compiler_controls:
        text += ["", "[Compiler precision controls](tables/compiler_precision_controls.csv) hold baseline weights, context, latent and time fixed while changing only the verified compiler profile. They retain full attention, final hidden features and velocities for the first five predetermined cases at steps 0/5/9. Their scalar differences measure a numerical contribution that can otherwise be mistaken for an effect of RLDT weights. These controls do not replace the native baseline, score behavior, or establish compiler-independent attention differences for the remaining 45 cases.", ""]
    independent_controls=[]
    for task in TASKS:
        path=root/"compiler_profile_integration"/f"{task}_expert_compiler_control_review_v1.json"
        value=read_json(path)
        if value and value.get("passed") and value.get("complete"):
            independent_controls.extend({"task":task,**row,"review":str(path.relative_to(root)),
                                         "review_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                                        for row in value["grouped"])
    write_csv(root/"tables/independent_expert_control_summary.csv",independent_controls,
              fields=None if independent_controls else ["task","policy","step","status"])
    object2_controls=[row for row in independent_controls if row["task"]=="object_2" and row["step"]==5]
    object2_outcomes=[row for row in data["success"] if row["task"]=="object_2" and row["split"]=="test" and row["policy"]!="pi05_base"]
    if len(object2_controls)==3 and len(object2_outcomes)==4 and all(row["status"]=="complete" and row["successes"]==0 for row in object2_outcomes):
        text += ["", "Object2 remains at **0/50 held-out successes** for LIBERO and all three checkpoints, despite clear expert changes in the five predetermined numerical-control cases. At nominal flow time 0.5, the existing independent audit reports:", "",
                 "| Checkpoint | Velocity L2 change | Same-weight compiler control | Attention L2 change | Same-weight compiler control |",
                 "|---|---:|---:|---:|---:|"]
        for row in object2_controls:
            text.append(f"| {row['policy']} | {row['checkpoint_velocity_l2_mean']:.4f} | {row['compiler_velocity_l2_mean']:.4f} | {row['checkpoint_attention_l2_mean']:.4f} | {row['compiler_attention_l2_mean']:.4f} |")
        text += ["", "These are copied five-state means from the [independent expert/control audit](tables/independent_expert_control_summary.csv), in normalized flow-velocity coordinates and native attention probabilities respectively. The measured changes exceed this specific compiler-only control, but they do not establish successful object grounding or task improvement. This is neither a universal numerical-error bound nor an estimate of explained behavioral variance.", ""]
    attention_pairs=[];layer_knockouts=[]
    for path in sorted((root/"trace_analysis").glob("*/paired_attention.json")):
        attention_pairs.extend(json.loads(path.read_text()))
    for path in sorted((root/"trace_analysis").glob("*/layer_knockout_actions.json")):
        layer_knockouts.extend(json.loads(path.read_text()))
    write_csv(root/"tables/paired_attention.csv",attention_pairs,fields=None if attention_pairs else ["task","policy","status"])
    write_csv(root/"tables/layer_knockouts.csv",layer_knockouts,fields=None if layer_knockouts else ["task","policy","status"])
    grounding_rows=[];grounding_index=[]
    for path in sorted((root/"trace_analysis").glob("*/state_attention.csv")):
        summary=read_json(path.parent/"summary.json")
        if not summary or not summary.get("complete"):continue
        with path.open(newline="") as stream:
            grounding_rows.extend(dict(row,aggregation="Per observation, mean all expert layers/heads/action queries",source=str(path.relative_to(root))) for row in csv.DictReader(stream))
        grounding_index.append({"task":path.parent.name,"per_layer_head":str((path.parent/"grounding.csv").relative_to(root)),
                                "routing_ratios_js_modality":str((path.parent/"routing.csv").relative_to(root)),
                                "text_tokens":str((path.parent/"text_grounding.csv").relative_to(root))})
    write_csv(root/"tables/grounding.csv",grounding_rows,fields=None if grounding_rows else ["task","policy","state_id","status"])
    write_csv(root/"tables/grounding_artifact_index.csv",grounding_index,fields=None if grounding_index else ["task","status"])
    if attention_pairs:
        text += ["", "[Paired attention intervals](tables/paired_attention.csv) resample whole matched observations after averaging all expert layers, heads and action queries. Heads and patches are not independent replicates. The full layer/head rows remain in each task's grounding table; the intervals are exploratory and unadjusted for multiple comparisons.", ""]
    toolkit_exports=[]
    for path in sorted((root/"trace_analysis").glob("*/vla_trace/*/summary.json")):
        value=read_json(path)
        if value and value.get("complete"):toolkit_exports.append({"task":value["task"],"policy":path.parent.name,"summary":str(path.relative_to(root)),"scope":value["scope"]})
    write_csv(root/"tables/vla_trace_exports.csv",toolkit_exports,fields=None if toolkit_exports else ["task","policy","status"])
    if toolkit_exports:
        text += ["", "The isolated official VLA-Trace CLI was invoked for camera overlays and attention metrics on the first predetermined matched test observation for each native policy. [Commands and artifact index](tables/vla_trace_exports.csv) retain every invocation. Its supplemental metrics use camera-conditional normalization, nearest-sampled binary masks and entropy in bits; the primary analyses above retain the full softmax denominator, fractional mask occupancy and entropy in nats.", ""]
    if layer_knockouts:
        text += ["", "Secondary [layer interventions](tables/layer_knockouts.csv) block image or text keys in early (0–5), middle (6–11), late (12–17), and each individual expert layer on the first five predetermined test observations. Full sampled action changes follow each intervened trajectory from identical initial noise; separate velocity tables compare the same baseline latent and time. Native no-op reconstruction and zero blocked attention are mandatory. These are direct key-pathway interventions, with contextual information still available through other keys.", ""]
    kinematics=[]
    for task in TASKS:
        path=root/"diagnostic_layouts"/task/"kinematic_audit.json"
        audit=read_json(path)
        if audit and audit.get("passed"):
            kinematics.append({"task":task,"states_checked":audit["states_checked"],"position_tolerance_m":audit["position_tolerance_m"],
                               "orientation_tolerance_rad":audit["orientation_tolerance_rad"],"collision_penetration_tolerance_m":audit["collision_penetration_tolerance_m"],
                               "passed":True,"policy_used":audit["policy_used"]})
    write_csv(root/"tables/ood_kinematic_audit.csv",kinematics,fields=None if kinematics else ["task","status"])
    if kinematics:
        text += ["", f"Native Panda Jacobian and joint-limit audits passed on {sum(r['states_checked'] for r in kinematics)} frozen custom OOD states across {len(kinematics)} tasks. Five approach/grasp-height waypoints per state satisfy 3 mm position and 0.06 rad orientation tolerances; new nonpermitted robot penetrations above 2 mm are rejected. The audit ignores pre-existing structural contacts and permits gripper–target contact. No policy outcomes were used. These local kinematic checks do not certify continuous collision-free trajectories or successful grasp dynamics. [Audit table](tables/ood_kinematic_audit.csv).", ""]
    from rl_vla.diagnostics.layout_support import layout_support
    support = layout_support(root)
    write_csv(root/"tables/ood_visibility_support.csv",support,fields=None if support else ["task","variant","camera","role","status"])
    if support:
        states=sum(row["states"] for row in support if row["camera"]=="image" and row["role"]=="target")
        either_pass=all(row["states_meeting_20_pixels_in_either_camera"]==row["states"] for row in support)
        text += ["", "[OOD camera visibility and spatial support](tables/ood_visibility_support.csv) come from the frozen generation records. The two novel role centers are fixed; continuous offsets widen the same neighborhoods (±0.025m versus ±0.01m). Variants reuse the same50 official base states and have overlapping support, so their outcomes are kept separate and must not be pooled as200 independent novel-region cases.",
                 f"Visibility records cover {states} frozen states across {len(support)//4} task/layout cells. Each relevant object meets20 pixels in at least one camera throughout." if either_pass else f"Visibility records cover {states} frozen states; the table records cases that lack20 pixels in either camera."]
        for row in support:
            if row["states_below_generation_threshold_20_pixels"]:
                text.append(f"In {row['task']} `{row['variant']}`, {row['role']} has fewer than20 pixels in `{row['camera']}` for {row['states_below_generation_threshold_20_pixels']}/{row['states']} cases. Per-camera evidence must retain this visibility limitation.")
        text += ["[Independent protocol/support review](scientific_protocol_review_v1.json) binds the frozen generation and feasibility evidence; it does not certify policy generalization.", ""]
    if (root/"ood_per_lane_release_review_v1/launch_manifest_v1.json").exists():
        text += ["", "OOD evaluation uses the [approved resource-release amendment](ood_per_lane_release_review_v1/scheduling_amendment_v1.json) and [exact replacement launch manifest](ood_per_lane_release_review_v1/launch_manifest_v1.json). A task's released main GPU can start its fixed cells before causal diagnostics finish; causal/trace resources join only after their own process-exit, completion and exclusive-device checks. Only resource order changed; all16 policy/layout cells,50 state IDs, seeds, checkpoints and native three-shard evaluators are preserved. Object1 binds the unchanged native shared-cache oracle and executable hashes to its new parent process. Final diagnostic readiness requires every task's16 claims/results and its completion/cache provenance, in addition to policy summaries. The superseded empty waiters and old cache watcher remain archived; their retirement is not an inference failure.", ""]
    ood_rows=[];ood_pairs=[];ood_reviews=[];ood_wrong=[];ood_taxonomy=[]
    variants=("both_new_regions","continuous_new_offsets","target_moved_distractor_fixed","distractor_moved_target_fixed")
    for task in TASKS:
        for variant in variants:
            baseline=None
            for policy in POLICIES[1:]:
                path=root/"ood"/task/policy/variant/"summary.json"
                value=read_json(path)
                if not value or not value.get("complete",value.get("completed")):continue
                if data["selection"][task]["status"] != "locked":
                    raise ValueError(f"OOD report exists before locked selection: {path}")
                episodes=checked_episodes(value,path,{"ood_state_ids":list(range(50))},"ood")
                review_rows,review_summary,taxonomy=ood_review_rows(task,policy,variant,path,value,reviews)
                ood_reviews.extend(review_rows);ood_wrong.append(review_summary);ood_taxonomy.extend(taxonomy)
                successes=sum(e["success"] for e in episodes);lo,hi=wilson(successes,50)
                ood_rows.append({"task":task,"policy":policy,"variant":variant,"successes":successes,"episodes":50,
                                 "success_rate":successes/50,"wilson95_low":lo,"wilson95_high":hi})
                if policy=="pi05_libero":baseline=episodes
                elif baseline is not None:ood_pairs.append(dict(task=task,policy=policy,variant=variant,**paired_success(baseline,episodes)))
    write_csv(root/"tables/ood_success.csv",ood_rows,fields=None if ood_rows else ["task","policy","variant","status"])
    write_csv(root/"tables/ood_paired.csv",ood_pairs,fields=None if ood_pairs else ["task","policy","variant","status"])
    write_csv(root/"tables/ood_failure_reviews.csv",ood_reviews,fields=None if ood_reviews else ["task","policy","variant","status"])
    write_csv(root/"tables/ood_wrong_object.csv",ood_wrong,fields=None if ood_wrong else ["task","policy","variant","status"])
    write_csv(root/"tables/ood_failure_taxonomy.csv",ood_taxonomy,fields=None if ood_taxonomy else ["task","policy","variant","failure_type","status"])
    reviewed_cases=[row for row in data["episodes"]+ood_reviews if row.get("reviewed") and not row["success"]]
    codebook=[{"failure_type":label,"review_rule":rule,
               "assigned_reviewed_failure_cases":sum(row["failure_type"]==label for row in reviewed_cases),
               "scope":"Existing verified main/OOD failure labels only; zero means no reviewed case assigned, not proof of zero underlying events. Automatic contact proxies excluded."}
              for label,rule in FAILURE_LABEL_RULES.items()]
    write_csv(root/"tables/failure_label_codebook.csv",codebook)
    text += ["", "The [five-label review codebook](tables/failure_label_codebook.csv) retains wrong object, correct target but failed grasp, correct grasp but placement failure, timeout, and other/ambiguous. Conservative reviewed cases may occupy only a subset; unused labels are retained with zero assigned reviewed cases. Those zeros are not estimates for unreviewed events. The280-step cap does not automatically mean timeout, and strict-contact misses or mixed sequences remain ambiguous. Native successes are separate; automatic contact proxies are not reviewed labels.", ""]
    from rl_vla.diagnostics.ood_figures import render as render_ood_figures
    ood_figure_manifests=[render_ood_figures(root,task,ood_rows,ood_pairs,ood_wrong,ood_taxonomy,ood_reviews) for task in TASKS]
    if ood_rows:
        text += ["", "Custom unseen-layout results below are separate from official PRO scores. The four variants move actual object centers beyond familiar support; frozen-state generation uses geometric/reset checks without policy-success filtering. Layout difficulty can vary in either direction. Success in these two novel-center/offset families leaves broader regional priors unresolved; compare policies within each variant and retain these task-specific support limits. Original-layout retention and standard PRO test outcomes remain in the action-performance table.","",
                 "| Task | Policy | Custom layout | Successes/episodes | Wilson95% |", "|---|---|---|---:|---|"]
        for row in ood_rows:text.append(f"| {row['task']} | {row['policy']} | {row['variant']} | {row['successes']}/50 | {100*row['wilson95_low']:.1f}–{100*row['wilson95_high']:.1f}% |")
        text += ["", "OOD [per-episode reviewed failures](tables/ood_failure_reviews.csv), [failure taxonomy](tables/ood_failure_taxonomy.csv), and [confirmed wrong-object failure fractions](tables/ood_wrong_object.csv) use exact report/state/seed identities and verified report/video/sheet hashes. Fractions use all50 episodes; incomplete review leaves the fraction and interval pending while preserving confirmed counts so far. Mixed sequences and insufficiently corroborated grasps remain ambiguous. These are **wrong-object failures**, not exhaustive first-choice errors among eventual successes. Automatic contact proxies remain a separate evidence stream.", "",
                 "| Task | Policy | Custom layout | Review status | Confirmed wrong failures /50 | Ambiguous | Unreviewed failures |", "|---|---|---|---|---:|---:|---:|"]
        for row in ood_wrong:text.append(f"| {row['task']} | {row['policy']} | {row['variant']} | {row['status']} | {row['confirmed_wrong_object_cases_so_far']}/50 | {row['reviewed_ambiguous_failures']} | {row['unreviewed_failures']} |")
    qualitative=[]
    for folder,pattern in (("rollout_trace_analysis","*/*/state_*/summary.json"),("rollout_trace_analysis/fixed_input_qualitative","*/*/summary.json")):
        for path in sorted((root/folder).glob(pattern)):
            value=read_json(path)
            if value and value.get("complete"):
                for figure in sorted(path.parent.glob("*.png")):
                    qualitative.append({"family":folder,"summary":str(path.relative_to(root)),"figure":str(figure.relative_to(root))})
    write_csv(root/"tables/qualitative_figures.csv",qualitative,fields=None if qualitative else ["family","summary","figure","status"])
    if qualitative:
        text += ["", "[Qualitative gallery](tables/qualitative_figures.csv) distinguishes two families. Actual rollout-conditioned phase maps use each policy's recorded pixels, sampling keys and own denoising trajectory; their inputs can differ after the initial decision. Recovered/unsolved comparison panels use matched fixed inputs and disclose their outcome-based case selection. A wrong-object label requires immutable video review evidence. These selected panels do not replace the full paired cohort statistics.", ""]
    phase_audit=read_json(root/"compiler_profile_integration/rollout_completion_independent_review_v1.json")
    if phase_audit and phase_audit.get("passed"):
        text += [f"The [independent rollout audit](compiler_profile_integration/rollout_completion_independent_review_v1.json) covers {phase_audit['episode_count']} sampled episodes and {phase_audit['phase_decisions']} phase decisions. Recorded inputs/actions match exactly, and raw attention agrees with exported views and rendered map arrays. {sum(phase_audit['absent_phase_counts'].values())} episodes have no basket-approach phase; no replacement phase is substituted. This covers sampled own-policy decisions, not every intermediate trajectory decision. Figure decoding is distinct from manual visual inspection, and grasp-phase candidates are not semantic failure labels.", ""]
    if (root/"compiler_profile_integration/rollout_phase_selected_visual_review_v1.json").exists():
        text += ["[Selected phase-panel visual review](compiler_profile_integration/rollout_phase_selected_visual_review_v1.json) inspected five panels separately from the audit of all 360 decoded figure files. Color scales are shared across phases and flow times within a panel, but their maxima differ between policy/camera panels. Cross-panel brightness is therefore not quantitative evidence of an attention change; use the recorded scales and raw measurements.", ""]
    supplemental=[]
    for task in TASKS:
        if not cell_ready(root,"supplemental_analysis",task,"all_policies"):
            continue
        path=root/"supplemental_analysis"/task/"summary.json"
        value=read_json(path)
        for artifact in value["artifacts"]:
            groups=[group for group, paths in value["required_artifact_groups"].items() if artifact["path"] in paths]
            supplemental.append({"task":task,"group":";".join(groups) or "provenance","path":str(Path(artifact["path"]).relative_to(root)),
                                 "sha256":artifact["sha256"],"kind":artifact["kind"],"summary":str(path.relative_to(root)),
                                 "amendment":value["amendment"]["path"],"amendment_sha256":value["amendment"]["sha256"]})
    write_csv(root/"tables/supplemental_artifacts.csv",supplemental,fields=None if supplemental else ["task","group","path","sha256","status"])
    if supplemental:
        text += ["", "The [supplemental figure and raw-data index](tables/supplemental_artifacts.csv) adds action-to-image/text/action modality curves, layer-intervention response plots, function-word/spatial/random-token controls, and action-vector comparisons. These CPU analyses were declared after held-out task outcomes were available and before their supplemental maps were inspected; they are exploratory additions, not prospectively selected confirmatory tests. The original fixed50-case cohort and first-five intervention subset are retained. Scaled translation/rotation requests describe native controller inputs, not achieved motion. First-command directional cosines are nominal world-frame alignment proxies, not predicted first-choice accuracy or observed target selection. Zero/undefined directions remain explicitly N/A. [Amendment](compiler_profile_integration/supplemental_analysis_amendment_v1.json); [independent Object0 review](compiler_profile_integration/object0_supplement_independent_review_v1.json).", ""]
    if scientific_detail:text += ["", "## Evidence-based interpretation", "", scientific_detail, ""]
    text += ["", "Per-cell evidence and reasons are in [tables/diagnostic_status.csv](tables/diagnostic_status.csv). Shared observations must have matched pixels, state, prompt, initial flow noise and latent/time before comparing expert attention. Region mass retains the full attention-softmax denominator; area-normalized density and camera-conditional mass are separate statistics. Prior-location masks are projected geometry; target/distractor masks are actual visible instances. In the official swap the distractor occupies the former target region, so these masks can overlap. Attention or ablation there alone cannot distinguish distractor identity from a spatial shortcut; the target-fixed, distractor-fixed and new-region conditions address that confound. Pixel controls use neutral background fill and mosaic with matched-area controls. Native attention knockouts require verified no-op fidelity. The native prefix permits bidirectional attention between valid image and text tokens. Blocking expert access to image keys can leave visual information in contextualized text keys; blocking text keys can leave instruction information in contextualized image keys. These are direct key-pathway knockouts, not complete removal of either modality's information. Pixel masking and any prefill-pathway knockout are distinct interventions. [Native mask topology audit](prefix_mask_interpretation_audit.json).", "",
             "Noun-query spans are resolved from saved token pieces, validity masks and captured prefix positions, with exact phrase boundaries and checks against any existing exported span. This repairs an empty standalone-token-ID lookup caused by context-dependent SentencePiece IDs; it does not retokenize inputs or alter attention values. Each task retains a `noun_span_resolution.json` source-hash audit. The [independent Object0 check](compiler_profile_integration/object0_noun_mapping_independent_review_v1.json) verified all 200 metadata/attention pairs.", "",
             "## Interpretation and limitations", "",
             "A successful localization probe establishes distributional decodability, not position-invariant object recognition or causal use. Fixed or nearly fixed training positions can confound same-layout localization; image-only/prefix comparisons and prompt changes alone do not remove that confounding. A failed probe does not prove missing information. Attention differences alone do not establish that the action head caused a behavioral change. Improved swap success with poor new-region generalization is consistent with a second position shortcut. A stronger routing conclusion requires unchanged frozen representations, matched-input expert changes, causal intervention sensitivity, and improved behavior at genuinely unseen locations.", "",
             "Failure causes without a verified video-review label remain **other/ambiguous — not reviewed**. Failed episodes are never automatically counted as semantic failures. New state IDs within trained spatial ranges do not establish unseen-region generalization. Single-seed and finite-state uncertainty limit population claims.", "",
             "**Spatial support limitation:** the pinned native sampler shrinks each placement range by the object radius. For alphabet soup, salad dressing and tomato sauce, radius 0.025 m in a 0.05 m range leaves a fixed XY center. Cream cheese and milk use reversed ±0.005 m native bounds, and the basket reversed ±0.015 m bounds. These official assets and native sampling semantics are preserved. Generated state/yaw diversity must therefore not be described as broad XY diversity. The separate custom diagnostic relocations change actual XY centers, audit footprint separation and measure distance from familiar target/distractor positions.", "",
             "## Compute and storage", "",
             f"Current run-local files occupy {data['resources']['run_local_file_bytes_excluding_symlink_targets']/2**30:.2f} GiB in logical bytes (shared symlink targets counted only at their original run-local path). This includes technical checks, recordings, diagnostics and any partial artifacts; pre-existing reference checkpoints are excluded. [resource_cost.json](resource_cost.json) records peak GPU memory and utilization-weighted GPU-hour/energy estimates from 30-second telemetry. These device-wide estimates include monitored idle power and are not training-only billing. Training interaction counts and optimizer updates are separate from wall-clock validation time.","",
             "## Reproduction and artifacts", "", "```bash", "cd /workspace/upenn/RL-VLA", "source scripts/env.sh", f"PYTHONPATH=src .venv/bin/python scripts/analyze_vla_trace_objects.py --run {root}", "```", "",
             "See [runbook.md](runbook.md) for launch commands and [manifest.json](manifest.json) for source and environment provenance. The [scientific evidence digest](evidence_digest.md) indexes existing means, paired intervals, pending cells and exact source hashes without adding statistics or conclusions. Tables are regenerated from completed raw artifacts; missing evidence remains explicitly pending.", ""]
    device_costs = data["resources"].get("devices", {})
    if (root / "resource_cost_final_v1.json").exists():
        resource_index = text.index("## Reproduction and artifacts")
        text[resource_index:resource_index] = ["The [immutable final resource snapshot](resource_cost_final_v1.json) binds the stopped telemetry stream, capture time and resource-accounting source. The [writer-freeze receipt](compiler_profile_integration/final_writer_freeze_v1/receipt.json) records the final monitor/contact-writer disposition; later report regeneration can add small files without changing that historical storage snapshot.", ""]
    if device_costs:
        resource_rows = ["| GPU | Telemetry samples | Peak reported GPU memory (GiB) | Utilization-weighted GPU-hours | Device energy (kWh) |",
                         "|---|---:|---:|---:|---:|"]
        for device, cost in sorted(device_costs.items(), key=lambda pair: int(pair[0])):
            resource_rows.append(f"| {device} | {cost['samples']} | {cost['peak_memory_mib']/1024:.2f} | {cost['utilization_weighted_gpu_hours']:.2f} | {cost['device_energy_kwh']:.2f} |")
        resource_rows += ["", "These are trapezoidal integrations between consecutive telemetry samples less than two minutes apart. Longer gaps are excluded; utilization-weighted hours are an occupancy estimate, not an exact allocation duration or hardware-operation count.", ""]
        resource_index = text.index("## Reproduction and artifacts")
        text[resource_index:resource_index] = resource_rows
    for path in sorted((root / "figures/task_comparison").glob("*.png")):
        text += [f"![{path.stem}]({path.relative_to(root)})", ""]
    for path in sorted((root/"probes").glob("*/*/confusion.png")):
        text += [f"![Diagnostic probe confusion: {path.parent.parent.name}, {path.parent.name}]({path.relative_to(root)})",""]
    for path in sorted((root/"probes_counterfactual").glob("*/*/*.png")):
        value=read_json(path.parent/"metrics.json")
        if value and value.get("complete"):
            text += [f"![Counterfactual probe: {path.parent.parent.name}, {path.parent.name}, {path.stem}]({path.relative_to(root)})",""]
    for pattern in ("*/fixed_observation_attention.png","*/layer_density_ratio.png","*/flow_time_evolution.png",
                    "*/head_attention_drift.png","*/weight_and_velocity_drift.png","*/text_grounding_layer*.png","*/mask_demonstration.png"):
        for path in sorted((root/"trace_analysis").glob(pattern)):
            text += [f"![{path.parent.name}: {path.stem}]({path.relative_to(root)})",""]
    for row in supplemental:
        if row["path"].endswith(".png"):
            text += [f"![Exploratory supplement: {row['task']}, {Path(row['path']).stem}]({row['path']})", ""]
    from rl_vla.diagnostics.causal_support_overlay_v1 import (
        readable_coverage as causal_figure_coverage,
    )
    causal_plots = causal_figure_coverage(root)
    text += [f"Readable causal figures: {causal_plots['completed_cells']}/12 validated cells. The original readable derivatives copy the original50-state means and intervals; accepted support-aware Object1 figures preserve that exact calculation and separately show unavailable closed-loop conditions. Axis ranges can differ between policy figures; compare recorded values rather than brightness or apparent span. Original figures and measurements remain available. [Derivative manifest](figures/causal/summary.json).", ""]
    if (root / "causal_analysis_structural_na_v1/summary.json").exists():
        text += ["The [original derivative marker](figures/causal/summary.json) retains its historical incomplete coverage; the four [accepted Object1 successor cells](causal_analysis_structural_na_v1/summary.json) supply the remaining selected coverage and explicit whole-condition N/A annotation.", ""]
    for cell in causal_plots["cells"]:
        if cell["status"] == "complete":
            for item in cell["artifacts"]:
                if Path(item["path"]).suffix == ".png":
                    text += [f"![Causal diagnostic: {cell['task']}, {cell['policy']}, {Path(item['path']).stem}]({Path(item['path']).relative_to(root)})", ""]
    for manifest in ood_figure_manifests:
        text += [f"OOD figures for {manifest['task']}: {manifest['native_cells']}/16 native cells, {manifest['paired_contrasts']}/12 paired contrasts and {manifest['complete_review_cells']}/16 complete review cells. Pending cells are placeholders, not zero estimates. [Figure and raw-table manifest](figures/ood/{manifest['task']}/summary.json).", ""]
        review_path = root / "compiler_profile_integration" / f"{manifest['task']}_final_ood_figure_review_v1.json"
        final_review = read_json(review_path)
        current_manifest = root / "figures/ood" / manifest["task"] / "summary.json"
        if (final_review and final_review.get("complete") is True and final_review.get("passed") is True
                and final_review.get("manifest", {}).get("path") == str(current_manifest)
                and final_review["manifest"]["sha256"] == hashlib.sha256(current_manifest.read_bytes()).hexdigest()):
            text += [f"The [independent final {manifest['task']} figure review]({review_path.relative_to(root)}) verifies the complete native, paired and reviewed-category tables and artifact hashes, and explicitly inspects both final PNGs. PDF decoding and numerical verification are recorded separately from visual inspection.", ""]
        for row in manifest["artifacts"]:
            if row["kind"] == "png":
                text += [f"![{manifest['task']}: {Path(row['path']).stem}]({Path(row['path']).relative_to(root)})", ""]
    if (root/"compiler_profile_integration/object_1_ood_figures_candidate_review_v2.json").exists():
        text += ["The [independent OOD presentation review](compiler_profile_integration/object_1_ood_figures_candidate_review_v2.json) verified candidate figure readability and exact table/source bindings while only one native cell was complete; it does not certify the remaining research outcomes or reviews.", ""]
    (root / "report.md").write_text("\n".join(text))


def snapshot_analysis_sources(root):
    """Keep immutable content-addressed source copies for the regenerated report."""
    repository=Path(__file__).resolve().parents[1]
    paths=sorted(set((repository/"scripts").glob("*vla_trace*.py"))
                 | set((repository/"scripts").glob("trace_pi05*.py"))
                 | set((repository/"src/rl_vla/diagnostics").glob("*.py")))
    records=[]
    for path in paths:
        payload=path.read_bytes();checksum=hashlib.sha256(payload).hexdigest()
        snapshot=root/"analysis_code"/checksum/path.name
        if snapshot.exists():
            if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=checksum:raise ValueError("Immutable analysis source snapshot changed")
        else:
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            snapshot.write_bytes(payload)
        records.append({"source":str(path.relative_to(repository)),"sha256":checksum,"snapshot":str(snapshot.relative_to(root))})
    write_json(root/"analysis_source_manifest.json",{"scope":"Sources used by the latest report/diagnostic tooling; frozen original training/input contracts remain separately recorded",
        "sources":records})


def _analyze(root):
    root = Path(root).resolve()
    snapshot_analysis_sources(root)
    reviews=reviewed_labels(root)
    data = collect(root,reviews=reviews)
    from rl_vla.diagnostics.training_coverage import final_training_reset_coverage
    data["training_reset_coverage"], training_visits = final_training_reset_coverage(root)
    write_csv(root / "tables/training_state_coverage.csv", data["training_reset_coverage"])
    write_csv(root / "tables/training_state_visit_counts.csv", training_visits,
              fields=None if training_visits else ["task", "state_id", "completed_episode_reset_count", "episode_log_sha256"])
    data["resources"]=resource_cost(root)
    write_json(root/"resource_cost.json",data["resources"])
    diagnostics = diagnostic_inventory(root)
    for key, name in (("success", "success"), ("paired", "paired_success"), ("episodes", "episodes"),
                      ("failures", "wrong_object_review"), ("learning", "learning_curve")):
        write_csv(root / "tables" / f"{name}.csv", data[key], fields=None if data[key] else ["task", "split", "policy", "status"])
    write_csv(root / "tables/diagnostic_status.csv", diagnostics)
    write_json(root / "analysis.json", {"training": data["training"], "selection": data["selection"],
                                        "paired": data["paired"], "diagnostics": diagnostics})
    figures(root, data)
    report(root, data, diagnostics,reviews=reviews)
    from rl_vla.diagnostics.evidence_digest import build_digest
    build_digest(root)
    return {"report": str(root / "report.md"), "measured_evaluation_cells": sum(r["status"] == "complete" for r in data["success"]),
            "pending_diagnostic_cells": sum(r["status"] == "pending" for r in diagnostics)}


def analyze(root):
    root = Path(root).resolve()
    with (root / ".analysis.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _analyze(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))
