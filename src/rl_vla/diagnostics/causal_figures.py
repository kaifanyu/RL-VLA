"""Presentation-only derivatives of already validated causal sensitivity figures."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rl_vla.diagnostics import figure_artifacts as original

TASKS = original.TASKS
POLICIES = original.POLICIES
CONDITIONS = (
    "none", "no_image", "no_text",
    *[f"{region}:{mode}" for region in ("target", "distractor", "random", "robot", "background", "old_target")
      for mode in ("background_fill", "mosaic")],
)
SCHEMA = "causal-presentation-derivative-v1"


def label(condition):
    if condition in ("none", "no_image", "no_text"):
        return {"none": "None (own-policy control)", "no_image": "No image keys", "no_text": "No text keys"}[condition]
    region, mode = condition.split(":")
    names = {"target": "Target", "distractor": "Distractor", "random": "Random area control",
             "robot": "Robot", "background": "Background", "old_target": "Prior target region"}
    return names[region] + " · " + mode.replace("_", " ")


def cell_folder(root, task, policy):
    return Path(root).resolve() / "figures/causal" / task / original.canonical(policy)


def binding(path):
    path = Path(path).resolve()
    before = original.identity(path)
    digest = original.sha(path)
    if original.identity(path) != before:
        raise ValueError(f"Source changed while binding: {path}")
    return {"path": str(path), "sha256": digest, "identity": before}


def verify_binding(row, *, full_hash=False):
    if original.identity(row["path"]) != row["identity"]:
        raise ValueError(f"Bound causal presentation input changed: {row['path']}")
    if full_hash and original.sha(row["path"]) != row["sha256"]:
        raise ValueError(f"Bound causal presentation bytes changed: {row['path']}")


def source_inputs(root, task, policy):
    root = Path(root).resolve(); policy = original.canonical(policy)
    source_manifest = original.manifest_path(root, "causal_analysis", task, policy)
    if not source_manifest.exists():
        return None
    validated = original.verify_manifest(source_manifest, full_hash=True)
    if (validated.get("task") != task or validated.get("policy") != policy
            or validated.get("domain") != "causal_analysis"):
        raise ValueError("Causal source manifest identity differs")
    padded = f"rldt_{int(policy.removeprefix('rldt_')):06d}" if policy.startswith("rldt_") else policy
    source = root / "causal_analysis" / task / padded
    summary = original.read(source / "summary.json")
    if (summary.get("complete") is not True or summary.get("task") != task
            or original.canonical(summary.get("policy", "")) != policy
            or summary.get("open_loop_states") != 50 or summary.get("open_loop_conditions") != 15
            or summary.get("paired_episodes_per_condition") != 10 or summary.get("closed_loop_conditions") != 9):
        raise ValueError("Causal source scope differs from the completed fixed protocol")
    values = summary["open_loop_aggregates"]
    if [row["condition"] for row in values] != list(CONDITIONS) or any(row["states"] != 50 for row in values):
        raise ValueError("Expected exact fifteen-condition fifty-state plotting rows")
    for row in values:
        lo, hi = row["paired_state_bootstrap95_mean"]
        mean = row["mean_decoded_controller_l2"]
        if not np.isfinite([lo, mean, hi]).all() or lo < 0 or not lo <= mean <= hi:
            raise ValueError("Invalid existing mean/bootstrap interval")
    with (source / "open_loop_actions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 750 or len({(row["state_id"], row["condition"]) for row in rows}) != 750:
        raise ValueError("Source table lacks exact fifteen-by-fifty coverage")
    # Validate row linkage without computing a new interval or statistical estimate.
    for value in values:
        subset = [row for row in rows if row["condition"] == value["condition"]]
        if sorted(int(row["state_id"]) for row in subset) != list(range(150, 200)):
            raise ValueError("Source plotting table dropped a predetermined state")
    files = [source_manifest, source / "summary.json", source / "open_loop_actions.csv",
             source / "open_loop_sensitivity.png", source / "open_loop_sensitivity.pdf"]
    repo = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), repo / "scripts/render_vla_trace_causal_figures.py", Path(original.__file__)]
    return {"values": values, "source_manifest": binding(source_manifest),
            "inputs": [binding(path) for path in files], "renderer_sources": [binding(path) for path in sources]}


def verify_cell(root, task, policy, *, full_hash=False):
    folder = cell_folder(root, task, policy)
    summary = original.read(folder / "summary.json")
    if (summary.get("complete") is not True or summary.get("passed") is not True
            or summary.get("schema") != SCHEMA or summary.get("task") != task
            or summary.get("policy") != original.canonical(policy)):
        raise ValueError("Causal derivative summary identity/completion differs")
    manifest_row = summary["manifest"]
    verify_binding(manifest_row, full_hash=full_hash)
    manifest_path = Path(manifest_row["path"])
    if not manifest_path.is_relative_to(folder / "versions"):
        raise ValueError("Causal derivative manifest escaped its version directory")
    manifest = original.read(manifest_path)
    if (manifest.get("complete") is not True or manifest.get("passed") is not True
            or manifest.get("schema") != SCHEMA or manifest.get("task") != task
            or manifest.get("policy") != original.canonical(policy)
            or manifest.get("states") != 50 or manifest.get("conditions") != list(CONDITIONS)
            or manifest.get("statistics_recomputed") is not False):
        raise ValueError("Causal derivative manifest scope differs")
    if summary.get("artifacts") != manifest["artifacts"] or summary.get("states") != 50 or summary.get("statistics_recomputed") is not False:
        raise ValueError("Causal summary artifact/scope pointers differ from immutable manifest")
    for row in manifest["inputs"] + manifest["renderer_sources"] + manifest["source_snapshots"] + manifest["artifacts"]:
        verify_binding(row, full_hash=full_hash)
    original.verify_manifest(manifest["original_validated_manifest"]["path"], full_hash=full_hash)
    expected_files = {"open_loop_sensitivity.png", "open_loop_sensitivity.pdf", "plot_values.json", "plot_values.csv"}
    if {Path(row["path"]).name for row in manifest["artifacts"]} != expected_files:
        raise ValueError("Missing derivative figure or exact plotting-data export")
    copied = original.read(manifest_path.parent / "plot_values.json")
    if copied != manifest["plot_values"]:
        raise ValueError("Copied plotting data differ from derivative manifest")
    source_summary = next(row for row in manifest["inputs"] if Path(row["path"]).name == "summary.json")
    if copied != original.read(source_summary["path"])["open_loop_aggregates"]:
        raise ValueError("Derivative values differ from the existing original statistics")
    return manifest


def verify_coverage(root, full_hash=False):
    """Pure CPU/read-only coverage check; pending and failed are distinct."""
    cells = []
    for task in TASKS:
        for policy in POLICIES:
            path = cell_folder(root, task, policy) / "summary.json"
            if not path.exists():
                cells.append({"task": task, "policy": policy, "status": "pending"})
                continue
            try:
                manifest = verify_cell(root, task, policy, full_hash=full_hash)
                cells.append({"task": task, "policy": policy, "status": "complete",
                              "summary": binding(path), "artifacts": manifest["artifacts"]})
            except (OSError, ValueError, KeyError, TypeError) as error:
                cells.append({"task": task, "policy": policy, "status": "failed", "reason": str(error)})
    failed = sum(row["status"] == "failed" for row in cells)
    pending = sum(row["status"] == "pending" for row in cells)
    complete = not failed and not pending
    return {"schema": SCHEMA, "complete": complete, "passed": False if failed else (True if complete else None),
            "expected_cells": 12, "completed_cells": 12 - failed - pending, "pending_cells": pending,
            "failed_cells": failed, "cells": cells,
            "scope": "Presentation-only derivatives of existing50-state open-loop statistics; paired10-episode closed-loop results remain separate."}


def render_cell(root, task, policy):
    root = Path(root).resolve(); policy = original.canonical(policy)
    folder = cell_folder(root, task, policy)
    if (folder / "summary.json").exists():
        return verify_cell(root, task, policy, full_hash=True)
    inputs = source_inputs(root, task, policy)
    if inputs is None:
        return None
    key_data = {name: inputs[name] for name in ("inputs", "renderer_sources")}
    version = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    output = folder / "versions" / version
    output.mkdir(parents=True, exist_ok=False)
    values = inputs["values"]
    original.write(output / "plot_values.json", values)
    with (output / "plot_values.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["condition", "states", "mean_decoded_controller_l2", "ci95_low", "ci95_high"])
        writer.writeheader()
        for row in values:
            writer.writerow({"condition": row["condition"], "states": row["states"],
                             "mean_decoded_controller_l2": row["mean_decoded_controller_l2"],
                             "ci95_low": row["paired_state_bootstrap95_mean"][0],
                             "ci95_high": row["paired_state_bootstrap95_mean"][1]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13.6, 8.7), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(3, 1.12))
    ax = fig.add_subplot(grid[0, 0]); numbers = fig.add_subplot(grid[0, 1], sharey=ax)
    means = np.array([row["mean_decoded_controller_l2"] for row in values])
    bounds = np.array([row["paired_state_bootstrap95_mean"] for row in values])
    y = np.arange(len(values))
    ax.errorbar(means, y, xerr=np.stack((means - bounds[:, 0], bounds[:, 1] - means)), fmt="o", capsize=4)
    ax.set(yticks=y, yticklabels=[label(row["condition"]) for row in values],
           xlabel="Mean action-chunk L2 change\n(controller command units)", ylim=(len(values) - .4, -.6))
    ax.grid(axis="x", alpha=.25); ax.axvline(0, color="gray", linewidth=.7)
    numbers.set_axis_off(); numbers.set_xlim(0, 1)
    numbers.text(.02, 1.02, "Mean [95% bootstrap interval]", transform=numbers.transAxes, fontsize=10)
    for index, (mean, (lo, hi)) in enumerate(zip(means, bounds, strict=True)):
        numbers.text(.02, index, f"{mean:.4g}  [{lo:.4g}, {hi:.4g}]", va="center", fontsize=10)
    policy_label = "LIBERO baseline" if policy == "pi05_libero" else f"RLDT {int(policy.removeprefix('rldt_'))}"
    fig.suptitle(f"{task.replace('_', ' ').title()} · {policy_label} · 50 fixed observations with matched noise\n"
                 "Open-loop sensitivity: existing means and intervals, unchanged", fontsize=14)
    fig.supxlabel("Nominal controller-command changes, not success rates or achieved motion.\n"
                  "Closed-loop intervention outcomes use a separate paired 10-episode subset.", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"open_loop_sensitivity.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    snapshots = []
    for index, row in enumerate(inputs["renderer_sources"]):
        path = output / f"source_{index:02d}_{Path(row['path']).name}"
        path.write_bytes(Path(row["path"]).read_bytes())
        if original.sha(path) != row["sha256"]:
            raise ValueError("Renderer source changed while snapshotting")
        snapshots.append(binding(path))
    artifacts = [original.artifact(output / name) for name in
                 ("open_loop_sensitivity.png", "open_loop_sensitivity.pdf", "plot_values.json", "plot_values.csv")]
    for row in inputs["inputs"] + inputs["renderer_sources"]:
        verify_binding(row, full_hash=True)
    manifest = {"schema": SCHEMA, "complete": True, "passed": True, "task": task, "policy": policy,
                "states": 50, "conditions": list(CONDITIONS), "statistics_recomputed": False,
                "original_validated_manifest": inputs["source_manifest"], "inputs": inputs["inputs"],
                "renderer_sources": inputs["renderer_sources"], "source_snapshots": snapshots,
                "plot_values": values, "artifacts": artifacts,
                "presentation_change": "Horizontal labels and numeric interval column replace clipped rotated/vertical text; originals retained byte-for-byte.",
                "scope": "Existing fixed-input50-state open-loop means/bootstrap intervals only; no new statistics, cases, scoring, interventions or claims of achieved motion."}
    original.write(output / "manifest.json", manifest)
    original.write(folder / "summary.json", {"schema": SCHEMA, "complete": True, "passed": True,
                   "task": task, "policy": policy, "manifest": binding(output / "manifest.json"),
                   "artifacts": artifacts, "states": 50, "statistics_recomputed": False})
    return verify_cell(root, task, policy, full_hash=True)
