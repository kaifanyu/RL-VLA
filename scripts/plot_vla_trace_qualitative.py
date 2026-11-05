#!/usr/bin/env python3
"""Render auditable rollout-phase and outcome-selected matched-input attention panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import textwrap
import time
from pathlib import Path

import numpy as np
from analyze_vla_trace_objects import checked_episodes, evaluation_path, reviewed_labels

from rl_vla.diagnostics.metrics import episode_identity, grounding_scores, patch_occupancy

POLICIES = ("pi05_libero", "rldt_000025", "rldt_000050", "rldt_000100")
CAMERAS = {"image": "base_0_rgb", "wrist_image": "left_wrist_0_rgb"}
OBJECTS = {
    "object_0": ("alphabet_soup_1", "butter_1"),
    "object_1": ("cream_cheese_1", "alphabet_soup_1"),
    "object_2": ("salad_dressing_1", "tomato_sauce_1"),
}
STEPS = (0, 5, 9)
COLORS = {"target": "#65ff32", "distractor": "#ff3acc", "old_target": "#00e5ff"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def complete(path):
    if not Path(path).exists():
        return False
    value = read(path)
    return bool(value.get("complete", value.get("completed", False)))


def camera_mask_arrays(path, task, camera, auxiliary=None):
    target, distractor = OBJECTS[task]
    with np.load(path, allow_pickle=False) as archive:
        result = {
            role: archive[f"{camera}__{name}"]
            for role, name in (("target", target), ("distractor", distractor))
        }
        if f"{camera}__old_target_prism" in archive:
            result["old_target"] = archive[f"{camera}__old_target_prism"]
    if auxiliary:
        with np.load(auxiliary, allow_pickle=False) as archive:
            result["old_target"] = archive[f"{camera}__old_target_prism"]
    return result


def mean_camera_map(directory, metadata, step, camera):
    span = next(span for span in metadata["camera_spans"] if span["name"] == CAMERAS[camera])
    if not span["valid"]:
        raise ValueError("Requested camera is masked in the native policy")
    with np.load(directory / f"views_step{step:02d}_none.npz", allow_pickle=False) as archive:
        values = archive[span["name"]].reshape(tuple(span["grid"]))
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid native full-denominator camera attention")
    return values


def toolkit_overlay(directory, metadata, camera, image, masks, task, output, *, step):
    """Supplementary toolkit view; primary panels retain a shared color scale."""
    executable = (
        Path(__file__).resolve().parents[1] / "third_party/vla_trace_tools/.venv/bin/vla-trace"
    )
    image_path = output.with_suffix(".npy")
    np.save(image_path, image, allow_pickle=False)
    span = next(span for span in metadata["camera_spans"] if span["name"] == CAMERAS[camera])
    command = [
        str(executable),
        "attention-overlay",
        "--image",
        str(image_path),
        "--attention",
        str(directory / f"views_step{step:02d}_none.npz"),
        "--attention-key",
        span["name"],
        "--masks",
        str(masks),
        "--mask-key",
        f"{camera}__{OBJECTS[task][0]}",
        "--grid-size",
        str(span["grid"][0]),
        "--output",
        str(output.with_suffix(".png")),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return {
        "command": command,
        "figure": str(output.with_suffix(".png")),
        "scope": "Supplementary pinned VLA-Trace top10-percent overlay; toolkit uses an individual color scale. Use shared-scale primary panels for magnitude comparisons.",
    }


def render_panel(rows, output, *, title, caption):
    """Use one unnormalized probability scale across every row/ODE column."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    arrays = [cell["heat"] for row in rows for cell in row["cells"] if cell is not None]
    if not arrays:
        raise ValueError("No attention arrays for panel")
    vmax = max(float(np.max(array)) for array in arrays)
    vmax = max(vmax, np.finfo(np.float32).tiny)
    fig, axes = plt.subplots(
        len(rows),
        len(STEPS),
        figsize=(10.3, 2.8 * len(rows) + 1.3),
        squeeze=False,
        layout="constrained",
    )
    exported, measurements = {}, []
    for row_index, row in enumerate(rows):
        for column, (step, cell) in enumerate(zip(STEPS, row["cells"], strict=True)):
            ax = axes[row_index, column]
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                ax.set_ylabel(
                    row["label"], fontsize=8, rotation=0, ha="right", va="center", labelpad=12
                )
            if row_index == 0:
                ax.set_title(f"ODE step {step} · t={1 - step / 10:.1f}", fontsize=10)
            if cell is None:
                ax.text(
                    0.5,
                    0.5,
                    "N/A\nIncompatible action contract",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.set_facecolor("#f2f2f2")
                continue
            image, heat = cell["image"], cell["heat"]
            if image.shape[:2] != (224, 224) or image.dtype != np.uint8:
                raise ValueError("Expected exact uint8 224×224 native policy pixels")
            if any(image.shape[index] % heat.shape[index] for index in (0, 1)):
                raise ValueError("Camera patch grid does not divide policy image")
            upsampled = np.repeat(
                np.repeat(heat, image.shape[0] // heat.shape[0], axis=0),
                image.shape[1] // heat.shape[1],
                axis=1,
            )
            ax.imshow(image)
            im = ax.imshow(
                upsampled, cmap="magma", vmin=0, vmax=vmax, alpha=0.43, interpolation="nearest"
            )
            for role, mask in cell["masks"].items():
                if np.any(mask) and not np.all(mask):
                    ax.contour(
                        mask.astype(float),
                        levels=[0.5],
                        colors=[COLORS[role]],
                        linewidths=1.0,
                        linestyles="--" if role == "old_target" else "-",
                    )
                measurements.append(
                    {
                        "row": row["label"],
                        "step": step,
                        "region": role,
                        **grounding_scores(heat, patch_occupancy(mask, heat.shape)),
                    }
                )
            exported[f"row{row_index:02d}_step{step:02d}"] = heat
    fig.suptitle(title, fontsize=11)
    legend = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2,
            ls="--" if role == "old_target" else "-",
            label=role.replace("_", " ") + (" prism" if role == "old_target" else ""),
        )
        for role, color in COLORS.items()
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.015), ncols=3, fontsize=8
    )
    fig.supxlabel("\n".join(textwrap.wrap(caption, 140)), fontsize=8)
    fig.colorbar(
        im,
        ax=axes,
        shrink=0.6,
        label="Mean native attention probability per patch (all-key denominator)",
    )
    paths = [output.with_suffix(suffix) for suffix in (".png", ".pdf")]
    for path in paths:
        fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    np.savez(output.with_suffix(".npz"), **exported)
    return {
        "figures": [str(path.resolve()) for path in paths],
        "raw_panel_maps": str(output.with_suffix(".npz").resolve()),
        "color_limits": [0.0, vmax],
        "aggregation": "mean over all18layers,8heads,10actionqueries; no per-camera or per-panel renormalization",
        "region_measurements": measurements,
        "caption": caption,
    }


def rollout_panels(trace, output):
    trace, output = Path(trace).resolve(), Path(output).resolve()
    summary = read(trace / "summary.json")
    if not summary.get("complete") or not summary.get("passed") or summary.get("replay_only"):
        raise ValueError("Rollout attention export must be complete and pass native replay")
    output.mkdir(parents=True, exist_ok=False)
    task, policy = summary["task"], summary["policy"]
    frames = [Path(row["output"]) for row in summary["phase_checks"]]
    entries, panels, toolkit = [], {}, []
    for camera in CAMERAS:
        rows = []
        for frame in frames:
            metadata = read(frame / "metadata.json")
            record = metadata["source_record"]
            if metadata["source_observation_sha256"] != sha(frame / "observation.npz") or metadata[
                "source_masks_sha256"
            ] != sha(frame / "masks.npz"):
                raise ValueError("Rollout plot inputs changed after exact action replay")
            with np.load(frame / "observation.npz", allow_pickle=False) as archive:
                image = archive[camera]
            with np.load(frame / "actions.npz", allow_pickle=False) as archive:
                own_path = archive["own_path"]
            masks = camera_mask_arrays(frame / "masks.npz", task, camera)
            cells = []
            for step in STEPS:
                with np.load(
                    frame / f"expert_step{step:02d}_none.npz", allow_pickle=False
                ) as archive:
                    if not np.array_equal(archive["x_t"], own_path[step]):
                        raise ValueError(
                            "Rollout panel does not follow the policy's actual denoising path"
                        )
                cells.append(
                    {
                        "image": image,
                        "masks": masks,
                        "heat": mean_camera_map(frame, metadata, step, camera),
                    }
                )
            label = f"Decision {record['decision_index']}\n" + "\n".join(
                textwrap.wrap(" / ".join(record["phase_names"]).replace("_", " "), 32)
            )
            rows.append({"label": label, "cells": cells})
            if record["decision_index"] == 0:
                toolkit.append(
                    toolkit_overlay(
                        frame,
                        metadata,
                        camera,
                        image,
                        frame / "masks.npz",
                        task,
                        output / f"toolkit_initial_{camera}",
                        step=0,
                    )
                )
            if camera == "image":
                entries.append(
                    {
                        "decision_index": record["decision_index"],
                        "phase_names": record["phase_names"],
                        "source_trace": str(frame),
                        "source_metadata_sha256": sha(frame / "metadata.json"),
                    }
                )
        state_id = read(frames[0] / "metadata.json")["source_record"]["initial_state_id"]
        caption = "Actual rollout-conditioned observations and own-policy latents. Phases follow predefined rules; grasp phases use contact/lift proxies. No attention/outcome filtering. Pixels may differ across policies. These descriptive maps are not causal effects."
        panels[camera] = render_panel(
            rows,
            output / f"phase_attention_{camera}",
            title=f"{task} · {policy} · state {state_id} · {camera}\nActual rollout phases; mean action-expert attention",
            caption=caption,
        )
    result = {
        "complete": True,
        "task": task,
        "policy": policy,
        "state_id": state_id,
        "trace_summary": str(trace / "summary.json"),
        "trace_summary_sha256": sha(trace / "summary.json"),
        "panels": panels,
        "toolkit_overlays": toolkit,
        "phases": entries,
        "absent_phases": sorted(
            {
                "initial_decision",
                "last_decision",
                "closest_requested_target_approach",
                "first_decision_after_sustained_grasp",
                "first_grasped_object_approach_to_basket",
            }
            - {name for entry in entries for name in entry["phase_names"]}
        ),
        "absence_rule": "A phase absent under the predefined contact/distance rule is omitted, never replaced with an attention-selected frame",
        "scope": "rollout-conditioned, not fixed-input comparison",
        "renderer_sha256": sha(__file__),
    }
    write_json(output / "summary.json", result)
    return result


def choose_cases(baseline, candidate, reviewed_wrong_ids=()):
    """Explicit outcome-selected illustrations; never estimates of population frequency."""
    left = {episode_identity(row): row for row in baseline}
    right = {episode_identity(row): row for row in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("Qualitative episode reset/noise identities are not paired")
    recovered = sorted(
        (key for key in left if not left[key]["success"] and right[key]["success"]),
        key=lambda key: key[1],
    )
    unsolved = sorted(
        (key for key in left if not left[key]["success"] and not right[key]["success"]),
        key=lambda key: key[1],
    )
    reviewed = [key for key in recovered if key[1] in reviewed_wrong_ids]
    return {
        "recovered": (reviewed or recovered)[0] if recovered else None,
        "unsolved": unsolved[0] if unsolved else None,
    }


def baseline_review_ready(root, task, report_path, baseline):
    """Do not freeze outcome-selected panels before all baseline failures are reviewed."""
    marker_path = root / "video_reviews" / task / "baseline_test_review_complete.json"
    if not marker_path.exists():
        return False
    marker = read(marker_path)
    if not marker.get("complete"):
        return False
    if marker.get("task") != task or marker.get("report_sha256") != sha(report_path):
        raise ValueError("Baseline review marker task/report identity differs")
    records = []
    for source in marker["review_records"]:
        path = Path(source["path"])
        if not path.is_absolute():
            path = root / path
        if sha(path) != source["sha256"]:
            raise ValueError("Reviewed label artifact changed after review-complete marker")
        records.extend(read(path)["records"])
    covered = {
        (row["state_id"], row["seed"])
        for row in records
        if row.get("reviewed")
        and row["task"] == task
        and Path(row["report"]).resolve() == report_path.resolve()
        and row.get("success") is False
        and row["failure_type"] != "success"
    }
    expected = {
        (row["reset"]["initial_state_id"], row["seed"]) for row in baseline if not row["success"]
    }
    if not expected <= covered:
        raise ValueError("Review-complete marker omits failed baseline test episodes")
    # The common report loader verifies report/video/sheet/evidence-manifest hashes.
    verified = reviewed_labels(root)
    if any(
        (task, str(report_path.resolve()), state, seed) not in verified for state, seed in expected
    ):
        raise ValueError(
            "Baseline review labels are absent from the verified immutable review store"
        )
    return True


def emit_review_requests(root, task, output):
    report_path = evaluation_path(root, task, "pi05_libero", "test")
    destination = output / "review_requests" / f"{task}.json"
    if destination.exists() or not report_path.exists():
        return
    definition = read(root / "tasks" / task / "protocol.json")["tasks"][task]
    episodes = checked_episodes(read(report_path), report_path, definition, "test")
    destination.parent.mkdir(exist_ok=True)
    write_json(
        destination,
        {
            "task": task,
            "report": str(report_path),
            "report_sha256": sha(report_path),
            "scope": "Review every failed baseline test episode before selecting illustrative wrong-object recovery cases",
            "episodes": [
                {
                    "state_id": row["reset"]["initial_state_id"],
                    "seed": row["seed"],
                    "video": row.get("video"),
                    "trajectory": row["trajectory"],
                }
                for row in episodes
                if not row["success"]
            ],
            "required_marker": str(
                root / "video_reviews" / task / "baseline_test_review_complete.json"
            ),
        },
    )


def verify_initial_observation(episode, bank):
    first = min(Path(episode["trajectory"]).glob("chunk_*.npz"))
    with np.load(first, allow_pickle=False) as archive:
        schema = json.loads(str(archive["metadata_json"].item()))["observation_schema"]
        fields = {
            parts[1]: archive[name][0] for name, parts in schema.items() if parts[0] == "openpi"
        }
    for key in ("image", "wrist_image", "state"):
        if not np.array_equal(fields[f"observation/{key}"], bank[key]):
            raise ValueError(f"Outcome-selected episode initial {key} differs from matched bank")
    if str(np.asarray(fields["prompt"]).item()) != str(bank["prompt"].item()):
        raise ValueError("Outcome-selected episode instruction differs from matched bank")
    return {
        "source_chunk": str(first.resolve()),
        "source_chunk_sha256": sha(first),
        "exact_initial_pixels_state_prompt": True,
    }


def fixed_case(root, task, state_id, reports, report_paths, reviews, output, *, case, candidate):
    output.mkdir(parents=True, exist_ok=False)
    bank = root / "observation_bank" / task / "test" / f"state_{state_id:06d}"
    bank_meta = read(bank / "metadata.json")
    if sha(bank / "observation.npz") != bank_meta["observation_sha256"]:
        raise ValueError("Frozen observation bank changed")
    if (
        sha(bank / "masks.npz") != bank_meta["masks_sha256"]
        or sha(bank / "initial_noise.npy") != bank_meta["initial_noise_sha256"]
    ):
        raise ValueError("Frozen mask or diagnostic noise changed")
    with np.load(bank / "observation.npz", allow_pickle=False) as archive:
        observation = {name: archive[name] for name in ("image", "wrist_image", "state", "prompt")}
    noise = np.load(bank / "initial_noise.npy", allow_pickle=False)
    source_records, outcomes = {}, {}
    directories = {policy: root / "traces" / task / policy / bank.name for policy in POLICIES}
    for policy in POLICIES:
        episode = next(
            row for row in reports[policy] if row["reset"]["initial_state_id"] == state_id
        )
        reviewed = reviews.get(
            (task, str(report_paths[policy].resolve()), state_id, episode["seed"])
        )
        outcomes[policy] = {
            "success": episode["success"],
            "seed": episode["seed"],
            "report": str(report_paths[policy]),
            "video": episode.get("video"),
            "trajectory": episode["trajectory"],
            "reviewed_failure_type": reviewed["failure_type"] if reviewed else None,
            "review_source": reviewed["label_source"] if reviewed else None,
        }
        source_records[policy] = verify_initial_observation(episode, observation)
        meta = read(directories[policy] / "metadata.json")
        if meta["bank_record"]["observation_sha256"] != bank_meta["observation_sha256"]:
            raise ValueError("Policy panel did not use the same frozen pixels/state/prompt")
        with np.load(directories[policy] / "actions.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["initial_noise"], noise):
                raise ValueError("Qualitative policies did not use identical diagnostic noise")
    panels, toolkit = {}, []
    for camera in CAMERAS:
        masks = camera_mask_arrays(
            bank / "masks.npz",
            task,
            camera,
            root / "observation_bank_aux" / task / "test" / bank.name / "old_target_region.npz",
        )
        rows = [{"label": "pi05_base", "cells": [None] * len(STEPS)}]
        for policy in POLICIES:
            directory = directories[policy]
            metadata = read(directory / "metadata.json")
            cells = []
            for step in STEPS:
                with (
                    np.load(
                        directory / f"expert_step{step:02d}_none.npz", allow_pickle=False
                    ) as archive,
                    np.load(
                        directories["pi05_libero"] / f"expert_step{step:02d}_none.npz",
                        allow_pickle=False,
                    ) as baseline,
                ):
                    if not np.array_equal(archive["x_t"], baseline["x_t"]) or not np.array_equal(
                        archive["t"], baseline["t"]
                    ):
                        raise ValueError(
                            "Fixed-input qualitative policies do not share latent and time"
                        )
                cells.append(
                    {
                        "image": observation[camera],
                        "masks": masks,
                        "heat": mean_camera_map(directory, metadata, step, camera),
                    }
                )
            outcome = "success" if outcomes[policy]["success"] else "failure"
            rows.append({"label": policy + f"\nPaired rollout: {outcome}", "cells": cells})
            if policy in ("pi05_libero", candidate):
                toolkit.append(
                    toolkit_overlay(
                        directory,
                        metadata,
                        camera,
                        observation[camera],
                        bank / "masks.npz",
                        task,
                        output / f"toolkit_{policy}_{camera}",
                        step=5,
                    )
                )
        caption = (
            f"Outcome-selected illustration ({case} relative to {candidate}); not a representative sample. "
            "All policies share bank pixels/state/prompt, predeclared PCG64 diagnostic noise, baseline latent and flow time. "
            "Outcomes come from paired native JAX-seeded rollouts with different action noise. These offline fixed-input maps are not recorded rollout attention."
        )
        panels[camera] = render_panel(
            rows,
            output / f"fixed_attention_{camera}",
            title=f"{task} · state {state_id} · {case} relative to {candidate} · {camera}\nSame observation and common baseline latent across all action policies",
            caption=caption,
        )
    result = {
        "complete": True,
        "task": task,
        "state_id": state_id,
        "category": case,
        "candidate": candidate,
        "panels": panels,
        "toolkit_overlays": toolkit,
        "outcomes": outcomes,
        "recorded_initial_observation_checks": source_records,
        "diagnostic_noise_sha256": sha(bank / "initial_noise.npy"),
        "noise_protocol": {
            "fixed_input_generator": bank_meta["noise_generator"],
            "fixed_input_noise_seed": bank_meta["noise_seed"],
            "fixed_input_noise_file": str(bank / "initial_noise.npy"),
            "paired_episode_rng": "Native JAX PRNGKey(recorded episode seed), split at each decision; distinct from the shared PCG64 diagnostic bank noise",
            "interpretation": "Same recorded initial pixels/state/prompt do not imply the diagnostic action sample is the recorded rollout action. Actual phase traces separately replay stored native sampling keys.",
        },
        "observation_sha256": bank_meta["observation_sha256"],
        "selection": "Outcome-selected minimum state ID per category/candidate; recovered prioritizes independently video-reviewed baseline wrong-object cases when available. No attention filtering.",
        "scope": "Illustration only; do not infer prevalence or causal grounding from selected panels",
    }
    write_json(output / "summary.json", result)
    return result


def fixed_task(root, task, output):
    if not complete(root / "traces" / task / "summary.json"):
        raise ValueError("Matched trace grid is incomplete")
    selection_path = root / "methods/rldt" / task / "selection.json"
    selection = read(selection_path)
    selection_hash = sha(selection_path)
    if not selection.get("locked") or selection.get("test_results_seen") is not False:
        raise ValueError("Qualitative test examples require locked validation-only selection")
    protocol = read(root / "tasks" / task / "protocol.json")["tasks"][task]
    report_paths = {policy: evaluation_path(root, task, policy, "test") for policy in POLICIES}
    report_hashes = {policy: sha(path) for policy, path in report_paths.items()}
    reports = {
        policy: checked_episodes(read(path), path, protocol, "test")
        for policy, path in report_paths.items()
    }
    if not baseline_review_ready(root, task, report_paths["pi05_libero"], reports["pi05_libero"]):
        raise ValueError("Baseline test failure reviews must finish before qualitative selection")
    review_marker = root / "video_reviews" / task / "baseline_test_review_complete.json"
    review_marker_hash = sha(review_marker)
    reviews = reviewed_labels(root)
    reviewed_wrong = [
        state
        for (row_task, report, state, _), value in reviews.items()
        if row_task == task
        and report == str(report_paths["pi05_libero"].resolve())
        and value["failure_type"] == "wrong object"
    ]
    output.mkdir(parents=True, exist_ok=False)
    cases, absent, unreviewed = [], [], {}
    for candidate in POLICIES[1:]:
        selected = choose_cases(reports["pi05_libero"], reports[candidate], reviewed_wrong)
        for category, identity in selected.items():
            if identity is None:
                absent.append(
                    {
                        "candidate": candidate,
                        "category": category,
                        "reason": "No paired test episode satisfies this outcome rule",
                    }
                )
                continue
            state_id = identity[1]
            destination = output / f"{candidate}_{category}_state_{state_id:06d}"
            result = fixed_case(
                root,
                task,
                state_id,
                reports,
                report_paths,
                reviews,
                destination,
                case=category,
                candidate=candidate,
            )
            for policy, outcome in result["outcomes"].items():
                if not outcome["success"] and outcome["reviewed_failure_type"] is None:
                    unreviewed[policy, state_id] = {
                        "task": task,
                        "policy": policy,
                        "state_id": state_id,
                        **outcome,
                    }
            cases.append(
                {
                    "candidate": candidate,
                    "category": category,
                    "state_id": state_id,
                    "summary": str(destination / "summary.json"),
                    "panels": result["panels"],
                }
            )
    if sha(selection_path) != selection_hash or any(
        sha(path) != report_hashes[policy] for policy, path in report_paths.items()
    ):
        raise ValueError("Selection or paired test report changed while rendering examples")
    if sha(review_marker) != review_marker_hash:
        raise ValueError("Review-complete marker changed while rendering examples")
    result = {
        "complete": True,
        "task": task,
        "cases": cases,
        "absent_categories": absent,
        "selection_sha256": selection_hash,
        "report_sha256": report_hashes,
        "renderer_sha256": sha(__file__),
        "baseline_review_complete_marker": str(review_marker),
        "baseline_review_complete_marker_sha256": review_marker_hash,
        "scope": "Outcome-selected illustrations only; all quantitative analyses use the complete frozen test set",
        "review_status": "Wrong-object labels appear only when immutable video reviews are verified; otherwise failure mechanism remains unreviewed",
        "unreviewed_failure_examples": list(unreviewed.values()),
        "review_complete": not unreviewed,
    }
    write_json(output / "summary.json", result)
    return result


def campaign(root, output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "plan.json",
        {
            "renderer_sha256": sha(__file__),
            "tasks": list(OBJECTS),
            "policies": list(POLICIES),
            "rollout_selection": "all available predefined phases from first five predetermined test episodes; no outcome filtering",
            "fixed_example_selection": "per candidate, minimum-ID recovered and jointly-unsolved cases; reviewed wrong-object recoveries prioritized when available; explicitly outcome-selected",
            "rendering": "mean18layers/8heads/10queries, one raw-probability scale per camera panel across all policies or phases and ODE times",
            "statistics": "Qualitative examples do not replace full-set quantitative comparisons",
            "baseline_review_gate": "video_reviews/<task>/baseline_test_review_complete.json; all failed baseline test episodes reviewed before outcome selection",
        },
    )
    pending = {(task, policy) for task in OBJECTS for policy in POLICIES}
    fixed_pending = set(OBJECTS)
    while pending or fixed_pending:
        for task, policy in sorted(pending):
            source = root / "rollout_traces" / task / policy
            if not complete(source / "summary.json"):
                continue
            records = [
                rollout_panels(Path(bank), output / task / policy / Path(bank).name)
                for bank in read(source / "summary.json")["banks"]
            ]
            write_json(
                output / task / policy / "summary.json",
                {
                    "complete": True,
                    "task": task,
                    "policy": policy,
                    "episodes": len(records),
                    "episode_summaries": [
                        str(
                            output / task / policy / f"state_{row['state_id']:06d}" / "summary.json"
                        )
                        for row in records
                    ],
                },
            )
            pending.remove((task, policy))
        for task in sorted(fixed_pending):
            emit_review_requests(root, task, output)
            if not complete(root / "traces" / task / "summary.json"):
                continue
            if any(not evaluation_path(root, task, policy, "test").exists() for policy in POLICIES):
                continue
            baseline_path = evaluation_path(root, task, "pi05_libero", "test")
            if not baseline_review_ready(
                root, task, baseline_path, read(baseline_path)["episodes"]
            ):
                continue
            fixed_task(root, task, output / "fixed_input_qualitative" / task)
            fixed_pending.remove(task)
        write_json(
            output / "status.json",
            {
                "complete": not pending and not fixed_pending,
                "rollout_pending": sorted(pending),
                "fixed_input_pending": sorted(fixed_pending),
            },
        )
        if pending or fixed_pending:
            time.sleep(30)
    write_json(
        output / "summary.json",
        {
            "complete": True,
            "tasks": list(OBJECTS),
            "policies": list(POLICIES),
            "rollout_figures": "<task>/<policy>/state_ID/phase_attention_<camera>.{png,pdf}",
            "matched_input_figures": "fixed_input_qualitative/<task>/<policy>_<category>_state_ID/fixed_attention_<camera>.{png,pdf}",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument(
        "--rollout",
        type=Path,
        help="Render one complete episode trace, including technical validation traces",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.run) == bool(args.rollout):
        parser.error("Choose exactly one of --run or --rollout")
    if args.rollout:
        rollout_panels(args.rollout, args.output)
    else:
        campaign(args.run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
