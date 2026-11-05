"""Build figures and a compact report from completed bottleneck diagnostics."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = ("object_0", "object_1", "object_2")
MODELS = ("pi05_base", "pi05_libero")


def read(path):
    return json.loads(path.read_text())


def head_maps(root, figures):
    matrices = []
    for task in TASKS:
        cells = defaultdict(list)
        with (root / "rank" / task / "heads.jsonl").open() as stream:
            for line in stream:
                if (
                    '"step": 5' not in line
                    or '"key_group": "visual"' not in line
                    or '"policy": "rldt_000100"' not in line
                ):
                    continue
                row = json.loads(line)
                cells[row["layer"], row["head"]].append(
                    (row["reference_spectrum"]["effective_rank"], row["effective_rank_delta"])
                )
        shape = (1 + max(k[0] for k in cells), 1 + max(k[1] for k in cells))
        baseline, delta = np.full(shape, np.nan), np.full(shape, np.nan)
        for key, values in cells.items():
            baseline[key], delta[key] = np.mean(values, axis=0)
        matrices.append((baseline, delta))
    limit = max(float(np.nanmax(np.abs(delta))) for _, delta in matrices)
    fig, axes = plt.subplots(2, 3, figsize=(11, 8), layout="constrained")
    for column, (task, (baseline, delta)) in enumerate(zip(TASKS, matrices, strict=True)):
        top = axes[0, column].imshow(baseline, vmin=1, vmax=10, aspect="auto", cmap="viridis")
        bottom = axes[1, column].imshow(
            delta, vmin=-limit, vmax=limit, aspect="auto", cmap="RdBu_r"
        )
        axes[0, column].set_title(task.replace("_", " ").title())
        for axis in axes[:, column]:
            axis.set(
                xlabel="Attention head (zero based)",
                ylabel="Transformer layer (zero based)",
                xticks=np.arange(baseline.shape[1]),
            )
    fig.colorbar(top, ax=axes[0].tolist(), label="Baseline effective rank")
    fig.colorbar(bottom, ax=axes[1].tolist(), label="Iteration 100 − baseline rank")
    fig.suptitle(
        "Visual attention rank by layer and head\nODE index 5; each cell averages the same 50 observations"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"attention_head_ranks.{suffix}", dpi=160)
    plt.close(fig)


def run(root):
    root = Path(root).resolve()
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    probe_rows = []
    for task in TASKS:
        for model in MODELS:
            result = read(root / "probes" / task / model / "summary.json")
            if not result["complete"]:
                raise ValueError("Incomplete probes")
            for row in result["results"]:
                probe_rows.append({"task": task, "model": model, **row})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, layout="constrained")
    colors = ("#1f77b4", "#d55e00", "#555555")
    for i, model in enumerate(MODELS):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            for k, (kind, control, label) in enumerate(
                (
                    ("image_features", "features", "Image embeddings"),
                    ("prefix_features", "features", "VLM final prefix"),
                    ("image_features", "position_only", "Position only"),
                )
            ):
                selected = [
                    next(
                        r
                        for r in probe_rows
                        if r["task"] == task
                        and r["model"] == model
                        and r["features"] == kind
                        and r["control"] == control
                        and r["split"] == split
                    )
                    for split in ("validation", "source")
                ]
                values = np.array([r["balanced_accuracy"] for r in selected])
                intervals = np.array([r["balanced_accuracy_state_bootstrap95"] for r in selected])
                x = np.arange(2) + (k - 1) * 0.24
                ax.bar(
                    x,
                    values,
                    0.22,
                    color=colors[k],
                    label=label,
                    yerr=np.maximum(
                        0, np.stack([values - intervals[:, 0], intervals[:, 1] - values])
                    ),
                    capsize=3,
                )
            ax.axhline(1 / 3, ls=":", color="gray", lw=1, label="3-class chance")
            ax.set(
                xticks=[0, 1],
                xticklabels=["Swap validation", "Original layout"],
                title=f"{task.replace('_', ' ').title()} · {model}",
                ylim=(0, 1.05),
            )
            if j == 0:
                ax.set_ylabel("Balanced accuracy")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "A position-only probe solves the swap-layout labels\n100 training states; 50 evaluation states per layout; state-bootstrap 95% intervals"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"probe_controls.{suffix}", dpi=160)
    plt.close(fig)

    rank, missing = {}, []
    for task in TASKS:
        path = root / "rank" / task / "summary.json"
        if path.exists() and read(path).get("complete"):
            rank[task] = read(path)
        else:
            missing.append(task)
    if len(rank) == 3:
        head_maps(root, figures)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
        for column, task in enumerate(TASKS):
            summary = rank[task]["summaries"]

            def cell(policy, step, metric, summary=summary):
                return next(
                    r
                    for r in summary
                    if r["policy"] == policy
                    and r["step"] == step
                    and r["key_group"] == "visual"
                    and r["metric"] == metric
                )

            for step, color in zip((0, 5, 9), ("#1f77b4", "#d55e00", "#009e73"), strict=True):
                policies = ("rldt_000025", "rldt_000050", "rldt_000100")
                baseline = cell(policies[0], step, "reference_effective_rank")
                rows = [baseline, *[cell(p, step, "candidate_effective_rank") for p in policies]]
                axes[0, column].plot(
                    [0, 25, 50, 100],
                    [r["mean"] for r in rows],
                    "o-",
                    color=color,
                    label=f"ODE index {step}",
                )
                rows = [cell(p, step, "relative_frobenius_delta") for p in policies]
                axes[1, column].plot(
                    [0, 25, 50, 100], [0, *[r["mean"] for r in rows]], "o-", color=color
                )
            axes[0, column].set_title(task.replace("_", " ").title())
            axes[0, column].set_ylabel("Visual attention effective rank")
            axes[1, column].set(xlabel="RLDT iteration", ylabel="Relative Frobenius drift")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            "Action → visual-prefix attention: rank and drift\nMeans over 50 matched states, 18 layers and 8 heads; rank ceiling = 10"
        )
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"attention_rank_drift.{suffix}", dpi=160)
        plt.close(fig)

    lines = [
        "# VLM / action-policy diagnostic results",
        "",
        (
            "The current evidence does not isolate the action policy as the sole failure source. "
            "A camera/patch-position-only probe almost perfectly solves swap-layout validation, "
            "so high in-layout object-decoding accuracy is not evidence that the VLM recognizes object identity."
        ),
        "",
        "## Frozen-feature probes",
        "",
        (
            "100 train states; 50 validation and 50 original-layout states per task/model. "
            "Percentages below are balanced accuracy across background/target/distractor. "
            "Each original-layout result uses the same probe fitted on swap-layout training only."
        ),
        "",
        "| Task | Model | Image: validation → original | VLM prefix: validation → original | Position: validation → original |",
        "|---|---|---:|---:|---:|",
    ]
    for task in TASKS:
        for model in MODELS:
            pairs = []
            for kind, control in (
                ("image_features", "features"),
                ("prefix_features", "features"),
                ("image_features", "position_only"),
            ):
                values = [
                    next(
                        r["balanced_accuracy"]
                        for r in probe_rows
                        if r["task"] == task
                        and r["model"] == model
                        and r["features"] == kind
                        and r["control"] == control
                        and r["split"] == split
                    )
                    for split in ("validation", "source")
                ]
                pairs.append(f"{values[0]:.1%} → {values[1]:.1%}")
            lines.append(f"| {task} | {model} | " + " | ".join(pairs) + " |")
    lines += [
        "",
        "![Probe controls](figures/probe_controls.png)",
        "",
        (
            "Three shuffled-label fits per feature kind are in each probe summary. They are negative "
            "controls, not a calibrated permutation p-value. Cross-layout probe failure cannot by itself "
            "prove that semantic information is absent, especially with a fixed low-dimensional projection."
        ),
        "",
        "## Attention rank and drift",
        "",
    ]
    # Keep the negative controls visible alongside the apparently positive probes.
    source_controls = [
        r
        for r in probe_rows
        if r["task"] == "object_2"
        and r["model"] == "pi05_libero"
        and r["features"] == "prefix_features"
        and r["split"] == "source"
    ]
    observed = next(r["balanced_accuracy"] for r in source_controls if r["control"] == "features")
    shuffled = [
        r["balanced_accuracy"] for r in source_controls if r["control"].startswith("shuffled_")
    ]
    lines[-2:-2] = [
        (
            f"For Object 2 / pi05_libero final-prefix transfer, the true-label probe scores {observed:.1%}; "
            f"the three shuffled-label fits span {min(shuffled):.1%}–{max(shuffled):.1%}. "
            "This probe does not establish robust transferable identity decoding. "
            "It also does not prove that the VLM lacks that information."
        ),
        "",
    ]
    if missing:
        lines += ["Pending: " + ", ".join(missing) + "."]
    else:
        count = sum(r["records"] for r in rank.values())
        lines += [
            (
                f"Completed {count:,} layer/head/key-group records: three tasks × 50 states × "
                "three checkpoints × three flow indices × 18 layers × eight heads × five key groups."
            ),
            "",
            "| Task | Baseline rank | Iteration 25 | Iteration 50 | Iteration 100 |",
            "|---|---:|---:|---:|---:|",
        ]
        for task in TASKS:
            selected = [
                r for r in rank[task]["summaries"] if r["step"] == 5 and r["key_group"] == "visual"
            ]
            base = next(r["mean"] for r in selected if r["metric"] == "reference_effective_rank")
            values = [
                next(
                    r["mean"]
                    for r in selected
                    if r["metric"] == "candidate_effective_rank" and r["policy"] == p
                )
                for p in ("rldt_000025", "rldt_000050", "rldt_000100")
            ]
            lines.append(
                f"| {task} | {base:.3f} | " + " | ".join(f"{v:.3f}" for v in values) + " |"
            )
        lines += [
            "",
            (
                "The table uses ODE index 5 (t=0.5) and averages per-head effective ranks. "
                "It is not the rank of an averaged heatmap. Rank is bounded by ten action queries; "
                "neither a low rank nor a higher rank establishes semantic competence. "
                "These are initial-decision observations; flow indices are internal denoising times, "
                "not later rollout frames."
            ),
            "",
            "![Attention rank and drift](figures/attention_rank_drift.png)",
            "",
            "![Attention rank by layer and head](figures/attention_head_ranks.png)",
            "",
            (
                "Tiny checkpoint differences must be compared against the campaign's compiler controls. "
                "Original per-layer/head spectra, absolute mass, zero-mass exclusions and query-conditional "
                "JS remain in `rank/<task>/heads.jsonl`; state-bootstrap intervals are in its summary."
            ),
        ]
    value_complete = all((root / "value_transport" / t / "summary.json").exists() for t in TASKS)
    compiler_path = root / "compiler_controls" / "summary.json"
    compiler_complete = compiler_path.exists() and read(compiler_path).get("complete", False)
    if compiler_complete:
        lines += [
            "",
            "## Compiler-only control",
            "",
            (
                "Same baseline weights under the original two compiler profiles, first five states, "
                "ODE index 5. These are descriptive controls; nonlinear drift distances must not be subtracted."
            ),
            "",
            "| Task | Relative attention drift | Effective-rank difference |",
            "|---|---:|---:|",
        ]
        for row in read(compiler_path)["summaries"]:
            if row["step"] == 5:
                lines.append(
                    f"| {row['task']} | {row['relative_frobenius_delta']:.4f} | {row['effective_rank_delta']:+.6f} |"
                )
    lines += [
        "",
        "## Hidden-value interface",
        "",
        (
            "Completed A@V analysis for the first five predetermined states in each task, all three "
            "checkpoints and flow indices. Exact saved native KV fingerprints are verified. This is "
            "float64 reconstruction before projection, residuals and MLPs; it does not claim native "
            "BF16 replay or causal sufficiency."
            if value_complete
            else "Value transport analysis is pending."
        ),
    ]
    if value_complete:
        value_groups = defaultdict(list)
        for task in TASKS:
            with (root / "value_transport" / task / "heads.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["step"] == 5 and row["key_group"] == "visual":
                        value_groups[task, row["policy"]].append(row)
        lines += [
            "",
            "Mean visual A@V contribution metrics at ODE index 5 (five states; no inference over heads).",
            "",
            "| Task | Checkpoint | Reference rank | Candidate rank | Relative A@V drift | Query CKA |",
            "|---|---|---:|---:|---:|---:|",
        ]
        value_summary = []
        for (task, policy), rows in sorted(value_groups.items()):
            a = np.mean([r["reference_spectrum"]["effective_rank"] for r in rows])
            b = np.mean([r["candidate_spectrum"]["effective_rank"] for r in rows])
            drift = np.mean(
                [
                    r["relative_frobenius_delta"]
                    for r in rows
                    if r["relative_frobenius_delta"] is not None
                ]
            )
            cka = np.mean(
                [r["query_linear_cka"] for r in rows if r["query_linear_cka"] is not None]
            )
            lines.append(f"| {task} | {policy} | {a:.3f} | {b:.3f} | {drift:.3f} | {cka:.4f} |")
            value_summary.append(
                {
                    "task": task,
                    "policy": policy,
                    "reference_rank": float(a),
                    "candidate_rank": float(b),
                    "relative_drift": float(drift),
                    "query_cka": float(cka),
                }
            )
        (root / "value_transport_summary.json").write_text(
            json.dumps(value_summary, indent=2, allow_nan=False) + "\n"
        )
    lines += [
        "",
        "## What is still needed",
        "",
        (
            "Use identity-balanced scenes with held-out positions and paired noun/relation instructions. "
            "Probe each layer's KV cache in addition to the final VLM hidden state. Then test a controlled "
            "KV/representation-patching rescue and evaluate closed-loop actions with the VLM frozen. "
            "The present data concerns familiar categories in altered layouts; it is not a novel-category "
            "or reasoning benchmark. All results here are exploratory reanalysis, with no new policy "
            "training, rollout, or checkpoint selection."
        ),
        "",
        (
            "See `docs/vlm-policy-bottleneck.md` for implementation, formulas, commands, causal criteria, "
            "and VLA-Trace/Act2Answer/planning-token/value-probing references."
        ),
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    (root / "report_status.json").write_text(
        json.dumps(
            {
                "complete": not missing and value_complete and compiler_complete,
                "rank_pending": missing,
                "value_complete": value_complete,
                "compiler_complete": compiler_complete,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
