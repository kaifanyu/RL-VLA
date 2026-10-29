"""Summarize existing PRO evaluations without loading policies or simulators."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

TASKS = ("spatial_1", "spatial_2", "object_0")
METHODS = ("rldt", "flow_sde", "flow_noise")
LABELS = {"rldt": "RLDT", "flow_sde": "Flow-SDE PPO", "flow_noise": "Flow-Noise PPO"}
CAVEAT = (
    "One training seed, three fixed conditions. Compare each method's paired before/after "
    "change: Torch and JAX use different sampling RNGs. Intervals quantify paired episode "
    "resampling, not variability across training seeds. Pending/failed evaluations are missing, "
    "never zero. Controlled training steps exclude evaluations and reset settling."
)
SPLIT_KEYS = (
    "episodes", "unique_initial_states", "frozen_success_rate", "trained_success_rate",
    "change_percentage_points", "new_successes", "lost_successes", "both_succeed", "both_fail",
    "paired_bootstrap_95_change_percentage_points", "mean_steps_frozen", "mean_steps_trained",
    "mean_step_change_on_paired_successes", "scope", "interval_caveat",
)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def selected_interactions(directory, iteration):
    """Read actual cumulative steps at selection; never extrapolate from the budget."""
    if iteration is None:
        return None
    for name in ("iteration_metrics.jsonl", "metrics.jsonl"):
        path = directory / "training" / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A writer may be appending the newest line during a manual refresh.
                continue
            if row.get("iteration", row.get("iterations")) == iteration:
                return row.get("primitive_steps")
    return None


def build_comparison(suite):
    """Keep the original results schema and add explicit states for every planned run."""
    suite = Path(suite)
    progress = read_json(suite / "pi_rl_queue/progress.json")
    failed = {(item["task"], item["method"]): item for item in progress.get("failures", [])}
    experiments, results = [], []
    for task in TASKS:
        for method in METHODS:
            directory = suite / "methods" / method / task
            path = directory / "comparison.json"
            comparison = read_json(path)
            summary = read_json(directory / "training/summary.json")
            current = read_json(directory / "status.json")
            failure = failed.get((task, method))
            complete = all(split in comparison for split in ("test", "source"))
            state = ("failed" if failure or current.get("state") == "failed" else
                     "completed" if complete else
                     "partial" if comparison else current.get("state", "pending"))
            # Completion of training alone must not imply completed paired evaluations.
            if state == "completed" and not complete:
                state = "partial"
            selection = read_json(directory / "selection.json")
            iteration = comparison.get("selected_iteration", selection.get("iteration"))
            row = {
                "method": method, "task": task, "status": state,
                "stage": current.get("stage"), "selected_iteration": iteration,
                "total_training_primitive_steps": summary.get("primitive_steps"),
                "selected_training_primitive_steps": selected_interactions(directory, iteration),
                "total_training_decisions": summary.get("decisions"),
                "training_elapsed_seconds": summary.get("elapsed_seconds"),
                "path": str(path) if path.exists() else None,
                "experiment_directory": str(directory),
                "evaluation_rng_backend": "jax" if method == "rldt" else "torch",
                "test": None, "source": None,
            }
            if failure:
                row["exit_code"] = failure.get("exit_code")
            if current.get("error"):
                row["error"] = current["error"]
            for split in ("test", "source"):
                if split in comparison:
                    row[split] = {key: comparison[split][key] for key in SPLIT_KEYS
                                  if key in comparison[split]}
            experiments.append(row)
            if path.exists():
                # Existing consumers iterate only available comparison reports.
                results.append({key: value for key, value in row.items()
                                if key not in ("test", "source") or value is not None})
    return {
        "results": results, "experiments": experiments, "caveat": CAVEAT,
        "generated_at": datetime.now(UTC).isoformat(),
        "planned_experiments": len(TASKS) * len(METHODS),
        "completed_experiments": sum(row["status"] == "completed" for row in experiments),
        "failed_experiments": sum(row["status"] == "failed" for row in experiments),
    }


def plot_comparison(report, path):
    """Plot within-method paired changes, with no plotted values for missing runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {"rldt": "#2475ad", "flow_sde": "#d27b15", "flow_noise": "#258653"}
    rows = {(row["task"], row["method"]): row for row in report["experiments"]}
    bounds = [0]
    for row in rows.values():
        for split in ("test", "source"):
            metric = row[split]
            if metric:
                bounds.append(metric["change_percentage_points"])
                bounds.extend(metric.get("paired_bootstrap_95_change_percentage_points", []))
    lower, upper = min(-10, min(bounds) - 8), max(10, max(bounds) + 8)
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for index, task in enumerate(TASKS):
        for column, split in enumerate(("test", "source")):
            ax = axes[index, column]
            ax.axvline(0, color="#919191", linewidth=1, linestyle="--")
            for method_index, method in enumerate(METHODS):
                y = 2 - method_index
                row, metric = rows[(task, method)], rows[(task, method)][split]
                if metric is None:
                    message = row["status"].replace("_", " ").capitalize()
                    split_label = "test" if split == "test" else "retention"
                    ax.text(.5, y, f"{message} — no {split_label} result", color="#777777",
                            ha="center", va="center", transform=ax.get_yaxis_transform())
                    continue
                change = metric["change_percentage_points"]
                interval = metric.get("paired_bootstrap_95_change_percentage_points")
                if interval:
                    ax.hlines(y, interval[0], interval[1], color=palette[method], linewidth=2)
                    ax.plot(interval, [y, y], "|", color=palette[method], markersize=8)
                ax.plot(change, y, "o", color=palette[method], markersize=7)
                ax.annotate(f"{change:+g} pp", (change, y), xytext=(0, 9),
                            textcoords="offset points", ha="center", fontsize=9)
                baseline, trained = (100 * metric[key] for key in
                                     ("frozen_success_rate", "trained_success_rate"))
                details = (f"{baseline:g}% → {trained:g}%  |  "
                           f"{metric['new_successes']} recovered, {metric['lost_successes']} lost")
                ax.text(.02, y - .30, details, transform=ax.get_yaxis_transform(), fontsize=8)
            title = task.replace("_", " ").title()
            title += " · held-out PRO test" if split == "test" else " · original-task retention"
            ax.set(title=title, xlim=(lower, upper), ylim=(-.55, 2.65),
                   yticks=[2, 1, 0], yticklabels=[LABELS[method] for method in METHODS])
            ax.grid(axis="x", alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            if index == 2:
                ax.set_xlabel("Trained − own baseline success (percentage points)")
    budgets = []
    for task in TASKS:
        entries = []
        for method in METHODS:
            row = rows[(task, method)]
            if row["selected_iteration"] is None:
                continue
            selected, total = (row[key] for key in
                               ("selected_training_primitive_steps", "total_training_primitive_steps"))
            steps = f"{selected:,}" if selected is not None else "unreported"
            total_text = f"{total:,}" if total is not None else "unreported"
            entries.append(f"{LABELS[method]}: iter {row['selected_iteration']}, "
                           f"{steps} / {total_text} steps")
        budgets.append(f"{task.replace('_', ' ').title()} — " + ("; ".join(entries) or "pending"))
    fig.suptitle("RLDT vs official π-RL: paired adaptation effects", fontsize=15)
    footer = ("Dots: paired changes; whiskers: 95% paired bootstrap intervals. "
              "One training seed; Torch/JAX RNGs differ.\n"
              "Measured controlled training steps below: selected checkpoint / full run.\n"
              + "\n".join(budgets))
    fig.text(.02, .015, footer, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, .14, 1, .96), h_pad=2)
    temporary = Path(path).with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(path)


def write_comparison(suite):
    suite = Path(suite)
    report = build_comparison(suite)
    temporary = suite / "comparison_matrix.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(suite / "comparison_matrix.json")
    plot_comparison(report, suite / "comparison_matrix.png")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    write_comparison(args.suite.resolve())
