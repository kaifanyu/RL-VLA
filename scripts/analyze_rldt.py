"""Plot RLDT iteration metrics and compare matched frozen/trained evaluations."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_records(path):
    records = []
    for line in path.read_text().splitlines(keepends=True):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if line.endswith("\n"):
                raise
            break  # A live writer may not have finished its last record.
    return records


def success_series(records, window=10):
    episodes = np.array([r["episodes"] for r in records])
    successes = np.array([r["successes"] for r in records])
    delta_episodes = np.diff(episodes, prepend=0)
    delta_successes = np.diff(successes, prepend=0)
    recent = []
    for i in range(len(records)):
        start = max(0, i + 1 - window)
        count = delta_episodes[start:i + 1].sum()
        recent.append(delta_successes[start:i + 1].sum() / count if count else np.nan)
    cumulative = np.divide(successes, episodes, out=np.full(len(episodes), np.nan),
                           where=episodes > 0)
    return cumulative, np.asarray(recent)


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count**2)) / denominator
    return [max(0., center - half), min(1., center + half)]


def compare_evaluations(before, after):
    for section in ("policy", "env", "rldt"):
        if before["config"][section] != after["config"][section]:
            raise ValueError(f"Evaluation {section} configurations differ")
    left, right = before["episodes"], after["episodes"]
    identity = lambda rows: [(r["seed"], r["reset"]) for r in rows]
    if not left or identity(left) != identity(right):
        raise ValueError("Evaluations must have identical nonempty seed/reset sequences")
    a = np.array([r["success"] for r in left], dtype=bool)
    b = np.array([r["success"] for r in right], dtype=bool)
    step_a = np.array([r["primitive_steps"] for r in left])
    step_b = np.array([r["primitive_steps"] for r in right])
    both = a & b
    return {
        "episodes": len(left),
        "unique_initial_states": len({r["reset"]["initial_state_id"] for r in left}),
        "frozen_success_rate": float(a.mean()), "trained_success_rate": float(b.mean()),
        "change_percentage_points": float(100 * (b.mean() - a.mean())),
        "frozen_wilson_95_interval": wilson(int(a.sum()), len(a)),
        "trained_wilson_95_interval": wilson(int(b.sum()), len(b)),
        "new_successes": int((~a & b).sum()), "lost_successes": int((a & ~b).sum()),
        "both_succeed": int(both.sum()), "both_fail": int((~a & ~b).sum()),
        "mean_steps_frozen": float(step_a.mean()),
        "mean_steps_trained": float(step_b.mean()),
        "mean_step_change_on_paired_successes": (
            float((step_b[both] - step_a[both]).mean()) if both.any() else None),
        "scope": "Matched reset/noise seeds on the training task, not held-out initial states.",
        "interval_caveat": "Episode-level intervals assume independent trials; initial states "
                           "can repeat, and this is one training seed.",
    }


def analyze(run, window=10):
    training = run / "training" if (run / "training").is_dir() else run
    records = load_records(training / "metrics.jsonl")
    if not records:
        raise ValueError("No completed training iteration yet")
    config = json.loads((training / "config.json").read_text())
    if config["algorithm"] != "rldt":
        raise ValueError("Expected an RLDT run")
    cumulative, recent = success_series(records, window)
    last = records[-1]
    report = {
        "training": str(training.resolve()),
        "training_complete": (training / "summary.json").exists(),
        "iteration": last["iterations"], "planned_iterations": config["train"]["iterations"],
        "decisions": last["decisions"], "primitive_steps": last["primitive_steps"],
        "episodes": last["episodes"], "successes": last["successes"],
        "actor_updates": last["actor_updates"], "critic_updates": last["critic_updates"],
        "training_success_rate": float(cumulative[-1]) if np.isfinite(cumulative[-1]) else None,
        "recent_training_success_rate": float(recent[-1]) if np.isfinite(recent[-1]) else None,
        "recent_window_iterations": window,
        "all_logged_metrics_finite": all(np.isfinite(v) for r in records
                                         for v in r["metrics"].values()),
        "last_metrics": last["metrics"],
        "metrics_note": "Losses are the last update in each iteration, not iteration averages. "
                        "Training success mixes changing policies and is diagnostic only.",
    }
    before, after = run / "evaluation_frozen.json", run / "evaluation_trained.json"
    if before.exists() and after.exists():
        report["evaluation"] = compare_evaluations(json.loads(before.read_text()),
                                                    json.loads(after.read_text()))
    temporary = run / "analysis.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(run / "analysis.json")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    x = np.array([r["iterations"] for r in records])
    plt.rcParams.update({"lines.marker": ".", "lines.markersize": 3})
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(x, cumulative, label="cumulative")
    axes[0].plot(x, recent, label=f"last {window} iterations")
    axes[0].set_ylim(-.05, 1.05)
    axes[0].set_title("Training success / finished episodes")
    panels = [
        (1, "Critic loss", ["critic_loss"]),
        (2, "Actor objective", ["actor_loss", "rl_loss"]),
        (3, "Regularizers (unweighted)", ["consistency_loss", "constraint_loss"]),
        (4, "Actor gradient and parameter update", ["actor_grad_norm", "actor_update_norm"]),
        (5, "Mean critic target", ["target_mean"]),
        (6, "Transport components", ["attraction_norm", "repulsion_norm"]),
    ]
    for index, title, keys in panels:
        axes[index].set_title(title)
        for key in keys:
            values = [r["metrics"].get(key, np.nan) for r in records]
            axes[index].plot(x, values, label=key)
    axes[7].plot(x, np.diff([r["elapsed_seconds"] for r in records], prepend=0), label="seconds")
    axes[7].set_title("Iteration duration (first includes setup/JIT)")
    for key in ("actor_updates", "critic_updates"):
        axes[8].plot(x, [r[key] for r in records], label=key)
    axes[8].set_title("Cumulative optimizer updates")
    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"RLDT · {run.name} · {last['iterations']}/"
                 f"{config['train']['iterations']} iterations · {last['episodes']} episodes")
    temporary_plot = run / "progress.tmp.png"
    fig.savefig(temporary_plot, dpi=130)
    plt.close(fig)
    temporary_plot.replace(run / "progress.png")
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", type=int, default=10, help="Recent success window in iterations")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    analyze(args.run, args.window)
