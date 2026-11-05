"""2x2 summary figure for one FlashSAC residual run (validation, training rollouts, held-out test).

Usage: .venv/bin/python scripts/plot_flashsac_run.py runs/flashsac_spatial_01_rldt_states [--eval-id test_best_osmesa]
Writes <run>/summary_plot.png.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREEN, RED, GRAY, LIGHT = "#1a7f64", "#c04a36", "#b8bfc9", "#dde1e6"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--eval-id", default="test_best_osmesa")
    p.add_argument("--block", type=int, default=20, help="training episodes per bar")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    run = a.run

    val = read_csv(run / "logs/validation_metrics.csv")
    eps = read_csv(run / "logs/train_episodes.csv")
    summary = json.loads((run / "logs/train_summary.json").read_text())
    test = json.loads((run / "evaluation_metrics.json").read_text())[a.eval_id]
    cfg = json.loads((run / "config/resolved_config.json").read_text())
    task = cfg.get("config", cfg).get("task", {})
    label = task.get("label") or f"{task.get('suite', '').replace('libero_', '').title()} {task.get('task_id', '')}"
    task_name = json.loads((run / "config/split_manifest.json").read_text())["task_name"].replace("_", " ")

    res = sorted((int(r["env_steps"]), 100 * float(r["success_rate"]), int(r["successes"]), int(r["episodes"]))
                 for r in val if r["variant"] == "residual")
    base_val = next(r for r in val if r["variant"] == "base")
    best_step = summary["best"]["env_steps"]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.8))
    n_val = int(base_val["episodes"])
    fig.suptitle(
        f"LIBERO-PRO {label} {task.get('category', '')} · {task_name}\n"
        f"FlashSAC residual on frozen pi05 · one training seed · {n_val} states per evaluation split",
        fontsize=15)

    # 1. Validation over env steps
    ax = axes[0, 0]
    style(ax)
    xs = [s for s, *_ in res]
    ax.plot(xs, [r for _, r, *_ in res], "-o", color=GREEN, lw=2, ms=8, label="FlashSAC residual")
    ax.axhline(100 * float(base_val["success_rate"]), color=GRAY, lw=2, ls="--",
               label=f"Frozen base ({base_val['successes']}/{n_val})")
    for s, r, k, n in res:
        if s == best_step:
            ax.plot(s, r, "o", ms=14, mfc="none", mec=GREEN, mew=2)
            ax.annotate(f"{k}/{n} · selected", (s, r), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=11)
    renderer = cfg.get("config", cfg).get("simulator", {}).get("renderer", "osmesa")
    ax.set(title=f"Validation over training ({renderer} renderer)", xlabel="Environment steps", ylabel="Success (%)",
           ylim=(0, 100), xticks=xs)
    ax.set_xticklabels([f"{x:,}" for x in xs])
    ax.legend(loc="upper right")

    # 2. Training rollout success per block of episodes
    ax = axes[0, 1]
    style(ax)
    ax.grid(axis="x", visible=False)
    labels, rates, counts = [], [], []
    for i in range(0, len(eps), a.block):
        blk = eps[i:i + a.block]
        k = sum(r["success"] in ("True", "1") for r in blk)
        labels.append(f"{int(blk[0]['episode']) + 1}–{int(blk[-1]['episode']) + 1}")
        rates.append(100 * k / len(blk))
        counts.append(f"{k}/{len(blk)}")
    bars = ax.bar(labels, rates, color=GREEN, width=0.6)
    ax.bar_label(bars, counts, padding=3, fontsize=11)
    ax.set(title="Training rollouts (stochastic policy)", xlabel="Training episodes", ylabel="Success (%)",
           ylim=(0, 100))

    # 3. Held-out test with Wilson intervals
    ax = axes[1, 0]
    style(ax)
    ax.grid(axis="x", visible=False)
    names, colors = ["Frozen base", "Base + residual"], [GRAY, GREEN]
    vs = [test["variants"]["base"], test["variants"]["residual"]]
    hs = [100 * v["success_rate"] for v in vs]
    err = [[h - 100 * v["success_95pct_wilson"][0] for h, v in zip(hs, vs)],
           [100 * v["success_95pct_wilson"][1] - h for h, v in zip(hs, vs)]]
    bars = ax.bar(names, hs, color=colors, width=0.55, yerr=err, capsize=6,
                  error_kw={"elinewidth": 1.5, "ecolor": "#555"})
    for b, v in zip(bars, vs):
        ax.annotate(f"{v['successes']}/{v['episodes']}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(10, 4), textcoords="offset points", ha="left", fontsize=11)
    po = test["paired_outcomes"]
    b, c = po["residual_only_succeeds"], po["base_only_succeeds"]
    from math import comb
    pval = min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))
    ax.set(title=f"Held-out test: selected checkpoint (step {best_step:,}, osmesa)", ylabel="Success (%)",
           ylim=(0, 100))
    ax.text(0.98, 0.95, f"95% Wilson intervals\nexact McNemar p = {pval:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#444")

    # 4. Paired outcomes on the test states
    ax = axes[1, 1]
    style(ax)
    ax.grid(axis="x", visible=False)
    cats = ["Gained\n(only base+residual)", "Lost\n(only base)", "Both succeed", "Both fail"]
    vals = [b, c, po["both_succeed"], po["both_fail"]]
    bars = ax.bar(cats, vals, color=[GREEN, RED, GRAY, LIGHT], width=0.6)
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set(title="Held-out test: paired outcomes per state", ylabel="States",
           ylim=(0, max(vals) * 1.2))

    fig.tight_layout()
    out = a.out or run / "summary_plot.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
