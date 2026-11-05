"""Failure analysis and training diagnostics for one FlashSAC residual run.

Usage: .venv/bin/python scripts/analyze_flashsac_run.py runs/flashsac_spatial_01_rldt_states [--eval-id test_best_osmesa]
Writes <run>/failure_analysis.png, <run>/training_diagnostics.png and <run>/failure_episodes.csv.

Failure modes come from the logged object poses (logging only, never a policy input). The target
and goal objects are read from the case BDDL goal, e.g. (On akita_black_bowl_1 plate_1) or
(In alphabet_soup_1 basket_1_contain_region). Thresholds are heuristics for sorting episodes,
not task predicates: check a few videos per mode before quoting the labels.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREEN, RED, GRAY, LIGHT = "#1a7f64", "#c04a36", "#b8bfc9", "#dde1e6"
MODES = ["success", "wrong object", "never lifted", "held at timeout", "dropped", "at goal, not placed"]
MODE_COLORS = [GREEN, "#6250d6", "#eda100", "#2a78d6", "#eb6834", "#e87ba4"]
PAIRS = ["gained", "lost", "both succeed", "both fail"]
PAIR_COLORS = [GREEN, RED, GRAY, "#8a919c"]

LIFT_M = 0.03       # object counts as lifted when 3 cm above its settled height
NEAR_GOAL_M = 0.10  # xy distance to the goal object's center


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)


def goal_objects(bddl: str) -> tuple[str, str]:
    text = Path(bddl).read_text()
    goal = text[text.index("(:goal"):]
    m = re.search(r"\((On|In)\s+(\S+)\s+(\S+?)\)", goal)
    if not m:
        raise ValueError(f"Unsupported goal in {bddl}")
    return m[2], re.sub(r"_contain_region$", "", m[3])


def load_episodes(run: Path, eval_id: str) -> dict:
    episodes = {}
    for path in sorted((run / "metadata").glob(f"*{eval_id}-init*.json")):
        meta = json.loads(path.read_text())
        if meta.get("eval_id") != eval_id:
            continue
        init = int(re.search(r"init(\d+)$", meta["episode_id"])[1])
        files = {k: run / ("logs" if k.endswith("csv") else "videos") / Path(v).name
                 for k, v in meta["files"].items() if k.endswith("csv")}
        episodes.setdefault(init, {})[meta["policy"]] = {"meta": meta, **files}
    return {i: e for i, e in episodes.items() if {"base", "residual"} <= set(e)}


def episode_features(ep: dict, target: str, goal: str) -> dict:
    objects = read_csv(ep["objects_csv"])
    steps = read_csv(ep["steps_csv"])
    names = sorted({c[:-2] for c in objects[0] if c.endswith("_z")})
    z = {n: np.array([float(r[f"{n}_z"]) for r in objects]) for n in names}
    xy = {n: np.array([[float(r[f"{n}_x"]), float(r[f"{n}_y"])] for r in objects]) for n in (target, goal)}
    dist = np.linalg.norm(xy[target] - xy[goal], axis=1)
    lifted = z[target] - z[target][0] > LIFT_M
    lift_step = int(np.argmax(lifted)) if lifted.any() else None
    others = [n for n in names if n not in (target, goal)]
    first_other = min((int(np.argmax(z[n] - z[n][0] > LIFT_M)) for n in others if (z[n] - z[n][0] > LIFT_M).any()),
                      default=None)
    wrong = first_other is not None and (lift_step is None or first_other < lift_step)
    outcome = ep["meta"]["outcome"]
    if outcome["success"]:
        mode = "success"
    elif wrong:
        mode = "wrong object"
    elif lift_step is None:
        mode = "never lifted"
    elif lifted[-1]:
        mode = "held at timeout"
    elif dist[-1] < NEAR_GOAL_M:
        mode = "at goal, not placed"
    else:
        mode = "dropped"
    corr = np.array([[float(r[f"correction_{a}"]) for a in "xyz"] for r in steps])
    return {"mode": mode, "success": bool(outcome["success"]), "steps": int(outcome["steps"]),
            "lift_step": lift_step, "initial_goal_distance": float(dist[0]), "final_goal_distance": float(dist[-1]),
            "final_offset": xy[target][-1] - xy[goal][-1],
            "goal_distance": dist, "translation_correction": np.linalg.norm(corr, axis=1)}


def pair_label(base: bool, residual: bool) -> str:
    return {(False, True): "gained", (True, False): "lost", (True, True): "both succeed"}.get((base, residual),
                                                                                             "both fail")


def median_band(ax, series, color, label, length):
    if not series:
        return
    padded = np.full((len(series), length), np.nan)
    for i, s in enumerate(series):  # after the episode ends the distance stays at its last value
        padded[i, :len(s)] = s
        padded[i, len(s):] = s[-1]
    t = np.arange(length)
    ax.plot(t, np.nanmedian(padded, 0), color=color, lw=2, label=f"{label} (n={len(series)})")
    ax.fill_between(t, *np.nanpercentile(padded, [25, 75], 0), color=color, alpha=0.18, lw=0)


def failure_figure(run, rows, feats, title, out):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    fig.suptitle(title, fontsize=15)

    # 1. Outcome mode per policy
    ax = axes[0, 0]
    style(ax)
    ax.grid(axis="x", visible=False)
    left = np.zeros(2)
    for mode, color in zip(MODES, MODE_COLORS):
        counts = np.array([sum(r[f"{p}_mode"] == mode for r in rows) for p in ("base", "residual")])
        bars = ax.barh(["Frozen base", "Base + residual"], counts, left=left, color=color, label=mode,
                       edgecolor="white", linewidth=2, height=0.55)
        ax.bar_label(bars, [str(c) if c >= 2 else "" for c in counts], label_type="center", fontsize=10,
                     color="white")
        left += counts
    ax.invert_yaxis()
    ax.set(title="Outcome by mode (held-out test)", xlabel="States", xlim=(0, len(rows)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9)

    # 2. Paired transitions: base mode -> residual mode
    ax = axes[0, 1]
    matrix = np.array([[sum(r["base_mode"] == a and r["residual_mode"] == b for r in rows) for b in MODES]
                       for a in MODES])
    ax.imshow(matrix, cmap="Greys", vmin=0, vmax=max(1, matrix.max()) * 1.6)
    for i in range(len(MODES)):
        for j in range(len(MODES)):
            if matrix[i, j]:
                ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=11,
                        color="white" if matrix[i, j] > matrix.max() * 0.6 else "#222")
    ax.set_xticks(range(len(MODES)), MODES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(MODES)), MODES, fontsize=9)
    ax.set(title="Same state: base mode → residual mode", xlabel="Base + residual", ylabel="Frozen base")

    # 3. Where the target ends, relative to the goal (top-down)
    ax = axes[0, 2]
    style(ax)
    for pair, policy, color, marker, name in (("gained", "base", GRAY, "o", "gained · frozen base"),
                                              ("gained", "residual", GREEN, "o", "gained · base + residual"),
                                              ("lost", "residual", RED, "X", "lost · base + residual")):
        pts = np.array([feats[r["initial_state_id"]][policy]["final_offset"] for r in rows if r["pair"] == pair])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=55, color=color, marker=marker, edgecolor="white", linewidth=1,
                       label=f"{name} (n={len(pts)})")
    ax.add_patch(plt.Circle((0, 0), NEAR_GOAL_M, fill=False, ls="--", color="#555"))
    ax.plot(0, 0, "+", color="#222", ms=12, mew=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(title=f"Target position at episode end (goal at +, circle {NEAR_GOAL_M:.2f} m)",
           xlabel="x − goal x (m)", ylabel="y − goal y (m)")
    ax.legend(loc="best", fontsize=9)

    # 4-5. Target-goal distance over time, gained and lost states
    length = max(r["base_steps"] for r in rows) + 1
    for ax, pair in ((axes[1, 0], "gained"), (axes[1, 1], "lost")):
        style(ax)
        ids = [r["initial_state_id"] for r in rows if r["pair"] == pair]
        median_band(ax, [feats[i]["base"]["goal_distance"] for i in ids], GRAY, "Frozen base", length)
        median_band(ax, [feats[i]["residual"]["goal_distance"] for i in ids], GREEN, "Base + residual", length)
        lifts = [feats[i][p]["lift_step"] for i in ids for p in ("base", "residual")
                 if feats[i][p]["lift_step"] is not None]
        if lifts:
            ax.axvline(np.median(lifts), color="#555", ls=":", lw=1.5, label="median lift step")
        ax.set(title=f"Target→goal distance over time: {pair} states (median, IQR)", xlabel="Env step",
               ylabel="xy distance (m)", ylim=(0, None))
        if ids:
            ax.legend(loc="upper right", fontsize=9)

    # 6. Where the residual acts: correction magnitude aligned at the lift step
    ax = axes[1, 2]
    style(ax)
    window = np.arange(-60, 121)
    for pair, color in (("gained", GREEN), ("lost", RED), ("both fail", "#8a919c")):
        traces = []
        for r in rows:
            f = feats[r["initial_state_id"]]["residual"]
            if r["pair"] != pair or f["lift_step"] is None:
                continue
            c = f["translation_correction"]
            idx = f["lift_step"] + window
            traces.append(np.where((idx >= 0) & (idx < len(c)), c[np.clip(idx, 0, len(c) - 1)], np.nan))
        if traces:
            ax.plot(window, np.nanmean(traces, 0), color=color, lw=2, label=f"{pair} (n={len(traces)})")
    ax.axvline(0, color="#555", ls=":", lw=1.5)
    ax.set(title="Residual translation correction around the lift", xlabel="Env steps relative to target lift",
           ylabel="mean ‖correction xyz‖ (normalized action)", ylim=(0, None))
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def training_figure(run, title, out):
    metrics = read_csv(run / "logs/train_metrics.csv")
    eps = read_csv(run / "logs/train_episodes.csv")
    val = read_csv(run / "logs/validation_metrics.csv")
    val_steps = sorted({int(r["env_steps"]) for r in val if r["variant"] == "residual" and int(r["env_steps"]) > 0})
    step = [int(r["env_steps"]) for r in metrics]

    def col(name):
        return [float(r[name]) if r[name] not in ("", "nan") else np.nan for r in metrics]

    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(title, fontsize=15)
    ep_step = [int(r["env_steps_total"]) for r in eps]
    succ = np.array([r["success"] in ("True", "1") for r in eps], float)
    window = 10
    panels = [
        (axes[0, 0], "Training success (rolling 10 episodes)", "Success (%)",
         [(ep_step[window - 1:], 100 * np.convolve(succ, np.ones(window) / window, "valid"), GREEN, None)]),
        (axes[0, 1], "Critic loss", "Loss", [(step, col("critic_loss"), GREEN, None)]),
        (axes[0, 2], "Q estimate vs target", "Q",
         [(step, col("q"), GREEN, "Q"), (step, col("target_q"), GRAY, "target Q")]),
        (axes[1, 0], "Entropy temperature α", "α", [(step, col("alpha"), GREEN, None)]),
        (axes[1, 1], "Policy entropy", "Entropy (nats)", [(step, col("entropy"), GREEN, None)]),
        (axes[1, 2], "Mean |correction| per training episode", "Normalized action units",
         [(ep_step, [float(r["mean_abs_correction"]) for r in eps], GREEN, None)]),
    ]
    for ax, name, ylabel, series in panels:
        style(ax)
        for x, y, color, label in series:
            ax.plot(x, y, color=color, lw=2, label=label)
        for s in val_steps:
            ax.axvline(s, color="#999", ls=":", lw=1)
        ax.set(title=name, xlabel="Environment steps", ylabel=ylabel)
        if any(label for *_, label in series):
            ax.legend(loc="best", fontsize=9)
    axes[0, 0].set_ylim(0, 100)
    axes[1, 0].set_yscale("log")
    fig.text(0.5, 0.005, "Dotted lines: validation checkpoints", ha="center", fontsize=10, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--eval-id", default="test_best_osmesa")
    a = p.parse_args()
    run = a.run

    identity = json.loads((run / "config/task_identity.json").read_text())
    target, goal = goal_objects(identity["case"]["bddl"])
    label = f"LIBERO-PRO {identity.get('label', '')} {identity['case']['category']}"

    episodes = load_episodes(run, a.eval_id)
    feats, rows = {}, []
    for init, ep in sorted(episodes.items()):
        feats[init] = {p: episode_features(ep[p], target, goal) for p in ("base", "residual")}
        b, r = feats[init]["base"], feats[init]["residual"]
        rows.append({"initial_state_id": init, "pair": pair_label(b["success"], r["success"]),
                     "initial_goal_distance": b["initial_goal_distance"],
                     **{f"{p}_{k}": feats[init][p][k] for p in ("base", "residual")
                        for k in ("mode", "steps", "lift_step", "final_goal_distance")}})

    with open(run / "failure_episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failure_figure(run, rows, feats,
                   f"{label} · failure analysis on {len(rows)} paired states ({a.eval_id})\n"
                   f"target {target} → goal {goal}; modes from logged object poses (heuristic)",
                   run / "failure_analysis.png")
    training_figure(run, f"{label} · FlashSAC training diagnostics", run / "training_diagnostics.png")

    for p in ("base", "residual"):
        counts = {m: sum(r[f"{p}_mode"] == m for r in rows) for m in MODES}
        print(p, {m: c for m, c in counts.items() if c})
    for pair in PAIRS:
        print(pair, sum(r["pair"] == pair for r in rows))
    for name in ("failure_analysis.png", "training_diagnostics.png", "failure_episodes.csv"):
        print(run / name)


if __name__ == "__main__":
    main()
