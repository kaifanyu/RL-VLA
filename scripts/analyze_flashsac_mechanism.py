"""Mechanism analysis for one FlashSAC residual run: object choice, grasp offset and release.

Usage: .venv/bin/python scripts/analyze_flashsac_mechanism.py runs/flashsac_object_00 \
    --eval-id ev20261104T004954 [--diag runs/flashsac_object_00_diag]
Writes <run>/mechanism_analysis.png and <run>/mechanism_episodes.csv and prints the statistics.

Needs a pick-and-place case with a distractor (Object 0: target alphabet_soup_1, distractor butter_1).
--diag reads short residual-only episodes (eval ids diag<step>) and reports which object each
checkpoint lifted. Thresholds sort episodes for analysis; they are not task predicates.
"""
import argparse
import csv
import json
import re
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

TARGET, DISTRACTOR, GOAL = "alphabet_soup_1", "butter_1", "basket_1"
LIFT_M = 0.03        # object counts as lifted this far above its start height
DETACH_M = 0.03      # object counts as detached once this much farther from the gripper than at the grasp
REGRASP_M = 0.03     # object this close to the gripper at the end is still held
RIM_Z = 0.10         # released object resting above this height sits on the basket, not the floor
OFF_CENTER_M = -0.024
GREEN, RED, GRAY, BLUE, PURPLE, ORANGE = "#1a7f64", "#c04a36", "#8b94a3", "#2a78d6", "#6450d8", "#eb6834"
CLASSES = ("slip in transport", "early release", "lodged on rim", "never released")
CLASS_COLORS = (RED, ORANGE, PURPLE, BLUE)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#ececec", linewidth=0.8)
    ax.set_axisbelow(True)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def columns(rows, names):
    return np.array([[float(r[n]) for n in names] for r in rows])


def load_episode(run: Path, meta: dict) -> dict:
    files = {k: run / "logs" / Path(v).name for k, v in meta["files"].items() if k.endswith("csv")}
    steps, objects = read_csv(files["steps_csv"]), read_csv(files["objects_csv"])
    pose = lambda name: columns(objects, [f"{name}_{a}" for a in "xyz"])  # noqa: E731
    return {"success": bool(meta["outcome"]["success"]), "steps": int(meta["outcome"]["steps"]),
            "eef": columns(steps, [f"eef_pos_{a}" for a in "xyz"]),
            "base": columns(steps, [f"base_{a}" for a in "xyz"]),
            "correction": columns(steps, [f"correction_{a}" for a in "xyz"]),
            "gripper": columns(steps, ["executed_gripper"])[:, 0],
            "target": pose(TARGET), "distractor": pose(DISTRACTOR), "goal": pose(GOAL)}


def load_eval(run: Path, eval_id: str) -> dict:
    episodes = {}
    for path in sorted((run / "metadata").glob(f"*{eval_id}-init*.json")):
        meta = json.loads(path.read_text())
        if meta.get("eval_id") == eval_id:
            init = int(re.search(r"init(\d+)$", meta["episode_id"])[1])
            episodes.setdefault(init, {})[meta["policy"]] = load_episode(run, meta)
    return episodes


def lift_step(z: np.ndarray):
    above = np.where(z - z[0] > LIFT_M)[0]
    return int(above[0]) if len(above) else None


def features(ep: dict) -> dict:
    n = min(len(ep["eef"]), len(ep["target"]))
    target, eef, goal, gripper = ep["target"][:n], ep["eef"][:n], ep["goal"][:n], ep["gripper"][:n]
    lift, wrong = lift_step(target[:, 2]), lift_step(ep["distractor"][:, 2])
    out = {"success": ep["success"], "lift": lift, "picked_target": lift is not None,
           "wrong_object": wrong is not None and (lift is None or wrong < lift), "grasp_offset_x": None,
           "detach": None, "detach_distance": None, "failure_class": ""}
    if lift is None or lift + 13 >= n:
        return out
    gap = np.linalg.norm(target - eef, axis=1)
    out["grasp_offset_x"] = float((target - eef)[lift + 5, 0])
    away = np.where(gap[lift:] > gap[lift + 3:lift + 13].mean() + DETACH_M)[0]
    if len(away):
        k = lift + int(away[0])
        out["detach"] = k
        out["detach_distance"] = float(np.linalg.norm(target[k, :2] - goal[k, :2]))
    if not ep["success"] and not out["wrong_object"]:
        if out["detach"] is None or gap[-1] < REGRASP_M:
            out["failure_class"] = "never released"
        elif gripper[max(out["detach"] - 8, 0):out["detach"] + 1].min() > 0:
            out["failure_class"] = "slip in transport"
        elif target[-1, 2] > RIM_Z:
            out["failure_class"] = "lodged on rim"
        else:
            out["failure_class"] = "early release"
    return out


def mcnemar(b: int, c: int) -> float:
    return min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))


def report(ids, episodes, feats):
    get = lambda p, k: np.array([feats[i][p][k] for i in ids])  # noqa: E731
    base_pick, res_pick = get("base", "picked_target") & ~get("base", "wrong_object"), \
        get("residual", "picked_target") & ~get("residual", "wrong_object")
    base_ok, res_ok = get("base", "success"), get("residual", "success")
    print(f"target picked: base {base_pick.sum()}/{len(ids)}, residual {res_pick.sum()}/{len(ids)}, "
          f"McNemar p={mcnemar((res_pick & ~base_pick).sum(), (base_pick & ~res_pick).sum()):.2e}")
    print(f"placed given picked: base {base_ok.sum()}/{base_pick.sum()}, residual {res_ok.sum()}/{res_pick.sum()}, "
          f"Fisher p={stats.fisher_exact([[base_ok.sum(), base_pick.sum() - base_ok.sum()], [res_ok.sum(), res_pick.sum() - res_ok.sum()]])[1]:.3f}")
    print(f"on the {base_pick.sum()} base-picked states: base {base_ok[base_pick].sum()}, residual "
          f"{res_ok[base_pick].sum()}, McNemar p={mcnemar((res_ok & ~base_ok & base_pick).sum(), (base_ok & ~res_ok & base_pick).sum()):.3f}; "
          f"on the {(~base_pick).sum()} other states residual {res_ok[~base_pick].sum()}")
    print("residual failure classes:", {c: [i for i in ids if feats[i]["residual"]["failure_class"] == c] for c in CLASSES})

    table = []
    for p in ("base", "residual"):
        grasps = [(feats[i][p]["grasp_offset_x"], feats[i][p]["success"]) for i in ids
                  if feats[i][p]["grasp_offset_x"] is not None and not feats[i][p]["wrong_object"]]
        off = np.array([g[0] for g in grasps])
        ok = np.array([g[1] for g in grasps])
        low = off < OFF_CENTER_M
        table.append([ok[low].sum(), (~ok[low]).sum()])
        print(f"{p}: off-center grasps {ok[low].sum()}/{low.sum()} succeed, centered {ok[~low].sum()}/{(~low).sum()} succeed, "
              f"median offset {np.median(off) * 100:.1f} cm")
    print(f"off-center grasps, base vs residual: Fisher p={stats.fisher_exact(table)[1]:.3f}")

    off, ok = get("residual", "grasp_offset_x").astype(float), res_ok
    print(f"residual offset, success vs failure: Mann-Whitney p={stats.mannwhitneyu(off[ok], off[~ok]).pvalue:.4f}")
    for name, column in (("x", 0), ("y", 1)):
        basket = np.array([episodes[i]["base"]["goal"][0, column] for i in ids])
        print(f"basket {name}, base wrong vs picked: {basket[~base_pick].mean():.4f} vs {basket[base_pick].mean():.4f}, "
              f"Mann-Whitney p={stats.mannwhitneyu(basket[~base_pick], basket[base_pick]).pvalue:.4f}")
    correction = np.array([episodes[i]["residual"]["correction"][:100] for i in ids])
    r2 = 1 - ((correction - correction.mean(0, keepdims=True)) ** 2).sum((0, 1)) / \
        ((correction - correction.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
    print("share of correction variance explained by time alone (x, y, z, first 100 steps):", r2.round(2))
    gap = np.array([np.linalg.norm(episodes[i]["residual"]["eef"][15] - episodes[i]["base"]["eef"][15]) for i in ids])
    print(f"gripper gap between base and residual at step 15: {gap.mean() * 100:.2f} cm "
          f"(range {gap.min() * 100:.2f}-{gap.max() * 100:.2f})")
    early = np.array([episodes[i]["residual"]["correction"][:45].sum(0) for i in ids]) * 0.05
    commanded = np.array([episodes[i]["residual"]["base"][:45].sum(0) for i in ids]) * 0.05
    print(f"first 45 steps, y command: correction {early[:, 1].mean() * 100:.1f} cm, base {commanded[:, 1].mean() * 100:.1f} cm")
    return base_pick


def read_diag(diag: Path) -> dict:
    results = {}
    for path in sorted((diag / "metadata").glob("*diag*-init*.json")):
        meta = json.loads(path.read_text())
        step = int(meta["eval_id"].removeprefix("diag"))
        f = features(load_episode(diag, meta))
        kind = "butter" if f["wrong_object"] else "soup" if f["picked_target"] else "neither"
        results.setdefault(step, {"soup": 0, "butter": 0, "neither": 0})[kind] += 1
    return dict(sorted(results.items()))


def figure(run, ids, episodes, feats, base_pick, diag, title):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    fig.suptitle(title, fontsize=15)

    # 1. Fork in the end-effector path
    ax = axes[0, 0]
    style(ax)
    t = np.arange(61)
    for i, picked in zip(ids, base_pick):
        ax.plot(t, episodes[i]["base"]["eef"][:61, 1], color=GRAY if picked else PURPLE, lw=0.8, alpha=0.6)
        ax.plot(t, episodes[i]["residual"]["eef"][:61, 1], color=GREEN, lw=0.8, alpha=0.5)
    for y, label in ((episodes[ids[0]]["base"]["target"][0, 1], "soup"),
                     (episodes[ids[0]]["base"]["distractor"][0, 1], "butter")):
        ax.axhline(y, color="#555", lw=1, ls=":")
        ax.text(60, y + 0.006, label, ha="right", fontsize=10, color="#444")
    ax.plot([], [], color=PURPLE, label="base, goes to butter")
    ax.plot([], [], color=GRAY, label="base, goes to soup")
    ax.plot([], [], color=GREEN, label="base + residual")
    ax.set(title="Gripper y: the base forks near step 15", xlabel="Env step", ylabel="Gripper y (m)")
    ax.legend(loc="upper right")

    # 2. Base commands in the same states, with and without the residual
    ax = axes[0, 1]
    style(ax)
    wrong = [i for i, picked in zip(ids, base_pick) if not picked]
    t = np.arange(45)
    ax.plot(t, np.mean([episodes[i]["base"]["base"][:45, 1] for i in wrong], 0), color=PURPLE, lw=2,
            label="base y command, base episode")
    ax.plot(t, np.mean([episodes[i]["residual"]["base"][:45, 1] for i in wrong], 0), color=GREEN, lw=2,
            label="base y command, residual episode")
    ax.plot(t, np.mean([episodes[i]["residual"]["correction"][:45, 1] for i in wrong], 0), color=ORANGE, lw=2,
            label="residual y correction")
    ax.axhline(0, color="#999", lw=1)
    ax.set(title=f"The frozen base changes its own command ({len(wrong)} butter states)", xlabel="Env step",
           ylabel="Normalized action", ylim=(-1.2, 0.15))
    ax.legend(loc="lower right")

    # 3. Time-indexed correction
    ax = axes[0, 2]
    style(ax)
    correction = np.array([episodes[i]["residual"]["correction"][:100] for i in ids])
    t = np.arange(100)
    for axis, name, color in ((0, "x", GREEN), (1, "y", BLUE), (2, "z", ORANGE)):
        mean, sd = correction[:, :, axis].mean(0), correction[:, :, axis].std(0)
        ax.plot(t, mean, color=color, lw=2, label=name)
        ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)
    ax.axhline(0, color="#999", lw=1)
    ax.set(title="Residual correction across all 50 states (mean ± SD)", xlabel="Env step",
           ylabel="Correction (normalized)")
    ax.legend(loc="upper right")

    # 4. Grasp offset against outcome
    ax = axes[1, 0]
    style(ax)
    rng = np.random.default_rng(0)
    for row, (p, label) in enumerate((("base", "Frozen base"), ("residual", "Base + residual"))):
        for success, color, marker in ((True, GREEN, "o"), (False, RED, "x")):
            xs = [feats[i][p]["grasp_offset_x"] * 100 for i in ids if feats[i][p]["success"] == success
                  and feats[i][p]["grasp_offset_x"] is not None and not feats[i][p]["wrong_object"]]
            ax.scatter(xs, row + rng.uniform(-0.15, 0.15, len(xs)), c=color, marker=marker, s=40,
                       label=("success" if success else "failure") if row == 0 else None)
    ax.axvline(OFF_CENTER_M * 100, color="#555", lw=1, ls="--")
    ax.set(title="Soup offset from the gripper just after lift", xlabel="Soup x − gripper x (cm)",
           yticks=[0, 1], yticklabels=["Frozen base", "Base + residual"], ylim=(-0.5, 1.5))
    ax.legend(loc="upper right")

    # 5. Where the soup left the gripper
    ax = axes[1, 1]
    style(ax)
    groups = [("success", [feats[i]["residual"]["detach_distance"] for i in ids
                           if feats[i]["residual"]["success"] and feats[i]["residual"]["detach_distance"]], GREEN)]
    groups += [(c, [feats[i]["residual"]["detach_distance"] for i in ids
                    if feats[i]["residual"]["failure_class"] == c and feats[i]["residual"]["detach_distance"]], color)
               for c, color in zip(CLASSES, CLASS_COLORS) if c != "never released"]
    for row, (name, values, color) in enumerate(groups):
        ax.scatter(np.array(values) * 100, row + rng.uniform(-0.15, 0.15, len(values)), c=color, s=40)
    ax.set(title="Soup distance from basket center when it left the gripper", xlabel="Distance (cm)",
           yticks=range(len(groups)), yticklabels=[f"{n} (n={len(v)})" for n, v, _ in groups])

    # 6. Object choice by checkpoint
    ax = axes[1, 2]
    style(ax)
    ax.grid(axis="x", visible=False)
    if diag:
        labels = [f"{s:,}" for s in diag]
        left = np.zeros(len(diag))
        for kind, color in (("soup", GREEN), ("butter", PURPLE), ("neither", GRAY)):
            counts = np.array([diag[s][kind] for s in diag])
            bars = ax.bar(labels, counts, bottom=left, color=color, label=kind, width=0.6, edgecolor="white", linewidth=2)
            ax.bar_label(bars, [str(c) if c else "" for c in counts], label_type="center", color="white", fontsize=11)
            left = left + counts
        ax.set(title="Object lifted by checkpoint (validation states, 100 steps)", xlabel="Checkpoint env step",
               ylabel="States", ylim=(0, 14))
        ax.legend(loc="upper center", ncol=3)
    else:
        ax.set_visible(False)

    fig.tight_layout()
    out = run / "mechanism_analysis.png"
    fig.savefig(out, dpi=150)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--eval-id", required=True)
    p.add_argument("--diag", type=Path)
    a = p.parse_args()

    episodes = load_eval(a.run, a.eval_id)
    ids = sorted(i for i, e in episodes.items() if {"base", "residual"} <= set(e))
    feats = {i: {k: features(episodes[i][k]) for k in ("base", "residual")} for i in ids}
    base_pick = report(ids, episodes, feats)

    diag = read_diag(a.diag) if a.diag else {}
    for step, counts in diag.items():
        print(f"checkpoint {step}: {counts}")

    with open(a.run / "mechanism_episodes.csv", "w", newline="") as f:
        names = ("success", "wrong_object", "lift", "grasp_offset_x", "detach", "detach_distance", "failure_class")
        writer = csv.DictWriter(f, fieldnames=["initial_state_id"] + [f"{p_}_{k}" for p_ in ("base", "residual") for k in names])
        writer.writeheader()
        for i in ids:
            writer.writerow({"initial_state_id": i, **{f"{p_}_{k}": feats[i][p_][k] for p_ in ("base", "residual") for k in names}})
    print(figure(a.run, ids, episodes, feats, base_pick, diag,
                 f"LIBERO-PRO Object 0 Position · mechanism analysis on {len(ids)} paired test states ({a.eval_id})"))


if __name__ == "__main__":
    main()
