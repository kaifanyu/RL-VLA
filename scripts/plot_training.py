"""Plot a train run's metrics.jsonl; with --watch, refresh until summary.json appears."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def load(run):
    records = []
    with (run / "metrics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # Last line may be mid-write.
                break
    return records


def rolling(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    sums = np.cumsum(np.insert(values, 0, 0.0))
    counts = np.minimum(np.arange(1, len(values) + 1), window)
    return (sums[1:] - sums[np.arange(1, len(values) + 1) - counts]) / counts


def series(records, key):
    rows = [(r["decisions"], r["metrics"][key]) for r in records if key in r.get("metrics", {})]
    return tuple(map(np.array, zip(*rows))) if rows else (np.array([]), np.array([]))


def plot(run, output, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load(run)
    if not records:
        return False
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    total = manifest.get("config", {}).get("train", {}).get("total_decisions")
    algorithm = manifest.get("config", {}).get("algorithm", "")
    episodes = [r for r in records if "episode_return" in r]
    ep_x = np.array([r["decisions"] for r in episodes])
    success = np.array([float(r["success"]) for r in episodes])
    returns = np.array([r["episode_return"] for r in episodes])
    lengths = np.array([r["episode_steps"] for r in episodes])
    x = np.array([r["decisions"] for r in records])

    fig, axes = plt.subplots(3, 3, figsize=(15, 10.5))
    axes = axes.ravel()

    def line(axis, xs, ys, title, label=None, log=False, **kwargs):
        if len(xs):
            axis.plot(xs, ys, label=label, linewidth=1.4, **kwargs)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        if log and len(ys) and np.all(np.asarray(ys) > 0):
            axis.set_yscale("log")

    axis = axes[0]
    if len(episodes):
        axis.scatter(ep_x, success, s=8, alpha=0.25, color="#888888", label="episode")
        axis.plot(ep_x, rolling(success, window), color="#246d94", linewidth=2,
                  label=f"rolling mean ({window} ep)")
        axis.legend(fontsize=8, loc="center right")
    axis.set_ylim(-0.05, 1.05)
    axis.set_title("Success", loc="left", fontsize=11, fontweight="bold")

    axis = axes[1]
    if len(episodes):
        axis.scatter(ep_x, returns, s=8, alpha=0.25, color="#888888")
        axis.plot(ep_x, rolling(returns, window), color="#b75138", linewidth=2)
    axis.set_title("Episode return (raw)", loc="left", fontsize=11, fontweight="bold")

    axis = axes[2]
    if len(episodes):
        axis.scatter(ep_x, lengths, s=8, alpha=0.25, color="#888888")
        axis.plot(ep_x, rolling(lengths, window), color="#4a8a3c", linewidth=2)
    axis.set_title("Episode length (motor steps)", loc="left", fontsize=11, fontweight="bold")

    line(axes[3], *series(records, "critic_loss"), "Critic loss", log=True)
    line(axes[4], *series(records, "q"), "Q estimates", label="Q (min of 2)")
    line(axes[4], *series(records, "target_q"), "Q estimates", label="target Q")
    axes[4].legend(fontsize=8)
    line(axes[5], *series(records, "actor_loss"), "Actor loss")
    line(axes[6], *series(records, "alpha"), "Temperature α", log=True)
    line(axes[7], *series(records, "entropy"), "Policy entropy")
    line(axes[8], x, [r["residual_rms"] for r in records], "Residual action", label="residual RMS")
    line(axes[8], x, [r["projection_fraction"] for r in records], "Residual action",
         label="projection fraction")
    axes[8].legend(fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        if total:
            axis.set_xlim(0, total)
    for axis in axes[6:]:
        axis.set_xlabel("Decisions")

    last = records[-1]
    done = (run / "summary.json").exists()
    rate = last["decisions"] / max(last["elapsed_seconds"], 1e-9)
    eta = (total - last["decisions"]) / rate if total and rate else None
    recent = (f"{success[-window:].mean() * 100:.0f}% success over last {min(window, len(success))} ep"
              if len(success) else "no finished episodes yet")
    status = "finished" if done else (f"ETA {eta / 3600:.1f} h" if eta is not None else "running")
    fig.suptitle(f"{algorithm.upper()} {run.name} — {last['decisions']}/{total or '?'} decisions, "
                 f"{last['episodes']} episodes, {recent}, {last['elapsed_seconds'] / 3600:.1f} h elapsed, "
                 f"{status}  (updated {time.strftime('%H:%M:%S')})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    fig.savefig(temporary, dpi=110)
    plt.close(fig)
    os.replace(temporary, output)  # Viewers never see a half-written image.
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory passed to `rl_vla train --output`")
    parser.add_argument("--output", type=Path, help="Image path (default: <run>/progress.png)")
    parser.add_argument("--window", type=int, default=20, help="Rolling window in episodes")
    parser.add_argument("--watch", action="store_true", help="Keep refreshing until the run finishes")
    parser.add_argument("--interval", type=float, default=60, help="Seconds between refreshes")
    args = parser.parse_args()
    output = args.output or args.run / "progress.png"
    while True:
        if (args.run / "metrics.jsonl").exists() and plot(args.run, output, args.window):
            print(f"{time.strftime('%H:%M:%S')} wrote {output}", flush=True)
        if not args.watch or (args.run / "summary.json").exists():
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
