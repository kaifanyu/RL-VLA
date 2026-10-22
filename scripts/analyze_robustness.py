"""Save a consistent, read-only analysis of an ongoing robustness evaluation."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from rl_vla.baseline import HORIZONS, atomic_json, wilson_interval
from rl_vla.robustness import ROOT, episode_seed


def stats(rows, planned=None):
    wins = sum(r["success"] for r in rows)
    latency = [value for row in rows for value in row["inference_latency_ms"]]
    result = {"episodes": len(rows), "successes": wins,
              "success_rate": wins / len(rows) if rows else None,
              "success_95pct_wilson": wilson_interval(wins, len(rows)) if rows else None,
              "failures": len(rows) - wins,
              "mean_primitive_steps": float(np.mean([r["primitive_steps"] for r in rows])) if rows else None,
              "mean_episode_seconds": float(np.mean([r["elapsed_seconds"] for r in rows])) if rows else None,
              "inference_latency_p50_ms": float(np.percentile(latency, 50)) if latency else None,
              "inference_latency_p95_ms": float(np.percentile(latency, 95)) if latency else None,
              "total_inference_calls": len(latency)}
    if planned is not None:
        result.update(planned_episodes=planned, status="complete" if len(rows) == planned else "partial")
    return result


def analyze(run, output):
    summary = json.loads((run / "summary.json").read_text())
    cutoff = summary["updated_unix_time"]
    paths = [p for p in (run / "episodes").glob("*.json") if p.stat().st_mtime <= cutoff]
    rows = [json.loads(path.read_text()) for path in paths]
    if len(rows) != summary["episodes"]:
        raise RuntimeError("Episode files and summary cutoff differ; take a fresh snapshot")
    plan = json.loads((run / "protocol.json").read_text())
    cases = {c["case_id"]: c for c in plan["cases"]}
    identifiers = set()
    failures = []
    for row in rows:
        case = cases[row["case_id"]]
        if row["episode_id"] in identifiers:
            raise ValueError("Repeated episode")
        identifiers.add(row["episode_id"])
        for name in ("benchmark", "suite", "task_id", "category", "difficulty"):
            if row[name] != case[name]:
                raise ValueError(f"Episode does not match protocol: {row['episode_id']} {name}")
        index = row["initial_state_id"]
        if not 0 <= index < case["episodes"] or row["seed"] != episode_seed(case["case_id"], index):
            raise ValueError("Invalid state index or seed")
        if row["success"] != row["terminated"] or row["return"] != float(row["success"]):
            raise ValueError("Success, termination and reward differ")
        if not 0 < row["primitive_steps"] <= HORIZONS[row["suite"]]:
            raise ValueError("Invalid controller step count")
        if not np.isfinite(row["inference_latency_ms"]).all() or not 0 <= row["projection_fraction"] <= 1:
            raise ValueError("Invalid inference/controller metrics")
        if not row["success"]:
            failures.append(row)
            if not row["truncated"] or row["primitive_steps"] != HORIZONS[row["suite"]]:
                raise ValueError("Recorded failure was not a suite-horizon timeout")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "live_summary_snapshot.json", summary)
    baseline = json.loads((ROOT / "runs/evaluation/pi05_libero_spatial_20260928/summary.json").read_text())
    groups = defaultdict(list)
    planned = Counter()
    for case in cases.values():
        keys = [case["benchmark"], f"{case['benchmark']}/{case['suite']}",
                f"{case['benchmark']}/{case['category']}",
                f"{case['benchmark']}/{case['category']}/{case['suite']}"]
        for key in keys:
            planned[key] += case["episodes"]
    for row in rows:
        keys = [row["benchmark"], f"{row['benchmark']}/{row['suite']}",
                f"{row['benchmark']}/{row['category']}",
                f"{row['benchmark']}/{row['category']}/{row['suite']}"]
        if row["benchmark"] == "plus":
            keys.append(f"plus/{row['category']}/{row['suite']}/difficulty:{row['difficulty']}")
        for key in keys:
            groups[key].append(row)
    metrics = {key: stats(group, planned.get(key)) for key, group in sorted(groups.items())}
    for name in ("plus", "pro"):
        if metrics[name]["episodes"] != summary["metrics"][name]["episodes"]:
            raise ValueError("Benchmark totals differ from frozen summary")
        if metrics[name]["successes"] != summary["metrics"][name]["successes"]:
            raise ValueError("Benchmark successes differ from frozen summary")
    published = [r for r in rows if r["benchmark"] == "pro" and r["category"] != "Environment"]
    generated = [r for r in rows if r["benchmark"] == "pro" and r["category"] == "Environment"]
    controls = [r for r in generated if r["suite"] == "libero_10" and r["task_name"].startswith("LIVING_ROOM_")]
    modified = [r for r in generated if r not in controls]
    per_case = defaultdict(list)
    for row in rows:
        if row["benchmark"] == "pro":
            per_case[row["case_id"]].append(row)
    task_metrics = [{"case_id": key, "suite": values[0]["suite"], "category": values[0]["category"],
                     "task": values[0]["task_name"], **stats(values, 50)}
                    for key, values in sorted(per_case.items())]
    recent = sum(path.stat().st_mtime > cutoff - 86400 for path in paths)
    remaining = summary["planned_episodes"] - len(rows)
    contract = json.loads((run / "queue_contract.json").read_text())
    code_matches = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                    for name, digest in contract["code_sha256"].items()}
    manifest = json.loads((run / "server_manifest.json").read_text())
    original = json.loads((ROOT / "runs/evaluation/pi05_libero_spatial_20260928/server_manifest.json").read_text())
    events = [json.loads(path.read_text()) for path in (run / "startup_events").glob("*/resource_stop.json")]
    result = {
        "snapshot_time": datetime.fromtimestamp(cutoff, ZoneInfo("America/New_York")).isoformat(),
        "run": str(run.resolve()), "queue_status_at_snapshot": summary["status"],
        "episodes": len(rows), "planned_episodes": summary["planned_episodes"],
        "completion_fraction": len(rows) / summary["planned_episodes"],
        "original_spatial_baseline": baseline, "metrics": metrics,
        "pro_published_only": stats(published, 8000), "pro_generated_environment": stats(generated, 2000),
        "pro_generated_unchanged_long_controls": stats(controls, 250),
        "pro_generated_modified_environments": stats(modified, 1750),
        "pro_per_task": task_metrics,
        "validation": {"unique_valid_episode_files": len(rows), "task_horizon_timeouts": len(failures),
                       "inconsistent_episode_files": 0, "checkpoint_matches_original_spatial":
                       manifest["checkpoint_sha256"] == original["checkpoint_sha256"],
                       "evaluation_code_hash_matches": code_matches,
                       "recorded_resource_stops": len(events)},
        "throughput": {"last_24h_completed_episodes": recent, "remaining_episodes": remaining,
                       "naive_remaining_days_at_last_24h_pace": remaining / recent if recent else None,
                       "caveat": "Unrun Plus Goal/Long suites and slow corruptions can increase runtime."},
        "limitations": ["Results are partial; Plus coverage is predominantly Spatial and some Object.",
                        "PRO Environment uses local generated states; five Long tasks are unchanged controls.",
                        "Original Object/Goal/Long baselines are absent.",
                        "Spatial comparisons use different case/state/noise sampling, not paired trials.",
                        "Episode Wilson intervals do not model clustering by task."]}
    atomic_json(output / "analysis.json", result)
    columns = ["episode_id", "benchmark", "category", "suite", "task_name", "difficulty",
               "initial_state_id", "seed", "state_origin", "success", "primitive_steps", "elapsed_seconds"]
    with (output / "episodes_snapshot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["episode_id"]))
    plot(result, output)
    print(json.dumps({"output": str(output.resolve()), "snapshot_time": result["snapshot_time"],
                      "episodes": len(rows), "published_pro": result["pro_published_only"],
                      "generated_controls": result["pro_generated_unchanged_long_controls"],
                      "modified_environment": result["pro_generated_modified_environments"],
                      "validation": result["validation"], "throughput": result["throughput"]}, indent=2))


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"wspace": 0.38})
    panels = [("plus", ["Light Conditions", "Background Textures", "Objects Layout", "Language Instructions",
                        "Sensor Noise", "Robot Initial States", "Camera Viewpoints"], "LIBERO-Plus: Spatial",
               "#246d94"), ("pro", ["Language", "Object", "Environment", "Position", "Task"],
                              "LIBERO-PRO: four suites, partial", "#b75138")]
    for axis, (benchmark, categories, title, color) in zip(axes, panels, strict=True):
        for index, category in enumerate(categories):
            key = f"{benchmark}/{category}" + ("/libero_spatial" if benchmark == "plus" else "")
            metric = result["metrics"][key]
            value = metric["success_rate"] * 100
            bar = axis.barh(index, value, color=color, height=0.62)[0]
            if category == "Environment":
                bar.set_hatch("///")
            suffix = "*" if metric["status"] == "partial" else ""
            axis.text(value + 1, index, f"{value:.1f}%{suffix}", va="center", fontsize=10)
        axis.set_yticks(range(len(categories)), categories)
        axis.invert_yaxis()
        axis.set_xlim(0, 114)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_xlabel("Success rate (%)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
    axes[0].axvline(99, color="#555555", linestyle="--", linewidth=1)
    fig.suptitle("π0.5 robustness evaluation — snapshot " + result["snapshot_time"][:10], fontsize=15)
    fig.text(0.06, 0.01, "Dashed line: original Spatial baseline, 99%.  * Partial group.  "
             "Hatching: locally generated PRO Environment cases.\n"
             "Plus and PRO panels use different suite coverage; compare categories within each panel.", fontsize=9)
    fig.subplots_adjust(left=0.17, right=0.96, top=0.85, bottom=0.2)
    fig.savefig(output / "robustness_snapshot.png", dpi=180)
    fig.savefig(output / "robustness_snapshot.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_ET")
    analyze(args.run, args.output or args.run / "analysis" / stamp)
