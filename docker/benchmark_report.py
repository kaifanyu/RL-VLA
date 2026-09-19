#!/usr/bin/env python3
"""Summarize residual-policy demo runs without training or plotting dependencies.

Usage: python docker/benchmark_report.py runs/gpu_demos_YYYYMMDD_HHMMSS
Safe to rerun while training: unfinished JSON/JSONL files are skipped or labelled.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

TASKS = ("toy", "weak_toy", "pendulum", "reacher", "panda_reach", "panda_push", "panda_pick_place")
COLORS = {"sac": "#1768ac", "ppo": "#c95120"}
CAVEATS = [
    "These are local learning checks with small training budgets. A single training seed does not establish an algorithm ranking.",
    "Compare return only within the same task and reward configuration. Higher return is better, including when both returns are negative.",
    "Training curves use sampled training episodes and a trailing mean of up to 20 episodes; evaluation uses separate episodes and may use deterministic actions.",
    "Pendulum and MuJoCo Reacher have no task success signal in these demos. Their success rate is unavailable, even if the generic evaluator writes zero.",
    "GPU device names come from run manifests. Peak allocated/reserved memory is PyTorch allocator memory from run summaries, where recorded; it excludes other GPU users and some CUDA overhead. Runtime includes the training loop's recorded overhead.",
]


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def read_json(path: Path, warnings: list[str], root: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise TypeError("expected a JSON object")
        return value
    except (OSError, ValueError, TypeError) as exc:
        warnings.append(f"{path.relative_to(root).as_posix()}: {exc}")
        return {}


def read_metrics(path: Path, warnings: list[str], root: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    malformed = 0
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError("expected object")
                    records.append(record)
                except (ValueError, TypeError):
                    malformed += 1
    except OSError as exc:
        warnings.append(f"{path.relative_to(root).as_posix()}: {exc}")
    if malformed:
        warnings.append(f"{path.relative_to(root).as_posix()}: skipped {malformed} incomplete or invalid metric lines")
    return records


def task_name(path: Path, root: Path, config: dict[str, Any]) -> str:
    for part in path.relative_to(root).parts:
        if part in TASKS:
            return part
    if root.name in TASKS:
        return root.name
    env = config.get("env", {})
    kwargs = env.get("kwargs", {})
    return str(kwargs.get("env_id") or kwargs.get("id") or env.get("kind") or "unknown")


def success_available(task: str, config: dict[str, Any]) -> bool:
    identifiers = [task.lower()]
    env = config.get("env", {})
    identifiers.extend(str(value).lower() for value in env.get("kwargs", {}).values())
    return not any("pendulum" in value or value.startswith("reacher") for value in identifiers)


def curve(records: list[dict[str, Any]], window: int = 20) -> list[list[float]]:
    points = []
    rewards: list[float] = []
    for record in records:
        reward, decisions = record.get("episode_return"), record.get("decisions")
        if numeric(reward) and numeric(decisions):
            rewards.append(float(reward))
            points.append([decisions, statistics.fmean(rewards[-window:])])
    # Limit report size while preserving the newest point.
    stride = max(1, math.ceil(len(points) / 1200))
    sampled = points[::stride]
    if points and sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def evaluation(path: Path, root: Path, warnings: list[str]) -> dict[str, Any] | None:
    value = read_json(path, warnings, root)
    if not value:
        return None
    config = value.get("config", {})
    task = task_name(path, root, config)
    results = value.get("results", [])
    returns = [result["return"] for result in results if numeric(result.get("return"))]
    steps = [result["primitive_steps"] for result in results if numeric(result.get("primitive_steps"))]
    is_base = value.get("controller") in ("base", "frozen_base", "base_only") or "base" in path.stem.lower()
    algorithm = "base" if is_base else config.get("algorithm", path.stem.split("_")[0])
    available = success_available(task, config)
    return {
        "task": task,
        "file": path.relative_to(root).as_posix(),
        "algorithm": algorithm,
        "sampling": "base" if is_base else value.get("residual_sampling", "unspecified"),
        "episodes": value.get("episodes", len(results)),
        "mean_return": value.get("mean_return", statistics.fmean(returns) if returns else None),
        "return_std": statistics.stdev(returns) if len(returns) > 1 else None,
        "mean_primitive_steps": statistics.fmean(steps) if steps else None,
        "success_available": available,
        "success_rate": value.get("success_rate") if available else None,
        "success_95pct_wilson": value.get("success_95pct_wilson") if available else None,
        "elapsed_seconds": value.get("elapsed_seconds"),
        "seeds": [result.get("seed") for result in results],
        "environment": config.get("env"),
        "chunk": config.get("chunk"),
        "base": config.get("base"),
    }


def collect(root: Path) -> dict[str, Any]:
    warnings: list[str] = []
    tasks: dict[str, dict[str, Any]] = {}

    def group(name: str) -> dict[str, Any]:
        return tasks.setdefault(name, {"task": name, "runs": [], "evaluations": [], "videos": []})

    # A directory containing only a metrics file is also a legitimate partial run.
    run_dirs = sorted({path.parent for name in ("manifest.json", "summary.json", "metrics.jsonl") for path in root.rglob(name)})
    for directory in run_dirs:
        manifest_path, summary_path = directory / "manifest.json", directory / "summary.json"
        manifest = read_json(manifest_path, warnings, root) if manifest_path.exists() else {}
        summary = read_json(summary_path, warnings, root) if summary_path.exists() else {}
        records = read_metrics(directory / "metrics.jsonl", warnings, root)
        config = manifest.get("config", {})
        task = task_name(directory, root, config)
        latest = {**(records[-1] if records else {}), **summary}
        decisions, elapsed = latest.get("decisions"), latest.get("elapsed_seconds")
        group(task)["runs"].append({
            "directory": directory.relative_to(root).as_posix(),
            "algorithm": config.get("algorithm", directory.name.split("_")[0]),
            "seed": config.get("train", {}).get("seed"),
            "status": "completed" if summary else "incomplete",
            "device": manifest.get("device", config.get("train", {}).get("device")),
            "cuda_device": manifest.get("cuda_device"),
            "cuda_peak_allocated_mib": latest.get("cuda_peak_allocated_mib"),
            "cuda_peak_reserved_mib": latest.get("cuda_peak_reserved_mib"),
            "decisions": decisions,
            "planned_decisions": config.get("train", {}).get("total_decisions"),
            "primitive_steps": latest.get("primitive_steps"),
            "episodes": latest.get("episodes"),
            "updates": latest.get("updates"),
            "elapsed_seconds": elapsed,
            "decisions_per_second": decisions / elapsed if numeric(decisions) and numeric(elapsed) and elapsed > 0 else None,
            "config": config,
            "training_return_curve": curve(records),
        })
    # Include optional *_eval_deterministic.json and *_eval_stochastic.json variants.
    eval_paths = sorted(path for path in root.rglob("*.json") if "eval" in path.stem.lower() and path.name not in ("manifest.json", "summary.json"))
    for path in eval_paths:
        result = evaluation(path, root, warnings)
        if result:
            group(result["task"])["evaluations"].append(result)
    for path in sorted(root.rglob("*.mp4")):
        if path.stem in ("base", "sac", "ppo") and path.is_file() and path.stat().st_size:
            group(task_name(path, root, {}))["videos"].append({
                "controller": path.stem,
                "file": path.relative_to(root).as_posix(),
            })
    for task in tasks.values():
        baselines = [item for item in task["evaluations"] if item["algorithm"] == "base"]
        for item in task["evaluations"]:
            matching = [base for base in baselines if all(base.get(key) == item.get(key) for key in ("environment", "chunk", "base"))]
            baseline = next((base for base in matching if Path(base["file"]).parent == Path(item["file"]).parent), matching[0] if matching else None)
            if item["algorithm"] != "base" and baseline:
                item["baseline_file"] = baseline["file"]
                item["baseline_mean_return"] = baseline["mean_return"]
                item["evaluation_seeds_match"] = bool(item["seeds"]) and None not in item["seeds"] and item["seeds"] == baseline["seeds"]
                item["return_gain_over_base"] = item["mean_return"] - baseline["mean_return"] if numeric(item["mean_return"]) and numeric(baseline["mean_return"]) else None
    order = {name: index for index, name in enumerate(TASKS)}
    return clean({
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "experiment_directory": str(root),
        "caveats": CAVEATS,
        "warnings": warnings,
        "tasks": sorted(tasks.values(), key=lambda task: (order.get(task["task"], len(TASKS)), task["task"])),
    })


def escape(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"))


def number(value: Any, digits: int = 2) -> str:
    return f"{value:,.{digits}f}" if numeric(value) else "—"


def svg_chart(runs: list[dict[str, Any]]) -> str:
    plotted = [run for run in runs if run["training_return_curve"]]
    points = [point for run in plotted for point in run["training_return_curve"]]
    if not points:
        return '<p class="muted">No completed training episodes recorded yet.</p>'
    width, height, left, top, right, bottom = 840, 280, 78, 18, 22, 52
    x_max = max(1, max(point[0] for point in points))
    y_min, y_max = min(point[1] for point in points), max(point[1] for point in points)
    pad = (y_max - y_min) * 0.08 or max(abs(y_max) * 0.08, 0.1)
    y_min, y_max = y_min - pad, y_max + pad

    def xy(point: list[float]) -> tuple[float, float]:
        return (left + point[0] / x_max * (width - left - right), top + (y_max - point[1]) / (y_max - y_min) * (height - top - bottom))

    elements = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Training episode return, trailing mean of up to 20 episodes">']
    for index in range(5):
        y_value = y_min + (y_max - y_min) * index / 4
        y = xy([0, y_value])[1]
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dfe5ee"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{y_value:.3g}</text>')
        x_value = x_max * index / 4
        x = xy([x_value, y_min])[0]
        elements.append(f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle">{x_value:,.0f}</text>')
    for run in plotted:
        color = COLORS.get(str(run["algorithm"]).lower(), "#6c589c")
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, run["training_return_curve"]))
        elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"><title>{escape(run["directory"])}</title></polyline>')
        if len(run["training_return_curve"]) == 1:
            x, y = xy(run["training_return_curve"][0])
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
    elements.append(f'<text x="{width/2}" y="{height-6}" text-anchor="middle">Training decisions</text></svg>')
    elements.append('<p class="legend">' + " · ".join(f'<span style="color:{COLORS.get(str(run["algorithm"]).lower(), "#6c589c")}">{escape(run["directory"])}</span>' for run in plotted) + '</p>')
    return "".join(elements)


def media_url(root: Path, relative_file: str, output_dir: Path) -> str:
    """Build a portable HTML URL, including spaces and reserved filename characters."""
    target = root / relative_file
    try:
        relative = Path(os.path.relpath(target, output_dir)).as_posix()
    except ValueError:
        # Windows cannot express a relative path across drive letters.
        return target.resolve().as_uri()
    return quote(relative, safe="/")


def render(report: dict[str, Any], output_dir: Path | None = None) -> str:
    root = Path(report["experiment_directory"])
    output_dir = output_dir if output_dir is not None else root
    sections = []
    for task in report["tasks"]:
        run_rows = []
        for run in task["runs"]:
            device = run.get("cuda_device") or run.get("device")
            run_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
                escape(run["directory"]), escape(run["status"]), escape(device), number(run["decisions"], 0),
                number(run["primitive_steps"], 0), number(run["elapsed_seconds"]) + " s", number(run["decisions_per_second"]),
                number(run.get("cuda_peak_allocated_mib"), 1), number(run.get("cuda_peak_reserved_mib"), 1),
            )) + "</tr>")
        eval_rows = []
        for item in task["evaluations"]:
            success = number(item["success_rate"] * 100, 1) + "%" if numeric(item["success_rate"]) else "N/A"
            gain = item.get("return_gain_over_base")
            gain_text = f"{gain:+.3f}" if numeric(gain) else "—"
            seed_match = "yes" if item.get("evaluation_seeds_match") else "—"
            eval_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
                escape(item["file"]), escape(item["algorithm"]), escape(item["sampling"]), number(item["episodes"], 0),
                number(item["mean_return"], 3), gain_text, success, seed_match,
            )) + "</tr>")
        videos = []
        for video in task.get("videos", []):
            url = escape(media_url(root, video["file"], output_dir))
            label = "Base controller" if video["controller"] == "base" else video["controller"].upper()
            videos.append(f'<figure><figcaption>{escape(label)}</figcaption><video controls preload="none" playsinline aria-label="{escape(task["task"])} {escape(label)} rollout"><source src="{url}" type="video/mp4">Your browser cannot play this clip.</video><a href="{url}">Open {escape(video["file"])}</a></figure>')
        video_section = ('<h3>One-episode demo clips</h3><p class="muted">The demo runner records one preselected episode (seed 100000), using mean residual actions for SAC/PPO. Clips are not selected for a favorable outcome; use the evaluation table to assess performance across episodes.</p><div class="videos">' + ''.join(videos) + '</div>') if videos else ''
        sections.append(f'<section><h2>{escape(task["task"])}</h2>'
            + '<h3>Training</h3><div class="scroll"><table><thead><tr><th>Run</th><th>Status</th><th>Device</th><th>Decisions</th><th>Primitive steps</th><th>Runtime</th><th>Decisions/s</th><th>Peak allocated MiB</th><th>Peak reserved MiB</th></tr></thead><tbody>'
            + ("".join(run_rows) or '<tr><td colspan="9">No training results yet.</td></tr>')
            + '</tbody></table></div><h3>Training episode return</h3>' + svg_chart(task["runs"])
            + '<h3>Evaluation</h3><div class="scroll"><table><thead><tr><th>File</th><th>Controller</th><th>Sampling</th><th>Episodes</th><th>Mean return</th><th>Gain over base</th><th>Success</th><th>Same eval seeds</th></tr></thead><tbody>'
            + ("".join(eval_rows) or '<tr><td colspan="8">No completed evaluations yet.</td></tr>')
            + '</tbody></table></div>' + video_section + '</section>')
    warnings = '<section><h2>Incomplete reads</h2><ul>' + ''.join(f'<li>{escape(warning)}</li>' for warning in report["warnings"]) + '</ul></section>' if report["warnings"] else ''
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Local SAC/PPO demo results</title><style>' + """
body{margin:0;background:#f4f6fa;color:#17273d;font:15px/1.5 system-ui,sans-serif}main{max-width:1220px;margin:32px auto;padding:0 22px}h1{font-size:30px;margin:0 0 6px}h2{font-size:23px;margin:0 0 18px}h3{font-size:16px;margin:22px 0 8px}section{background:white;border:1px solid #dfe5ee;border-radius:12px;margin:22px 0;padding:22px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid #e8ecf1;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:#47596f;background:#f5f7fa}.scroll{overflow-x:auto}svg{display:block;max-width:100%;width:900px}svg text{font:12px system-ui;fill:#54647a}.muted,.legend{color:#617084;font-size:13px}.legend{margin:0 0 10px}li{margin:7px 0}code{overflow-wrap:anywhere}.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}.videos figure{margin:0;min-width:0}.videos figcaption{font-weight:600;margin-bottom:6px}.videos video{width:100%;display:block;background:#17273d;border-radius:6px;aspect-ratio:4/3}.videos a{display:inline-block;font-size:13px;margin-top:7px;overflow-wrap:anywhere}
""" + '</style></head><body><main><h1>Local SAC / PPO demo results</h1>' \
        + f'<p class="muted">Generated {escape(report["generated_at_utc"])} · <code>{escape(report["experiment_directory"])}</code></p>' \
        + '<p>Rerun the report command to refresh results as sequential experiments complete.</p>' \
        + '<section><h2>How to read these results</h2><ul>' + ''.join(f'<li>{escape(caveat)}</li>' for caveat in report["caveats"]) + '</ul></section>' \
        + (''.join(sections) or '<section>No experiment output found yet.</section>') + warnings + '</main></body></html>'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Report destination; defaults to experiment directory")
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    if not root.is_dir():
        parser.error(f"Experiment directory does not exist: {root}")
    report = collect(root)
    output = args.output_dir.resolve() if args.output_dir else root
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("report.json", json.dumps(report, indent=2, allow_nan=False) + "\n"), ("report.html", render(report, output))):
        destination = output / name
        temporary = output / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
        print(destination)
    print(f"{len(report['tasks'])} task(s), {sum(len(task['runs']) for task in report['tasks'])} training run(s), {sum(len(task['evaluations']) for task in report['tasks'])} evaluation(s), {len(report['warnings'])} read warning(s)")


if __name__ == "__main__":
    main()
