"""Fixed OOD presentation of existing statistics, with independent byte validation."""
import csv
import hashlib
import json
from pathlib import Path

from rl_vla.diagnostics.figure_artifacts import artifact, identity, read, sha, write
from rl_vla.diagnostics.ood_completion import POLICIES, VARIANTS

TASKS = ("object_0", "object_1", "object_2")
CATEGORIES = ("success", "wrong object", "correct grasp but placement failure", "correct target but failed grasp",
              "timeout", "other/ambiguous", "not reviewed")
COLORS = ("#278657", "#bd3548", "#df982f", "#b46a32", "#7c6289", "#8d74b8", "#b9b9b9")
POLICY_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")
LABELS = ("Both new regions", "Continuous new offsets", "Target moved; distractor fixed", "Distractor moved; target fixed")


def indexed(rows, task):
    result = {}
    for row in rows:
        if row["task"] != task:
            continue
        key = row["policy"], row["variant"]
        if key in result or key[0] not in POLICIES or key[1] not in VARIANTS:
            raise ValueError("OOD figure duplicate or unknown evidence cell")
        result[key] = row
    return result


def inputs(root, task, successes, pairs, reviews, taxonomy, episodes):
    score, pair, review = (indexed(rows, task) for rows in (successes, pairs, reviews))
    categories = {}
    for row in taxonomy:
        if row["task"] != task:
            continue
        key = row["policy"], row["variant"], row["failure_type"]
        if (key in categories or key[0] not in POLICIES or key[1] not in VARIANTS or key[2] not in CATEGORIES
                or type(row["count"]) is not int or not 0 <= row["count"] <= 50):
            raise ValueError("OOD figure taxonomy is duplicated or unsupported")
        categories[key] = row["count"]
    sources = {}
    for row in episodes:
        if row["task"] != task:
            continue
        for field in ("report", "review_source"):
            if row.get(field):
                sources[row[field]] = sha(row[field])
    source_rows, pair_rows, category_rows, review_rows = [], [], [], []
    for variant in VARIANTS:
        for policy in POLICIES:
            key = policy, variant
            base = {"task": task, "policy": policy, "variant": variant}
            source_rows.append({**base, "status": "complete" if key in score else "pending", **score.get(key, {})})
            review_rows.append({**base, "status": "pending_report", **review.get(key, {})})
            if policy != "pi05_libero":
                pair_rows.append({**base, "status": "complete" if key in pair else "pending", **pair.get(key, {})})
            counts = [categories.get((*key, kind), 0) for kind in CATEGORIES] if key in score else [None] * len(CATEGORIES)
            if key in score and (score[key]["episodes"] != 50 or sum(counts) != 50 or counts[0] != score[key]["successes"]):
                raise ValueError("OOD figure taxonomy does not reproduce all 50 native outcomes")
            if key in review and counts[-1] != review[key]["unreviewed_failures"]:
                raise ValueError("OOD figure reviewed coverage conflicts with taxonomy")
            category_rows.extend({**base, "category": kind, "count": count,
                                  "status": "complete" if key in score else "pending"}
                                 for kind, count in zip(CATEGORIES, counts, strict=True))
    return {"task": task, "success": source_rows, "paired": pair_rows, "categories": category_rows,
            "review_coverage": review_rows, "sources": sources,
            "renderer_sha256": sha(__file__),
            "scope": "Existing predeclared cell statistics only; no pooling across correlated variants, no new intervals/tests, no automatic contact labels in reviewed categories. Raw-base actions N/A."}


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def complete_data(data):
    task = data["task"]
    score, pairs, reviews = (indexed(data[name], task) for name in ("success", "paired", "review_coverage"))
    expected = {(policy, variant) for policy in POLICIES for variant in VARIANTS}
    if set(score) != expected or set(reviews) != expected or set(pairs) != {key for key in expected if key[0]!="pi05_libero"}:
        raise ValueError("OOD figure raw tables do not cover the fixed matrix")
    return (all(row["status"] == "complete" for row in list(score.values()) + list(pairs.values()))
            and all(row["status"] == "complete_review" and row.get("review_coverage_complete") is True
                    and row.get("episodes") == 50 and row.get("unreviewed_failures") == 0
                    and data["sources"].get(row.get("report")) == row.get("report_sha256")
                    and row.get("report_sha256") is not None for row in reviews.values()))


def draw(folder, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    score = {(row["policy"], row["variant"]): row for row in data["success"]}
    pair = {(row["policy"], row["variant"]): row for row in data["paired"]}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    for column, variant in enumerate(VARIANTS):
        top, bottom = axes[:, column]
        for index, policy in enumerate(POLICIES):
            row = score[policy, variant]
            if row["status"] == "complete":
                y = row["success_rate"]
                top.errorbar(index, y, yerr=[[max(0., y-row["wilson95_low"])], [max(0., row["wilson95_high"]-y)]], fmt="o", capsize=4, color=POLICY_COLORS[index])
                top.annotate(f"{row['successes']}/50", (index, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
            else:
                top.text(index, .48, "pending", rotation=90, ha="center", color=".5", fontsize=8)
            if index:
                row = pair[policy, variant]
                if row["status"] == "complete":
                    y = row["difference"]
                    low, high = row["paired_bootstrap95"]
                    bottom.errorbar(index, y, yerr=[[max(0., y-low)], [max(0., high-y)]], fmt="o", capsize=4, color=POLICY_COLORS[index])
                    bottom.annotate(f"R{row['recovered']}/L{row['regressed']}", (index, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
                else:
                    bottom.text(index, .1, "pending", rotation=90, ha="center", color=".5", fontsize=8)
        top.set(title=LABELS[column], ylim=(-.03, 1.1), ylabel="Native success · Wilson 95%")
        bottom.set(ylim=(-1.08, 1.08), ylabel="Paired change vs LIBERO · 95% interval")
        bottom.axhline(0, color=".6", linewidth=.7)
        for axis in (top, bottom):
            axis.set(xticks=range(4), xticklabels=("LIBERO", "25", "50", "100"), xlim=(-.5, 3.5))
            axis.grid(axis="y", alpha=.2)
    fig.suptitle(f"{data['task']} · Custom OOD layouts kept separate · R/L = recovered/regressed\nTwo fixed new centers with overlapping offset neighborhoods; no pooled 200-region estimate · raw base: N/A")
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"success_and_paired.{suffix}", dpi=170)
    plt.close(fig)
    category = {(row["policy"], row["variant"], row["category"]): row["count"] for row in data["categories"]}
    fig, axes = plt.subplots(1, 4, figsize=(17, 5), layout="constrained")
    for axis, variant, title in zip(axes, VARIANTS, LABELS, strict=True):
        for index, policy in enumerate(POLICIES):
            if score[policy, variant]["status"] != "complete":
                axis.bar(index, 1, color="none", edgecolor=".7", hatch="///")
                axis.text(index, .5, "pending", ha="center", rotation=90, color=".4")
                continue
            base = 0.
            for kind, color in zip(CATEGORIES, COLORS, strict=True):
                count = category[policy, variant, kind]
                if count:
                    axis.bar(index, count/50, bottom=base, color=color, hatch="///" if kind=="not reviewed" else None)
                    axis.text(index, base+count/100, str(count), ha="center", va="center", fontsize=8)
                    base += count/50
        axis.set(title=title, ylim=(0, 1.03), ylabel="Cases / all 50 native episodes", xticks=range(4), xticklabels=("LIBERO", "25", "50", "100"))
    fig.suptitle(f"{data['task']} · Native success and reviewed failure categories\nSelected-frame AI review; ambiguous and not-reviewed remain distinct · not exhaustive first-choice errors")
    fig.legend(handles=[Patch(facecolor=color, label=kind, hatch="///" if kind=="not reviewed" else None)
                        for kind, color in zip(CATEGORIES, COLORS, strict=True)], loc="outside lower center", ncols=4, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"reviewed_outcomes.{suffix}", dpi=170)
    plt.close(fig)


def render(root, task, successes, pairs, reviews, taxonomy, episodes):
    root = Path(root).resolve()
    data = inputs(root, task, successes, pairs, reviews, taxonomy, episodes)
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    key = hashlib.sha256(payload).hexdigest()
    task_root = root / "figures/ood" / task
    folder = task_root / "versions" / key
    if (folder / "manifest.json").exists():
        manifest = read(folder / "manifest.json")
        for value in manifest["artifacts"]:
            artifact(value["path"], expected_sha=value["sha256"])
        write(task_root / "summary.json", manifest)
        return manifest
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "inputs.json").write_bytes(payload)
    for name in ("success", "paired", "categories", "review_coverage"):
        save_csv(folder / f"{name}.csv", data[name])
    draw(folder, data)
    complete = complete_data(data)
    manifest = {"task": task, "complete": complete, "passed": True if complete else None,
                "expected_native_cells": 16, "expected_paired_contrasts": 12, "expected_review_cells": 16,
                "native_cells": sum(row["status"]=="complete" for row in data["success"]),
                "paired_contrasts": sum(row["status"]=="complete" for row in data["paired"]),
                "complete_review_cells": sum(row["status"]=="complete_review" for row in data["review_coverage"]),
                "input_sha256": key, "renderer": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
                "sources": data["sources"], "scope": data["scope"],
                "artifacts": [artifact(path) for path in sorted(folder.iterdir())]}
    write(folder / "manifest.json", manifest)
    write(task_root / "summary.json", manifest)
    return manifest


def verify(root, task, *, full_hash=False):
    path = Path(root) / "figures/ood" / task / "summary.json"
    if not path.exists():
        return {"task": task, "complete": False, "passed": None, "reason": "OOD figures pending"}
    try:
        value = read(path)
        if not isinstance(value, dict):
            raise TypeError("OOD figure summary is not an object")
        if value.get("complete") is not True:
            return {"task": task, "complete": False, "passed": None, "reason": "OOD figures retain pending native/paired/review cells"}
        if value["task"] != task or value["passed"] is not True or (value["native_cells"],value["paired_contrasts"],value["complete_review_cells"]) != (16,12,16):
            raise ValueError("OOD figure coverage is incomplete")
        if value["renderer"]["sha256"] != sha(__file__):
            raise ValueError("OOD figures use a different renderer source")
        expected = {"inputs.json", "success.csv", "paired.csv", "categories.csv", "review_coverage.csv",
                    "success_and_paired.png", "success_and_paired.pdf", "reviewed_outcomes.png", "reviewed_outcomes.pdf"}
        if {Path(row["path"]).name for row in value["artifacts"]} != expected or len(value["artifacts"]) != len(expected):
            raise ValueError("OOD required figures/raw tables missing")
        for source, expected_sha in value["sources"].items():
            if sha(source) != expected_sha:
                raise ValueError("OOD figure source report/review changed")
        for row in value["artifacts"]:
            if identity(row["path"]) != row["identity"] or full_hash and sha(row["path"]) != row["sha256"]:
                raise ValueError("OOD figure artifact changed")
        input_row = next(row for row in value["artifacts"] if Path(row["path"]).name=="inputs.json")
        data = read(input_row["path"])
        if (input_row["sha256"] != value["input_sha256"] or sha(input_row["path"]) != value["input_sha256"]
                or data["task"] != task or data["renderer_sha256"] != sha(__file__) or data["sources"] != value["sources"]
                or not complete_data(data)):
            raise ValueError("OOD complete figure marker does not match its full reviewed input matrix")
        return {"task": task, "complete": True, "passed": True, "manifest": str(path), "sha256": sha(path)}
    except (OSError, KeyError, TypeError, ValueError) as error:
        return {"task": task, "complete": False, "passed": False, "reason": str(error)}
