"""Versioned review inventory accepting the original immutable marker contract.

The v1 producer and all existing markers remain unchanged; only comparison of
optional marker metadata differs from the original inventory consumer.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from analyze_vla_trace_objects import audit_selection, checked_episodes, evaluation_path

POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")
VARIANTS = ("both_new_regions", "continuous_new_offsets", "target_moved_distractor_fixed", "distractor_moved_target_fixed")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Hashes:
    def __init__(self):
        self.cache = {}

    def get(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self.cache.get(path, (None, None))[0] != identity:
            result = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(1024**2):
                    result.update(block)
            self.cache[path] = (identity, result.hexdigest())
        return self.cache[path][1]


def verified_reviews(root, task, hashes):
    labels = {}
    sources = {}
    for path in sorted((root / "video_reviews/reviewed_labels").glob("*.json")):
        records = [row for row in read(path)["records"] if row["task"] == task and row.get("reviewed") is True]
        if not records:
            continue
        sources[str(path)] = hashes.get(path)
        for row in records:
            for field in ("report", "video", "sheet", "evidence_manifest"):
                if hashes.get(row[field]) != row[field + "_sha256"]:
                    raise ValueError(f"Review evidence changed: {row[field]}")
            key = (str(Path(row["report"]).resolve()), row["state_id"], row["seed"])
            if key in labels and labels[key]["failure_type"] != row["failure_type"]:
                raise ValueError("Conflicting immutable review labels")
            labels[key] = {**row, "label_source": str(path), "label_source_sha256": sources[str(path)]}
    return labels, sources


def expected_reports(root, task):
    rows = []
    for split in ("test", "source", "validation"):
        for policy in POLICIES:
            rows.append({"domain": "main", "task": task, "policy": policy, "split": split,
                         "report": str(evaluation_path(root, task, policy, split)),
                         "selection_required": split != "validation"})
    for policy in POLICIES:
        for variant in VARIANTS:
            rows.append({"domain": "ood", "task": task, "policy": policy, "variant": variant,
                         "report": str(root / "ood" / task / policy / variant / "summary.json"),
                         "selection_required": True})
    return rows


def validate_baseline_marker(actual, expected):
    """Require original core bindings and validate richer metadata if supplied."""
    required = {"complete", "task", "report_sha256", "review_records"}
    if not isinstance(actual, dict) or not required.issubset(actual):
        raise ValueError("Existing immutable baseline-review marker lacks required core fields")
    for key, value in actual.items():
        if key not in expected or value != expected[key]:
            raise ValueError(f"Existing immutable baseline-review marker differs at {key}")


def inspect(root, task, hashes, *, prepare_sheets=False):
    selection = audit_selection(root / "methods/rldt" / task)
    definition = read(root / "tasks" / task / "protocol.json")["tasks"][task]
    labels, sources = verified_reviews(root, task, hashes)
    rows = expected_reports(root, task)
    for row in rows:
        path = Path(row["report"])
        row["expected_episodes"] = 50
        if row["selection_required"] and selection["status"] != "locked":
            row["status"] = "pending_locked_selection"
            continue
        value = read(path)
        if value is None or (row["domain"] == "ood" and value.get("complete") is not True):
            row["status"] = "pending_report"
            continue
        episodes = checked_episodes(value, path, definition if row["domain"] == "main" else {"ood_state_ids": list(range(50))},
                                    row.get("split", "ood"))
        if len(episodes) != 50:
            raise ValueError("Review report is not the complete fifty-case cohort")
        failed = [episode for episode in episodes if not episode["success"]]
        covered = []
        remaining = []
        for episode in failed:
            key = (str(path.resolve()), episode["reset"]["initial_state_id"], episode["seed"])
            reviewed = labels.get(key)
            if reviewed:
                if reviewed["success"] is not False or reviewed["video"] != episode.get("video"):
                    raise ValueError("Reviewed episode identity does not match native failed episode")
                covered.append(reviewed)
            else:
                remaining.append({"state_id": key[1], "seed": key[2], "video": episode.get("video")})
        review_sources = {record["label_source"]: record["label_source_sha256"] for record in covered}
        row.update(report_sha256=hashes.get(path), episodes=50, native_successes=50 - len(failed),
                   failed_episodes=len(failed), reviewed_failures=len(covered), unreviewed_failures=len(remaining),
                   remaining_cases=remaining, review_records=[{"path": path, "sha256": checksum} for path, checksum in sorted(review_sources.items())],
                   status="pending_review" if remaining else "complete_review")
        if not remaining and row["domain"] == "main" and row["split"] == "test" and row["policy"] == "pi05_libero":
            marker = root / "video_reviews" / task / "baseline_test_review_complete.json"
            record = {"complete": True, "task": task, "report": str(path.resolve()), "report_sha256": row["report_sha256"],
                      "review_records": row["review_records"], "failed_episodes": len(failed),
                      "native_successes": 50 - len(failed), "no_failure_review_required": not failed}
            if marker.exists():
                validate_baseline_marker(read(marker), record)
            if not marker.exists():
                write(marker, record)
        if remaining and prepare_sheets:
            label = row.get("variant", row.get("split"))
            output = root / "video_reviews" / task / "prepared" / row["domain"] / row["policy"] / label
            if not output.exists():
                command = [sys.executable, str(Path(__file__).with_name("build_vla_trace_review_sheets.py")),
                           "--report", str(path), "--task", task, "--output", str(output)]
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.with_suffix(".log").open("x") as stream:
                    subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            evidence = read(output / "manifest.json")
            if not evidence or evidence["report_sha256"] != row["report_sha256"]:
                raise ValueError("Prepared review sheets are incomplete or bind another report")
            row["evidence_manifest"] = str(output / "manifest.json")
            row["evidence_manifest_sha256"] = hashes.get(output / "manifest.json")
            row["sheets_to_inspect"] = sorted({str(output / entry["sheet"]) for entry in evidence["entries"]})
    result = {"task": task, "expected_reports": 28, "expected_episodes": 1400,
              "complete": all(row["status"] == "complete_review" for row in rows),
              "complete_review_reports": sum(row["status"] == "complete_review" for row in rows),
              "selection": selection, "reports": rows, "existing_review_sources": sources,
              "scope": "Evidence inventory only; no automatic failure labels. Every failed case requires explicit selected-frame and native-evidence review. Native-success-only cohorts need no invented failure labels."}
    write(root / "video_reviews" / task / "review_inventory.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("object_0", "object_1", "object_2"))
    parser.add_argument("--prepare-sheets", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    cache = Hashes()
    while True:
        result = inspect(args.run.resolve(), args.task, cache, prepare_sheets=args.prepare_sheets)
        print(json.dumps({"task": args.task, "complete": result["complete"], "complete_review_reports": result["complete_review_reports"]}), flush=True)
        if not args.watch or result["complete"]:
            break
        time.sleep(60)
