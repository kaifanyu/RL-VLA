"""Prepare immutable review evidence as native main/OOD reports complete.

This utility never creates review labels or review-completion markers. All
primary and supplemental evidence remains reviewed:false until a separate
reviewer actually inspects the images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from analyze_vla_trace_contacts import report_episodes, report_plan
from build_vla_trace_review_sheets import build as build_primary
from build_vla_trace_review_supplements import build as build_supplement


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def research_items(root, task):
    return [item for item in report_plan(root)
            if item["task"] == task and (item["domain"] == "ood" or
                item["domain"] == "main" and item["variant"] in {"test", "source"})]


def existing_evidence(root, task, report_path, digest):
    """Reuse only complete evidence for the exact source path and bytes."""
    for path in sorted((root / "video_reviews" / task).glob("*/manifest.json")):
        manifest = read(path)
        if Path(manifest.get("report", "")).resolve() != report_path.resolve():
            continue
        if manifest.get("report_sha256") != digest:
            raise ValueError(f"Previously prepared report changed: {report_path}")
        if manifest.get("reviewed") is not False:
            raise ValueError("Evidence manifest must not claim explicit review")
        if any(not (path.parent / row["sheet"]).is_file() for row in manifest["entries"]):
            raise ValueError(f"Previously prepared sheet missing: {path}")
        return path
    return None


def prepare(root, item):
    path = Path(item["report"])
    if not path.exists():
        return None
    report = read(path)
    episodes = report_episodes(root, item, report)
    if episodes is None:
        return None
    digest = sha(path)
    manifest_path = existing_evidence(root, item["task"], path, digest)
    if manifest_path is None:
        slug = f'{item["domain"]}_{item["policy"]}_{item["variant"]}_ai_prepare_objects_{digest[:12]}'
        output = root / "video_reviews" / item["task"] / slug
        # Existing partial directories are retained as evidence, never overwritten.
        build_primary(path, item["task"], output)
        manifest_path = output / "manifest.json"
    manifest = read(manifest_path)
    if manifest["report_sha256"] != digest or sha(path) != digest:
        raise ValueError("Report changed during evidence generation")
    expected_failures = {(row["reset"]["initial_state_id"], row["seed"])
                         for row in episodes if row["success"] is False}
    actual_failures = {(row["state_id"], row["seed"]) for row in manifest["entries"]}
    if actual_failures != expected_failures or len(manifest["entries"]) != len(expected_failures):
        raise ValueError("Review evidence does not cover exactly the native failures")
    ambiguous = [row for row in manifest["entries"]
                 if len(row["evidence"]["sustained_grasp_candidates"]) != 1]
    supplement_path = manifest_path.parent / "supplemental" / "manifest.json"
    if ambiguous and not supplement_path.exists():
        build_supplement(manifest_path, supplement_path.parent)
    supplements = None
    if supplement_path.exists():
        supplement = read(supplement_path)
        if supplement.get("reviewed") is not False:
            raise ValueError("Supplemental evidence must remain explicitly unreviewed")
        if supplement["primary_manifest_sha256"] != sha(manifest_path):
            raise ValueError("Supplemental evidence belongs to another primary manifest")
        supplements = {"path": str(supplement_path), "sha256": sha(supplement_path)}
    if sha(path) != digest:
        raise ValueError("Report changed during supplemental generation")
    return {"task": item["task"], "domain": item["domain"], "policy": item["policy"],
            "variant": item["variant"], "report": str(path), "report_sha256": digest,
            "episodes": len(episodes), "native_failures": len(expected_failures),
            "manifest": {"path": str(manifest_path), "sha256": sha(manifest_path)},
            "supplemental_manifest": supplements, "reviewed": False,
            "requires_explicit_image_review": True}


def run(root, task, poll_seconds):
    output = root / "video_reviews" / task / "evidence_queue"
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), Path(__file__).with_name("analyze_vla_trace_contacts.py"),
               Path(__file__).with_name("review_vla_trace_episodes.py"),
               Path(__file__).with_name("build_vla_trace_review_sheets.py"),
               Path(__file__).with_name("build_vla_trace_review_supplements.py")]
    source_hashes = {str(path): sha(path) for path in sources}
    atomic(output / "inputs.json", {"task": task, "run": str(root), "sources": source_hashes,
                                    "reviewed": False, "scope": "Evidence preparation only"})
    completed, transient = {}, {}
    while True:
        if any(sha(path) != digest for path, digest in source_hashes.items()):
            raise ValueError("Review preparation source changed while watcher was active")
        items = research_items(root, task)
        assert len(items) == 24
        for item in items:
            key = f'{item["domain"]}/{item["policy"]}/{item["variant"]}'
            if key in completed:
                if sha(completed[key]["report"]) != completed[key]["report_sha256"]:
                    raise ValueError("Completed review source changed")
                continue
            try:
                result = prepare(root, item)
            except (json.JSONDecodeError, FileNotFoundError) as error:
                record = transient.setdefault(key, {"first_seen": time.time(), "error": str(error)})
                if time.time() - record["first_seen"] > 300:
                    raise ValueError(f"Persistent invalid review producer {key}: {error}") from error
                continue
            transient.pop(key, None)
            if result is not None:
                completed[key] = result
                atomic(output / "index.json", {"task": task, "complete": len(completed) == 24,
                       "reports_expected": 24, "reports_prepared": len(completed),
                       "reviewed": False, "entries": list(completed.values())})
        atomic(output / "status.json", {"state": "complete" if len(completed) == 24 else "watching",
               "reports_prepared": len(completed), "reports_expected": 24,
               "updated_unix_seconds": time.time(), "pending_producers": transient,
               "reviewed": False})
        if len(completed) == 24:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["object_0", "object_1", "object_2"])
    parser.add_argument("--poll-seconds", type=float, default=45)
    args = parser.parse_args()
    run(args.run.resolve(), args.task, args.poll_seconds)
