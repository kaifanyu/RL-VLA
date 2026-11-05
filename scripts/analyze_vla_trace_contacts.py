"""Incremental, cached native contact evidence for all192 scientific reports."""

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from rl_vla.diagnostics.contact_evidence import (
    DISTRACTORS,
    RULE,
    TARGETS,
    causal_steps,
    evidence,
    native_steps,
    stat_record,
)

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")
OOD = ("both_new_regions", "continuous_new_offsets", "target_moved_distractor_fixed", "distractor_moved_target_fixed")
CAUSAL = ("none", "no_image", "no_text", "target:background_fill", "distractor:background_fill",
          "random:background_fill", "target:mosaic", "distractor:mosaic", "random:mosaic")
EXPECTED_REPORTS, EXPECTED_EPISODES = 192, 5280


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            digest.update(data)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def write_csv(path, rows, fields):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), allow_nan=False) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in fields})
    temporary.replace(path)


class FileCache:
    def __init__(self, records):
        self.records = records

    def hash(self, path):
        key = str(Path(path).resolve())
        current = stat_record(path)
        previous = self.records.get(key)
        if previous and previous["stat"] == current:
            return previous["sha256"]
        digest = checksum(path)
        if current != stat_record(path):
            raise RuntimeError(f"File changed while hashing: {path}")
        self.records[key] = {"stat": current, "sha256": digest}
        return digest


def report_plan(root):
    """Enumerate canonical cells once; selected checkpoint aliases never add cells."""
    result = []
    for task in TARGETS:
        method = root/"methods/rldt"/task
        selection_path = method/"selection.json"
        selection = read(selection_path) if selection_path.exists() else None
        for policy in POLICIES:
            iteration = int(policy.split("_")[1]) if policy.startswith("rldt_") else None
            for split in ("validation", "test", "source"):
                standard = root/"evaluations"/task/policy/split/"summary.json"
                if iteration is None:
                    fallback = method/f"baseline_{split}.json"
                elif split == "validation":
                    fallback = method/f"validation_{iteration:06d}.json"
                else:
                    fallback = method/f"checkpoint_{iteration:06d}_{split}.json"
                    if not fallback.exists() and selection and selection["iteration"] == iteration:
                        fallback = method/f"selected_{split}.json"
                result.append({"domain": "main", "task": task, "policy": policy, "variant": split,
                               "report": str(standard if standard.exists() else fallback), "episodes": 50})
            for variant in OOD:
                result.append({"domain": "ood", "task": task, "policy": policy, "variant": variant,
                               "report": str(root/"ood"/task/policy/variant/"summary.json"), "episodes": 50})
            for condition in CAUSAL:
                result.append({"domain": "causal", "task": task, "policy": policy, "variant": condition,
                               "report": str(root/"interventions"/task/policy/condition.replace(":", "_")/"summary.json"), "episodes": 10})
    assert len(result) == EXPECTED_REPORTS and sum(row["episodes"] for row in result) == EXPECTED_EPISODES
    return result


def report_episodes(root, item, report):
    if report.get("technical_only") or report.get("technical_smoke"):
        raise ValueError("Technical smoke cannot enter scientific contact coverage")
    if item["domain"] != "main" and not report.get("complete", report.get("completed", False)):
        return None
    episodes = report.get("episodes")
    if not isinstance(episodes, list):
        raise TypeError("Completed report lacks explicit episode records")
    definition = read(root/"tasks"/item["task"]/"protocol.json")["tasks"][item["task"]]
    expected_ids = (list(range(50)) if item["domain"] == "ood" else
                    definition["test_state_ids"][:10] if item["domain"] == "causal" else
                    definition[f"{item['variant']}_state_ids"])
    actual_ids = [episode.get("initial_state_id", episode.get("reset", {}).get("initial_state_id")) for episode in episodes]
    if len(episodes) != item["episodes"] or actual_ids != expected_ids:
        raise ValueError("Contact source is not the exact complete predetermined state set")
    if [episode["seed"] for episode in episodes] != list(range(100000, 100000+len(episodes))):
        raise ValueError("Contact source differs from frozen episode noise seeds")
    if any(type(episode["success"]) is not bool or not 0 < episode["primitive_steps"] <= 280 for episode in episodes):
        raise ValueError("Contact source has invalid native outcome or controlled-step count")
    iteration = int(item["policy"].split("_")[1]) if item["policy"].startswith("rldt_") else None
    expected_checkpoint = root/"methods/rldt"/item["task"]/"training"/f"iteration_{iteration:06d}" if iteration is not None else None
    supplied = [episode.get("checkpoint") for episode in episodes] if item["domain"] == "causal" else [report.get("checkpoint")]
    if any((value not in (None, "pi05_libero") if expected_checkpoint is None else
            value is None or Path(value).resolve() != expected_checkpoint.resolve()) for value in supplied):
        raise ValueError("Contact report checkpoint does not match its policy identity")
    for episode in episodes:
        if item["domain"] == "causal" and episode["condition"] != item["variant"]:
            raise ValueError("Causal report condition differs")
        if episode.get("technical_smoke"):
            raise ValueError("Technical episode cannot enter scientific contact coverage")
        reset = episode.get("reset", {})
        expected_split = "test" if item["domain"] == "causal" else item["variant"]
        if reset.get("task_id") != int(item["task"][-1]) or reset.get("experiment_split") != expected_split:
            raise ValueError("Contact episode task or environment split differs from report identity")
    return episodes


def process_report(root, item, report, payload, destination, code_hash):
    episodes = report_episodes(root, item, report)
    if episodes is None:
        return None
    digest = hashlib.sha256(payload).hexdigest()
    artifact = destination/"reports"/digest
    if artifact.exists():
        manifest = read(artifact/"summary.json")
        if manifest["code_hash"] != code_hash or manifest["identity"] != {k: item[k] for k in ("domain", "task", "policy", "variant")}:
            raise ValueError("Existing contact artifact was produced with different code or report identity")
        return manifest
    temporary = destination/"staging"/f"{digest}.{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)  # our unpublished temporary calculation only
    temporary.mkdir(parents=True)
    rows, sources = [], []
    try:
        for index, episode in enumerate(episodes):
            target, distractor = TARGETS[item["task"]], DISTRACTORS[item["task"]]
            if item["domain"] == "causal":
                if episode["episode"] != index:
                    raise ValueError("Causal episode ordinal differs from telemetry filename")
                telemetry = Path(item["report"]).parent/f"episode_{index:03d}_telemetry.json"
                steps = causal_steps(telemetry, episode, sources)
                trajectory = None
            else:
                telemetry = None
                trajectory = episode["trajectory"]
                steps = native_steps(trajectory, episode, sources)
            measured = evidence(steps, target=target, distractor=distractor,
                                primitive_steps=episode["primitive_steps"], success=episode["success"],
                                reset_telemetry=episode.get("reset", {}).get("object_telemetry"))
            if not Path(episode["video"]).is_file():
                raise FileNotFoundError(episode["video"])
            rows.append({**{k: item[k] for k in ("domain", "task", "policy", "variant", "report")},
                         "report_sha256": digest, "episode": index,
                         "state_id": episode.get("initial_state_id", episode.get("reset", {}).get("initial_state_id")),
                         "seed": episode["seed"], "success": episode["success"], "target": target,
                         "distractor": distractor, "video": episode["video"], "trajectory": trajectory,
                         "telemetry_path": str(telemetry) if telemetry else None,
                         "contact_artifact": str(artifact/"episodes.jsonl"), **measured})
        if checksum(item["report"]) != digest:
            raise ValueError("Completed source report changed while contact evidence was extracted")
        (temporary/"report.json").write_bytes(payload)
        (temporary/"episodes.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows))
        (temporary/"source_inputs.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in sources))
        summary = {"complete": True, "identity": {k: item[k] for k in ("domain", "task", "policy", "variant")},
                   "report": item["report"], "report_sha256": digest, "episodes": len(rows),
                   "telemetry_complete_episodes": sum(row["telemetry_complete"] for row in rows),
                   "artifact": str(artifact), "code_hash": code_hash, "rule": RULE,
                   "evidence_sha256": checksum(temporary/"episodes.jsonl"),
                   "source_inputs_sha256": checksum(temporary/"source_inputs.jsonl"),
                   "automatic_only": True, "reviewed": False}
        atomic_json(temporary/"summary.json", summary)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(artifact)
        return summary
    except BaseException:
        if temporary.exists():
            failed = destination/"failed_attempts"/f"{digest}.{time.time_ns()}"
            failed.parent.mkdir(parents=True, exist_ok=True)
            temporary.rename(failed)
        raise


def manual_reviews(root, files):
    """Verify true reviewed sources independently; never infer a reviewed flag."""
    labels, source_hashes = {}, {}
    for path in sorted((root/"video_reviews/reviewed_labels").glob("*.json")):
        records = read(path)
        source_hashes[str(path)] = files.hash(path)
        for row in records["records"]:
            if row.get("reviewed") is not True:
                continue
            for name in ("report", "video", "sheet", "evidence_manifest"):
                if files.hash(row[name]) != row[f"{name}_sha256"]:
                    raise ValueError(f"Manual review evidence hash changed: {row[name]}")
            key = (row["task"], row["report_sha256"], row["state_id"], row["seed"])
            if key in labels and labels[key]["failure_type"] != row["failure_type"]:
                raise ValueError("Conflicting manual labels require explicit reviewer resolution")
            labels[key] = {**row, "label_source": str(path), "label_source_sha256": source_hashes[str(path)]}
    return labels, source_hashes


def publish(root, destination, manifests, labels, manual_hashes, errors, pending):
    signature = canonical_hash({"reports": [(m["report_sha256"], m["evidence_sha256"]) for m in manifests],
                                "manual": manual_hashes, "errors": errors, "pending": pending})
    current = destination/"summary.json"
    if current.exists() and read(current).get("evidence_signature") == signature:
        return read(current)
    cases, calibration, groups = [], [], defaultdict(list)
    for manifest in manifests:
        for line in (Path(manifest["artifact"])/"episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            manual = labels.get((row["task"], row["report_sha256"], row["state_id"], row["seed"]))
            if manual and (manual["success"] != row["success"] or manual["video"] != row["video"]):
                raise ValueError("Manual review and automatic case identity differ")
            row.update(manual_reviewed=bool(manual), manual_failure_type=manual["failure_type"] if manual else None,
                       manual_grasped_object_id=manual.get("grasped_object_id") if manual else None,
                       manual_label_source=manual["label_source"] if manual else None,
                       manual_label_source_sha256=manual["label_source_sha256"] if manual else None,
                       manual_notes=manual.get("notes") if manual else None)
            cases.append(row)
            groups[row["domain"], row["task"], row["policy"], row["variant"]].append(row)
            if manual:
                calibration.append({k: row[k] for k in ("domain", "task", "policy", "variant", "state_id", "seed",
                    "report", "report_sha256", "success", "video", "telemetry_complete", "sustained_grasp_candidates",
                    "operational_outcome", "provisional_failure_type", "manual_failure_type", "manual_grasped_object_id",
                    "manual_label_source", "manual_label_source_sha256", "manual_notes")})
    rates = []
    for key, rows in sorted(groups.items()):
        complete = [row for row in rows if row["telemetry_complete"]]
        manual = [row for row in rows if row["manual_reviewed"]]
        rate = {**dict(zip(("domain", "task", "policy", "variant"), key, strict=True)),
                "episodes": len(rows), "native_successes": sum(row["success"] for row in rows),
                "telemetry_complete_episodes": len(complete), "manual_reviewed_episodes": len(manual),
                "manual_wrong_object_episodes": sum(row["manual_failure_type"] == "wrong object" for row in manual),
                "manual_placement_failure_episodes": sum(row["manual_failure_type"] == "correct grasp but placement failure" for row in manual),
                "manual_ambiguous_episodes": sum(row["manual_failure_type"] == "other/ambiguous" for row in manual),
                "automatic_labels_reviewed": False, "candidate_denominator": "telemetry_complete_episodes"}
        for name in ("target_grasp_candidate", "non_target_grasp_candidate", "mixed_target_non_target_candidates",
                     "no_grasp_candidate", "first_confirmation_tie"):
            count = sum(row[name] for row in complete)
            rate[name+"_count"] = count
            rate[name+"_rate"] = count/len(complete) if complete else None
        rate["first_target_candidate_count"] = sum(row["first_candidate_role"] == "target" for row in complete)
        rate["first_target_candidate_rate"] = rate["first_target_candidate_count"]/len(complete) if complete else None
        rates.append(rate)
    tables = root/"tables"
    tables.mkdir(exist_ok=True)
    fields = ["domain", "task", "policy", "variant", "report", "report_sha256", "episode", "state_id", "seed", "success",
              "target", "distractor", "video", "trajectory", "telemetry_path", "contact_artifact", "primitive_steps", "measured_steps",
              "telemetry_complete", "sustained_grasp_candidates", "non_target_candidates", "target_grasp_candidate", "non_target_grasp_candidate",
              "mixed_target_non_target_candidates", "multiple_candidate_objects", "no_grasp_candidate", "first_contact_confirmed_grasp_id",
              "first_contact_confirmed_grasp_ids", "first_confirmation_tie", "first_candidate_confirmation_step",
              "first_candidate_confirmation_sim_seconds", "first_candidate_confirmation_elapsed_sim_seconds", "first_candidate_role",
              "operational_outcome", "provisional_failure_type", "reviewed", "automatic", "manual_reviewed", "manual_failure_type",
              "manual_grasped_object_id", "manual_label_source", "manual_label_source_sha256"]
    write_csv(tables/"contact_cases.csv", cases, fields)
    write_csv(tables/"contact_candidate_rates.csv", rates, list(rates[0]) if rates else ["domain", "task", "policy", "variant", "episodes"])
    write_csv(tables/"contact_manual_calibration.csv", calibration, list(calibration[0]) if calibration else ["domain", "task", "state_id", "manual_failure_type"])
    summary = {"schema_version": 1, "evidence_signature": signature,
               "complete": len(manifests) == EXPECTED_REPORTS and len(cases) == EXPECTED_EPISODES and all(row["telemetry_complete"] for row in cases) and not errors,
               "reports_processed": len(manifests), "reports_expected": EXPECTED_REPORTS,
               "episodes_processed": len(cases), "episodes_expected": EXPECTED_EPISODES,
               "telemetry_complete_episodes": sum(row["telemetry_complete"] for row in cases),
               "reports_by_domain": dict(Counter(m["identity"]["domain"] for m in manifests)),
               "episodes_by_domain": dict(Counter(row["domain"] for row in cases)),
               "manual_reviewed_episodes": len(calibration), "automatic_labels_reviewed": False,
               "manual_failure_types": dict(Counter(row["manual_failure_type"] for row in calibration)),
               "manual_ambiguous_without_strict_candidate": [{k: row[k] for k in ("task", "state_id", "seed", "manual_notes", "report_sha256")}
                     for row in calibration if row["manual_failure_type"] == "other/ambiguous" and not row["sustained_grasp_candidates"]],
               "manual_mixed_candidate_cases": [{k: row[k] for k in ("task", "state_id", "seed", "sustained_grasp_candidates", "manual_notes", "report_sha256")}
                     for row in calibration if row["operational_outcome"] == "mixed_target_and_non_target_candidates"],
               "rule": RULE, "manual_label_sources": manual_hashes, "pending_producers": pending, "persistent_errors": errors,
               "reports": manifests, "tables": {name: {"path": str(tables/name), "sha256": checksum(tables/name)}
                   for name in ("contact_cases.csv", "contact_candidate_rates.csv", "contact_manual_calibration.csv")},
               "limitations": ["Automatic contact-confirmed candidates are not manually reviewed semantic failures.",
                   "The strict contact/lift rule can miss transported objects and can produce mixed sequences; absence of a candidate is not proof of no grasp.",
                   "Manual review is a separately verified selected subset. Its operational-rule calibration is not an independent precision/sensitivity estimate.",
                   "Cached report artifacts bind the exact metadata/timestamps read. Unchanged report hashes are not reparsed; image arrays are not loaded."]}
    atomic_json(current, summary)
    return summary


def refresh(root, destination, cache, code_hash):
    files = FileCache(cache.setdefault("files", {}))
    processed = cache.setdefault("reports", {})
    retry = cache.setdefault("retry", {})
    manifests, errors, pending = [], [], []
    for item in report_plan(root):
        path = Path(item["report"])
        if not path.exists():
            continue
        try:
            before = stat_record(path)
            payload = path.read_bytes()
            report = json.loads(payload)
            if before != stat_record(path):
                raise RuntimeError("Producer changed summary during read")
            digest = hashlib.sha256(payload).hexdigest()
            old = processed.get(str(path))
            if old and old["report_sha256"] != digest:
                raise ValueError("Previously completed source report changed; prior immutable evidence preserved")
            manifest = old or process_report(root, item, report, payload, destination, code_hash)
            if manifest is None:
                continue
            processed[str(path)] = manifest
            manifests.append(manifest)
            retry.pop(str(path), None)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            current = stat_record(path) if path.exists() else None
            prior = retry.get(str(path), {})
            count = min(3, prior.get("count", 0)+1) if prior.get("stat") == current else 1
            retry[str(path)] = {"stat": current, "count": count, "error": str(exc)}
            row = {**{k: item[k] for k in ("domain", "task", "policy", "variant", "report")}, "error": str(exc)}
            (errors if count >= 3 else pending).append(row)
    try:
        labels, manual_hashes = manual_reviews(root, files)
        cache["manual"] = {"labels": list(labels.values()), "sources": manual_hashes}
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        pending.append({"domain": "manual_reviews", "error": str(exc)})
        previous = cache.get("manual", {"labels": [], "sources": {}})
        labels, manual_hashes = {}, {}
        for row in previous["labels"]:
            try:
                if (files.hash(row["label_source"]) != row["label_source_sha256"] or
                        any(files.hash(row[name]) != row[f"{name}_sha256"] for name in ("report", "video", "sheet", "evidence_manifest"))):
                    continue
            except OSError:
                continue
            labels[row["task"], row["report_sha256"], row["state_id"], row["seed"]] = row
            manual_hashes[row["label_source"]] = row["label_source_sha256"]
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        errors.append({"domain": "manual_reviews", "error": str(exc)})
        labels, manual_hashes = {}, {}
    result = publish(root, destination, manifests, labels, manual_hashes, errors, pending)
    atomic_json(destination/"cache.json", cache)
    return result


def validate_baseline(root, destination):
    path = ROOT/"scripts/review_vla_trace_episodes.py"
    spec = importlib.util.spec_from_file_location("root_contact_rule", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compared, mixed = 0, []
    for task in TARGETS:
        report_path = root/"methods/rldt"/task/"baseline_validation.json"
        digest = checksum(report_path)
        rows = [json.loads(line) for line in (destination/"reports"/digest/"episodes.jsonl").read_text().splitlines()]
        if len(rows) != 50:
            raise ValueError("Baseline audit requires exactly50 validation episodes per task")
        for row in rows:
            legacy = module.episode_evidence(row["trajectory"], row["target"])
            if sorted(legacy["sustained_grasp_candidates"]) != row["sustained_grasp_candidates"] or legacy["measured_steps"] != row["measured_steps"]:
                raise ValueError("Sidecar rule differs from original root review rule")
            for name, state in legacy["objects"].items():
                for field in ("contact_steps", "max_consecutive_lift_contact", "max_lift_m", "min_gripper_distance_m", "first_contact_step"):
                    if state[field] != row["objects"][name][field]:
                        raise ValueError("Sidecar object measurements differ from root rule")
            if not row["telemetry_complete"]:
                raise ValueError("Baseline native telemetry is incomplete")
            if row["mixed_target_non_target_candidates"]:
                mixed.append({"task": task, "state_id": row["state_id"], "candidates": row["sustained_grasp_candidates"]})
            compared += 1
    summary = read(destination/"summary.json")
    result = {"passed": compared == 150, "scope": "All150 completed baseline validation episodes; no test episodes inspected",
              "episodes": compared, "root_rule_source_sha256": checksum(path), "candidate_and_object_measurements_match": True,
              "mixed_cases_explicitly_preserved": mixed, "manual_reviewed_cases": summary["manual_reviewed_episodes"],
              "manual_ambiguous_without_strict_candidate": len(summary["manual_ambiguous_without_strict_candidate"]),
              "manual_mixed_cases": len(summary["manual_mixed_candidate_cases"])}
    atomic_json(destination/"baseline_validation_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--validate-baseline", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    root = args.run.resolve()
    destination = root/"contact_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__).resolve(), ROOT/"src/rl_vla/diagnostics/contact_evidence.py", ROOT/"scripts/review_vla_trace_episodes.py"]
    source_hashes = {str(path): checksum(path) for path in paths}
    code_hash = canonical_hash(source_hashes)
    with (destination/"worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = destination/"source_manifest.json"
        if manifest_path.exists() and read(manifest_path)["code_hash"] != code_hash:
            raise ValueError("Contact worker source changed; coordinate a new versioned sidecar instead of overwriting evidence")
        if not manifest_path.exists():
            atomic_json(manifest_path, {"code_hash": code_hash, "sources": source_hashes, "rule": RULE})
            source_root = destination/"sources"
            source_root.mkdir()
            for path in paths:
                shutil.copy2(path, source_root/path.name)
        cache = read(destination/"cache.json") if (destination/"cache.json").exists() else {}
        while True:
            if any(checksum(path) != expected for path, expected in source_hashes.items()):
                raise ValueError("Contact worker source changed while running")
            started = time.monotonic()
            result = refresh(root, destination, cache, code_hash)
            print(json.dumps({key: result[key] for key in ("complete", "reports_processed", "episodes_processed", "manual_reviewed_episodes", "persistent_errors")}), flush=True)
            if args.validate_baseline:
                print(json.dumps(validate_baseline(root, destination)), flush=True)
                args.validate_baseline = False
            atomic_json(destination/"worker_status.json", {"pid": os.getpid(), "state": "watching" if args.watch else "idle",
                        "refresh_seconds": time.monotonic()-started, "updated_unix_seconds": time.time(),
                        "evidence_signature": result["evidence_signature"], "source_hash": code_hash})
            if not args.watch:
                break
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
