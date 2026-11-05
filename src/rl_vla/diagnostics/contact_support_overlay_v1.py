"""Versioned contact evidence with strict-support NA kept outside outcomes.

The native extractor, its rule, original watcher and cached artifacts are read
unchanged. Only completed native resumed cases reach the contact classifier.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from rl_vla.diagnostics import causal_structural_validation_v1 as validation
from rl_vla.diagnostics.contact_evidence import DISTRACTORS, RULE, TARGETS, causal_steps, evidence

ROOT = Path(__file__).resolve().parents[3]
LEGACY_SOURCE = ROOT / "scripts/analyze_vla_trace_contacts.py"
LEGACY_SCRIPT_SHA = "b7d13400dd34dceef6fd147deec1c8e675e9348322514437fd506667bfe63d27"
LEGACY_SHA = "69c0f6c41cb1d7524031c3661d3434f8cbf9c825ed9312fed92c84854889af35"
DESTINATION = "contact_analysis_structural_na_v1"
EXPECTED_REPORTS, EXPECTED_CASES, MAIN_OOD_CASES = 192, 5280, 4200
_LEGACY = None
_VERIFY_FILES = {}


def sha(path):
    return validation.sha(path)


def read(path):
    return validation.read(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def legacy():
    global _LEGACY
    if sha(LEGACY_SOURCE) != LEGACY_SCRIPT_SHA:
        raise ValueError("Frozen native contact extractor changed")
    if _LEGACY is None:
        spec = importlib.util.spec_from_file_location(
            "_frozen_native_contact_extractor_v1", LEGACY_SOURCE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LEGACY = module
    return _LEGACY


def report_key(row):
    return tuple(row[k] for k in ("domain", "task", "policy", "variant"))


def case_key(row):
    return (*report_key(row), row["episode"])


def bind(path, inputs, files, expected=None):
    path = Path(path).resolve()
    checksum = files.hash(path)
    if expected is not None and checksum != expected:
        raise ValueError(f"Changed contact overlay input: {path}")
    inputs[str(path)] = checksum
    return {"path": str(path), "sha256": checksum}


def verify_legacy_raw_sources(path, files):
    """Preserve the original metadata-only hash scope; check raw file identity."""
    stat_record = legacy().stat_record
    for line in Path(path).read_text().splitlines():
        source = json.loads(line)
        current = stat_record(source["path"])
        if any(current[name] != source[name] for name in ("bytes", "mtime_ns", "inode")):
            raise ValueError(f"Native source changed since contact extraction: {source['path']}")
        if "sha256" in source and files.hash(source["path"]) != source["sha256"]:
            raise ValueError(f"Native telemetry/header bytes changed: {source['path']}")


def _legacy_reports(run, summary, inputs, files, *, replace_object1):
    reports = []
    for manifest in summary["reports"]:
        identity = manifest["identity"]
        if replace_object1 and identity["domain"] == "causal" and identity["task"] == "object_1":
            continue
        if (
            manifest["code_hash"] != LEGACY_SHA
            or manifest["rule"] != RULE
            or manifest["complete"] is not True
        ):
            raise ValueError("Native contact cache came from another extractor/rule")
        folder = Path(manifest["artifact"])
        if read(folder / "summary.json") != manifest:
            raise ValueError("Native contact cache manifest differs from published lineage")
        bind(folder / "summary.json", inputs, files)
        bind(manifest["report"], inputs, files, manifest["report_sha256"])
        bind(folder / "episodes.jsonl", inputs, files, manifest["evidence_sha256"])
        bind(folder / "source_inputs.jsonl", inputs, files, manifest["source_inputs_sha256"])
        verify_legacy_raw_sources(folder / "source_inputs.jsonl", files)
        rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
        if len(rows) != manifest["episodes"] or any(
            report_key(row) != report_key(identity) for row in rows
        ):
            raise ValueError("Cached native case count/identity differs")
        reports.append(
            {"identity": identity, "rows": rows, "unavailable": [], "manifest": manifest}
        )
    return reports


def _support_certificate(run, inputs, files, cache):
    base = run / "compiler_profile_integration/causal_structural_na_v1"
    path = base / "execution_coverage.json"
    if not path.exists():
        return None
    signature = bind(path, inputs, files)["sha256"]
    coverage = read(path)
    if cache.get("validated_execution_sha256") != signature:
        # Independently recompute the raw360-case certificate once per immutable
        # publication; never trust a completed flag alone.
        actual = validation.validate_execution(run)
        if actual is None:
            raise ValueError("Execution coverage exists without raw completion")
        details = read(validation.verify(coverage["case_records"]))
        expected = {k: v for k, v in actual.items() if k not in ("cases", "native_artifacts")}
        if {k: v for k, v in coverage.items() if k != "case_records"} != expected:
            raise ValueError(
                "Published coverage differs from independently recomputed raw evidence"
            )
        if (
            details["cases"] != actual["cases"]
            or details["native_artifacts"] != actual["native_artifacts"]
        ):
            raise ValueError("Published case records differ from raw execution")
        cache["validated_execution_sha256"] = signature
    detail_path = bind(
        coverage["case_records"]["path"], inputs, files, coverage["case_records"]["sha256"]
    )
    details = read(detail_path["path"])
    for name in (
        "root_acceptance",
        "amendment",
        "queue_completion",
        "validator_source",
        "native_key_recurrence",
    ):
        bind(coverage[name]["path"], inputs, files, coverage[name]["sha256"])
    for item in details["native_artifacts"]:
        bind(item["path"], inputs, files, item["sha256"])
    # Root/source/proof lineage is explicit even after a cached raw validation.
    acceptance = read(coverage["root_acceptance"]["path"])
    for name in (
        "reuse_manifest",
        "resume_proof",
        "independent_review",
        "resume_plan",
        "execution_validator_source",
    ):
        bind(acceptance[name]["path"], inputs, files, acceptance[name]["sha256"])
    for path, checksum in acceptance["source_input_sha256"].items():
        bind(path, inputs, files, checksum)
    reuse = read(acceptance["reuse_manifest"]["path"])
    for item in reuse["inputs"].values():
        bind(item["path"], inputs, files, item["sha256"])
    return coverage, details


def _resumed_reports(run, destination, coverage, details, inputs, files):
    by_condition = defaultdict(list)
    for case in details["cases"]:
        by_condition[case["policy"], case["condition"]].append(case)
    reports = []
    for condition in coverage["conditions"]:
        policy, variant = condition["policy"], condition["condition"]
        identity = {"domain": "causal", "task": "object_1", "policy": policy, "variant": variant}
        report = condition["summary"]
        bind(report["path"], inputs, files, report["sha256"])
        cases = by_condition[policy, variant]
        if len(cases) != 10 or [c["episode"] for c in cases] != list(range(10)):
            raise ValueError("Resumed contact group differs from original ten cases")
        cache_id = digest({"report": report, "cases": cases, "rule": RULE, "extractor": LEGACY_SHA})
        folder = destination / "reports" / cache_id
        if not folder.exists():
            rows, unavailable, sources = [], [], []
            for case in cases:
                row = case["row"]
                common = {
                    **identity,
                    "report": report["path"],
                    "report_sha256": report["sha256"],
                    "episode": case["episode"],
                    "state_id": case["state_id"],
                    "seed": case["seed"],
                    "source_case_sha256": digest(case),
                    "execution_case_status": case["status"],
                }
                if case["status"] == "structural_NA":
                    if row["success"] is not None:
                        raise ValueError("Structural prefix cannot have a native outcome")
                    witness = read(case["structural_evidence"]["path"])
                    unavailable.append(
                        {
                            **common,
                            "status": "structural_NA",
                            "success": None,
                            "outcome_available": False,
                            "partial_telemetry_excluded_from_contact_classification": True,
                            "structural_evidence": case["structural_evidence"],
                            "decision": witness["decision"],
                            "first_failing_camera": witness["first_failing_camera"],
                            "support": witness["support"],
                            "partial_video": row.get("video"),
                            "reviewed": False,
                            "manual_reviewed": False,
                        }
                    )
                    continue
                if case["status"] != "completed" or type(row["success"]) is not bool:
                    raise ValueError("Only completed native episodes reach contact extraction")
                telemetry = case["artifacts"]["telemetry"]["path"]
                measured = evidence(
                    causal_steps(telemetry, row, sources),
                    target=TARGETS["object_1"],
                    distractor=DISTRACTORS["object_1"],
                    primitive_steps=row["primitive_steps"],
                    success=row["success"],
                    reset_telemetry=row["reset"]["object_telemetry"],
                )
                rows.append(
                    {
                        **common,
                        "success": row["success"],
                        "target": TARGETS["object_1"],
                        "distractor": DISTRACTORS["object_1"],
                        "video": row["video"],
                        "trajectory": None,
                        "telemetry_path": telemetry,
                        "contact_artifact": str(folder / "episodes.jsonl"),
                        **measured,
                    }
                )
            folder.parent.mkdir(parents=True, exist_ok=True)
            temporary = folder.with_name(folder.name + f".tmp.{os.getpid()}")
            temporary.mkdir()
            (temporary / "episodes.jsonl").write_text(
                "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
            )
            atomic(temporary / "support_unavailable.json", unavailable)
            atomic(temporary / "source_inputs.json", sources)
            atomic(
                temporary / "summary.json",
                {
                    "complete": True,
                    "identity": identity,
                    "rule": RULE,
                    "native_cases": len(rows),
                    "structural_na_cases": len(unavailable),
                    "planned_cases": 10,
                    "case_input_sha256": digest(cases),
                    "report": report,
                },
            )
            temporary.rename(folder)
        manifest = read(folder / "summary.json")
        if manifest["case_input_sha256"] != digest(cases) or manifest["rule"] != RULE:
            raise ValueError("Immutable resumed contact artifact differs")
        for name in (
            "summary.json",
            "episodes.jsonl",
            "support_unavailable.json",
            "source_inputs.json",
        ):
            bind(folder / name, inputs, files)
        for case in cases:
            if case["status"] == "structural_NA":
                item = case["structural_evidence"]
                bind(item["path"], inputs, files, item["sha256"])
                for artifact in read(item["path"])["artifacts"]:
                    bind(artifact["path"], inputs, files, artifact["sha256"])
        reports.append(
            {
                "identity": identity,
                "rows": [
                    json.loads(line)
                    for line in (folder / "episodes.jsonl").read_text().splitlines()
                ],
                "unavailable": read(folder / "support_unavailable.json"),
                "manifest": manifest,
            }
        )
    return reports


def assemble(reports, plan, labels):
    """Pure unique-case accounting; unavailable prefixes never become contact rows."""
    expected = {report_key(item): item["episodes"] for item in plan}
    native, unavailable, rates, calibration, seen = [], [], [], [], set()
    for report in sorted(reports, key=lambda item: report_key(item["identity"])):
        key = report_key(report["identity"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unplanned contact report")
        seen.add(key)
        rows, na = report["rows"], report["unavailable"]
        all_rows = rows + na
        if len(all_rows) != expected[key] or {r["episode"] for r in all_rows} != set(
            range(expected[key])
        ):
            raise ValueError("Contact report lacks exact resolved episode ordinals")
        if any(report_key(row) != key for row in all_rows):
            raise ValueError("Contact case differs from logical report identity")
        if any(
            type(row.get("success")) is not bool
            or row.get("execution_case_status") == "structural_NA"
            for row in rows
        ):
            raise ValueError(
                "Undefined or non-Boolean native outcomes cannot enter contact evidence"
            )
        if any(
            row.get("status") != "structural_NA" or row.get("success") is not None for row in na
        ):
            raise ValueError("Support events require explicit structuralNA and null outcome")
        if na and (key[0], key[1]) != ("causal", "object_1"):
            raise ValueError("Structural amendment does not cover this report")
        for original in rows:
            row = dict(original)
            if row.get("reviewed") is not False or row.get("automatic") is not True:
                raise ValueError("Automatic contact labels must remain unreviewed")
            manual = labels.get((row["task"], row["report_sha256"], row["state_id"], row["seed"]))
            if manual and (manual["success"] != row["success"] or manual["video"] != row["video"]):
                raise ValueError("Manual review and native contact identity differ")
            row.update(
                manual_reviewed=bool(manual),
                manual_failure_type=manual["failure_type"] if manual else None,
                manual_grasped_object_id=manual.get("grasped_object_id") if manual else None,
                manual_label_source=manual["label_source"] if manual else None,
                manual_label_source_sha256=manual["label_source_sha256"] if manual else None,
            )
            native.append(row)
            if manual:
                calibration.append(
                    {
                        k: row.get(k)
                        for k in (
                            "domain",
                            "task",
                            "policy",
                            "variant",
                            "state_id",
                            "seed",
                            "report",
                            "report_sha256",
                            "video",
                            "success",
                            "sustained_grasp_candidates",
                            "provisional_failure_type",
                            "manual_failure_type",
                            "manual_grasped_object_id",
                            "manual_label_source",
                            "manual_label_source_sha256",
                        )
                    }
                )
        unavailable.extend(na)
        complete = [r for r in rows if r["telemetry_complete"]]
        rate = {
            **report["identity"],
            "planned_cases": expected[key],
            "native_completed": len(rows),
            "structural_na_cases": len(na),
            "resolved_cases": len(all_rows),
            "primary_outcome_available": not na,
            "native_successes": sum(r["success"] for r in rows) if not na else None,
            "observed_native_successes": sum(r["success"] for r in rows),
            "telemetry_complete_episodes": len(complete),
            "automatic_labels_reviewed": False,
            "manual_reviewed_episodes": sum(
                bool(labels.get((r["task"], r["report_sha256"], r["state_id"], r["seed"])))
                for r in rows
            ),
            "candidate_denominator": "unavailable_entire_condition"
            if na
            else "telemetry_complete_episodes",
        }
        for name in (
            "target_grasp_candidate",
            "non_target_grasp_candidate",
            "mixed_target_non_target_candidates",
            "no_grasp_candidate",
            "first_confirmation_tie",
        ):
            count = sum(r[name] for r in complete)
            rate[name + "_count"] = count
            rate[name + "_rate"] = count / len(complete) if complete and not na else None
        rate["first_target_candidate_count"] = sum(
            r["first_candidate_role"] == "target" for r in complete
        )
        rate["first_target_candidate_rate"] = (
            rate["first_target_candidate_count"] / len(complete) if complete and not na else None
        )
        rates.append(rate)
    keys = [case_key(row) for row in native + unavailable]
    if len(keys) != len(set(keys)):
        raise ValueError("A native or unavailable logical case was counted twice")
    return native, unavailable, rates, calibration


def refresh(run, cache):
    run = Path(run).resolve()
    destination = run / DESTINATION
    destination.mkdir(parents=True, exist_ok=True)
    old = legacy()
    files = old.FileCache(cache.setdefault("files", {}))
    inputs = {}
    for path in (
        Path(__file__),
        LEGACY_SOURCE,
        Path(validation.__file__),
        ROOT / "src/rl_vla/diagnostics/contact_evidence.py",
        ROOT / "scripts/analyze_vla_trace_contacts_support_v1.py",
    ):
        bind(path, inputs, files)
    legacy_source_manifest = run / "contact_analysis/source_manifest.json"
    original_sources = read(legacy_source_manifest)
    if (
        original_sources["code_hash"] != LEGACY_SHA
        or digest(original_sources["sources"]) != LEGACY_SHA
    ):
        raise ValueError("Original native contact source manifest differs")
    bind(legacy_source_manifest, inputs, files)
    for path, checksum in original_sources["sources"].items():
        bind(path, inputs, files, checksum)
    for task in ("object_0", "object_1", "object_2"):
        bind(run / "tasks" / task / "protocol.json", inputs, files)
    source = run / "contact_analysis/summary.json"
    payload = source.read_bytes()
    original = json.loads(payload)
    source_hash = hashlib.sha256(payload).hexdigest()
    snapshot = destination / "snapshots" / f"legacy_{source_hash}.json"
    if not snapshot.exists():
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(payload)
    bind(snapshot, inputs, files, source_hash)
    support = _support_certificate(run, inputs, files, cache)
    reports = _legacy_reports(run, original, inputs, files, replace_object1=support is not None)
    if support:
        reports.extend(_resumed_reports(run, destination, *support, inputs, files))
    labels, manual_hashes = old.manual_reviews(run, files)
    for path, checksum in manual_hashes.items():
        bind(path, inputs, files, checksum)
    for label in labels.values():
        for name in ("report", "video", "sheet", "evidence_manifest"):
            bind(label[name], inputs, files, label[name + "_sha256"])
    native, na, rates, calibration = assemble(reports, old.report_plan(run), labels)
    errors = original.get("persistent_errors", [])
    signature = digest(
        {"inputs": inputs, "errors": errors, "reports": [r["identity"] for r in reports]}
    )
    current = destination / "summary.json"
    if current.exists() and read(current).get("evidence_signature") == signature:
        return read(current)
    generation = destination / "generations" / signature
    generation.mkdir(parents=True, exist_ok=False)
    tables = generation / "tables"
    tables.mkdir()
    fields = list(
        dict.fromkeys(
            k
            for row in native
            for k in row
            if k not in ("objects", "rule", "interpretation", "missing_telemetry_steps")
        )
    )
    for filename, rows, columns in (
        ("contact_cases.csv", native, fields),
        ("contact_candidate_rates.csv", rates, list(rates[0]) if rates else ["domain"]),
        (
            "contact_manual_calibration.csv",
            calibration,
            list(calibration[0]) if calibration else ["task"],
        ),
        (
            "support_unavailable_cases.csv",
            na,
            list(na[0])
            if na
            else [
                "domain",
                "task",
                "policy",
                "variant",
                "episode",
                "state_id",
                "seed",
                "status",
                "success",
            ],
        ),
    ):
        old.write_csv(tables / filename, rows, columns)
    atomic(
        generation / "inputs.json",
        {"schema": "contact-support-overlay-inputs-v1", "inputs": inputs},
    )
    counts = Counter(row["domain"] for row in native)
    resolved = len(native) + len(na)
    complete = (
        support is not None
        and len(reports) == EXPECTED_REPORTS
        and resolved == EXPECTED_CASES
        and counts["main"] + counts["ood"] == MAIN_OOD_CASES
        and counts["causal"] + len(na) == 1080
        and all(row["telemetry_complete"] for row in native)
        and not errors
    )
    summary = {
        "schema": "contact-support-overlay-v1",
        "complete": complete,
        "evidence_signature": signature,
        "planned_reports": EXPECTED_REPORTS,
        "planned_cases": EXPECTED_CASES,
        "main_ood_planned_cases": MAIN_OOD_CASES,
        "resolved_reports": len(reports),
        "resolved_cases": resolved,
        "completed_native_cases": len(native),
        "completed_main_ood_native_cases": counts["main"] + counts["ood"],
        "completed_causal_native_cases": counts["causal"],
        "structural_na_cases": len(na),
        "native_cases_by_domain": dict(counts),
        "telemetry_complete_native_cases": sum(row["telemetry_complete"] for row in native),
        "manual_reviewed_cases": len(calibration),
        "automatic_labels_reviewed": False,
        "support_execution_adopted": support is not None,
        "execution_coverage": validation.binding(
            run / "compiler_profile_integration/causal_structural_na_v1/execution_coverage.json"
        )
        if support
        else None,
        "persistent_errors": errors,
        "rule": RULE,
        "source_manifest": validation.binding(generation / "inputs.json"),
        "tables": {p.name: validation.binding(p) for p in tables.iterdir()},
        "interpretation": [
            "Native outcomes and strict contact candidates remain separate from selected-frame AI reviews.",
            "StructuralNA prefixes are excluded from contact classification, native outcomes, and manual failure denominators.",
            "Any structuralNA disables all primary candidate rates for its entire original ten-case condition; valid-case counts remain descriptive only.",
            "Immutable prior contact artifacts are reused; original watcher/source/tree are unchanged.",
            "Original native NPZ evidence retains exact metadata/timestamp hash scope and bytes/mtime/inode guards; image/action arrays are not reparsed. Full hashes are rechecked for recorded headers/telemetry and resumed completed artifacts. Only separately reviewed videos have manual visual-evidence bindings.",
        ],
    }
    atomic(generation / "summary.json", summary)
    aliases = destination / "tables"
    aliases.mkdir(exist_ok=True)
    for path in tables.iterdir():
        alias = aliases / path.name
        temporary = alias.with_name(alias.name + f".tmp.{os.getpid()}")
        temporary.symlink_to(path)
        temporary.replace(alias)
    atomic(current, summary)
    return summary


def verify_overlay(run, require_complete=False):
    """Return verified compact coverage, or None before the overlay is published."""
    run = Path(run).resolve()
    path = run / DESTINATION / "summary.json"
    if not path.exists():
        return None
    value = read(path)
    if (
        value.get("schema") != "contact-support-overlay-v1"
        or value.get("automatic_labels_reviewed") is not False
    ):
        raise ValueError("Unrecognized contact overlay lineage")
    files = legacy().FileCache(_VERIFY_FILES)
    manifest = value["source_manifest"]
    if files.hash(manifest["path"]) != manifest["sha256"]:
        raise ValueError("Changed contact overlay input manifest")
    bound_inputs = read(manifest["path"])["inputs"]
    for required in (
        Path(__file__).resolve(),
        LEGACY_SOURCE,
        Path(validation.__file__).resolve(),
        ROOT / "src/rl_vla/diagnostics/contact_evidence.py",
        ROOT / "scripts/analyze_vla_trace_contacts_support_v1.py",
    ):
        if bound_inputs.get(str(required)) != files.hash(required):
            raise ValueError("Overlay lacks current extraction/consumer source provenance")
    for source, checksum in bound_inputs.items():
        if files.hash(source) != checksum:
            raise ValueError(f"Changed contact overlay evidence: {source}")
        if source.endswith("/source_inputs.jsonl"):
            verify_legacy_raw_sources(source, files)
    for table in value["tables"].values():
        if files.hash(table["path"]) != table["sha256"]:
            raise ValueError("Changed contact overlay table")
    required_tables = {
        "contact_cases.csv",
        "contact_candidate_rates.csv",
        "contact_manual_calibration.csv",
        "support_unavailable_cases.csv",
    }
    if set(value["tables"]) != required_tables or value.get("rule") != RULE:
        raise ValueError("Overlay table or operational rule identity differs")

    def rows(name):
        with Path(value["tables"][name]["path"]).open(newline="") as stream:
            return list(csv.DictReader(stream))

    native = rows("contact_cases.csv")
    unavailable = rows("support_unavailable_cases.csv")
    rates = rows("contact_candidate_rates.csv")
    manual = rows("contact_manual_calibration.csv")
    expected = {report_key(item): item["episodes"] for item in legacy().report_plan(run)}
    grouped = defaultdict(list)
    for row in native + unavailable:
        key = report_key(row)
        if key not in expected:
            raise ValueError("Overlay table contains an unplanned report")
        protocol = read(run / "tasks" / row["task"] / "protocol.json")["tasks"][row["task"]]
        ids = (
            list(range(50))
            if row["domain"] == "ood"
            else protocol["test_state_ids"][:10]
            if row["domain"] == "causal"
            else protocol[row["variant"] + "_state_ids"]
        )
        episode = int(row["episode"])
        if (
            not 0 <= episode < len(ids)
            or int(row["state_id"]) != ids[episode]
            or int(row["seed"]) != 100000 + episode
        ):
            raise ValueError("Contact table differs from frozen state IDs/noise seeds")
        report_path = str(Path(row["report"]).resolve())
        if bound_inputs.get(report_path) != row["report_sha256"]:
            raise ValueError("Contact row lacks bound original report provenance")
        grouped[key].append(row)
    for key, cases in grouped.items():
        if len(cases) != expected[key] or {int(row["episode"]) for row in cases} != set(
            range(expected[key])
        ):
            raise ValueError("Overlay table contains duplicate or unresolved cases")
    if any(
        row["success"] not in ("True", "False")
        or row["reviewed"] != "False"
        or row["automatic"] != "True"
        for row in native
    ):
        raise ValueError("Contact table silently classified an unavailable case or manual label")
    if any(
        row["status"] != "structural_NA"
        or row["success"] != ""
        or (row["domain"], row["task"]) != ("causal", "object_1")
        for row in unavailable
    ):
        raise ValueError("Unavailable contact table contains a native outcome")
    if len(rates) != len(grouped) or {report_key(row) for row in rates} != set(grouped):
        raise ValueError("Contact rate table lacks unique resolved report groups")
    for rate in rates:
        key = report_key(rate)
        missing = sum(report_key(row) == key for row in unavailable)
        if int(rate["structural_na_cases"]) != missing:
            raise ValueError("Contact condition availability differs from case evidence")
        available = [row for row in native if report_key(row) == key]
        complete_rows = [row for row in available if row["telemetry_complete"] == "True"]
        if (
            int(rate["native_completed"]) != len(available)
            or int(rate["resolved_cases"]) != len(available) + missing
        ):
            raise ValueError("Contact rate native/resolved counts differ")
        for field in (
            "target_grasp_candidate",
            "non_target_grasp_candidate",
            "mixed_target_non_target_candidates",
            "no_grasp_candidate",
            "first_confirmation_tie",
            "first_target_candidate",
        ):
            count = sum(
                (
                    row["first_candidate_role"] == "target"
                    if field == "first_target_candidate"
                    else row[field] == "True"
                )
                for row in complete_rows
            )
            expected_rate = count / len(complete_rows) if complete_rows and not missing else None
            if int(rate[field + "_count"]) != count or (
                rate[field + "_rate"] != ""
                if expected_rate is None
                else float(rate[field + "_rate"]) != expected_rate
            ):
                raise ValueError("Contact candidate rate differs from native case evidence")
        if missing and (
            rate["primary_outcome_available"] != "False"
            or rate["native_successes"] != ""
            or any(v != "" for k, v in rate.items() if k.endswith("_rate"))
        ):
            raise ValueError("Undefined condition emitted a primary contact/outcome rate")
    counts = Counter(row["domain"] for row in native)
    expected_counts = {
        "completed_native_cases": len(native),
        "structural_na_cases": len(unavailable),
        "resolved_cases": len(native) + len(unavailable),
        "resolved_reports": len(grouped),
        "completed_main_ood_native_cases": counts["main"] + counts["ood"],
        "completed_causal_native_cases": counts["causal"],
        "manual_reviewed_cases": len(manual),
        "telemetry_complete_native_cases": sum(
            row["telemetry_complete"] == "True" for row in native
        ),
    }
    if any(value.get(k) != v for k, v in expected_counts.items()):
        raise ValueError("Overlay compact counts differ from its concrete tables")
    if value["completed_native_cases"] + value["structural_na_cases"] != value["resolved_cases"]:
        raise ValueError("Contact overlay double-counted or lost resolved cases")
    if value["complete"] and (
        value["resolved_reports"] != 192
        or value["resolved_cases"] != 5280
        or value["completed_main_ood_native_cases"] != 4200
        or value["completed_causal_native_cases"] + value["structural_na_cases"] != 1080
        or value["persistent_errors"]
        or value["telemetry_complete_native_cases"] != value["completed_native_cases"]
        or value["support_execution_adopted"] is not True
    ):
        raise ValueError("Complete contact overlay has incomplete native/NA coverage")
    if value["support_execution_adopted"]:
        coverage_record = value["execution_coverage"]
        expected_path = (
            run / "compiler_profile_integration/causal_structural_na_v1/execution_coverage.json"
        )
        if (
            Path(coverage_record["path"]).resolve() != expected_path
            or bound_inputs.get(str(expected_path)) != coverage_record["sha256"]
        ):
            raise ValueError("Contact overlay adopted an unbound support execution certificate")
        coverage = read(expected_path)
        if coverage.get("complete") is not True or coverage.get("passed") is not True:
            raise ValueError("Contact overlay adopted incomplete support execution")
        object1 = sum(row["domain"] == "causal" and row["task"] == "object_1" for row in native)
        if coverage["coverage"] != {
            "planned_cases": 360,
            "native_completed": object1,
            "structural_NA": len(unavailable),
            "total_resolved": 360,
        }:
            raise ValueError("Contact overlay differs from independently validated support counts")
    elif unavailable:
        raise ValueError("Contact overlay classified unavailable cases without adopted execution")
    if require_complete and value["complete"] is not True:
        return None
    return value
