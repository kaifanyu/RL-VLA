"""One explicit, byte-preserving contact-cache lineage replacement after GPU6 repair.

The contact extraction worker/rule is deliberately unchanged. This migration
requires its lock, all four archived invalid controls, and the repaired full
rollout audit. It preserves every old cache artifact and republishes no scores.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def repaired_proofs(root):
    path = root / "compiler_profile_integration/object1_repaired_none_rollouts_v1/summary.json"
    summary = read(path)
    if summary.get("complete") is not True or summary.get("passed") is not True or summary.get("expected_cells") != 4:
        raise ValueError("Repaired full-rollout control proof is incomplete")
    bindings = [{"path": str(path), "sha256": sha(path)}]
    reports = {}
    for row in summary["cells"]:
        result = Path(row["result"]["path"])
        if sha(result) != row["result"]["sha256"]:
            raise ValueError("Repaired full-rollout proof bytes changed")
        value = read(result)
        if value.get("task") != "object_1" or value.get("policy") not in POLICIES or value.get("passed") is not True:
            raise ValueError("Repaired proof identity differs")
        if any(value.get(key) != 10 for key in ("paired_episodes", "outcomes_equal", "full_normalized_action_sequences_exact",
            "full_executed_action_sequences_exact", "full_simulator_state_sequences_exact", "full_step_metadata_sequences_exact")):
            raise ValueError("Repaired proof does not cover every complete trajectory")
        if value["policy"] in reports:
            raise ValueError("Duplicate repaired control proof")
        expected = root / "interventions/object_1" / value["policy"] / "none/summary.json"
        sources = [item for item in value["sources"] if Path(item["path"]) == expected]
        if len(sources) != 1 or sha(expected) != sources[0]["sha256"]:
            raise ValueError("Repaired proof does not bind current causal report")
        reports[value["policy"]] = sources[0]
        bindings.append(row["result"])
    if set(reports) != set(POLICIES):
        raise ValueError("Missing repaired policy proof")
    return bindings, reports


def build_plan(root):
    destination = root / "contact_analysis"
    cache = read(destination / "cache.json")
    code = read(destination / "source_manifest.json")
    for path, expected in code["sources"].items():
        if sha(path) != expected:
            raise ValueError("Original contact extraction source changed")
    proof_bindings, proof_reports = repaired_proofs(root)
    adjudication = root / "technical_attempts/gpu6_descriptor_profile_miss_v1/failure_adjudication.json"
    judged = read(adjudication)
    if judged.get("native_main_evaluations_changed") is not False or judged.get("retry_cases_seeds_conditions_change") is not False:
        raise ValueError("Supersession lacks unchanged native outcome/case contract")
    invalid_root = root / "technical_attempts/gpu6_descriptor_profile_miss_v1/object_1"
    current_root = root / "interventions/object_1"
    mappings, invalid = [], []
    for path, manifest in sorted(cache["reports"].items()):
        identity = manifest["identity"]
        if identity["domain"] != "causal" or identity["task"] != "object_1":
            continue
        source = Path(path)
        if source != current_root / identity["policy"] / identity["variant"].replace(":", "_") / "summary.json":
            raise ValueError("Cached causal identity/path mismatch")
        artifact = Path(manifest["artifact"])
        if artifact != destination / "reports" / manifest["report_sha256"]:
            raise ValueError("Unexpected immutable contact artifact path")
        if sha(artifact / "report.json") != manifest["report_sha256"] or sha(artifact / "episodes.jsonl") != manifest["evidence_sha256"] or sha(artifact / "source_inputs.jsonl") != manifest["source_inputs_sha256"]:
            raise ValueError("Preserved contact evidence bytes changed")
        current_hash = sha(source)
        changed = current_hash != manifest["report_sha256"]
        archived_report = invalid_root / source.relative_to(current_root)
        if changed:
            if identity["variant"] != "none" or sha(archived_report) != manifest["report_sha256"]:
                raise ValueError("Changed cache entry is not the explicitly archived invalid none control")
            if current_hash != proof_reports[identity["policy"]]["sha256"]:
                raise ValueError("Repaired control hash differs from its exact trajectory proof")
            invalid.append(identity["policy"])
        source_inputs = []
        for line in (artifact / "source_inputs.jsonl").read_text().splitlines():
            item = json.loads(line)
            old_path = Path(item["path"])
            bound_path = invalid_root / old_path.relative_to(current_root) if changed else old_path
            if sha(bound_path) != item["sha256"]:
                raise ValueError("Archived/current telemetry does not match original contact extraction")
            source_inputs.append({"old_path": str(old_path), "preserved_source": str(bound_path), "sha256": item["sha256"]})
        mappings.append({"identity": identity, "canonical_report": path, "old_report_sha256": manifest["report_sha256"],
                         "current_report_sha256": current_hash, "old_artifact": str(artifact),
                         "old_artifact_files": {child.name: sha(child) for child in artifact.iterdir() if child.is_file()},
                         "archived_invalid_report": str(archived_report) if changed else None,
                         "source_inputs": source_inputs,
                         "disposition": "excluded_invalid_GPU6_attempt" if changed else "valid_repaired_cohort_reextracted_in_new_lineage"})
    if sorted(invalid) != sorted(POLICIES):
        raise ValueError("Supersession must identify exactly the four archived invalid controls")
    errors = read(destination / "summary.json")["persistent_errors"]
    expected_errors = {str(current_root / policy / "none/summary.json") for policy in POLICIES}
    if {row["report"] for row in errors} != expected_errors or any("Previously completed source report changed" not in row["error"] for row in errors):
        raise ValueError("Additional contact errors require separate adjudication")
    return {"schema": "contact-sidecar-explicit-gpu6-supersession-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "extraction_code_hash": code["code_hash"], "unchanged_sources": code["sources"], "rule": code["rule"],
            "adjudication": {"path": str(adjudication), "sha256": sha(adjudication)},
            "repaired_full_rollout_proofs": proof_bindings, "mappings": mappings,
            "preserved_errors": errors, "expected_final_reports": 192, "expected_final_episodes": 5280,
            "scope": "Every current Object1 causal cache entry is re-extracted; all future Object1 conditions use unchanged extraction. Four invalid old controls are excluded, never relabeled or overwritten. Other task/main/OOD cached evidence is retained byte-for-byte."}


def apply(root):
    destination = root / "contact_analysis"
    archive = root / "technical_attempts/contact_sidecar_gpu6_supersession_v1"
    stage = root / "contact_analysis_rebind_v1"
    if archive.exists() or stage.exists():
        raise FileExistsError("Supersession destinations already exist; do not overwrite prior lineage")
    with (destination / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = build_plan(root)
        cache = read(destination / "cache.json")
        old_cache_hash = sha(destination / "cache.json")
        excluded = {row["canonical_report"] for row in plan["mappings"]}
        retained = {path: value for path, value in cache["reports"].items() if path not in excluded}
        stage.mkdir()
        for name in ("sources", "source_manifest.json", "baseline_validation_audit.json"):
            source = destination / name
            if source.is_dir():
                shutil.copytree(source, stage / name)
            elif source.exists():
                shutil.copy2(source, stage / name)
        (stage / "reports").mkdir()
        for manifest in retained.values():
            shutil.copytree(manifest["artifact"], stage / "reports" / manifest["report_sha256"])
        new_cache = {**cache, "reports": retained,
                     "retry": {key: value for key, value in cache.get("retry", {}).items() if key not in excluded},
                     "files": {key: value for key, value in cache.get("files", {}).items() if not key.startswith(str(root / "interventions/object_1") + "/")}}
        plan.update(archive=str(archive), prior_cache_sha256=old_cache_hash, prior_cached_reports=len(cache["reports"]),
                    retained_reports=len(retained), reextract_cached_reports=len(excluded),
                    migration_source={"path": str(Path(__file__).resolve()), "sha256": sha(__file__)})
        archive.mkdir(parents=True)
        write(archive / "supersession.json", plan)
        new_cache["supersession"] = {"path": str(archive / "supersession.json"), "sha256": sha(archive / "supersession.json")}
        write(stage / "cache.json", new_cache)
        write(stage / "supersession.json", new_cache["supersession"])
        table_archive = archive / "tables"
        table_archive.mkdir()
        for path in (root / "tables").glob("contact_*.csv"):
            shutil.copy2(path, table_archive / path.name)
        for name in ("contact_sidecar_worker.json", "contact_sidecar_watch.log"):
            if (root / name).exists():
                shutil.copy2(root / name, archive / name)
        prior_files = {str(path.relative_to(destination)): sha(path) for path in destination.rglob("*") if path.is_file()}
        write(archive / "prior_contact_files.json", prior_files)
        destination.rename(archive / "contact_analysis")
        stage.rename(destination)
        for name, expected in prior_files.items():
            if sha(archive / "contact_analysis" / name) != expected:
                raise ValueError("Archived contact lineage changed during replacement")
        return {"complete": True, "supersession": new_cache["supersession"], "preserved_files": len(prior_files),
                "retained_reports": len(retained), "reextract_cached_reports": len(excluded), "extraction_sources_unchanged": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.run.resolve()
    result = apply(root) if args.apply else build_plan(root)
    print(json.dumps(result, indent=2))
