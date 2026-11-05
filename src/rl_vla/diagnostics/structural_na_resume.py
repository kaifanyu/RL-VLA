"""Immutable episode reuse and strict random-control availability.

This module does not change a valid pixel intervention. A control requiring more
unique background pixels than exist has no defined policy input or outcome.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")
CONDITIONS = ("none", "no_image", "no_text", "target:background_fill",
              "distractor:background_fill", "random:background_fill",
              "target:mosaic", "distractor:mosaic", "random:mosaic")
NONZERO_PROOFS = {"pi05_libero": (5, "target:mosaic"), "rldt_25": (8, "target:mosaic"),
                  "rldt_50": (2, "target:mosaic"), "rldt_100": (7, "random:background_fill")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "sha256": sha(path),
            "identity": [stat.st_size, stat.st_mtime_ns, stat.st_ino]}


def verify_record(item, *, full_hash=True):
    path = Path(item["path"])
    stat = path.stat()
    if [stat.st_size, stat.st_mtime_ns, stat.st_ino] != item["identity"]:
        raise ValueError(f"Immutable resume input identity changed: {path}")
    if full_hash and sha(path) != item["sha256"]:
        raise ValueError(f"Immutable resume input bytes changed: {path}")


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class StructuralNA(Exception):
    """The strict, unique-pixel matched-background control does not exist."""

    def __init__(self, support):
        super().__init__("insufficient_unowned_background_for_exact_target_area")
        self.support = support


def strict_support(raw, condition, target):
    if condition not in {"random:background_fill", "random:mosaic"}:
        return []
    support = []
    for camera in ("image", "wrist_image"):
        masks = raw["diagnostic_masks"]
        target_mask = np.asarray(masks[f"{camera}__{target}"], bool)
        background = np.asarray(masks[f"{camera}__unowned_background"], bool)
        if target_mask.ndim != 2 or background.shape != target_mask.shape:
            raise ValueError("Malformed simulator mask; not structural inapplicability")
        support.append({"camera": camera, "requested_pixels": int(target_mask.sum()),
                        "available_pixels": int((background & ~target_mask).sum()),
                        "exact_match_possible": bool((background & ~target_mask).sum() >= target_mask.sum())})
    if not all(item["exact_match_possible"] for item in support):
        raise StructuralNA(support)
    return support


def episode_identity(row, index, condition, checkpoint, selection_sha):
    if (row.get("episode") != index or row.get("seed") != 100000 + index
            or row.get("initial_state_id") != 150 + index
            or row.get("reset", {}).get("initial_state_id") != 150 + index
            or row.get("condition") != condition or row.get("checkpoint") != checkpoint
            or row.get("selection_sha256") != selection_sha
            or row.get("technical_smoke") is not False):
        raise ValueError("Reused episode differs from frozen identity/selection")


def collect_reuse(original, *, selection_sha, checkpoints):
    """Hash and validate only complete JSONL rows; ignore interrupted orphan files."""
    original = Path(original).resolve()
    inputs, policies = {}, {}

    def bind(path):
        item = record(path)
        inputs[item["path"]] = item
        return item

    total = whole = 0
    for policy in POLICIES:
        folder = original / policy
        plan_path = folder / "case_plan.json"
        plan = json.loads(plan_path.read_text())
        bind(plan_path)
        if (plan["task"] != "object_1" or plan["split"] != "test"
                or plan["state_ids"] != list(range(150, 160))
                or plan["seeds"] != list(range(100000, 100010))
                or plan["conditions"] != list(CONDITIONS)
                or plan["checkpoint"] != checkpoints[policy]
                or plan["selection_sha256"] != selection_sha):
            raise ValueError("Original case plan differs from the frozen experiment")
        for path, digest in plan["source_sha256"].items():
            if bind(path)["sha256"] != digest:
                raise ValueError("Original numerical/input source changed")
        conditions = {}
        for condition in CONDITIONS:
            cell = folder / condition.replace(":", "_")
            rows, artifacts, raw_lines = [], [], []
            log = cell / "episodes.jsonl"
            if log.exists():
                bind(log)
                for index, line in enumerate(log.read_text().splitlines()):
                    row = json.loads(line)
                    episode_identity(row, index, condition, checkpoints[policy], selection_sha)
                    if type(row.get("success")) is not bool or not 0 < row["primitive_steps"] <= 280:
                        raise ValueError("Only fully completed native outcomes can be reused")
                    action_path = cell / f"episode_{index:03d}_actions.npz"
                    telemetry_path = cell / f"episode_{index:03d}_telemetry.json"
                    video_path = Path(row["video"])
                    with np.load(action_path, allow_pickle=False) as archive:
                        actions = archive["normalized"]
                        if actions.shape != ((row["primitive_steps"] + 4) // 5, 10, 32) or not np.isfinite(actions).all():
                            raise ValueError("Reused action chunks are incomplete/nonfinite")
                    telemetry = json.loads(telemetry_path.read_text())
                    if len(telemetry) != row["primitive_steps"] or [s["primitive_steps"] for s in telemetry] != list(range(1, row["primitive_steps"] + 1)):
                        raise ValueError("Reused telemetry is not a complete primitive prefix")
                    artifacts.append({"episode": index, "row_sha256": hashlib.sha256(line.encode()).hexdigest(),
                                      "actions": bind(action_path), "telemetry": bind(telemetry_path),
                                      "video": bind(video_path)})
                    rows.append(row)
                    raw_lines.append(line)
                if rows:
                    proof = json.loads((cell / "no_op.json").read_text())
                    if proof != {"passed": True, "max_abs_difference": 0.0,
                                 "persisted_noise_recurrence_max_abs_difference": 0.0}:
                        raise ValueError("Original no-op proof is not exact")
                    bind(cell / "no_op.json")
            if len(rows) > 10:
                raise ValueError("Too many original episodes")
            summary = cell / "summary.json"
            complete = summary.exists()
            if complete:
                previous = json.loads(summary.read_text())
                if (previous.get("complete") is not True or len(rows) != 10
                        or previous.get("episodes") != rows or previous.get("n") != 10
                        or previous.get("successes") != sum(row["success"] for row in rows)):
                    raise ValueError("Original complete condition is inconsistent")
                bind(summary)
                whole += 1
            conditions[condition] = {"original_directory": str(cell), "whole_condition_complete": complete,
                                     "rows": rows, "episode_artifacts": artifacts,
                                     "raw_lines": raw_lines,
                                     "next_episode": len(rows)}
            total += len(rows)
        policies[policy] = {"original_case_plan": str(plan_path), "conditions": conditions}
    result = {"schema": "strict-background-resume-inputs-v1", "task": "object_1",
              "original_root": str(original), "completed_episodes": total,
              "whole_conditions_complete": whole, "inputs": inputs, "policies": policies}
    for item in inputs.values():
        verify_record(item)
    return result


def condition_summary(condition, rows, *, amendment, reuse_manifest, elapsed):
    if len(rows) != 10 or [row["episode"] for row in rows] != list(range(10)):
        raise ValueError("Structural availability requires all ten predetermined cases")
    unavailable = [row for row in rows if row.get("status") == "structural_NA"]
    for row in rows:
        if row in unavailable:
            if row.get("success") is not None or not row.get("structural_evidence"):
                raise ValueError("Structural NA must preserve evidence and no success value")
        elif type(row.get("success")) is not bool:
            raise ValueError("Available outcomes must be Boolean")
    count = None if unavailable else sum(row["success"] for row in rows)
    return {"complete": True, "completed": True, "execution_coverage_complete": True,
            "condition": condition, "episodes": rows, "n": 10, "attempted_cases": 10,
            "completed_outcome_cases": 10 - len(unavailable), "structural_na_cases": len(unavailable),
            "primary_outcome_available": not unavailable, "successes": count,
            "primary_success_rate": count / 10 if count is not None else None,
            "primary_paired_comparison": None, "technical_smoke": False,
            "amendment": amendment, "reuse_manifest": reuse_manifest, "elapsed_seconds": elapsed,
            "interpretation": "Any structural NA makes the original ten-case primary success/comparison unavailable. No complete-case estimate, fallback mask, failed-outcome imputation or replacement case."}


def validate_amendment(value, *, execution):
    controls = value.get("control_semantics", {})
    required = ("strict_matched_area_definition_unchanged", "no_area_capping",
                "no_replacement_sampling", "no_semantic_pixel_fallback",
                "no_silent_camera_skip", "normal_cases_use_original_control_implementation")
    if value.get("schema") != "causal-strict-support-amendment-v1" or any(controls.get(k) is not True for k in required):
        raise ValueError("Not the explicit strict-control no-fallback amendment")
    if execution and value.get("approved_for_execution") is not True:
        raise ValueError("The concrete resume amendment has not been approved for execution")
    if value.get("preservation", {}).get("completed_episodes_expected") != 256 or value["preservation"].get("whole_conditions_expected") != 23:
        raise ValueError("Amendment belongs to another interrupted attempt")


def verify_acceptance(path, *, amendment, reuse_manifest, resume_proof):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get("approved") is not True or value.get("passed") is not True:
        raise ValueError("Root has not accepted this concrete resume implementation")
    for key, expected in (("amendment", amendment), ("reuse_manifest", reuse_manifest), ("resume_proof", resume_proof)):
        item = value[key]
        if Path(item["path"]).resolve() != Path(expected).resolve() or sha(expected) != item["sha256"]:
            raise ValueError("Acceptance binds another protocol/input/proof")
    root = Path(__file__).resolve().parents[3]
    required = (root / "scripts/run_vla_trace_interventions_gpu6_resume_v1.py",
                root / "scripts/queue_vla_trace_gpu6_resume_v1.py", Path(__file__).resolve())
    for source in required:
        if value["source_input_sha256"].get(str(source)) != sha(source):
            raise ValueError("Acceptance does not bind the current resume sources")
    for source, digest in value["source_input_sha256"].items():
        if sha(source) != digest:
            raise ValueError("Accepted source/input changed")
    validator = value["execution_validator_source"]
    if sha(validator["path"]) != validator["sha256"]:
        raise ValueError("Accepted independent coverage validator changed")
    return record(path)


def validate_structural_evidence(evidence):
    if evidence.get("condition") not in {"random:background_fill", "random:mosaic"}:
        raise ValueError("Structural availability is only defined for strict random controls")
    for item in evidence["artifacts"]:
        verify_record(item)
    artifacts = {Path(item["path"]).name: item["path"] for item in evidence["artifacts"]}
    with np.load(artifacts["masks.npz"], allow_pickle=False) as masks:
        actual = strict_support_counts(masks, evidence["target"])
    if actual != evidence["support"] or all(item["exact_match_possible"] for item in actual):
        raise ValueError("The saved masks do not prove strict support exhaustion")
    if evidence["first_failing_camera"] != next(item["camera"] for item in actual if not item["exact_match_possible"]):
        raise ValueError("First unsupported camera is inconsistent")
    decision = evidence["decision"]
    index = evidence["episode"]
    if (type(decision) is not int or decision < 0 or evidence["initial_state_id"] != 150 + index
            or evidence["seed"] != 100000 + index
            or evidence["random_mask_seed"] != 100000 + index * 10000 + decision):
        raise ValueError("Structural event identity or mask seed changed")
    with np.load(artifacts["observation.npz"], allow_pickle=False) as data:
        for camera in ("image", "wrist_image"):
            if data[camera].shape != (224, 224, 3) or data[camera].dtype != np.uint8:
                raise ValueError("Missing original policy RGB")
        if (data["normalized_action_prefix"].shape != (decision, 10, 32)
                or data["sampling_key"].shape != (2,) or data["sampling_key"].dtype != np.uint32
                or data["key_after_split"].shape != (2,) or data["initial_noise"].shape != (1, 10, 32)):
            raise ValueError("Structural event lacks exact native key/action prefix")
        for key in ("state", "simulator_state", "normalized_action_prefix", "initial_noise"):
            if not np.isfinite(data[key]).all():
                raise ValueError("Nonfinite retained structural evidence")
        simulator_state = data["simulator_state"]
    telemetry = json.loads(Path(artifacts["partial_telemetry.json"]).read_text())
    if len(telemetry) != decision * 5 or evidence["primitive_steps_before_undefined_input"] != len(telemetry):
        raise ValueError("Partial trajectory does not end at the declared decision boundary")
    latest = telemetry[-1] if telemetry else evidence["reset"]["object_telemetry"]
    if not np.array_equal(simulator_state, latest["simulator_state"]):
        raise ValueError("Retained simulator state differs from its partial trajectory")
    if evidence.get("success") is not None or evidence.get("outcome_available") is not False:
        raise ValueError("Structural event was silently assigned an outcome")


def strict_support_counts(masks, target):
    result = []
    for camera in ("image", "wrist_image"):
        target_mask = np.asarray(masks[f"{camera}__{target}"], bool)
        allowed = np.asarray(masks[f"{camera}__unowned_background"], bool)
        if target_mask.shape != (224, 224) or allowed.shape != target_mask.shape:
            raise ValueError("Structural evidence uses another camera geometry")
        needed, available = int(target_mask.sum()), int((allowed & ~target_mask).sum())
        result.append({"camera": camera, "requested_pixels": needed, "available_pixels": available,
                       "exact_match_possible": available >= needed})
    return result


def verify_resume_proof(path, *, run):
    """Recompute exact two-decision actions/state comparisons from saved proof files."""
    path = Path(path).resolve()
    proof = json.loads(path.read_text())
    run = Path(run).resolve()
    if (proof.get("complete") is not True or proof.get("passed") is not True
            or proof.get("technical_only") is not True or proof.get("decisions") != 2
            or proof.get("run") != str(run)):
        raise ValueError("Missing exact technical resume replay proof")
    root = Path(__file__).resolve().parents[3]
    required = (root / "scripts/run_vla_trace_interventions_gpu6_resume_v1.py", Path(__file__).resolve(),
                root / "scripts/run_vla_trace_interventions_gpu6_v1.py")
    inputs = {str(path): record(path)}
    for source in required:
        if proof["source_input_sha256"].get(str(source)) != sha(source):
            raise ValueError("Resume replay used another numerical/availability implementation")
        inputs[str(source)] = record(source)
    rows = proof.get("comparisons", [])
    expected = {(p, c, 0) for p in POLICIES for c in CONDITIONS} | {(p, c, i) for p, (i, c) in NONZERO_PROOFS.items()}
    if {(r.get("policy"), r.get("condition"), r.get("episode_index")) for r in rows} != expected or len(rows) != 40:
        raise ValueError("Resume proof lacks all conditions and the nonzero original episode index")
    for row in rows:
        snapshots = {}
        index = row["episode_index"]
        checkpoint = "pi05_libero" if row["policy"] == "pi05_libero" else str(run / "methods/rldt/object_1/training" / f"iteration_{int(row['policy'].split('_')[1]):06d}")
        for kind in ("candidate", "reference"):
            artifacts = row[kind]
            for item in artifacts.values():
                verify_record(item)
                inputs[item["path"]] = item
            if kind == "reference":
                if index == 0:
                    expected_folder = run / "compiler_profile_integration/gpu6_concurrency_v1" / f"worker_{POLICIES.index(row['policy'])}" / row["condition"].replace(":", "_")
                else:
                    expected_folder = run / "interventions/object_1" / row["policy"] / row["condition"].replace(":", "_")
                reference_files = [("actions", f"episode_{index:03d}_actions.npz"),
                                   ("telemetry", f"episode_{index:03d}_telemetry.json"), ("no_op", "no_op.json")]
                reference_files.append(("summary", "summary.json") if index == 0 else ("episode_log", "episodes.jsonl"))
                for name, filename in reference_files:
                    if Path(artifacts[name]["path"]) != expected_folder / filename:
                        raise ValueError("Replay reference is not the immutable original artifact")
            else:
                case_plan = json.loads(Path(artifacts["case_plan"]["path"]).read_text())
                if (case_plan.get("policy") != row["policy"] or case_plan.get("checkpoint") != checkpoint
                        or case_plan.get("technical_smoke") is not True
                        or case_plan.get("state_ids") != [150 + index] or case_plan.get("seeds") != [100000 + index]
                        or case_plan.get("reset_progression_state_ids") != list(range(150, 151 + index))):
                    raise ValueError("Candidate replay did not use the original index and reset progression")
                for source in required[:2]:
                    if case_plan["source_sha256"].get(str(source)) != sha(source):
                        raise ValueError("Candidate replay ran another producer implementation")
                expected_profile = "native_baseline_gpu6_v1" if row["policy"] == "pi05_libero" else "native_trained_gpu6_v1"
                if case_plan["compiler_profile"]["profile_id"] != expected_profile:
                    raise ValueError("Candidate replay used another compiler/device profile")
            with np.load(artifacts["actions"]["path"], allow_pickle=False) as archive:
                actions = archive["normalized"]
                if kind == "reference" and index != 0:
                    actions = actions[:2]
                if actions.shape != (2, 10, 32) or not np.isfinite(actions).all():
                    raise ValueError("Replay actions do not cover two finite native decisions")
            telemetry = json.loads(Path(artifacts["telemetry"]["path"]).read_text())
            no_op = json.loads(Path(artifacts["no_op"]["path"]).read_text())
            if no_op != {"passed": True, "max_abs_difference": 0.0,
                         "persisted_noise_recurrence_max_abs_difference": 0.0}:
                raise ValueError("Replay sampler no-op is not exact")
            if kind == "reference" and index != 0:
                telemetry = telemetry[:10]
                episode = json.loads(Path(artifacts["episode_log"]["path"]).read_text().splitlines()[index])
                if type(episode.get("success")) is not bool or episode.get("technical_smoke") is not False:
                    raise ValueError("Nonzero reference is not a completed original episode")
            else:
                report = json.loads(Path(artifacts["summary"]["path"]).read_text())
                if not report.get("technical_smoke") or len(report.get("episodes", [])) != 1:
                    raise ValueError("Replay scope differs")
                episode = report["episodes"][0]
                if episode["success"] is not None:
                    raise ValueError("Technical replay improperly scores a native outcome")
            if len(telemetry) != 10:
                raise ValueError("Replay scope differs")
            if (episode["episode"] != index or episode["initial_state_id"] != 150 + index
                    or episode["seed"] != 100000 + index or episode["checkpoint"] != checkpoint
                    or episode.get("status") == "structural_NA"):
                raise ValueError("Replay is not the original unscored technical case")
            snapshots[kind] = (actions, telemetry, episode)
        candidate, reference = snapshots["candidate"], snapshots["reference"]
        if (not np.array_equal(candidate[0], reference[0]) or candidate[1] != reference[1]
                or candidate[2]["reset"] != reference[2]["reset"]
                or candidate[2]["checkpoint"] != reference[2]["checkpoint"]
                or candidate[2]["intervention_severity"] != reference[2]["intervention_severity"][:2]
                or candidate[2]["condition"] != row["condition"] or reference[2]["condition"] != row["condition"]):
            raise ValueError("Resume changed original actions, simulator trajectory or reset identity")
    return inputs
