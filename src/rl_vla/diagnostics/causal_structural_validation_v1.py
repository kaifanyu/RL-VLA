"""Independent CPU validation of strict-control execution coverage (no outcomes imputed).

This version is deliberately separate from plotting, contact classification and the
producer. It does not initialize JAX or a simulator and never publishes GPU release.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

TASK = "object_1"
POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")
CONDITIONS = ("none", "no_image", "no_text", "target:background_fill",
              "distractor:background_fill", "random:background_fill", "target:mosaic",
              "distractor:mosaic", "random:mosaic")
RANDOM = ("random:background_fill", "random:mosaic")
APPROVAL = "compiler_profile_integration/causal_structural_na_v1/root_acceptance.json"
RAW = "interventions_structural_na_v1/object_1"
KEY_AUDIT_SHA = "1091dcd280ba644a7386fc6fbe0896de016ff963a3f8ed3ebbefc655f2443c13"


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def verify(item):
    path = Path(item["path"])
    if sha(path) != item["sha256"]:
        raise ValueError(f"Changed execution evidence: {path}")
    if "identity" in item:
        stat = path.stat()
        if item["identity"] != [stat.st_size, stat.st_mtime_ns, stat.st_ino]:
            raise ValueError(f"Changed execution evidence identity: {path}")
    return path


def same_binding(left, right):
    return Path(left["path"]).resolve() == Path(right["path"]).resolve() and left["sha256"] == right["sha256"]


def accepted_amendment(run):
    path = Path(run).resolve() / APPROVAL
    acceptance = read(path)
    if acceptance.get("approved") is not True or acceptance.get("passed") is not True:
        raise ValueError("Concrete strict-support execution has not been accepted")
    amendment_path = verify(acceptance["amendment"])
    amendment = read(amendment_path)
    if amendment.get("schema") != "causal-strict-support-amendment-v1" or amendment.get("approved_for_execution") is not True:
        raise ValueError("Strict-support amendment is not approved")
    controls = ("strict_matched_area_definition_unchanged", "no_area_capping", "no_replacement_sampling",
                "no_semantic_pixel_fallback", "no_silent_camera_skip", "normal_cases_use_original_control_implementation")
    if any(amendment["control_semantics"].get(name) is not True for name in controls):
        raise ValueError("Original strict control definition changed")
    aggregations = ("any_structural_na_makes_condition_primary_success_unavailable",
                    "any_structural_na_makes_condition_primary_paired_comparison_unavailable",
                    "no_complete_case_mean_as_full_predeclared_effect", "no_imputation",
                    "partial_trajectory_not_full_episode_outcome")
    if any(amendment["primary_aggregation"].get(name) is not True for name in aggregations):
        raise ValueError("Undefined controls cannot enter complete-case primary estimates")
    # The explicit proxy restriction is bound as text as well as the native-outcome rule.
    scope = amendment["primary_aggregation"]
    if scope.get("structural_na_disables_condition_proxy_rates") is not True:
        raise ValueError("The accepted amendment must also prohibit partial-condition proxy rates")
    for key, expected in (("main_and_ood_native_cases_required", 4200), ("causal_planned_cases", 1080), ("all_planned_cases", 5280)):
        if amendment["coverage"].get(key) != expected:
            raise ValueError("Planned scientific case counts changed")
    if amendment["preservation"].get("completed_episodes_expected") != 256 or amendment["preservation"].get("whole_conditions_expected") != 23:
        raise ValueError("Accepted preservation belongs to another interrupted attempt")
    for item in acceptance["source_input_sha256"].items():
        verify(dict(zip(("path", "sha256"), item, strict=True)))
    validator = acceptance["execution_validator_source"]
    if not same_binding(validator, binding(__file__)):
        raise ValueError("Root acceptance binds another execution validator")
    for name in ("reuse_manifest", "resume_proof", "independent_review", "resume_plan"):
        verify(acceptance[name])
    return acceptance, amendment, {"root_acceptance": binding(path), "amendment": binding(amendment_path)}


def key_recurrence(run, acceptance):
    path = Path(run) / "compiler_profile_integration/causal_structural_na_v1/native_key_recurrence_v1.json"
    if sha(path) != KEY_AUDIT_SHA or not same_binding(acceptance["native_key_recurrence"], binding(path)):
        raise ValueError("Accepted native key recurrence proof differs")
    audit = read(path)
    expected = {"complete": True, "passed": True, "schema": "native-jax-key-recurrence-v1",
                "jax": "0.5.3", "jaxlib": "0.5.3", "backend": "cpu", "jax_default_prng_impl": "threefry2x32",
                "jax_threefry_partitionable": True, "dtype": "uint32", "shape": [10, 56, 2],
                "seeds": list(range(100000, 100010)), "decisions": list(range(56)), "saved_native_recorded_key_checks": 298}
    if any(audit.get(k) != value for k, value in expected.items()) or len(audit["references"]) != 298:
        raise ValueError("Native key recurrence provenance or coverage differs")
    verify(audit["source"])
    with np.load(verify(audit["table"]), allow_pickle=False) as data:
        table = {name: _finite(data[name], (10, 56, 2), np.dtype(np.uint32)).copy() for name in ("sampling_key", "key_after_split")}
    for reference in audit["references"]:
        with np.load(verify(reference), allow_pickle=False) as observation:
            if not np.array_equal(observation["sampling_key"], table["sampling_key"][reference["state_id"] - 150, reference["decision"]]):
                raise ValueError("Native recurrence table differs from actual recorded native sampling keys")
    return table, binding(path)


def _finite(array, shape=None, dtype=None):
    array = np.asarray(array)
    if (shape is not None and array.shape != shape) or (dtype is not None and array.dtype != dtype):
        raise ValueError("Retained native array has an unexpected shape or dtype")
    if not np.isfinite(array).all():
        raise ValueError("Retained native array is nonfinite")
    return array


def validate_structural_case(row, *, policy, condition, index, amendment, expected_prompt=None, key_table=None):
    """Independently prove the exact original control is undefined, never its outcome."""
    if condition not in RANDOM or row.get("status") != "structural_NA" or row.get("success", "missing") is not None:
        raise ValueError("Structural NA requires a strict random control and null native outcome")
    if row.get("availability_reason") != "insufficient_unowned_background_for_exact_target_area":
        raise ValueError("An arbitrary runtime error is not structural inapplicability")
    evidence_path = verify(row["structural_evidence"])
    evidence = read(evidence_path)
    expected = {"task": TASK, "policy": policy, "condition": condition, "episode": index,
                "initial_state_id": 150 + index, "seed": 100000 + index, "target": "cream_cheese_1",
                "distractor": "alphabet_soup_1", "status": "structural_NA", "complete": True,
                "success": None, "outcome_available": False}
    if any(evidence.get(key, "missing") != value for key, value in expected.items()):
        raise ValueError("Structural witness identity/outcome contract differs")
    if evidence["reset"] != row["reset"] or not same_binding(evidence["amendment"], amendment):
        raise ValueError("Structural witness reset/amendment differs")
    verify(evidence["amendment"])
    decision = evidence["decision"]
    if type(decision) is not int or not 0 <= decision < 56 or evidence["random_mask_seed"] != 100000 + index * 10000 + decision:
        raise ValueError("Structural witness decision or original mask seed differs")
    if evidence["primitive_steps_before_undefined_input"] != decision * 5 or row["primitive_steps"] != decision * 5 or evidence["normalized_chunks_before_undefined_input"] != decision:
        raise ValueError("Structural interruption is not the next controlled decision boundary")
    artifacts = {}
    for item in evidence["artifacts"]:
        path = verify(item)
        if path.name in artifacts or path.parent.resolve() != evidence_path.parent.resolve():
            raise ValueError("Structural artifact is duplicated or outside its immutable witness directory")
        artifacts[path.name] = path
    if set(artifacts) != {"observation.npz", "masks.npz", "partial_telemetry.json", "partial_intervention_severity.json"}:
        raise ValueError("Structural witness lacks the full retained input/prefix evidence")
    actual_support = []
    instances = evidence["mask_metadata"]["instances"]
    if not {"cream_cheese_1", "alphabet_soup_1"}.issubset(instances):
        raise ValueError("Mask metadata lacks original named object instances")
    with np.load(artifacts["masks.npz"], allow_pickle=False) as masks:
        for camera in ("image", "wrist_image"):
            owned = np.zeros((224, 224), bool)
            for instance in instances:
                owned |= _finite(masks[f"{camera}__{instance}"], (224, 224), np.dtype(bool))
            background = _finite(masks[f"{camera}__unowned_background"], (224, 224), np.dtype(bool))
            if not np.array_equal(background, ~owned):
                raise ValueError("Eligible support is not the exact original unowned instance/robot complement")
            target = masks[f"{camera}__cream_cheese_1"]
            requested, available = int(target.sum()), int((background & ~target).sum())
            actual_support.append({"camera": camera, "requested_pixels": requested,
                                   "available_pixels": available, "exact_match_possible": available >= requested})
    if evidence["support"] != actual_support or all(item["exact_match_possible"] for item in actual_support):
        raise ValueError("Saved original masks do not prove mathematical support exhaustion")
    if evidence["first_failing_camera"] != next(item["camera"] for item in actual_support if not item["exact_match_possible"]):
        raise ValueError("First failing camera differs from the actual strict control order")
    telemetry = read(artifacts["partial_telemetry.json"])
    if len(telemetry) != decision * 5 or [step["primitive_steps"] for step in telemetry] != list(range(1, decision * 5 + 1)):
        raise ValueError("Partial telemetry is not a complete controlled prefix")
    if len(read(artifacts["partial_intervention_severity.json"])) != decision:
        raise ValueError("Partial intervention history differs from preceding decisions")
    with np.load(artifacts["observation.npz"], allow_pickle=False) as data:
        for camera in ("image", "wrist_image"):
            _finite(data[camera], (224, 224, 3), np.dtype(np.uint8))
        _finite(data["state"], (8,))
        _finite(data["normalized_action_prefix"], (decision, 10, 32))
        _finite(data["sampling_key"], (2,), np.dtype(np.uint32))
        _finite(data["key_after_split"], (2,), np.dtype(np.uint32))
        if key_table is not None and any(not np.array_equal(data[name], key_table[name][index, decision]) for name in ("sampling_key", "key_after_split")):
            raise ValueError("Retained keys differ from the original native seed/split recurrence")
        _finite(data["initial_noise"], (1, 10, 32))
        simulator_state = _finite(data["simulator_state"])
        previous = telemetry[-1] if telemetry else evidence["reset"]["object_telemetry"]
        if not np.array_equal(simulator_state, previous["simulator_state"]):
            raise ValueError("Retained simulator state differs from the last controlled prefix state")
        if data["prompt"].shape != () or (expected_prompt is not None and str(data["prompt"].item()) != expected_prompt):
            raise ValueError("Retained original instruction differs")
    return {"policy": policy, "condition": condition, "episode": index, "state_id": 150 + index,
            "seed": 100000 + index, "decision": decision, "support": actual_support,
            "evidence": binding(evidence_path), "artifacts": evidence["artifacts"],
            "coverage_scope": "Undefined original input with retained native prefix; no full-episode outcome or contact proxy"}


def _identity(row, index, condition, checkpoint, selection_sha):
    expected = {"episode": index, "initial_state_id": 150 + index, "seed": 100000 + index,
                "condition": condition, "checkpoint": checkpoint, "selection_sha256": selection_sha,
                "target": "cream_cheese_1", "distractor": "alphabet_soup_1", "technical_smoke": False}
    if any(row.get(key, "missing") != value for key, value in expected.items()):
        raise ValueError("Resolved case differs from the frozen identity/checkpoint/selection")
    reset = row["reset"]
    if reset.get("initial_state_id") != 150 + index or reset.get("task_id") != 1 or reset.get("experiment_split") != "test":
        raise ValueError("Resolved case reset differs from the exact frozen test task")


def _native_artifacts(folder, row):
    index, steps = row["episode"], row["primitive_steps"]
    if type(row.get("success")) is not bool or row.get("status", "completed") != "completed" or type(steps) is not int or not 0 < steps <= 280:
        raise ValueError("Completed native outcomes require Boolean success and full controlled horizon")
    actions = folder / f"episode_{index:03d}_actions.npz"
    telemetry_path = folder / f"episode_{index:03d}_telemetry.json"
    video = folder / f"episode_{index:03d}.mp4"
    if video.resolve() != Path(row["video"]).resolve() or video.stat().st_size == 0:
        raise ValueError("Completed episode video is missing or points to another case")
    with np.load(actions, allow_pickle=False) as data:
        _finite(data["normalized"], ((steps + 4) // 5, 10, 32))
    telemetry = read(telemetry_path)
    if len(telemetry) != steps or [s["primitive_steps"] for s in telemetry] != list(range(1, steps + 1)):
        raise ValueError("Completed native telemetry is not the full controlled episode")
    state_shape = np.asarray(row["reset"]["object_telemetry"]["simulator_state"]).shape
    for step in telemetry:
        _finite(step["action"], (7,))
        _finite(step["simulator_state"], state_shape)
        if not {"cream_cheese_1", "alphabet_soup_1"}.issubset(step["objects"]):
            raise ValueError("Completed telemetry lacks required native object identities")
    return [binding(path) for path in (actions, telemetry_path, video)]


def validate_execution(run):
    """Return a full certificate only once the new raw queue is complete; otherwise None."""
    run = Path(run).resolve()
    directory = run / RAW
    queue_path = directory / "queue_complete.json"
    if not queue_path.exists():
        return None
    queue = read(queue_path)
    if queue.get("complete") is not True or queue.get("passed") is not True or queue.get("execution_coverage_complete") is not True or queue.get("canonical_completion_published") is not False:
        raise ValueError("The independent resume queue does not claim resolved execution coverage")
    acceptance, _, approved = accepted_amendment(run)
    key_table, key_binding = key_recurrence(run, acceptance)
    plan_path = verify(queue["queue_plan"])
    if plan_path.resolve() != (directory / "queue_plan.json").resolve():
        raise ValueError("Queue completion binds another plan")
    plan = read(plan_path)
    if plan.get("task") != TASK or plan.get("states") != list(range(150, 160)) or plan.get("seeds") != list(range(100000, 100010)) or plan.get("conditions") != list(CONDITIONS) or plan.get("planned_cases") != 360 or plan.get("original_parent") != 363111:
        raise ValueError("Resume changed the original case plan")
    if queue["parent"] != plan["parent"] or queue["source_input_sha256"] != plan["source_input_sha256"]:
        raise ValueError("Queue parent/source provenance differs")
    for path, digest in queue["source_input_sha256"].items():
        verify({"path": path, "sha256": digest})
    for key in ("amendment", "reuse_manifest"):
        if not same_binding(queue[key], plan[key]) or not same_binding(queue[key], acceptance[key]):
            raise ValueError("Resume queue uses unaccepted input/amendment")
        verify(queue[key])
    if not same_binding(plan["resume_proof"], acceptance["resume_proof"]):
        raise ValueError("Resume queue proof differs from root acceptance")
    reuse = read(verify(queue["reuse_manifest"]))
    if reuse.get("task") != TASK or reuse.get("completed_episodes") != 256 or reuse.get("whole_conditions_complete") != 23:
        raise ValueError("Resume did not bind all original complete outcomes")
    for item in reuse["inputs"].values():
        verify(item)
    if [row.get("policy") for row in queue["policy_summaries"]] != list(POLICIES) and {row.get("policy") for row in queue["policy_summaries"]} != set(POLICIES):
        raise ValueError("Queue lacks exact four policy summaries")
    if len(queue["policy_summaries"]) != 4 or len(plan["jobs"]) != 4 or {j["name"] for j in plan["jobs"]} != set(POLICIES):
        raise ValueError("Duplicate or missing policy execution")
    selection_sha = sha(run / "methods/rldt/object_1/selection.json")
    bound_artifacts, structural_cases, conditions, cases = [], [], [], []
    native_total = reused_total = whole_total = 0
    for policy in POLICIES:
        folder = directory / policy
        checkpoint = "pi05_libero" if policy == "pi05_libero" else str(run / "methods/rldt/object_1/training" / f"iteration_{int(policy.split('_')[1]):06d}")
        job = next(j for j in plan["jobs"] if j["name"] == policy)
        if job["checkpoint"] != checkpoint or Path(job["destination"]).resolve() != folder:
            raise ValueError("Queue policy checkpoint/destination differs")
        queue_policy = next(item for item in queue["policy_summaries"] if item["policy"] == policy)
        top_path = verify(queue_policy["summary"])
        if top_path.resolve() != (folder / "summary.json").resolve():
            raise ValueError("Queue binds another policy summary")
        top = read(top_path)
        case_plan = read(folder / "case_plan.json")
        if top.get("complete") is not True or top.get("execution_coverage_complete") is not True or top["case_plan"] != case_plan:
            raise ValueError("Policy coverage/case plan is incomplete")
        if case_plan.get("task") != TASK or case_plan.get("policy") != policy or case_plan.get("checkpoint") != checkpoint or case_plan.get("technical_smoke") is not False or case_plan.get("conditions") != list(CONDITIONS) or case_plan.get("state_ids") != list(range(150, 160)) or case_plan.get("seeds") != list(range(100000, 100010)):
            raise ValueError("Policy changed the original case plan")
        for path, digest in case_plan["source_sha256"].items():
            verify({"path": path, "sha256": digest})
        policy_native = policy_na = 0
        summarized = []
        for condition in CONDITIONS:
            cell = folder / condition.replace(":", "_")
            summary_path = cell / "summary.json"
            report = read(summary_path)
            rows = report.get("episodes", [])
            lines = (cell / "episodes.jsonl").read_text().splitlines()
            if report.get("complete") is not True or report.get("technical_smoke") is not False or report.get("condition") != condition or report.get("n") != 10 or len(rows) != 10 or [json.loads(line) for line in lines] != rows:
                raise ValueError("Condition did not resolve all10 exact original episodes")
            old = reuse["policies"][policy]["conditions"][condition]
            if rows[:old["next_episode"]] != old["rows"] or lines[:old["next_episode"]] != old["raw_lines"]:
                raise ValueError("A previously complete native episode was rewritten")
            if old["whole_condition_complete"]:
                whole_total += 1
                if summary_path.resolve() != Path(old["original_directory"]) / "summary.json":
                    raise ValueError("A whole completed condition was not retained byte-identically")
            reused_total += old["next_episode"]
            for old_item in old["episode_artifacts"]:
                for kind, suffix in (("actions", "actions.npz"), ("telemetry", "telemetry.json"), ("video", None)):
                    expected_path = cell / (f"episode_{old_item['episode']:03d}_{suffix}" if suffix else f"episode_{old_item['episode']:03d}.mp4")
                    if sha(expected_path) != old_item[kind]["sha256"]:
                        raise ValueError("Previously complete native episode artifact was changed")
            native = na = 0
            for index, row in enumerate(rows):
                _identity(row, index, condition, checkpoint, selection_sha)
                if row.get("status") == "structural_NA":
                    prompt_path = run / "observation_bank/object_1/test" / f"state_{150 + index:06d}" / "metadata.json"
                    prompt_metadata = read(prompt_path)
                    witness = validate_structural_case(row, policy=policy, condition=condition, index=index, amendment=queue["amendment"],
                                                       expected_prompt=prompt_metadata["prompt"], key_table=key_table)
                    witness["original_prompt_metadata"] = binding(prompt_path)
                    structural_cases.append(witness)
                    cases.append({"task": TASK, "policy": policy, "condition": condition, "episode": index,
                                  "state_id": 150 + index, "seed": 100000 + index, "status": "structural_NA",
                                  "row": row, "structural_evidence": witness["evidence"]})
                    na += 1
                    with np.load(cell / f"episode_{index:03d}_actions.npz", allow_pickle=False) as data, np.load(Path(witness["evidence"]["path"]).parent / "observation.npz", allow_pickle=False) as retained:
                        if not np.array_equal(data["normalized"], retained["normalized_action_prefix"]):
                            raise ValueError("NA standard artifact differs from retained partial action prefix")
                    if read(cell / f"episode_{index:03d}_telemetry.json") != read(Path(witness["evidence"]["path"]).parent / "partial_telemetry.json"):
                        raise ValueError("NA standard artifact differs from retained partial telemetry")
                else:
                    native += 1
                    native_files = _native_artifacts(cell, row)
                    bound_artifacts.extend(native_files)
                    cases.append({"task": TASK, "policy": policy, "condition": condition, "episode": index,
                                  "state_id": 150 + index, "seed": 100000 + index, "status": "completed",
                                  "row": row, "artifacts": dict(zip(("actions", "telemetry", "video"), native_files, strict=True))})
            if na:
                if report.get("primary_outcome_available") is not False or any(report.get(key, "missing") is not None for key in ("successes", "primary_success_rate", "primary_paired_comparison")):
                    raise ValueError("A structurally incomplete original condition has a primary estimate")
            elif report.get("successes") != sum(row["success"] for row in rows):
                raise ValueError("Native completed condition outcome count differs")
            if not old["whole_condition_complete"]:
                if report.get("attempted_cases") != 10 or report.get("completed_outcome_cases") != native or report.get("structural_na_cases") != na or report.get("primary_outcome_available") != (na == 0) or not same_binding(report["amendment"], queue["amendment"]) or not same_binding(report["reuse_manifest"], queue["reuse_manifest"]):
                    raise ValueError("Resumed condition availability/accounting differs")
                if not na and report.get("primary_success_rate") != sum(row["success"] for row in rows) / 10:
                    raise ValueError("Native resumed success rate differs")
            no_op = read(cell / "no_op.json")
            if no_op != {"passed": True, "max_abs_difference": 0.0, "persisted_noise_recurrence_max_abs_difference": 0.0}:
                raise ValueError("Condition lacks exact original native no-op fidelity")
            bound_artifacts.extend(binding(path) for path in (summary_path, cell / "episodes.jsonl", cell / "no_op.json"))
            conditions.append({"policy": policy, "condition": condition, "planned_cases": 10,
                               "native_completed": native, "structural_NA": na, "total_resolved": 10,
                               "primary_outcome_available": na == 0, "summary": binding(summary_path)})
            summarized.append({k: v for k, v in report.items() if k != "episodes"})
            policy_native += native
            policy_na += na
        if top["conditions"] != summarized or top.get("primary_outcomes_all_available") != (policy_na == 0):
            raise ValueError("Policy summary does not match all resolved original conditions")
        if queue_policy["native_completed"] != policy_native or queue_policy["structural_NA"] != policy_na or queue_policy["resolved"] != 90:
            raise ValueError("Queue policy accounting differs from raw cases")
        native_total += policy_native
    if reused_total != 256 or whole_total != 23:
        raise ValueError("Original valid execution coverage was not preserved")
    coverage = {"planned_cases": 360, "native_completed": native_total, "structural_NA": len(structural_cases), "total_resolved": 360}
    if native_total + len(structural_cases) != 360 or queue["coverage"] != coverage:
        raise ValueError("Queue availability counts differ from validated case identities")
    return {"schema": "causal-structural-execution-coverage-v1", "complete": True, "passed": True, "task": TASK,
            "queue_completion": binding(queue_path), **approved, "coverage": coverage,
            "policy_summaries": queue["policy_summaries"], "validator_source": binding(__file__),
            "native_key_recurrence": key_binding,
            "preserved_native_cases": reused_total, "preserved_whole_conditions": whole_total,
            "conditions": conditions, "cases": cases, "structural_cases": structural_cases, "native_artifacts": bound_artifacts,
            "scope": "Full raw planned-case resolution and preserved native artifacts. NA prefixes are not full outcomes. Keys independently checked against accepted pinned native recurrence table; no simulator replay, Gaussian-noise regeneration, video interpretation or GPU/process-release claim."}


def publish_execution(run, certificate):
    """Publish immutable compact coverage plus separately bound per-case evidence."""
    directory = Path(run).resolve() / "compiler_profile_integration/causal_structural_na_v1"
    output = dict(certificate)
    cases = output.pop("cases")
    native_artifacts = output.pop("native_artifacts")
    case_path = directory / "execution_cases.json"
    detail = {"schema": "causal-structural-execution-cases-v1", "task": TASK,
              "complete": True, "cases": cases, "native_artifacts": native_artifacts,
              "validator_source": output["validator_source"]}

    def immutable(path, value):
        data = json.dumps(value, indent=2, allow_nan=False) + "\n"
        if path.exists():
            if path.read_text() != data:
                raise ValueError(f"An existing immutable execution certificate differs: {path}")
            return
        temporary = path.with_suffix(".tmp")
        with temporary.open("x") as stream:
            stream.write(data)
        temporary.replace(path)

    immutable(case_path, detail)
    output["case_records"] = binding(case_path)
    immutable(directory / "execution_coverage.json", output)
    return output
