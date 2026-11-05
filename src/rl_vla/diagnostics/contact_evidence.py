"""Read-only sustained contact/lift evidence, distinct from visual review labels."""

import hashlib
import json
from pathlib import Path

import numpy as np

RULE = {"version": 1, "minimum_lift_m": .03, "minimum_consecutive_steps": 3,
        "contact": "native robosuite both_fingerpads_contact", "excluded_candidates": ["basket_1"],
        "meaning": "Operational grasp candidates only; contact confirmation is not manual visual confirmation."}
TARGETS = {"object_0": "alphabet_soup_1", "object_1": "cream_cheese_1", "object_2": "salad_dressing_1"}
DISTRACTORS = {"object_0": "butter_1", "object_1": "alphabet_soup_1", "object_2": "tomato_sauce_1"}


def stat_record(path):
    value = Path(path).stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns, "inode": value.st_ino}


def native_steps(trajectory, episode, sources):
    """Read only metadata/timestamps from NPZ archives; never load image arrays."""
    directory = Path(trajectory)
    header_path = directory/"episode.json"
    payload = header_path.read_bytes()
    header = json.loads(payload)
    if (header.get("complete") is not True or header["primitive_steps"] != episode["primitive_steps"] or
            header["metadata"]["seed"] != episode["seed"] or
            header["metadata"]["reset"]["initial_state_id"] != episode["reset"]["initial_state_id"] or
            header["summary"]["success"] != episode["success"]):
        raise ValueError("Native trajectory header differs from completed report episode")
    sources.append({"path": str(header_path), "sha256": hashlib.sha256(payload).hexdigest(), **stat_record(header_path)})
    chunks = sorted(directory.glob("chunk_*.npz"))
    if len(chunks) != header["chunks"]:
        raise ValueError("Completed native trajectory has missing or extra chunks")
    expected_first = 0
    for number, path in enumerate(chunks, start=1):
        before = stat_record(path)
        with np.load(path, allow_pickle=False) as archive:
            text = str(archive["metadata_json"].item())
            metadata = json.loads(text)
            timestamps = archive["step_unix_ns"].copy() if "step_unix_ns" in archive else None
        infos = metadata["step_infos"]
        if (metadata["chunk"] != number or metadata["first_primitive_step"] != expected_first or
                metadata["executed_steps"] != len(infos) or (timestamps is not None and len(timestamps) != len(infos))):
            raise ValueError("Native contact records are not contiguous controlled steps")
        if stat_record(path) != before:
            raise ValueError("Native trajectory changed while metadata was read")
        sources.append({"path": str(path), **before,
                        "metadata_json_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "step_unix_ns_sha256": hashlib.sha256(timestamps.tobytes()).hexdigest() if timestamps is not None else None,
                        "hash_scope": "Exact metadata string and timestamp array used for evidence; image/action arrays not decompressed"})
        for offset, info in enumerate(infos, start=1):
            yield {"step": expected_first+offset, "telemetry": info.get("object_telemetry"),
                   "unix_ns": int(timestamps[offset-1]) if timestamps is not None else None}
        expected_first += len(infos)
    if expected_first != episode["primitive_steps"]:
        raise ValueError("Native trajectory controlled-step total differs from report")


def causal_steps(path, episode, sources):
    path = Path(path)
    before = stat_record(path)
    payload = path.read_bytes()
    entries = json.loads(payload)
    if stat_record(path) != before:
        raise ValueError("Causal telemetry changed while read")
    sources.append({"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), **before})
    previous = 0
    for entry in entries:
        step = entry["primitive_steps"]
        if type(step) is not int or not previous < step <= episode["primitive_steps"]:
            raise ValueError("Causal telemetry step IDs are duplicated, reordered or out of range")
        previous = step
        yield {"step": step, "telemetry": entry, "unix_ns": None}


def evidence(steps, *, target, distractor, primitive_steps, success, reset_telemetry=None):
    """Apply the root review rule, adding first-confirmation times and mixed cases."""
    objects, observed_steps, complete_steps = {}, [], []
    expected_names = set(reset_telemetry.get("objects", {})) if reset_telemetry else set()
    reset_time = None
    if reset_telemetry and reset_telemetry.get("simulator_state"):
        reset_time = float(reset_telemetry["simulator_state"][0])
    previous = 0
    for entry in steps:
        index = entry["step"]
        if type(index) is not int or not previous < index <= primitive_steps:
            raise ValueError("Expected strictly increasing one-based controlled steps")
        telemetry = entry["telemetry"]
        if index != previous+1 or not telemetry:
            for row in objects.values():
                row["consecutive_lift_contact"] = 0
        previous = index
        if not telemetry:
            continue
        states = telemetry.get("objects", {})
        if not expected_names:
            expected_names = set(states)
        observed_steps.append(index)
        if target in states and expected_names.issubset(states):
            complete_steps.append(index)
        for name in set(objects)-set(states):
            objects[name]["consecutive_lift_contact"] = 0
        for name, state in states.items():
            contact, lift, distance = state["both_fingerpads_contact"], float(state["lift_from_reset_m"]), float(state["gripper_distance_m"])
            if type(contact) is not bool or not np.isfinite([lift, distance]).all() or distance < 0:
                raise ValueError("Invalid native contact/lift/distance measurement")
            row = objects.setdefault(name, {"observed_steps": 0, "contact_steps": 0,
                "consecutive_lift_contact": 0, "max_consecutive_lift_contact": 0,
                "max_lift_m": 0., "min_gripper_distance_m": None, "first_contact_step": None,
                "first_candidate_onset_step": None, "first_candidate_confirmation_step": None,
                "first_candidate_confirmation_sim_seconds": None,
                "first_candidate_confirmation_elapsed_sim_seconds": None,
                "first_candidate_confirmation_unix_ns": None})
            row["observed_steps"] += 1
            row["contact_steps"] += int(contact)
            row["max_lift_m"] = max(row["max_lift_m"], lift)
            row["min_gripper_distance_m"] = distance if row["min_gripper_distance_m"] is None else min(row["min_gripper_distance_m"], distance)
            if contact and row["first_contact_step"] is None:
                row["first_contact_step"] = index
            row["consecutive_lift_contact"] = row["consecutive_lift_contact"]+1 if contact and lift >= RULE["minimum_lift_m"] else 0
            row["max_consecutive_lift_contact"] = max(row["max_consecutive_lift_contact"], row["consecutive_lift_contact"])
            if name not in RULE["excluded_candidates"] and row["consecutive_lift_contact"] >= RULE["minimum_consecutive_steps"] and row["first_candidate_confirmation_step"] is None:
                sim_time = float(telemetry["simulator_state"][0]) if telemetry.get("simulator_state") else None
                row.update(first_candidate_onset_step=index-RULE["minimum_consecutive_steps"]+1,
                           first_candidate_confirmation_step=index,
                           first_candidate_confirmation_sim_seconds=sim_time,
                           first_candidate_confirmation_elapsed_sim_seconds=sim_time-reset_time if sim_time is not None and reset_time is not None else None,
                           first_candidate_confirmation_unix_ns=entry.get("unix_ns"))
    complete = observed_steps == list(range(1, primitive_steps+1)) and complete_steps == observed_steps
    candidates = sorted(name for name, row in objects.items() if row["first_candidate_confirmation_step"] is not None)
    first_step = min((objects[name]["first_candidate_confirmation_step"] for name in candidates), default=None)
    first_ids = [name for name in candidates if objects[name]["first_candidate_confirmation_step"] == first_step]
    first_id = first_ids[0] if len(first_ids) == 1 else None
    non_target = [name for name in candidates if name != target]
    mixed = target in candidates and bool(non_target)
    if success:
        taxonomy, provisional = "native_success", "success"
    elif not complete:
        taxonomy, provisional = "insufficient_telemetry", "other/ambiguous"
    elif mixed:
        taxonomy, provisional = "mixed_target_and_non_target_candidates", "other/ambiguous"
    elif target in candidates:
        taxonomy, provisional = "target_candidate_goal_failed", "correct grasp but placement failure"
    elif non_target:
        taxonomy, provisional = "non_target_candidate_goal_failed", "wrong object"
    else:
        taxonomy, provisional = "no_sustained_candidate_goal_failed", "other/ambiguous"
    first = objects[first_ids[0]] if first_ids else {}
    return {"measured_steps": len(observed_steps), "primitive_steps": primitive_steps,
            "telemetry_complete": complete, "missing_telemetry_steps": sorted(set(range(1, primitive_steps+1))-set(observed_steps)),
            "objects": objects, "sustained_grasp_candidates": candidates, "non_target_candidates": non_target,
            "target_grasp_candidate": target in candidates, "distractor_grasp_candidate": distractor in candidates,
            "non_target_grasp_candidate": bool(non_target), "mixed_target_non_target_candidates": mixed,
            "multiple_candidate_objects": len(candidates) > 1, "no_grasp_candidate": not candidates,
            "first_contact_confirmed_grasp_id": first_id, "first_contact_confirmed_grasp_ids": first_ids,
            "first_confirmation_tie": len(first_ids) > 1, "first_candidate_confirmation_step": first_step,
            "first_candidate_confirmation_sim_seconds": first.get("first_candidate_confirmation_sim_seconds"),
            "first_candidate_confirmation_elapsed_sim_seconds": first.get("first_candidate_confirmation_elapsed_sim_seconds"),
            "first_candidate_confirmation_unix_ns": first.get("first_candidate_confirmation_unix_ns"),
            "first_candidate_role": "tie" if len(first_ids)>1 else "none" if first_id is None else "target" if first_id==target else "distractor" if first_id==distractor else "other_object",
            "operational_outcome": taxonomy, "provisional_failure_type": provisional,
            "reviewed": False, "automatic": True, "rule": RULE,
            "interpretation": "Contact-confirmed means this operational sustained contact/lift criterion only. Native success is preserved; provisional failure type is not a manual semantic label."}
