"""Extract native contact evidence and a video review queue without semantic guesses."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

TARGETS = {"object_0": "alphabet_soup_1", "object_1": "cream_cheese_1", "object_2": "salad_dressing_1"}


def episode_evidence(trajectory, target):
    directory = Path(trajectory)
    objects = {}
    measured_steps = 0
    for chunk in sorted(directory.glob("chunk_*.npz")):
        with np.load(chunk, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        for info in metadata["step_infos"]:
            telemetry = info.get("object_telemetry")
            if not telemetry:
                continue
            measured_steps += 1
            for name, state in telemetry["objects"].items():
                row = objects.setdefault(name, {"contact_steps": 0, "consecutive_lift_contact": 0,
                    "max_consecutive_lift_contact": 0, "max_lift_m": 0., "min_gripper_distance_m": float("inf"),
                    "first_contact_step": None})
                contact = state["both_fingerpads_contact"]
                row["contact_steps"] += int(contact)
                row["max_lift_m"] = max(row["max_lift_m"], state["lift_from_reset_m"])
                row["min_gripper_distance_m"] = min(row["min_gripper_distance_m"], state["gripper_distance_m"])
                if contact and row["first_contact_step"] is None:
                    row["first_contact_step"] = measured_steps
                row["consecutive_lift_contact"] = row["consecutive_lift_contact"]+1 if contact and state["lift_from_reset_m"] >= .03 else 0
                row["max_consecutive_lift_contact"] = max(row["max_consecutive_lift_contact"], row["consecutive_lift_contact"])
    candidates = [name for name, row in objects.items()
                  if name != "basket_1" and row["max_consecutive_lift_contact"] >= 3]
    if target in candidates:
        suggestion = "correct grasp but placement failure"
    elif candidates:
        suggestion = "wrong object"
    else:
        suggestion = "other/ambiguous"
    return {"measured_steps": measured_steps, "objects": objects,
            "sustained_grasp_candidates": candidates, "provisional_failure_type": suggestion,
            "rule": "Both native fingerpad contacts plus >=3cm lift above reset for >=3 consecutive controlled steps; provisional evidence requiring video review",
            "reviewed": False}


def collect(root):
    rows = []
    for task, target in TARGETS.items():
        for report_path in sorted((root/"methods/rldt"/task).glob("*.json")):
            report = json.loads(report_path.read_text())
            if not isinstance(report, dict) or not isinstance(report.get("episodes"), list):
                continue
            for index, episode in enumerate(report["episodes"]):
                if not episode.get("trajectory"):
                    continue
                evidence = episode_evidence(episode["trajectory"], target)
                if episode["success"]:
                    evidence["provisional_failure_type"] = "success"
                rows.append({"task": task, "report": str(report_path), "episode": index+1,
                    "state_id": episode["reset"]["initial_state_id"], "seed": episode["seed"],
                    "success": episode["success"], "target": target, "video": episode.get("video"),
                    "trajectory": episode["trajectory"], **evidence})
    tables = root/"tables"
    tables.mkdir(exist_ok=True)
    temporary = tables/"contact_evidence.tmp"
    temporary.write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows))
    temporary.replace(tables/"contact_evidence.jsonl")
    fields = ["task", "report", "episode", "state_id", "seed", "success", "target", "video", "trajectory",
              "measured_steps", "sustained_grasp_candidates", "provisional_failure_type", "reviewed"]
    with (tables/"video_review_queue.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k]) if isinstance(row[k], list) else row[k] for k in fields})
    return {"episodes_with_recordings": len(rows), "video_review_required": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.run.resolve()), indent=2))
