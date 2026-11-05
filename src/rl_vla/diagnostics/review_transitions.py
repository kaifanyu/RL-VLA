"""Descriptive paired failure-label transitions; no new scoring or inference."""
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
from pathlib import Path

TASKS = ("object_0", "object_1", "object_2")
POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")
LABELS = {"wrong object", "correct grasp but placement failure", "other/ambiguous"}


def category(row):
    if row["success"]:
        return "success"
    if not row.get("reviewed"):
        return "not reviewed"
    if row["failure_type"] not in LABELS:
        raise ValueError("Unsupported reviewed failure category")
    return row["failure_type"]


def transitions(episodes, *, expected_count=50):
    groups = defaultdict(dict)
    hashes = {}

    def sha(path):
        path = Path(path).resolve()
        if path not in hashes:
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes[path]

    for row in episodes:
        key = row["task"], row["split"], row["policy"]
        identity = row["initial_state_id"], row["seed"]
        if identity in groups[key]:
            raise ValueError("Duplicate paired review identity")
        groups[key][identity] = row
    cases, counts, summaries = [], [], []
    for task in TASKS:
        for split in ("validation", "test", "source"):
            for before_policy, after_policy in combinations(POLICIES, 2):
                base = {"task": task, "split": split, "before_policy": before_policy, "after_policy": after_policy}
                before = groups.get((task, split, before_policy))
                after = groups.get((task, split, after_policy))
                if not before or not after:
                    summaries.append({**base, "status": "pending_reports"})
                    continue
                if len(before) != expected_count or before.keys() != after.keys():
                    raise ValueError("Failure transition comparison lacks complete identical state/seed pairs")
                pair_cases = []
                for identity in sorted(before):
                    a, b = before[identity], after[identity]
                    if a.get("bddl_sha256") != b.get("bddl_sha256"):
                        raise ValueError("Failure transitions cross different task layouts")
                    row = {**base, "state_id": identity[0], "seed": identity[1],
                           "before_category": category(a), "after_category": category(b),
                           "before_report": a["report"], "before_report_sha256": sha(a["report"]),
                           "after_report": b["report"], "after_report_sha256": sha(b["report"]),
                           "before_video": a.get("video"), "after_video": b.get("video"),
                           "before_review_source": a.get("review_source"), "after_review_source": b.get("review_source"),
                           "before_review_source_sha256": sha(a["review_source"]) if a.get("review_source") else None,
                           "after_review_source_sha256": sha(b["review_source"]) if b.get("review_source") else None}
                    pair_cases.append(row)
                cases.extend(pair_cases)
                matrix = Counter((row["before_category"], row["after_category"]) for row in pair_cases)
                unreviewed = sum("not reviewed" in (row["before_category"], row["after_category"]) for row in pair_cases)
                status = "complete_review" if unreviewed == 0 else "pending_review"
                counts.extend({**base, "status": status, "before_category": a, "after_category": b,
                               "count": count, "paired_episodes": expected_count} for (a, b), count in sorted(matrix.items()))
                summaries.append({**base, "status": status, "paired_episodes": expected_count,
                                  "pairs_with_unreviewed_failure": unreviewed,
                                  "recovered": sum(count for (a, b), count in matrix.items() if a != "success" and b == "success"),
                                  "regressed": sum(count for (a, b), count in matrix.items() if a == "success" and b != "success"),
                                  "wrong_failure_to_success": matrix["wrong object", "success"],
                                  "wrong_failure_to_placement_failure": matrix["wrong object", "correct grasp but placement failure"],
                                  "placement_failure_to_success": matrix["correct grasp but placement failure", "success"],
                                  "success_to_placement_failure": matrix["success", "correct grasp but placement failure"],
                                  "success_to_wrong_failure": matrix["success", "wrong object"],
                                  "both_success": matrix["success", "success"],
                                  "scope": "Descriptive exact paired report/state/seed join; failure labels describe unsuccessful episodes, not exhaustive first-choice behavior among successes. Ambiguous/unreviewed cases remain distinct."})
    return cases, counts, summaries
