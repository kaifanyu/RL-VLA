"""Read-only acceptance of the versioned OOD dispatch and native-cache lineage."""
import hashlib
import json
import math
from pathlib import Path

POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")
VARIANTS = ("both_new_regions", "continuous_new_offsets", "target_moved_distractor_fixed", "distractor_moved_target_fixed")
VERSION = "per_lane_release_v1"
ACCEPTED_GPU6_ORACLE_SHA256 = "d217c613a7d6361348e97b6fa7362daa2401fdf64de338537603c85429da497e"


def completion(root, task):
    root = Path(root).resolve()
    folder = root / "ood" / task
    path = folder / "queue_complete.json"
    sources = {}

    def load(file, expected=None):
        file = Path(file).resolve()
        payload = file.read_bytes()
        checksum = hashlib.sha256(payload).hexdigest()
        if expected is not None and checksum != expected:
            raise ValueError(f"OOD bound artifact changed: {file}")
        sources[str(file)] = checksum
        return json.loads(payload) if file.suffix == ".json" else None

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def bound(value):
        return load(value["path"], value["sha256"])

    if not path.exists():
        status_path = folder / "queue_status.json"
        status = load(status_path) if status_path.exists() else {}
        failed = status.get("stage") == "failed" or status.get("state") == "failed"
        return {"task": task, "complete": False, "passed": False if failed else None,
                "reason": "OOD producer failed" if failed else "All16 per-lane OOD cells and final provenance checks pending", "sources": sources}
    try:
        queue = load(path)
        if queue.get("complete") is not True:
            return {"task": task, "complete": False, "passed": False if queue.get("passed") is False else None,
                    "reason": "OOD queue has not passed completion", "sources": sources}
        require(queue.get("passed") is True and queue.get("technical_only") is False and queue.get("version") == VERSION,
                "OOD queue completion is not a passed research dispatch")
        require(queue.get("task") == task and queue.get("episodes") == 800 and set(queue["policies"]) == set(POLICIES)
                and set(queue["variants"]) == set(VARIANTS), "OOD queue scientific matrix differs")
        review_root = root / "ood_per_lane_release_review_v1"
        launch = load(review_root / "launch_manifest_v1.json")
        amendment = load(launch["scheduling_amendment"], launch["scheduling_amendment_sha256"])
        acceptance = bound(amendment["root_acceptance"])
        independent = bound(amendment["independent_review"])
        require(launch.get("complete") is True and acceptance.get("passed") is True and independent.get("passed") is True,
                "OOD scheduling launch/review did not pass")
        require(amendment.get("approved_for_scheduling") is True and amendment.get("cases_unchanged") is True
                and amendment.get("outcome_dependent_scheduling") is False and amendment.get("version") == VERSION,
                "OOD scheduling amendment changes scientific conditions")
        release_path = review_root / "release_plans" / f"{task}.json"
        release = load(release_path, amendment["release_plan_sha256"][task])
        require(queue["release_plan_sha256"] == amendment["release_plan_sha256"][task]
                and queue["scheduling_amendment_sha256"] == launch["scheduling_amendment_sha256"], "OOD completion uses a different release amendment")
        plan = load(folder / "queue_plan.json")
        parent = launch["new_queues"][task]["parent"]
        require(plan["pid"] == parent["pid"] and plan["version"] == VERSION and plan["resources"] == queue["resources"] == release["resources"],
                "OOD parent/resource identity differs from launch")
        require(plan["state_ids"] == list(range(50)) and plan["noise_seeds"] == list(range(100000, 100050))
                and plan["selection_can_change"] is False, "OOD state/noise/selection contract differs")
        # Rehash code and compact immutable input bindings; checkpoints are
        # separately covered by the native runtime and acceptance Gate C.
        contract = load(folder / "input_manifest.json")
        for source, expected in contract["sources"].items():
            load(source, expected)
        for source, expected in contract["inputs"].items():
            if Path(source).name == "expert.npz":
                state = Path(source).stat()
                require(contract["checkpoint_stats"].get(source) == {
                    "bytes": state.st_size, "mtime_ns": state.st_mtime_ns, "inode": state.st_ino},
                    "OOD frozen expert changed; full checkpoint hashes belong to Gate C")
                continue
            load(source, expected)
        require(all(contract["sources"].get(source) == expected for source, expected in launch["source_sha256"].items()),
                "OOD native freeze differs from approved dispatcher sources")
        require(contract["dispatcher_version"] == VERSION and contract["inputs"].get(str(release_path)) == queue["release_plan_sha256"],
                "OOD native freeze does not bind release plan")
        expected_cells = {(policy, variant) for policy in POLICIES for variant in VARIANTS}
        claims, claim_data = {}, {}
        for row in queue["claims"]:
            claim = bound(row)
            identity = claim["policy"], claim["variant"]
            require(identity in expected_cells and identity not in claims and claim["task"] == task,
                    "OOD duplicate or unexpected claim")
            require(claim["state_ids"] == list(range(50)) and claim["noise_seeds"] == list(range(100000, 100050))
                    and claim["episodes"] == 50 and claim["native_shards"] == 3 and claim["primitive_step_limit"] == 280,
                    "OOD claim differs from exact50-case native protocol")
            expected_checkpoint = None if identity[0] == "pi05_libero" else str(root / "methods/rldt" / task / "training" / f"iteration_{int(identity[0].split('_')[1]):06d}")
            require(claim["checkpoint"] == expected_checkpoint and claim["input_manifest"] == str(folder / "input_manifest.json"),
                    "OOD checkpoint/native manifest differs")
            guard = bound(claim["lane_release_guard"])
            load(guard["path"], guard["sha256"])
            require(guard["owned_processes_exited"] is True and guard["gpu_compute_processes_at_acquisition"] == []
                    and guard["lane"] == claim["lane"] and guard["resource"] == claim["resource"]
                    and guard["resource"] == queue["resources"][claim["lane"]]
                    and guard["gpu_uuid"] == claim["gpu_uuid"], "OOD lane release ownership is not proven")
            claims[identity] = row
            claim_data[identity] = claim
        require(set(claims) == expected_cells and len(queue["claims"]) == 16, "OOD missing claim coverage")
        results = set()
        for row in queue["completed"]:
            identity = row["policy"], row["variant"]
            require(identity in claims and identity not in results and row["claim"] == claims[identity]["path"]
                    and row["claim_sha256"] == claims[identity]["sha256"], "OOD result/claim pairing differs")
            require(Path(row["summary"]) == folder / identity[0] / identity[1] / "summary.json", "OOD result path differs")
            persisted = load(folder / "dispatch_results" / f"{identity[0]}__{identity[1]}.json")
            require(persisted == row, "OOD completed result differs from its immutable dispatch record")
            report = load(row["summary"], row["summary_sha256"])
            require(report.get("complete") is True and report["task"] == task and report["variant"] == identity[1], "OOD report identity incomplete")
            require(report["checkpoint"] == claim_data[identity]["checkpoint"]
                    and report["source_config"] == claim_data[identity]["config"]
                    and contract["inputs"].get(report["source_config"]) == report["source_config_sha256"],
                    "OOD report checkpoint/config differs from native freeze")
            load(report["source_config"], report["source_config_sha256"])
            episodes = report["episodes"]
            require(len(episodes) == 50 and [episode["reset"]["initial_state_id"] for episode in episodes] == list(range(50))
                    and [episode["seed"] for episode in episodes] == list(range(100000, 100050)), "OOD report lacks exact paired cases")
            require(all(type(episode["success"]) is bool and 0 < episode["primitive_steps"] <= 280 for episode in episodes),
                    "OOD report has invalid native outcomes/budgets")
            require(all(episode["reset"]["experiment_split"] == identity[1] and episode["reset"]["task_id"] == int(task[-1])
                        and "object_telemetry" in episode["reset"] and episode.get("video") and episode.get("trajectory")
                        for episode in episodes), "OOD report lacks matching task/layout or native evidence")
            require(all(Path(episode["video"]).is_file() and Path(episode["trajectory"]).is_dir() for episode in episodes),
                    "OOD native video/trajectory evidence missing")
            successes = sum(episode["success"] for episode in episodes)
            from rl_vla.diagnostics.metrics import wilson
            interval = wilson(successes, 50)
            require(report["episode_count"] == 50 and report["successes"] == successes
                    and math.isclose(report["success_rate"], successes / 50, abs_tol=1e-12, rel_tol=0)
                    and len(report["wilson95"]) == 2
                    and all(math.isclose(float(a), b, abs_tol=1e-12, rel_tol=0) for a, b in zip(report["wilson95"], interval)),
                    "OOD aggregate success/interval differs from exact native episode outcomes")
            results.add(identity)
        require(results == expected_cells and len(queue["completed"]) == 16, "OOD result coverage differs")
        if task == "object_1":
            binding_path = folder / "gpu6_native_cache_binding.json"
            binding = load(binding_path, contract["inputs"][str(binding_path)])
            cache_complete = load(folder / "gpu6_native_cache_completion.json")
            require(cache_complete.get("complete") is True and cache_complete.get("passed") is True
                    and cache_complete["checks_performed"] >= 1, "OOD native cache final check missing")
            for value in (binding, cache_complete):
                require(value["parent_pid"] == parent["pid"] and value["parent_start_ticks"] == parent["start_ticks"],
                        "OOD cache proof belongs to a superseded parent")
            require(cache_complete["input_sha256"] == binding["inputs"], "OOD cache completion changes its bound inputs")
            require(len(binding["binary_sha256"]) == 3 and all(binding["inputs"].get(path) == expected for path, expected in binding["binary_sha256"].items()),
                    "OOD cache lacks its three original executable identities")
            require(binding["oracle"] == str(root / "compiler_profile_integration/gpu6_native_cache_oracle_v1/summary.json")
                    and binding["oracle_sha256"] == ACCEPTED_GPU6_ORACLE_SHA256,
                    "OOD native cache oracle differs from independently accepted exact16-case evidence")
            for source, expected in binding["inputs"].items():
                load(source, expected)
            oracle = load(binding["oracle"], binding["oracle_sha256"])
            require(oracle.get("passed") is True and oracle.get("complete") is True
                    and oracle.get("original_native_cache_compatibility_verified") is True, "OOD original native cache oracle failed")
        return {"task": task, "complete": True, "passed": True, "cells": 16, "episodes": 800,
                "version": VERSION, "reason": "Exact per-lane results/release/amendment and native-cache provenance verified", "sources": sources}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        return {"task": task, "complete": False, "passed": False, "reason": str(exc), "sources": sources}
