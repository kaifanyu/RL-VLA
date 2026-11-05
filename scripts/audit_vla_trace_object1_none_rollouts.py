"""Read-only complete-rollout comparison of repaired Object1 GPU6 none controls.

This creates evidence only. It never imports JAX, initializes a simulator, or
changes any policy, report, cache, diagnostic source, or statistical setting.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TASKS = ("object_1",)
POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")
PREDECESSOR_SHA256 = "4df22f64e0ab60fe6b2027544385fc570a781a1190fd58e5af1237be5732291c"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def native_report(root, task, policy):
    folder = root / "methods/rldt" / task
    if policy == "pi05_libero":
        return folder / "baseline_test.json"
    iteration = int(policy.split("_")[1])
    candidate = folder / f"checkpoint_{iteration:06d}_test.json"
    if candidate.exists():
        return candidate
    selection = folder / "selection.json"
    if selection.exists() and read(selection)["iteration"] == iteration:
        return folder / "selected_test.json"
    return candidate


def compare_sequences(causal, native):
    """Shape/finite-sensitive exact comparison; unequal lengths never pass."""
    causal, native = np.asarray(causal), np.asarray(native)
    if causal.ndim == 0 or native.ndim == 0 or causal.shape[1:] != native.shape[1:]:
        raise ValueError("Incompatible sequence element shapes")
    common = min(len(causal), len(native))
    finite = bool(np.all(np.isfinite(causal)) and np.all(np.isfinite(native)))
    equal_rows = np.all(causal[:common] == native[:common], axis=tuple(range(1, causal.ndim)))
    different = np.flatnonzero(~equal_rows)
    prefix = int(different[0]) if len(different) else common
    delta = (np.abs(causal[:common].astype(np.float64) - native[:common].astype(np.float64))
             if finite else np.empty((0,)))
    return {
        "shape_causal": list(causal.shape), "shape_native": list(native.shape),
        "common_rows": common, "finite": finite,
        "full_exact": bool(finite and causal.shape == native.shape and np.array_equal(causal, native)),
        "common_prefix_exact_rows": prefix, "different_common_rows": int(len(different)),
        "max_abs_difference_common": float(delta.max()) if delta.size and finite else None,
        "max_abs_difference_first_row": float(delta[0].max()) if common and finite else None,
    }


def compare_episode(causal_episode, native_episode, folder):
    index = causal_episode["episode"]
    actions_path = folder / f"episode_{index:03d}_actions.npz"
    telemetry_path = folder / f"episode_{index:03d}_telemetry.json"
    sources = [binding(actions_path), binding(telemetry_path)]
    with np.load(actions_path, allow_pickle=False) as arrays:
        causal_actions = arrays["normalized"]
    telemetry = read(telemetry_path)
    if [row["primitive_steps"] for row in telemetry] != list(range(1, len(telemetry) + 1)):
        raise ValueError("Causal telemetry primitive steps missing, repeated, or reordered")
    normalized, executed, states, keys, native_telemetry = [], [], [], [], []
    chunks = sorted(Path(native_episode["trajectory"]).glob("chunk_*.npz"))
    if not chunks:
        raise ValueError("Native trajectory contains no chunks")
    next_step = 0
    for index, chunk in enumerate(chunks):
        sources.append(binding(chunk))
        with np.load(chunk, allow_pickle=False) as arrays:
            normalized.append(arrays["normalized_actions"])
            executed.append(arrays["executed_actions"])
            metadata = json.loads(str(arrays["metadata_json"]))
        if (metadata["chunk"] != index + 1 or metadata["decision"]["decision"] != index
                or metadata["first_primitive_step"] != next_step
                or metadata["executed_steps"] != len(executed[-1])
                or len(metadata["step_infos"]) != len(executed[-1])):
            raise ValueError("Native chunk/decision/primitive sequence incomplete or unordered")
        next_step += metadata["executed_steps"]
        states.extend(row["object_telemetry"]["simulator_state"] for row in metadata["step_infos"])
        native_telemetry.extend(row["object_telemetry"] for row in metadata["step_infos"])
        keys.append(metadata["decision"]["sampling_key"])
    if next_step != native_episode["primitive_steps"] or len(telemetry) != causal_episode["primitive_steps"]:
        raise ValueError("Recorded sequence does not cover the full episode")
    checks = {
        "normalized_actions": compare_sequences(causal_actions, np.stack(normalized)),
        "executed_controller_actions": compare_sequences([r["action"] for r in telemetry], np.concatenate(executed)),
        "simulator_states": compare_sequences([r["simulator_state"] for r in telemetry], states),
    }
    reset_equal = causal_episode["reset"] == native_episode["reset"]
    outcome_equal = causal_episode["success"] == native_episode["success"]
    causal_telemetry = [{key: value for key, value in row.items() if key not in ("primitive_steps", "action")}
                        for row in telemetry]
    equal_metadata = [a == b for a, b in zip(causal_telemetry, native_telemetry)]
    metadata_exact = len(causal_telemetry) == len(native_telemetry) and all(equal_metadata)
    row = {
        "initial_state_id": causal_episode["initial_state_id"], "seed": causal_episode["seed"],
        "causal_episode": causal_episode["episode"], "native_trajectory": native_episode["trajectory"],
        "initial_reset_metadata_exact": reset_equal,
        "initial_simulator_state_exact": causal_episode["reset"]["object_telemetry"]["simulator_state"] == native_episode["reset"]["object_telemetry"]["simulator_state"],
        "causal_success": causal_episode["success"], "native_success": native_episode["success"],
        "outcome_exact": outcome_equal, "causal_primitive_steps": len(telemetry), "native_primitive_steps": next_step,
        **checks, "native_sampling_keys": keys,
        "step_metadata": {"full_exact": metadata_exact, "causal_rows": len(causal_telemetry),
                          "native_rows": len(native_telemetry), "equal_common_rows": sum(equal_metadata),
                          "first_differing_primitive_step": next((i + 1 for i, same in enumerate(equal_metadata) if not same), None),
                          "fields": sorted(native_telemetry[0])},
        "direct_causal_raw_input_and_key_retention": False,
        "passed": bool(reset_equal and outcome_equal and metadata_exact and all(v["full_exact"] for v in checks.values())),
        "sources": sources,
    }
    # Detect mutation even when source bytes were consumed only partially via NPZ.
    for source in sources:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError(f"Source changed during comparison: {source['path']}")
    return row


def compare_cell(root, task, policy):
    report_path = native_report(root, task, policy)
    folder = root / "interventions" / task / policy / "none"
    summary_path = folder / "summary.json"
    if not report_path.exists() or not summary_path.exists():
        return None
    report, summary = read(report_path), read(summary_path)
    if len(report.get("episodes", [])) != 50 or summary.get("complete") is not True:
        return None
    if report.get("technical_only") or summary.get("technical_smoke"):
        raise ValueError("Technical-only artifacts cannot serve as research controls")
    protocol_path = root / "tasks" / task / "protocol.json"
    config_path = root / "tasks" / task / "test.toml"
    selection_path = root / "methods/rldt" / task / "selection.json"
    plan_path = folder.parent / "case_plan.json"
    plan = read(plan_path)
    expected_profile = "native_baseline_gpu6_v1" if policy == "pi05_libero" else "native_trained_gpu6_v1"
    if plan["compiler_profile"]["profile_id"] != expected_profile:
        raise ValueError("Only the explicitly repaired GPU6 attempt belongs in this audit")
    definition = read(protocol_path)["tasks"][task]
    expected_states = definition["test_state_ids"]
    expected_keys = list(zip(expected_states[:10], range(100000, 100010)))
    native_keys = [(e["reset"]["initial_state_id"], e["seed"]) for e in report["episodes"]]
    causal_keys = [(e["initial_state_id"], e["seed"]) for e in summary["episodes"]]
    if native_keys != list(zip(expected_states, range(100000, 100050))) or causal_keys != expected_keys:
        raise ValueError("Reports do not contain the declared complete ordered paired cohort")
    if (plan["state_ids"] != expected_states[:10] or plan["seeds"] != list(range(100000, 100010))
            or plan["protocol_sha256"] != sha(protocol_path) or plan["selection_sha256"] != sha(selection_path)
            or plan["config_sha256"] != sha(config_path)
            or plan.get("technical_smoke") or summary["condition"] != "none"):
        raise ValueError("Causal case-plan binding is invalid")
    expected_checkpoint = "pi05_libero" if report["checkpoint"] is None else str(Path(report["checkpoint"]).resolve())
    if plan["checkpoint"] != expected_checkpoint:
        raise ValueError("Main and causal checkpoint identities differ")
    if any(e["checkpoint"] != expected_checkpoint or e["condition"] != "none" or e.get("technical_smoke")
           for e in summary["episodes"]):
        raise ValueError("Causal episode policy/condition identity differs from its plan")
    sources = [binding(p) for p in (report_path, summary_path, protocol_path, config_path, selection_path, plan_path)]
    episodes = [compare_episode(c, n, folder) for c, n in zip(summary["episodes"], report["episodes"][:10])]
    for source in sources:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Cell source changed during comparison")
    return {
        "schema": "object1-repaired-none-native-full-rollout-cell-v1", "complete": True,
        "task": task, "policy": policy, "physical_gpu": 6,
        "comparison_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(row["passed"] for row in episodes), "paired_episodes": len(episodes),
        "successes_causal": sum(row["causal_success"] for row in episodes),
        "successes_native": sum(row["native_success"] for row in episodes),
        "outcomes_equal": sum(row["outcome_exact"] for row in episodes),
        "full_normalized_action_sequences_exact": sum(row["normalized_actions"]["full_exact"] for row in episodes),
        "full_executed_action_sequences_exact": sum(row["executed_controller_actions"]["full_exact"] for row in episodes),
        "full_simulator_state_sequences_exact": sum(row["simulator_states"]["full_exact"] for row in episodes),
        "full_step_metadata_sequences_exact": sum(row["step_metadata"]["full_exact"] for row in episodes),
        "compiler_profile_id": plan["compiler_profile"]["profile_id"],
        "compiler_profile": plan["compiler_profile"], "sources": sources, "episodes": episodes,
        "auditor_source": binding(__file__),
    }


def refresh(root, output):
    predecessor = Path(__file__).with_name("audit_vla_trace_none_rollouts.py")
    if sha(predecessor) != PREDECESSOR_SHA256:
        raise ValueError("Documented healthy-device auditor predecessor changed")
    cells, pending = [], []
    for task in TASKS:
        for policy in POLICIES:
            result_path = output / "cells" / f"{task}__{policy}.json"
            if result_path.exists():
                result = read(result_path)
                if result["auditor_source"] != binding(__file__):
                    raise ValueError("Auditor changed; create a separate versioned evidence destination")
                # Completed evidence is immutable; bind its compact inputs on each refresh.
                for source in result["sources"]:
                    if sha(source["path"]) != source["sha256"]:
                        raise ValueError("Completed cell input changed")
            else:
                result = compare_cell(root, task, policy)
                if result is None:
                    pending.append({"task": task, "policy": policy, "native_report": str(native_report(root, task, policy)),
                                    "causal_summary": str(root / "interventions" / task / policy / "none/summary.json")})
                    continue
                publish(result_path, result)
                print(json.dumps({"task": task, "policy": policy, "passed": result["passed"]}), flush=True)
            cells.append({key: result[key] for key in ("task", "policy", "physical_gpu", "passed", "paired_episodes", "successes_causal", "successes_native", "outcomes_equal", "full_normalized_action_sequences_exact", "full_executed_action_sequences_exact", "full_simulator_state_sequences_exact", "full_step_metadata_sequences_exact", "compiler_profile_id")} | {"result": binding(result_path)})
    summary = {"schema": "object1-repaired-none-native-full-rollout-audit-v1", "complete": not pending,
               "passed": False if any(not row["passed"] for row in cells) else (True if not pending else None),
               "scope": "All four repaired Object1 baseline/25/50/100 none controls; first ten frozen test states/seeds; full recorded actions, states and shared telemetry metadata",
               "cells": cells, "pending": pending, "expected_cells": 4, "auditor_source": binding(__file__),
               "updated_utc": datetime.now(timezone.utc).isoformat(), "gpu_jobs_launched": False,
               "predecessor": binding(predecessor),
               "limitations": ["Causal raw RGB/proprioceptive inputs and actual PRNG keys were not retained; this audit does not directly compare them.",
                               "Stored native PRNG keys are retained as evidence; key schedule inference is not labeled a causal key observation.",
                               "Simulator state includes every controlled primitive step plus reset; this does not inspect unrecorded internal MuJoCo substeps.",
                               "Step metadata comparison covers every shared retained object telemetry field. Causal files do not retain per-step native reward/done/success flags or native discounted episode return; these are unavailable for direct comparison. Final success, primitive count, seed and reset metadata are compared exactly.",
                               "Exactness applies only to these paired first-ten none controls on the recorded physical devices, not all policies, devices, or intervention conditions."]}
    publish(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    while True:
        summary = refresh(args.run.resolve(), args.output.resolve())
        if not args.watch or summary["complete"] or summary["passed"] is False:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
