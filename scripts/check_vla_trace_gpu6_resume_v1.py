#!/usr/bin/env python3
"""Unscored exact replay for the isolated structural-NA resume producer."""
from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

import queue_vla_trace_gpu6_resume_v1 as queue

from rl_vla.diagnostics.compiler_profile_gpu6_v1 import (
    child_environment,
    load_profile,
    observed_device,
)
from rl_vla.diagnostics.structural_na_resume import (
    CONDITIONS,
    NONZERO_PROOFS,
    POLICIES,
    record,
    verify_record,
    verify_resume_proof,
    write,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Stopping owned technical replay workers")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Requires root's explicit technical-only launch approval")
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    sources = queue.source_contract()
    sources[str(Path(__file__).resolve())] = record(__file__)
    for path in (args.reuse_manifest, args.amendment):
        sources[str(path.resolve())] = record(path)
    reuse = json.loads(args.reuse_manifest.read_text())
    profiles = {name: load_profile(args.run / "compiler_profiles" / profile / "manifest.json")
                for name, profile in (("baseline", "native_baseline_gpu6_v1"), ("trained", "native_trained_gpu6_v1"))}
    stage_jobs = {}
    for stage in ("initial", "nonzero"):
        jobs = []
        for policy in POLICIES:
            index = 0 if stage == "initial" else NONZERO_PROOFS[policy][0]
            profile = profiles["baseline" if policy == "pi05_libero" else "trained"]
            env, contract = child_environment(profile, Path(profile["manifest"]["isolated_cache_root"]) / "structural_na_replay_v1" / policy)
            contract_path = args.output / "compiler_contracts" / f"{stage}_{policy}.json"
            destination = args.output / stage / policy
            command = [str(ROOT / "third_party/openpi/.venv/bin/python"), str(queue.CHILD),
                       "--config", str(args.run / "tasks/object_1/test.toml"),
                       "--selection", str(args.run / "methods/rldt/object_1/selection.json"),
                       "--policy", policy, "--output", str(destination),
                       "--token-metadata", str(args.run / "gate_b_actual/metadata.json"),
                       "--reuse-manifest", str(args.reuse_manifest.resolve()), "--amendment", str(args.amendment.resolve()),
                       "--compiler-profile", profile["manifest_path"], "--compiler-contract", str(contract_path),
                       "--technical-smoke", "--episodes", str(index + 1), "--proof-episode-index", str(index)]
            if policy != "pi05_libero":
                command.extend(("--checkpoint", str(args.run / "methods/rldt/object_1/training" / f"iteration_{int(policy.split('_')[1]):06d}")))
            environment = {key: env[key] for key in contract["environment"]}
            environment.update(XLA_PYTHON_CLIENT_MEM_FRACTION="0.20", RL_VLA_OBJECT_TELEMETRY="1")
            jobs.append({"name": policy, "destination": str(destination), "command": command,
                         "environment": environment, "compiler_contract": contract,
                         "compiler_contract_path": str(contract_path)})
        stage_jobs[stage] = jobs
    plan = {"technical_only": True, "research_outcomes_scored": False,
            "initial_state": 150, "initial_seed": 100000, "nonzero_cases": NONZERO_PROOFS,
            "sampling_decisions": 2, "initial_conditions": list(CONDITIONS),
            "source_input_sha256": {p: v["sha256"] for p, v in sources.items()},
            "stage_jobs": stage_jobs, "gpu": "6", "cpus": "8-15", "output": str(args.output)}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for jobs in stage_jobs.values():
        for job in jobs:
            write(job["compiler_contract_path"], job["compiler_contract"])
    write(args.output / "plan.json", plan)

    def guard():
        for item in [*sources.values(), *reuse["inputs"].values()]:
            verify_record(item, full_hash=False)

    comparisons, exits, descendants, groups = [], [], [], []
    with queue.prior.native.exclusive_gpu("6"):
        device = observed_device()
        for stage in ("initial", "nonzero"):
            stage_dir = args.output / stage
            stage_dir.mkdir()
            _, stage_exits, stage_descendants, stage_groups = queue.run_children(stage_jobs[stage],
                queue.prior.native.cpu_lanes("8-15", 4), stage_dir, guard, lambda job: {"policy": job["name"]})
            exits.extend(stage_exits)
            descendants.extend(stage_descendants)
            groups.extend(stage_groups)
            for number, policy in enumerate(POLICIES):
                index = 0 if stage == "initial" else NONZERO_PROOFS[policy][0]
                for condition in (CONDITIONS if index == 0 else (NONZERO_PROOFS[policy][1],)):
                    candidate = stage_dir / policy / condition.replace(":", "_")
                    reference = (args.run / "compiler_profile_integration/gpu6_concurrency_v1" / f"worker_{number}" / condition.replace(":", "_")
                                 if index == 0 else args.run / "interventions/object_1" / policy / condition.replace(":", "_"))
                    row = {"policy": policy, "condition": condition, "episode_index": index}
                    for kind, folder in (("candidate", candidate), ("reference", reference)):
                        row[kind] = {"actions": record(folder / f"episode_{index:03d}_actions.npz"),
                                     "telemetry": record(folder / f"episode_{index:03d}_telemetry.json"),
                                     "no_op": record(folder / "no_op.json"),
                                     "case_plan": record(folder.parent / "case_plan.json")}
                        if kind == "reference" and index != 0:
                            row[kind]["episode_log"] = record(folder / "episodes.jsonl")
                        else:
                            row[kind]["summary"] = record(folder / "summary.json")
                    comparisons.append(row)
        guard()
        summary = {"complete": True, "passed": True, "technical_only": True, "decisions": 2,
                   "run": str(args.run), "source_input_sha256": plan["source_input_sha256"],
                   "comparisons": comparisons, "device": device, "child_exits": exits,
                   "observed_descendants": descendants, "owned_process_groups": sorted(set(groups)),
                   "parent_pid": os.getpid(), "research_outcomes_scored": False}
        candidate_path = args.output / "proof_candidate.json"
        write(candidate_path, summary)
        verified = verify_resume_proof(candidate_path, run=args.run)
        for item in verified.values():
            verify_record(item)
    summary["gpu_lock_released"] = True
    write(args.output / "summary.json", summary)
    print(args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
