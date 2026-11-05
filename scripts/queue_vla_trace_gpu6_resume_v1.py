#!/usr/bin/env python3
"""Isolated strict-support resume; never publishes canonical resource release."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import queue_vla_trace_gpu6_interventions_v1 as prior

from rl_vla.diagnostics.compiler_profile_gpu6_v1 import (
    child_environment,
    load_profile,
    observed_device,
)
from rl_vla.diagnostics.device_fidelity_gpu6_v1 import (
    policy_fidelity_contract,
    verify_task_fidelity,
)
from rl_vla.diagnostics.structural_na_resume import (
    CONDITIONS,
    NONZERO_PROOFS,
    POLICIES,
    collect_reuse,
    condition_summary,
    episode_identity,
    record,
    sha,
    validate_amendment,
    verify_acceptance,
    verify_record,
    verify_resume_proof,
    write,
)

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "scripts/run_vla_trace_interventions_gpu6_resume_v1.py"


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = text[text.rindex(")") + 2:].split()
    return {"pid": int(pid), "start_ticks": int(fields[19]), "ppid": int(fields[1]),
            "pgid": int(fields[2]), "state": fields[0]}


def group_gone(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False


def source_contract():
    if sha(prior.__file__) != "a21485a9d5e41891405f4d63151f000075b7f2fe95129d85b134bd7d92d3a08a":
        raise ValueError("GPU6 predecessor changed")
    paths = [Path(__file__).resolve(), CHILD, ROOT / "src/rl_vla/diagnostics/structural_na_resume.py"]
    paths.extend(Path(p) for p in prior.source_contract())
    return {str(path): record(path) for path in paths}


def validate_policy(job, selection_sha, amendment, reuse):
    root = Path(job["destination"])
    summary = json.loads((root / "summary.json").read_text())
    if not summary.get("execution_coverage_complete") or not summary.get("complete"):
        raise ValueError("Resume policy coverage is incomplete")
    native_completed = unavailable = 0
    for condition in CONDITIONS:
        path = root / condition.replace(":", "_") / "summary.json"
        cell = json.loads(path.read_text())
        rows = cell.get("episodes", [])
        if len(rows) != 10 or not cell.get("complete"):
            raise ValueError("A condition lacks all ten resolved cases")
        for index, row in enumerate(rows):
            episode_identity(row, index, condition, job["checkpoint"], selection_sha)
            if row.get("status") == "structural_NA":
                unavailable += 1
                evidence_item = row["structural_evidence"]
                verify_record(evidence_item)
                evidence = json.loads(Path(evidence_item["path"]).read_text())
                if (condition not in ("random:background_fill", "random:mosaic")
                        or evidence.get("status") != "structural_NA" or evidence.get("success") is not None
                        or evidence.get("episode") != index or evidence.get("condition") != condition
                        or evidence.get("policy") != job["name"] or evidence.get("amendment") != amendment):
                    raise ValueError("Structural evidence has inconsistent identity")
                from rl_vla.diagnostics.structural_na_resume import validate_structural_evidence
                validate_structural_evidence(evidence)
            elif type(row.get("success")) is bool:
                native_completed += 1
            else:
                raise ValueError("Arbitrary unavailable outcomes are forbidden")
        if any(row.get("status") == "structural_NA" for row in rows):
            expected = condition_summary(condition, rows, amendment=amendment, reuse_manifest=reuse,
                                         elapsed=cell["elapsed_seconds"])
            if cell != expected:
                raise ValueError("NA condition emitted a primary estimate or wrong coverage")
    return {"policy": job["name"], "summary": record(root / "summary.json"),
            "native_completed": native_completed, "structural_NA": unavailable, "resolved": native_completed + unavailable}


def run_children(jobs, lanes, output, guard, validate):
    active, launched, exits, observed, results = [], [], [], {}, []
    owned_groups = set()

    def observe(*, best_effort=False):
        for job in jobs:
            journal = Path(job["destination"]) / "owned_processes.jsonl"
            if journal.exists():
                # A writer can be between write and flush: only complete lines publish events.
                try:
                    data = journal.read_text()
                except OSError:
                    if best_effort:
                        continue
                    raise
                for line in data.splitlines(keepends=True):
                    if not line.endswith("\n"):
                        continue
                    try:
                        identity = json.loads(line)
                        if not all(type(identity.get(key)) is int and identity[key] > 0 for key in ("pid", "start_ticks", "pgid", "ppid")):
                            raise ValueError("Malformed child process identity")
                    except (ValueError, TypeError, AttributeError):
                        if best_effort:
                            continue
                        raise
                    observed[f'{identity["pid"]}:{identity["start_ticks"]}'] = identity
                    owned_groups.add(identity["pgid"])
        try:
            paths = list(Path("/proc").iterdir())
        except OSError:
            if best_effort:
                return
            raise
        table = {}
        for path in paths:
            if not path.name.isdigit():
                continue
            try:
                identity = process_identity(int(path.name))
            except OSError:
                if best_effort:
                    continue
                raise
            if identity:
                table[identity["pid"]] = identity
            if identity and identity["pgid"] in owned_groups:
                observed[f'{identity["pid"]}:{identity["start_ticks"]}'] = identity
        known_pids = {item["pid"] for item in [*launched, *observed.values()]
                      if item["pid"] in table and table[item["pid"]]["start_ticks"] == item["start_ticks"]}
        changed = True
        while changed:
            changed = False
            for identity in table.values():
                if identity["ppid"] in known_pids and identity["pid"] not in known_pids:
                    known_pids.add(identity["pid"])
                    owned_groups.add(identity["pgid"])
                    observed[f'{identity["pid"]}:{identity["start_ticks"]}'] = identity
                    changed = True
    try:
        for job, lane in zip(jobs, lanes, strict=True):
            guard()
            if Path(job["destination"]).exists():
                raise FileExistsError(job["destination"])
            env = dict(os.environ, **job["environment"])
            env.update(RLDT_CPUSET=lane, LP_NUM_THREADS=str(len(lane.split(","))))
            command = ["taskset", "-c", lane, *job["command"]]
            stream = (output / f"{job['name']}.log").open("x")
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                stream.close()
                raise
            active.append((process, stream, job))
            identity = process_identity(process.pid)
            if identity is None:
                raise RuntimeError("Child exited before its process identity was recorded")
            job["process_identity"] = identity
            owned_groups.add(identity["pgid"])
            launched.append({"policy": job["name"], **identity, "command": command})
            write(output / "child_launches.json", launched, replace=True)
        while active:
            guard()
            observe()
            for process, stream, job in list(active):
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(f"Resume child {job['name']} exited {code}")
                process.wait()
                identity = job["process_identity"]
                if not group_gone(identity["pgid"]):
                    raise RuntimeError("A reaped worker left an owned process group alive")
                results.append(validate(job))
                observe()
                exits.append({"policy": job["name"], **identity, "returncode": code,
                              "reaped": True, "process_group_gone": True})
                active.remove((process, stream, job))
                stream.close()
            write(output / "queue_status.json", {"state": "running", "active": [j["name"] for _, _, j in active],
                  "child_exits": exits, "observed_descendants": list(observed.values())}, replace=True)
            if active:
                time.sleep(1)
        observe()
        if any(not group_gone(group) for group in owned_groups):
            raise RuntimeError("Owned environment/video process groups remain alive")
        for identity in observed.values():
            current = process_identity(identity["pid"])
            if current and current["start_ticks"] == identity["start_ticks"]:
                raise RuntimeError("An observed owned descendant remains alive")
        return results, exits, list(observed.values()), sorted(owned_groups)
    finally:
        # Discover separately grouped children before terminating their parents.
        observe(best_effort=True)
        try:
            prior.native.stop_children([p for p, _, _ in active])
        except BaseException as error:  # noqa: BLE001 - remaining groups must be cleaned even during interruption
            # A failure in one cleanup operation must not skip the other groups.
            print(f"Primary cleanup raised {error!r}; continuing owned-group cleanup", flush=True)
        observe(best_effort=True)
        for group in owned_groups:
            if group != os.getpgrp() and not group_gone(group):
                try:
                    os.killpg(group, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 10
        while any(not group_gone(group) for group in owned_groups) and time.monotonic() < deadline:
            time.sleep(.1)
        for group in owned_groups:
            if group != os.getpgrp() and not group_gone(group):
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for _, stream, _ in active:
            stream.close()


def run(args):
    args.run = args.run.resolve()
    if str(args.gpu) != "6" or args.cpuset != "8-15":
        raise ValueError("Resume resources are the original physical GPU6 and CPUs8–15")
    startup = source_contract()
    observed_device()
    profiles = {"baseline": load_profile(args.baseline_compiler_profile), "trained": load_profile(args.compiler_profile)}
    amendment = json.loads(args.amendment.read_text())
    validate_amendment(amendment, execution=not args.prepare_only)
    method = args.run / "methods/rldt/object_1"
    checkpoints = {"pi05_libero": "pi05_libero", **{f"rldt_{i}": str(method / "training" / f"iteration_{i:06d}") for i in (25, 50, 100)}}
    selection = method / "selection.json"
    selection_sha = sha(selection)
    if args.prepare_only:
        folder = args.plan_directory.resolve()
        folder.mkdir(parents=True, exist_ok=False)
        reuse = collect_reuse(args.run / "interventions/object_1", selection_sha=selection_sha, checkpoints=checkpoints)
        if reuse["completed_episodes"] != 256 or reuse["whole_conditions_complete"] != 23:
            raise ValueError("Unexpected original partial coverage")
        write(folder / "reuse_manifest.json", reuse)
        write(folder / "plan.json", {"version": "strict_background_structural_na_resume_v1", "execution": False,
              "source_input_sha256": {p: v["sha256"] for p, v in startup.items()},
              "reuse_manifest": record(folder / "reuse_manifest.json"), "amendment": record(args.amendment),
              "destination": str(args.run / "interventions_structural_na_v1/object_1"),
              "expected_resolved_cases": 360, "completed_reuse": 256, "missing_cases": 104,
              "proof_plan": "All36 first-state two-decision condition/policy exact replays plus the last completed episode of each interrupted condition; strict support unit tests; no research launch before root acceptance.",
              "nonzero_proof_cases": NONZERO_PROOFS})
        print(folder / "plan.json", flush=True)
        return
    if args.reuse_manifest is None or args.resume_proof is None or args.root_acceptance is None:
        raise ValueError("Reviewed reuse manifest and actual exact replay proof required")
    reuse = json.loads(args.reuse_manifest.read_text())
    accepted = verify_acceptance(args.root_acceptance, amendment=args.amendment,
                                 reuse_manifest=args.reuse_manifest, resume_proof=args.resume_proof)
    startup[accepted["path"]] = accepted
    proof_inputs = verify_resume_proof(args.resume_proof, run=args.run)
    prior.concurrency_contract(args.concurrency_proof, profiles)
    fidelity = {kind: policy_fidelity_contract(args.run, "object_1", profile) for kind, profile in profiles.items()}
    if any(value is None for value in fidelity.values()):
        raise ValueError("Original exact GPU6 numerical gate is not complete")
    output = args.run / "interventions_structural_na_v1/object_1"
    output.mkdir(parents=True, exist_ok=False)
    args.output_owned = output
    jobs = []
    for policy in POLICIES:
        profile = profiles["baseline" if policy == "pi05_libero" else "trained"]
        env, contract = child_environment(profile, Path(profile["manifest"]["isolated_cache_root"]) / "causal_structural_na_v1" / policy)
        contract_path = output / "compiler_contracts" / f"{policy}.json"
        write(contract_path, contract)
        command = [str(ROOT / "third_party/openpi/.venv/bin/python"), str(CHILD),
                   "--policy", policy, "--config", str(args.run / "tasks/object_1/test.toml"),
                   "--selection", str(selection), "--token-metadata", str(args.run / "gate_b_actual/metadata.json"),
                   "--output", str(output / policy), "--reuse-manifest", str(args.reuse_manifest),
                   "--amendment", str(args.amendment), "--resume-proof", str(args.resume_proof),
                   "--root-acceptance", str(args.root_acceptance),
                   "--compiler-profile", profile["manifest_path"], "--compiler-contract", str(contract_path)]
        if policy != "pi05_libero":
            command.extend(("--checkpoint", checkpoints[policy]))
        environment = {key: env[key] for key in contract["environment"]}
        environment.update(XLA_PYTHON_CLIENT_MEM_FRACTION="0.20", RL_VLA_OBJECT_TELEMETRY="1")
        jobs.append({"name": policy, "checkpoint": checkpoints[policy], "destination": str(output / policy),
                     "command": command, "environment": environment, "compiler_contract": contract})
        startup[str(contract_path)] = record(contract_path)
    for path in (args.amendment, args.reuse_manifest, args.resume_proof, selection,
                 method / "checkpoint_audit.json", args.run / "tasks/object_1/protocol.json",
                 args.run / "tasks/object_1/test.toml", args.run / "gate_b_actual/metadata.json"):
        startup[str(path)] = record(path)
    startup.update(proof_inputs)

    def guard(full=False):
        for item in [*startup.values(), *reuse["inputs"].values()]:
            verify_record(item, full_hash=full)
        for value in fidelity.values():
            verify_task_fidelity(value, full_hash=full)

    guard(full=True)
    parent = process_identity(os.getpid())
    plan = {"version": "strict_background_structural_na_resume_v1", "task": "object_1", "parent": parent,
            "original_parent": 363111, "jobs": jobs, "states": list(range(150, 160)), "seeds": list(range(100000, 100010)),
            "conditions": list(CONDITIONS), "planned_cases": 360, "reuse_manifest": record(args.reuse_manifest),
            "amendment": record(args.amendment), "resume_proof": record(args.resume_proof),
            "source_input_sha256": {p: v["sha256"] for p, v in startup.items()},
            "protocol_sha256": sha(args.run / "tasks/object_1/protocol.json"),
            "compiler_profiles": {k: v["manifest_path"] for k, v in profiles.items()}}
    write(output / "queue_plan.json", plan)
    with prior.native.exclusive_gpu("6"):
        guard()
        observed_device()
        results, exits, descendants, groups = run_children(jobs, prior.native.cpu_lanes("8-15", 4), output, guard,
            lambda job: validate_policy(job, selection_sha, plan["amendment"], plan["reuse_manifest"]))
        guard(full=True)
    native_count = sum(item["native_completed"] for item in results)
    na_count = sum(item["structural_NA"] for item in results)
    if native_count + na_count != 360:
        raise ValueError("Resume did not resolve the exact case plan")
    write(output / "queue_complete.json", {"complete": True, "passed": True, "execution_coverage_complete": True,
          "canonical_completion_published": False, "parent": parent, "child_exits": exits,
          "owned_process_groups": groups,
          "child_process_journals": [record(output / p / "owned_processes.jsonl") for p in POLICIES],
          "observed_descendants": descendants, "coverage": {"planned_cases": 360,
          "native_completed": native_count, "structural_NA": na_count, "total_resolved": 360},
          "policy_summaries": results, "queue_plan": record(output / "queue_plan.json"),
          "reuse_manifest": plan["reuse_manifest"], "amendment": plan["amendment"],
          "source_input_sha256": plan["source_input_sha256"]})
    write(output / "queue_status.json", {"state": "completed", "canonical_completion_published": False}, replace=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--cpuset", default="8-15")
    parser.add_argument("--compiler-profile", type=Path, required=True)
    parser.add_argument("--baseline-compiler-profile", type=Path, required=True)
    parser.add_argument("--concurrency-proof", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--reuse-manifest", type=Path)
    parser.add_argument("--resume-proof", type=Path)
    parser.add_argument("--root-acceptance", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--plan-directory", type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("Stopping owned resume workers")))
    try:
        run(args)
    except BaseException as error:
        if getattr(args, "output_owned", None):
            write(args.output_owned / "queue_status.json", {"state": "failed", "error": repr(error)}, replace=True)
        raise
