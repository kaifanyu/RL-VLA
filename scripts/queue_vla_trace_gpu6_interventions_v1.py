"""Versioned Object1/GPU6 causal queue; native numerical bodies are unchanged."""

import argparse
import ast
import json
from pathlib import Path
import signal
import time
import uuid

import queue_vla_trace_diagnostics as native
from rl_vla.diagnostics.compiler_profile_gpu6_v1 import child_environment, load_profile, observed_device
from rl_vla.diagnostics.device_fidelity_gpu6_v1 import policy_fidelity_contract, verify_task_fidelity

ROOT = Path(__file__).resolve().parents[1]
NATIVE_QUEUE_SHA256 = "232a5502e23a99514a285b155cf91c757c7f3f0a949e0eaad42d79ac089d74da"
CHILD = ROOT / "scripts/run_vla_trace_interventions_gpu6_v1.py"


def source_contract():
    old_queue = ROOT / "scripts/queue_vla_trace_diagnostics.py"
    if native.file_hash(old_queue) != NATIVE_QUEUE_SHA256:
        raise ValueError("Versioned GPU6 queue requires its exact frozen native predecessor")
    old_child = ROOT / "scripts/run_vla_trace_interventions.py"
    functions = lambda path: {node.name: ast.dump(node, include_attributes=False)
                             for node in ast.parse(path.read_text()).body
                             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "main"}
    if functions(old_child) != functions(CHILD):
        raise ValueError("GPU6 repair changed a native intervention numerical/helper function")
    paths = [Path(__file__).resolve(), old_queue, old_child, CHILD,
             ROOT / "src/rl_vla/diagnostics/compiler_profile_gpu6_v1.py",
             ROOT / "src/rl_vla/diagnostics/device_fidelity_gpu6_v1.py",
             ROOT / "src/rl_vla/diagnostics/pi05_attention.py",
             ROOT / "src/rl_vla/diagnostics/interventions.py"]
    return {str(path): native.file_hash(path) for path in paths}


def policy_jobs(args, policies, selection, output, profiles):
    # Reuse the unchanged official case/condition/seed/output command generator.
    jobs = native.policy_jobs(args, policies, selection, output)
    for job in jobs:
        job["command"][1] = str(CHILD)
        profile = profiles["baseline" if job["name"] == "pi05_libero" else "trained"]
        environment, contract = child_environment(profile,
            Path(profile["manifest"]["isolated_cache_root"]) / "causal_repair_v1" / job["name"],
            base_env=job["environment"])
        job["environment"].update({key: environment[key] for key in contract["environment"]})
        contract_path = output / "compiler_contracts" / f'{job["name"]}.json'
        native.write(contract_path, contract)
        job["command"] += ["--compiler-profile", profile["manifest_path"],
                           "--compiler-contract", str(contract_path)]
        job["compiler_profile"] = contract
    return jobs


def concurrency_contract(directory, profiles):
    import numpy as np

    directory = Path(directory).resolve()
    summary_path = directory / "summary.json"
    summary = native.read(summary_path)
    validate_concurrency_summary(summary)
    if summary.get("gpu_uuid") != observed_device()["uuid"]:
        raise ValueError("Concurrency proof was not measured on the repaired physical GPU6")
    release_path = directory / "release_verification.json"
    release = native.read(release_path)
    if (not release or release.get("physical_gpu_lock_acquired_after_smoke") is not True
            or release.get("live_worker_pids") != []
            or summary.get("release_verification_sha256") != native.file_hash(release_path)):
        raise ValueError("GPU6 technical worker processes are not released")
    for role, profile in profiles.items():
        row = summary["compiler_profiles"][role]
        if any(row.get(key) != profile[key] for key in
               ("manifest_path", "manifest_sha256", "profile_sha256", "helper_sha256")):
            raise ValueError("Concurrency proof used another derived profile")
    plan_path = directory / "plan.json"
    plan = native.read(plan_path)
    required_sources = [CHILD, ROOT / "src/rl_vla/diagnostics/pi05_attention.py",
                        ROOT / "src/rl_vla/diagnostics/interventions.py",
                        ROOT / "src/rl_vla/diagnostics/compiler_profile_gpu6_v1.py"]
    for path in required_sources:
        if plan["source_input_sha256"].get(str(path)) != native.file_hash(path):
            raise ValueError(f"GPU6 concurrency numerical source changed: {path}")
    hashes = {str(path): native.file_hash(path) for path in (summary_path, release_path, plan_path)}
    case_plans = sorted(directory.glob("worker_*/case_plan.json"))
    if len(case_plans) != 4:
        raise ValueError("GPU6 concurrency proof lacks four child case plans")
    campaign = Path(profiles["baseline"]["manifest_path"]).resolve().parents[2]
    expected_checkpoints = ["pi05_libero", *[str(campaign / "methods/rldt/object_1/training" / f"iteration_{i:06d}")
                                              for i in (25, 50, 100)]]
    if summary.get("loaded_experts") != expected_checkpoints:
        raise ValueError("GPU6 concurrency workers did not load all four exact campaign policies")
    for worker, path in enumerate(case_plans):
        case_plan = native.read(path)
        if (path.parent.name != f"worker_{worker}" or case_plan.get("checkpoint") != expected_checkpoints[worker]
                or case_plan.get("state_ids") != [150] or case_plan.get("seeds") != [100000]
                or case_plan.get("conditions") != list(native.CONDITIONS)
                or case_plan.get("technical_smoke") is not True or case_plan.get("smoke_decisions") != 2):
            raise ValueError("GPU6 concurrency case plan differs from fixed policy/case/conditions")
        hashes[str(path)] = native.file_hash(path)
        for source, digest in case_plan["source_sha256"].items():
            if native.file_hash(source) != digest:
                raise ValueError("GPU6 concurrency child source/input changed")
            hashes[source] = digest
    for row in summary["conditions"]:
        for worker in range(4):
            folder = directory / f"worker_{worker}" / row["condition"].replace(":", "_")
            action_path = folder / "episode_000_actions.npz"
            action_hash = native.file_hash(action_path)
            if action_hash != row["action_artifact_sha256"][worker]:
                raise ValueError("GPU6 concurrency action artifact changed after proof")
            no_op_path = folder / "no_op.json"
            if native.read(no_op_path) != row["native_no_op"][worker]:
                raise ValueError("GPU6 concurrency no-op artifact differs from exact proof")
            report_path = folder / "summary.json"
            condition_report = native.read(report_path)
            if (not condition_report or condition_report.get("completed") is not True
                    or condition_report.get("technical_smoke") is not True
                    or len(condition_report.get("episodes", [])) != 1):
                raise ValueError("GPU6 concurrency condition report is not the declared technical case")
            episode = condition_report["episodes"][0]
            if (episode.get("success") is not None or episode.get("initial_state_id") != 150
                    or episode.get("seed") != 100000 or episode.get("primitive_steps") != 10
                    or episode.get("checkpoint") != expected_checkpoints[worker]):
                raise ValueError("GPU6 concurrency episode identity/budget differs")
            telemetry_path = folder / "episode_000_telemetry.json"
            telemetry = native.read(telemetry_path)
            if len(telemetry) != 10:
                raise ValueError("GPU6 concurrency telemetry does not cover two complete decisions")
            with np.load(action_path, allow_pickle=False) as archive:
                actions = archive["normalized"].copy()
            if len(actions) != 2 or not np.isfinite(actions).all():
                raise ValueError("GPU6 concurrent actions are incomplete or nonfinite")
            for artifact in (action_path, no_op_path, report_path, telemetry_path, Path(episode["video"])):
                hashes[str(artifact)] = native.file_hash(artifact)
            if row["condition"] == "none":
                replay = row["none_recorded_native_replay"][worker]
                reference_report = native.read(Path(replay["recorded_report"]))
                if (reference_report.get("checkpoint") or "pi05_libero") != expected_checkpoints[worker]:
                    raise ValueError("GPU6 none reference belongs to another native checkpoint")
                reference = reference_report["episodes"][0]
                if [Path(chunk["path"]).resolve() for chunk in replay["source_chunks"]] != [
                        Path(reference["trajectory"]).resolve() / f"chunk_{i:06d}.npz" for i in (1, 2)]:
                    raise ValueError("GPU6 none reference chunks do not belong to the paired first native episode")
                reference_actions, reference_motor, reference_states = [], [], []
                for chunk in replay["source_chunks"]:
                    with np.load(chunk["path"], allow_pickle=False) as archive:
                        reference_actions.append(archive["normalized_actions"])
                        reference_motor.extend(archive["executed_actions"])
                        metadata = json.loads(str(archive["metadata_json"].item()))
                        reference_states.extend(info["object_telemetry"]["simulator_state"] for info in metadata["step_infos"])
                    hashes[chunk["path"]] = chunk["sha256"]
                if (not np.array_equal(actions, np.asarray(reference_actions))
                        or not np.array_equal(np.asarray([item["action"] for item in telemetry]), np.asarray(reference_motor))
                        or not np.array_equal(np.asarray([item["simulator_state"] for item in telemetry]), np.asarray(reference_states))
                        or episode["reset"] != reference["reset"]):
                    raise ValueError("GPU6 concrete none artifacts fail independent recorded-action/state replay")
                hashes[replay["recorded_report"]] = replay["report_sha256"]
    return hashes


def validate_concurrency_summary(summary):
    if (not isinstance(summary, dict) or summary.get("complete") is not True
            or summary.get("passed") is not True or summary.get("workers") != 4
            or summary.get("memory_fraction_per_worker") != 0.2
            or summary.get("peak_gpu_processes", 0) < 4
            or summary.get("physical_gpu_lock_released") is not True):
        raise ValueError("GPU6 concurrency requires four actual released workers at .20 memory")
    rows = summary.get("conditions", [])
    if [row.get("condition") for row in rows] != list(native.CONDITIONS):
        raise ValueError("GPU6 concurrency proof must cover all nine ordered conditions")
    expected_policies = ["pi05_libero", "rldt_25", "rldt_50", "rldt_100"]
    for row in rows:
        checks = row.get("native_no_op", [])
        if (row.get("four_workers_finite") is not True or len(checks) != 4
                or any(check.get("passed") is not True or check.get("max_abs_difference") != 0.0
                       or check.get("persisted_noise_recurrence_max_abs_difference") != 0.0 for check in checks)
                or len(row.get("action_artifact_sha256", [])) != 4):
            raise ValueError("GPU6 concurrency condition lacks four finite exact no-op checks")
        replay = row.get("none_recorded_native_replay", [])
        if row["condition"] == "none":
            if ([item.get("policy") for item in replay] != expected_policies
                    or [item.get("worker") for item in replay] != list(range(4))):
                raise ValueError("GPU6 none control must replay all four actual native test policies")
            for item in replay:
                required = {"normalized_actions_exact", "executed_actions_exact", "simulator_states_exact", "reset_metadata_exact"}
                if set(item.get("checks", {})) != required or any(item["checks"][key] is not True for key in required):
                    raise ValueError("GPU6 none control differs from actual recorded native actions/state/reset")
                if native.file_hash(item["recorded_report"]) != item["report_sha256"]:
                    raise ValueError("GPU6 none reference report changed")
                if len(item.get("source_chunks", [])) != 2:
                    raise ValueError("GPU6 none reference must bind both native decisions")
                for chunk in item["source_chunks"]:
                    if native.file_hash(chunk["path"]) != chunk["sha256"]:
                        raise ValueError("GPU6 none reference decision changed")
        elif replay:
            raise ValueError("Intervened actions must not be mislabeled as native none replay")


def run(args):
    if args.task != "object_1" or str(args.gpu) != "6" or args.workers != 4:
        raise ValueError("This repair is exclusively four policies on Object1 physical GPU6")
    startup_sources = source_contract()
    observed_device()
    profiles = {"baseline": load_profile(args.baseline_compiler_profile),
                "trained": load_profile(args.compiler_profile)}
    if profiles["baseline"]["manifest"]["profile_id"] != "native_baseline_gpu6_v1" or profiles["trained"]["manifest"]["profile_id"] != "native_trained_gpu6_v1":
        raise ValueError("Both explicit derived baseline/trained profiles are required")
    method = args.run / "methods/rldt/object_1"
    selection = method / "selection.json"
    native.wait_file(selection, failure_path=method / "status.json")
    audit = native.wait_file(method / "checkpoint_audit.json", failure_path=method / "status.json")
    checkpoint_files = native.checkpoint_contract(method, audit)
    if native.wait_file(args.run / "moving_mask_alignment_gate.json").get("passed") is not True:
        raise ValueError("Native moving observation/mask gate failed")
    fidelity = {}
    while len(fidelity) != 2:
        native.verify_queue_sources(startup_sources)
        for kind, profile in profiles.items():
            proof = policy_fidelity_contract(args.run, args.task, profile)
            if proof is not None:
                fidelity[kind] = proof
        if len(fidelity) != 2:
            time.sleep(30)
    hashes = concurrency_contract(args.concurrency_proof, profiles)
    output = args.run / "interventions/object_1"
    output.mkdir(parents=True, exist_ok=False)
    args.output_owned = True
    policies = {"pi05_libero": None, **{f"rldt_{iteration}": method / "training" / f"iteration_{iteration:06d}"
                                        for iteration in (25, 50, 100)}}
    jobs = policy_jobs(args, policies, selection, output, profiles)
    definition = native.read(args.run / "tasks/object_1/protocol.json")["tasks"]["object_1"]
    ids, seeds = definition["test_state_ids"][:10], list(range(100000, 100010))
    lanes = native.cpu_lanes(args.cpuset, 4)
    selection_sha = native.file_hash(selection)
    paths = [selection, method / "checkpoint_audit.json", args.run / "tasks/object_1/protocol.json",
             args.run / "tasks/object_1/test.toml", args.run / "gate_b_actual/metadata.json",
             args.run / "moving_mask_alignment_gate.json", *sorted((output / "compiler_contracts").glob("*.json"))]
    hashes.update(startup_sources)
    hashes.update({str(path): native.file_hash(path) for path in paths})

    def immutable_check():
        native.verify_queue_sources(hashes)
        native.verify_checkpoint_contract(checkpoint_files)
        for proof in fidelity.values():
            verify_task_fidelity(proof)

    plan = {"version": "gpu6_causal_repair_v1", "task": args.task, "jobs": jobs,
            "cpu_lanes": lanes, "max_workers": 4, "max_primary_workers": 4,
            "input_sha256": hashes, "checkpoint_files": checkpoint_files,
            "compiler_profiles": {kind: profile["manifest_path"] for kind, profile in profiles.items()},
            "policy_fidelity": fidelity, "states": ids, "seeds": seeds,
            "conditions": list(native.CONDITIONS), "overflow": None,
            "scope": "Same predeclared cases/conditions/seeds; isolated device-descriptor startup repair"}
    native.write(output / "queue_plan.json", plan)
    with native.exclusive_gpu(args.gpu):
        immutable_check()
        observed_device()
        native.run_jobs(jobs, lanes, output, immutable_check=immutable_check,
                        validate=lambda job: native.validate_policy(job, ids, seeds, selection_sha))
    native.verify_checkpoint_contract(checkpoint_files, full_hash=True)
    for proof in fidelity.values():
        verify_task_fidelity(proof, full_hash=True)
    native.write(output / "queue_complete.json", {**plan, "complete": True, "passed": True,
                 "technical_only": False, "policies": list(policies), "episodes_per_condition": 10,
                 "selection_sha256": selection_sha, "max_concurrent_policies": 4,
                 "checkpoint_hashes_verified_start_and_final": True})
    native.write(output / "queue_status.json", {"state": "completed", "version": "gpu6_causal_repair_v1"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--task", default="object_1")
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--cpuset", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compiler-profile", required=True, type=Path)
    parser.add_argument("--baseline-compiler-profile", required=True, type=Path)
    parser.add_argument("--concurrency-proof", required=True, type=Path)
    args = parser.parse_args()
    args.run = args.run.resolve()
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit("Stopping owned GPU6 workers")))
    try:
        run(args)
    except BaseException as error:
        # Keep failed/stopped-attempt evidence; a restart needs a new reviewed archive.
        failure_path = (args.run / "interventions/object_1/queue_status.json" if getattr(args, "output_owned", False)
                        else args.run / "compiler_profile_integration" / f"gpu6_queue_failure_{uuid.uuid4().hex}.json")
        native.write(failure_path,
                     {"state": "failed", "version": "gpu6_causal_repair_v1", "error": repr(error)})
        raise
