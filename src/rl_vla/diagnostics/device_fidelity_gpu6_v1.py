"""Exact recorded-test-action gate for the isolated GPU6 diagnostic repair."""

import json
import math
from pathlib import Path

from rl_vla.diagnostics.compiler_profile import sha256
from rl_vla.diagnostics.compiler_profile_gpu6_v1 import load_profile, verify_effective_environment
from rl_vla.diagnostics.trained_fidelity import (
    EXPECTED_CHECKS, policy_fidelity_contract as parent_contract,
    reject_failed_campaign,
)

ROOT = Path(__file__).resolve().parents[3]
POLICIES = ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")
CASES = [(150, 0), (150, 1), (151, 0), (151, 1)]
SOURCES = ("scripts/check_pi05_gpu6_fidelity.py", "src/rl_vla/rldt_pi05.py",
           "src/rl_vla/diagnostics/pi05_attention.py", "src/rl_vla/diagnostics/layer_knockout.py",
           "src/rl_vla/diagnostics/compiler_profile_gpu6_v1.py",
           "scripts/check_pi05_trained_sampler_fidelity.py", "scripts/check_pi05_native_baseline_oracle.py")


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def stat(path):
    record = Path(path).stat()
    return {"size": record.st_size, "mtime_ns": record.st_mtime_ns, "inode": record.st_ino}


def exact(check):
    return (isinstance(check, dict) and check.get("exact") is True and check.get("finite") is True
            and type(check.get("different_elements")) is int and check["different_elements"] == 0
            and type(check.get("max_abs_difference")) in (int, float)
            and math.isfinite(check["max_abs_difference"]) and check["max_abs_difference"] == 0)


def policy_fidelity_contract(run, task, profile):
    run = Path(run).resolve()
    profile = load_profile(profile)
    if task != "object_1":
        raise ValueError("GPU6 v1 proof applies only to Object1")
    base = parent_contract(run, task, profile["parent_profile"])
    if base is None:
        return None
    directory = run / "compiler_profile_integration/gpu6_fidelity_v1"
    aggregate_path = directory / "summary.json"
    aggregate = read(aggregate_path)
    if aggregate and aggregate.get("passed") is False:
        raise ValueError("GPU6 derived-profile exact numerical proof failed")
    if not aggregate or aggregate.get("complete") is not True:
        launch = read(directory / "launch.json")
        if launch and any(job.get("returncode") not in (None, 0) for job in launch.get("jobs", [])):
            raise ValueError("GPU6 numerical proof child failed")
        return None
    if (aggregate.get("passed") is not True or aggregate.get("task") != task
            or aggregate.get("policies") != list(POLICIES)
            or set(aggregate.get("proofs", {})) != set(POLICIES)
            or aggregate.get("total_cases") != 16 or aggregate.get("total_exact_checks") != 320):
        raise ValueError("Incomplete GPU6 policy/case proof coverage")
    hashes, stats = {}, {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key in hashes:
            if expected is not None and hashes[key] != expected:
                raise ValueError("GPU6 proof contains conflicting input hashes")
            return
        before = stat(path)
        digest = sha256(path)
        if before != stat(path) or expected is not None and digest != expected:
            raise ValueError(f"GPU6 numerical proof input changed: {path}")
        hashes[key], stats[key] = digest, before

    for path, digest in base["input_sha256"].items():
        bind(path, digest)
    bind(aggregate_path)
    bind(__file__)
    for field in ("manifest_path", "profile_path", "helper_path"):
        bind(profile[field], profile[field.replace("_path", "_sha256")])
    bind(profile["manifest"]["descriptor_derivation"]["audit_path"],
         profile["manifest"]["descriptor_derivation"]["audit_sha256"])
    selected = (POLICIES[0],) if profile["manifest"]["profile_id"] == "native_baseline_gpu6_v1" else POLICIES[1:]
    audit_path = run / "methods/rldt/object_1/checkpoint_audit.json"
    audit = read(audit_path)
    if not audit or audit.get("passed") is not True:
        raise ValueError("GPU6 requires passed native checkpoint audit")
    bind(audit_path)
    for policy in selected:
        row = aggregate["proofs"][policy]
        path = Path(row["path"]).resolve()
        if path != directory / policy / "summary.json":
            raise ValueError("GPU6 policy proof escaped its declared directory")
        bind(path, row["sha256"])
        proof = read(path)
        iteration = None if policy == "pi05_libero" else int(policy.removeprefix("rldt_"))
        checkpoint = None if iteration is None else run / "methods/rldt/object_1/training" / f"iteration_{iteration:06d}"
        if (proof.get("passed") is not True or proof.get("complete") is not True
                or proof.get("task") != task or proof.get("policy") != policy
                or proof.get("iteration") != iteration or proof.get("cases_expected") != 4
                or proof.get("checkpoint") != (str(checkpoint) if checkpoint else None)
                or len(proof.get("cases", [])) != 4 or proof.get("checks_per_case") != 20
                or proof.get("compiler_profile_manifest_sha256") != profile["manifest_sha256"]
                or proof.get("autotune_sha256") != profile["profile_sha256"]):
            raise ValueError("GPU6 policy proof identity/coverage/profile differs")
        compiler = proof.get("compiler_profile", {})
        verify_effective_environment(profile, compiler, environment=compiler.get("environment", {}))
        compiler_audit = proof.get("compiler_audit", {})
        if (compiler_audit.get("passed") is not True or compiler_audit.get("versions_checked") is not True
                or compiler_audit.get("observed_device") != profile["manifest"]["device"]):
            raise ValueError("GPU6 proof lacks verified actual device/version startup evidence")
        sources = proof["source_input_sha256"]
        for source in SOURCES:
            bind(ROOT / source, sources.get(str(ROOT / source), "missing"))
        if checkpoint:
            for name in ("expert.npz", "rldt_pi05.json"):
                expected = audit["checkpoints"][str(iteration)]["files"][name]["sha256"]
                if sources.get(str(checkpoint / name)) != expected:
                    raise ValueError("GPU6 proof expert differs from native audited checkpoint")
        for source, digest in sources.items():
            bind(source, digest)
        selection_path = run / "methods/rldt/object_1/selection.json"
        selection = read(selection_path)
        report_name = "baseline_test.json" if checkpoint is None else (
            "selected_test.json" if selection["iteration"] == iteration
            else f"checkpoint_{iteration:06d}_test.json")
        report_path = run / "methods/rldt/object_1" / report_name
        report = read(report_path)
        if (not report or report.get("checkpoint") != (str(checkpoint) if checkpoint else None)
                or len(report.get("episodes", [])) != 50
                or [entry["reset"]["initial_state_id"] for entry in report["episodes"]] != list(range(150, 200))
                or [entry["seed"] for entry in report["episodes"]] != list(range(100000, 100050))):
            raise ValueError("GPU6 oracle is not the exact completed native main test report")
        bind(selection_path, sources.get(str(selection_path), "missing"))
        bind(report_path, sources.get(str(report_path), "missing"))
        identities = []
        for item in proof["cases"]:
            case_path = Path(item["path"]).resolve()
            if not case_path.is_relative_to(path.parent):
                raise ValueError("GPU6 case escaped its policy proof directory")
            bind(case_path, item["sha256"])
            case = read(case_path)
            identity = (case.get("state_id"), case.get("decision_index"))
            identities.append(identity)
            if (case.get("passed") is not True or case.get("complete") is not True
                    or case.get("task") != task or case.get("policy") != policy
                    or case.get("iteration") != iteration or identity not in CASES
                    or case.get("seed") != 100000 + identity[0] - 150
                    or not exact(case.get("recorded_action_replay"))
                    or set(case.get("checks", {})) != EXPECTED_CHECKS
                    or any(not exact(check) for check in case["checks"].values())
                    or case.get("compiler_profile") != proof.get("compiler_profile")):
                raise ValueError("GPU6 case is not exact to recorded native actions and all twenty checks")
            expected_trajectory = Path(report["episodes"][identity[0] - 150]["trajectory"]).resolve()
            if (Path(case["recorded_report"]).resolve() != report_path
                    or Path(case["recorded_chunk"]).resolve() != expected_trajectory / f"chunk_{identity[1] + 1:06d}.npz"):
                raise ValueError("GPU6 recorded decision does not belong to the paired native test episode")
            arrays = case["arrays"]
            if Path(arrays["path"]).resolve() != case_path.parent / "arrays.npz":
                raise ValueError("GPU6 proof arrays escaped their case directory")
            bind(arrays["path"], arrays["sha256"])
            bind(case["recorded_chunk"], case["recorded_chunk_sha256"])
            bind(case["recorded_report"], case["recorded_report_sha256"])
            for source, digest in case["source_input_sha256"].items():
                bind(source, digest)
            if not case.get("raw_input_arrays") or not case.get("context_leaf_sha256"):
                raise ValueError("GPU6 proof lacks recorded raw/context identity evidence")
        if identities != CASES:
            raise ValueError("GPU6 proof cases differ from the four predeclared decisions")
    return {"passed": True, "task": task, "profile_id": profile["manifest"]["profile_id"],
            "autotune_sha256": profile["profile_sha256"], "manifest_sha256": profile["manifest_sha256"],
            "parent_contract": base, "aggregate": str(aggregate_path), "policies": list(selected),
            "input_sha256": hashes, "input_stat": stats,
            "scope": "Exact actual recorded-test-input actions and20component checks; concurrency separately gated"}


def verify_task_fidelity(contract, *, full_hash=False):
    parent = contract["parent_contract"]
    reject_failed_campaign(parent["global_failure_path"], parent["autotune_sha256"])
    for path, digest in contract["input_sha256"].items():
        if stat(path) != contract["input_stat"][path] or full_hash and sha256(path) != digest:
            raise ValueError(f"Frozen GPU6 fidelity prerequisite changed: {path}")
