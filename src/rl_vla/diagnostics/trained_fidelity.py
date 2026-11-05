"""Bind diagnostic readiness to immutable recorded-action fidelity evidence."""

import json
import math
from pathlib import Path

from rl_vla.diagnostics.compiler_profile import sha256

ROOT = Path(__file__).resolve().parents[3]
ITERATIONS = (25, 50, 100)
EXPECTED_CASES = [(100, 0), (100, 1), (101, 0), (101, 1)]
EXPECTED_CHECKS = {"native_actions_only", "explicit_native_actions", "intervention_noop_actions",
                   "intervention_noop_path", "captured_actions", "captured_path", "layer_noop_actions", "layer_noop_path"}
EXPECTED_CHECKS.update(f"step{index:02d}_{name}" for index in (0, 5, 9)
                       for name in ("capture_velocity", "projection", "layer_velocity", "euler_path"))
SOURCES = ("src/rl_vla/rldt_pi05.py", "src/rl_vla/diagnostics/pi05_attention.py",
           "src/rl_vla/diagnostics/layer_knockout.py", "src/rl_vla/diagnostics/compiler_profile.py",
           "scripts/check_pi05_native_sampler_oracle.py",
           "scripts/check_pi05_trained_sampler_fidelity.py")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def reject_failed_campaign(path, autotune_sha256):
    campaign = read(Path(path))
    if (campaign and campaign.get("passed") is False
            and campaign.get("autotune_sha256") == autotune_sha256):
        raise ValueError("Trained-checkpoint fidelity campaign failed for this compiler profile")


def task_fidelity_contract(run, task, profile):
    """Return None while pending; reject inconsistent completed proof or source bytes."""
    run = Path(run).resolve()
    failure_path = run / "trained_checkpoint_sampler_fidelity.json"
    reject_failed_campaign(failure_path, profile["profile_sha256"])
    ready_path = run / "trained_sampler_fidelity/task_readiness" / f"{task}.json"
    ready = read(ready_path)
    if ready and ready.get("passed") is False:
        raise ValueError(f"Trained-checkpoint fidelity failed: {ready_path}")
    if not ready or ready.get("complete") is not True:
        return None
    if (ready.get("passed") is not True or ready.get("task") != task
            or ready.get("required_iterations") != list(ITERATIONS)
            or set(ready.get("proofs", {})) != {str(i) for i in ITERATIONS}
            or ready.get("autotune_sha256") != profile["profile_sha256"]
            or ready.get("compiler_profile_manifest_sha256") != profile["manifest_sha256"]):
        raise ValueError("Trained fidelity readiness differs from task/profile/checkpoints")
    method = run / "methods/rldt" / task
    audit_path = method / "checkpoint_audit.json"
    audit = read(audit_path)
    if audit is None:
        return None
    if audit.get("passed") is not True:
        raise ValueError("Trained fidelity requires the passed native checkpoint audit")
    hashes = {str(ready_path): sha256(ready_path), str(audit_path): sha256(audit_path),
              str(Path(__file__).resolve()): sha256(Path(__file__))}
    amendment = ready.get("amendment", {})
    amendment_path = Path(amendment.get("path", "")).resolve()
    expected_amendment = run / "compiler_profile_integration/fidelity_protocol_amendment_v1.json"
    if amendment_path != expected_amendment or sha256(amendment_path) != amendment.get("sha256"):
        raise ValueError("Missing or changed declared native-oracle fidelity protocol")
    hashes[str(amendment_path)] = amendment["sha256"]
    oracle_source = ROOT / "scripts/check_pi05_native_sampler_oracle.py"
    oracle_source_sha = sha256(oracle_source)
    if ready.get("native_oracle_source_sha256") != oracle_source_sha:
        raise ValueError("Readiness uses a different independent native oracle")
    for iteration in ITERATIONS:
        row = ready["proofs"][str(iteration)]
        path = Path(row["path"]).resolve()
        proof = read(path)
        if (sha256(path) != row["sha256"] or not proof or proof.get("passed") is not True
                or proof.get("complete") is not True or proof.get("task") != task
                or proof.get("iteration") != iteration or proof.get("cases_expected") != 4
                or len(proof.get("cases", [])) != 4
                or proof.get("amendment") != amendment
                or proof.get("native_oracle_source_sha256") != oracle_source_sha
                or proof.get("autotune_sha256") != profile["profile_sha256"]):
            raise ValueError("Invalid or changed trained-checkpoint fidelity proof")
        hashes[str(path)] = row["sha256"]
        source_hashes = proof["source_input_sha256"]
        for relative in SOURCES:
            source = ROOT / relative
            expected = source_hashes.get(str(source))
            if expected is None or sha256(source) != expected:
                raise ValueError(f"Numerical proof source changed: {source}")
            hashes[str(source)] = expected
        if hashes[str(ROOT / SOURCES[-1])] != ready["auditor_source_sha256"]:
            raise ValueError("Readiness names a different numerical auditor")
        checkpoint = method / "training" / f"iteration_{iteration:06d}"
        for name in ("expert.npz", "rldt_pi05.json"):
            expected = audit["checkpoints"][str(iteration)]["files"][name]["sha256"]
            item = row["checkpoint_files"][name]
            if (Path(item["path"]).resolve() != checkpoint / name or item["sha256"] != expected
                    or source_hashes.get(str(checkpoint / name)) != expected):
                raise ValueError("Numerical proof checkpoint differs from native checkpoint audit")
        identities = []
        for item in proof["cases"]:
            case_path = Path(item["path"]).resolve()
            case = read(case_path)
            if (sha256(case_path) != item["sha256"] or case.get("passed") is not True
                    or case.get("complete") is not True or set(case.get("checks", {})) != EXPECTED_CHECKS
                    or case.get("task") != task or case.get("iteration") != iteration
                    or case.get("amendment") != amendment
                    or case.get("context_matches_independent_oracle") is not True
                    or case.get("noise_matches_independent_oracle") is not True
                    or not all(c.get("exact") is True and c.get("finite") is True
                               and c.get("max_abs_difference") == 0
                               and c.get("different_elements") == 0 for c in case["checks"].values())):
                raise ValueError("Raw numerical fidelity case no longer passes exactly")
            hashes[str(case_path)] = item["sha256"]
            identities.append((case.get("state_id"), case.get("decision_index")))
            arrays = case_path.parent / "arrays.npz"
            if sha256(arrays) != case.get("arrays_sha256"):
                raise ValueError("Numerical fidelity raw arrays changed")
            hashes[str(arrays)] = case["arrays_sha256"]
            oracle = case.get("oracle", {})
            oracle_path = Path(oracle.get("summary_path", "")).resolve()
            oracle_arrays = Path(oracle.get("arrays_path", "")).resolve()
            if (not oracle_path.is_file() or oracle_arrays != oracle_path.parent / "arrays.npz"
                    or oracle.get("cache_hits_verified") is not True
                    or oracle.get("native_oracle_source_sha256") != oracle_source_sha
                    or sha256(oracle_path) != oracle.get("summary_sha256")):
                raise ValueError("Missing or changed independent cached-native oracle")
            native = read(oracle_path)
            if (native.get("complete") is not True or native.get("passed") is not True
                    or native.get("cache_hits_verified") is not True
                    or native.get("native_oracle_source_sha256") != oracle_source_sha
                    or native.get("amendment") != amendment
                    or any(native.get(key) != case.get(key) for key in ("task", "iteration", "state_id", "decision_index"))
                    or native.get("arrays_sha256") != oracle.get("arrays_sha256")
                    or sha256(oracle_arrays) != oracle.get("arrays_sha256")):
                raise ValueError("Independent native oracle identity or arrays differ")
            native_inputs = native.get("source_input_sha256", {})
            for source in (oracle_source, ROOT / "src/rl_vla/rldt_pi05.py",
                           checkpoint / "expert.npz", checkpoint / "rldt_pi05.json"):
                if native_inputs.get(str(source)) != source_hashes.get(str(source)):
                    raise ValueError("Independent native oracle source or checkpoint differs")
            hashes[str(oracle_path)] = oracle["summary_sha256"]
            hashes[str(oracle_arrays)] = oracle["arrays_sha256"]
            executables = native.get("cached_executables", [])
            if (len(executables) < 2
                    or len({row.get("cache_key") for row in executables}) != len(executables)
                    or len({row.get("path") for row in executables}) != len(executables)):
                raise ValueError("Independent native oracle lacks cached executable provenance")
            for executable in executables:
                executable_path = Path(executable["path"]).resolve()
                expected = executable.get("sha256")
                if not executable.get("cache_key") or sha256(executable_path) != expected:
                    raise ValueError("Independent native cached executable changed")
                hashes[str(executable_path)] = expected
            historical = case.get("historical_recorded_replay", {})
            if historical != native.get("historical_recorded_replay"):
                raise ValueError("Historical replay comparison differs from independent oracle")
            if (type(historical.get("exact")) is not bool or historical.get("finite") is not True
                    or type(historical.get("max_abs_difference")) not in (int, float)
                    or not math.isfinite(historical["max_abs_difference"])
                    or historical["max_abs_difference"] < 0
                    or type(historical.get("different_elements")) is not int
                    or historical["different_elements"] < 0
                    or (historical["max_abs_difference"] == 0) != (historical["different_elements"] == 0)
                    or historical["exact"] != (historical["max_abs_difference"] == 0 and historical["different_elements"] == 0)):
                raise ValueError("Historical replay comparison must be preserved separately and consistently")
        if identities != EXPECTED_CASES:
            raise ValueError("Numerical fidelity cases differ from predetermined validation decisions")
    return {"task": task, "profile_manifest_sha256": profile["manifest_sha256"],
            "autotune_sha256": profile["profile_sha256"], "input_sha256": hashes,
            "global_failure_path": str(failure_path),
            "scope": "Three requested experts, four predetermined recorded validation decisions each; numerical sources/checkpoints match."}


def verify_task_fidelity(contract):
    reject_failed_campaign(contract["global_failure_path"], contract["autotune_sha256"])
    for path, expected in contract["input_sha256"].items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Frozen trained-fidelity prerequisite changed: {path}")


BASELINE_SOURCES = (
    "src/rl_vla/rldt_pi05.py", "src/rl_vla/diagnostics/pi05_attention.py",
    "src/rl_vla/diagnostics/layer_knockout.py", "src/rl_vla/diagnostics/compiler_profile_baseline.py",
    "src/rl_vla/diagnostics/compiler_dispatch.py", "scripts/check_pi05_native_baseline_oracle.py",
    "scripts/check_pi05_baseline_fidelity.py",
)


def baseline_fidelity_contract(run, task, profile):
    """Bind four unchanged-baseline decisions to their own independent native oracle."""
    import hashlib

    run = Path(run).resolve()
    failure_path = run / "baseline_sampler_fidelity.json"
    reject_failed_campaign(failure_path, profile["profile_sha256"])
    ready_path = run / "baseline_sampler_fidelity/task_readiness" / f"{task}.json"
    ready = read(ready_path)
    if ready and ready.get("passed") is False:
        raise ValueError("Baseline native fidelity proof failed")
    if not ready or ready.get("complete") is not True:
        return None
    if (ready.get("passed") is not True or ready.get("task") != task
            or ready.get("policy") != "pi05_libero"
            or ready.get("autotune_sha256") != profile["profile_sha256"]
            or ready.get("compiler_profile_manifest_sha256") != profile["manifest_sha256"]):
        raise ValueError("Baseline readiness differs from task, policy, or compiler profile")
    hashes = {str(ready_path): sha256(ready_path), str(Path(__file__).resolve()): sha256(Path(__file__))}
    amendment = ready.get("amendment", {})
    amendment_path = Path(amendment.get("path", "")).resolve()
    if (amendment_path != run / "compiler_profile_integration/baseline_dispatch_protocol_amendment_v1.json"
            or sha256(amendment_path) != amendment.get("sha256")):
        raise ValueError("Baseline proof lacks its unchanged declared separate-process amendment")
    hashes[str(amendment_path)] = amendment["sha256"]
    oracle_source = ROOT / "scripts/check_pi05_native_baseline_oracle.py"
    oracle_source_sha = sha256(oracle_source)
    if ready.get("native_oracle_source_sha256") != oracle_source_sha:
        raise ValueError("Baseline readiness names a different native oracle")
    row = ready.get("proof", {})
    proof_path = Path(row.get("path", "")).resolve()
    if proof_path != run / "baseline_sampler_fidelity" / task / "summary.json":
        raise ValueError("Baseline readiness names an unexpected proof path")
    proof = read(proof_path)
    if (not proof or sha256(proof_path) != row.get("sha256")
            or proof.get("complete") is not True or proof.get("passed") is not True
            or proof.get("task") != task or proof.get("policy") != "pi05_libero"
            or proof.get("cases_expected") != 4 or len(proof.get("cases", [])) != 4
            or proof.get("amendment") != amendment
            or proof.get("native_oracle_source_sha256") != oracle_source_sha
            or proof.get("autotune_sha256") != profile["profile_sha256"]
            or proof.get("compiler_profile_manifest_sha256") != profile["manifest_sha256"]):
        raise ValueError("Baseline proof is incomplete, changed, or misattributed")
    hashes[str(proof_path)] = row["sha256"]
    source_hashes = proof.get("source_input_sha256", {})
    for relative in BASELINE_SOURCES:
        source = ROOT / relative
        expected = source_hashes.get(str(source))
        if expected is None or sha256(source) != expected:
            raise ValueError(f"Baseline numerical proof source changed: {source}")
        hashes[str(source)] = expected
    if hashes[str(ROOT / BASELINE_SOURCES[-1])] != ready.get("auditor_source_sha256"):
        raise ValueError("Baseline readiness names a different numerical auditor")
    checkpoint = ROOT / "checkpoints/openpi/pi05_libero"
    assets = sorted(path for path in (checkpoint / "assets").rglob("*") if path.is_file())
    if not assets:
        raise ValueError("Baseline proof lacks normalization assets")
    baseline_inputs = [checkpoint / "params/_METADATA", checkpoint / "params/manifest.ocdbt", *assets]
    for path in baseline_inputs:
        expected = source_hashes.get(str(path))
        if expected is None or sha256(path) != expected:
            raise ValueError("Baseline model metadata, manifest, or normalization assets changed")
        hashes[str(path)] = expected
    model = proof.get("baseline_model_contract", {})
    expected_model_hashes = {
        "checkpoint_metadata_sha256": hashes[str(baseline_inputs[0])],
        "checkpoint_manifest_sha256": hashes[str(baseline_inputs[1])],
        "normalization_assets_sha256": hashlib.sha256(json.dumps([
            (str(path.relative_to(checkpoint / "assets")), hashes[str(path)]) for path in assets
        ]).encode()).hexdigest(),
    }
    if (model.get("model_config") != "pi05_libero" or Path(model.get("checkpoint", "")).resolve() != checkpoint
            or any(model.get(key) != expected for key, expected in expected_model_hashes.items())):
        raise ValueError("Baseline model contract differs from the actual checkpoint and assets")
    identities = []
    for item in proof["cases"]:
        case_path = Path(item["path"]).resolve()
        case = read(case_path)
        if (not case or sha256(case_path) != item.get("sha256")
                or case.get("passed") is not True or case.get("complete") is not True
                or case.get("task") != task or case.get("policy") != "pi05_libero"
                or case.get("amendment") != amendment or set(case.get("checks", {})) != EXPECTED_CHECKS
                or case.get("context_matches_independent_oracle") is not True
                or case.get("noise_matches_independent_oracle") is not True
                or not all(check.get("exact") is True and check.get("finite") is True
                           and check.get("max_abs_difference") == 0 and check.get("different_elements") == 0
                           for check in case["checks"].values())):
            raise ValueError("Baseline numerical case does not pass all exact checks")
        identities.append((case.get("state_id"), case.get("decision_index")))
        arrays = case_path.parent / "arrays.npz"
        if sha256(arrays) != case.get("arrays_sha256"):
            raise ValueError("Baseline numerical evidence arrays changed")
        hashes[str(case_path)] = item["sha256"]
        hashes[str(arrays)] = case["arrays_sha256"]
        oracle = case.get("oracle", {})
        oracle_path = Path(oracle.get("summary_path", "")).resolve()
        oracle_arrays = Path(oracle.get("arrays_path", "")).resolve()
        if (not oracle_path.is_file() or oracle_arrays != oracle_path.parent / "arrays.npz"
                or oracle.get("cache_hits_verified") is not True
                or oracle.get("native_oracle_source_sha256") != oracle_source_sha
                or sha256(oracle_path) != oracle.get("summary_sha256")):
            raise ValueError("Baseline independent cached-native oracle is missing or changed")
        native = read(oracle_path)
        if (native.get("complete") is not True or native.get("passed") is not True
                or native.get("cache_hits_verified") is not True
                or native.get("native_oracle_source_sha256") != oracle_source_sha
                or native.get("amendment") != amendment or native.get("baseline_model_contract") != model
                or any(native.get(key) != case.get(key) for key in ("task", "policy", "state_id", "decision_index"))
                or native.get("arrays_sha256") != oracle.get("arrays_sha256")
                or sha256(oracle_arrays) != oracle.get("arrays_sha256")):
            raise ValueError("Baseline independent native oracle identity, model, or arrays differ")
        native_inputs = native.get("source_input_sha256", {})
        for source in (oracle_source, ROOT / "src/rl_vla/rldt_pi05.py", *baseline_inputs):
            if native_inputs.get(str(source)) != source_hashes.get(str(source)):
                raise ValueError("Baseline native oracle source or model inputs differ")
        hashes[str(oracle_path)] = oracle["summary_sha256"]
        hashes[str(oracle_arrays)] = oracle["arrays_sha256"]
        executables = native.get("cached_executables", [])
        if (len(executables) < 2
                or len({entry.get("cache_key") for entry in executables}) != len(executables)
                or len({entry.get("path") for entry in executables}) != len(executables)):
            raise ValueError("Baseline native oracle lacks distinct cached executable provenance")
        for executable in executables:
            path = Path(executable["path"]).resolve()
            if not executable.get("cache_key") or sha256(path) != executable.get("sha256"):
                raise ValueError("Baseline native cached executable changed")
            hashes[str(path)] = executable["sha256"]
        historical = case.get("historical_recorded_replay", {})
        if (historical != native.get("historical_recorded_replay")
                or type(historical.get("exact")) is not bool or historical.get("finite") is not True
                or type(historical.get("max_abs_difference")) not in (int, float)
                or not math.isfinite(historical["max_abs_difference"]) or historical["max_abs_difference"] < 0
                or type(historical.get("different_elements")) is not int or historical["different_elements"] < 0
                or (historical["max_abs_difference"] == 0) != (historical["different_elements"] == 0)
                or historical["exact"] != (historical["different_elements"] == 0)):
            raise ValueError("Baseline historical replay must remain separate, finite, and consistent")
    if identities != EXPECTED_CASES:
        raise ValueError("Baseline fidelity cases differ from the four predetermined validation decisions")
    return {"task": task, "policy": "pi05_libero", "profile_manifest_sha256": profile["manifest_sha256"],
            "autotune_sha256": profile["profile_sha256"], "input_sha256": hashes,
            "global_failure_path": str(failure_path),
            "scope": "Unchanged baseline policy, four predetermined validation decisions; exact independent native-oracle comparison."}


def policy_fidelity_contract(run, task, profile):
    """Select evidence by the verified manifest, without pooling compiler profiles."""
    identifier = profile.get("manifest", {}).get("profile_id")
    if identifier == "native_baseline_v1":
        return baseline_fidelity_contract(run, task, profile)
    if identifier == "native_trained_v2":
        return task_fidelity_contract(run, task, profile)
    raise ValueError(f"Unsupported diagnostic compiler profile: {identifier}")
