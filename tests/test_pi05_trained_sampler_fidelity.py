"""Manifest guards for trained sampler evidence; numerical proof uses real checkpoints."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("trained_fidelity", Path(__file__).parents[1] / "scripts/check_pi05_trained_sampler_fidelity.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture_summary(tmp_path):
    tuning = tmp_path / "compiler_profiles/native_trained_v2/autotune.textproto"
    tuning.parent.mkdir(parents=True)
    tuning.write_text("version: 3\n")
    (tuning.parent / "manifest.json").write_text("{}\n")
    amendment = tmp_path / "amendment.json"
    amendment.write_text("{}\n")
    args = SimpleNamespace(run=tmp_path, output=tmp_path / "proofs", autotune_sha256="a"*64, autotune=tuning, amendment=amendment)
    base = args.output / "object_0/iteration_000025"
    cases = []
    for state, decision in [(100, 0), (100, 1), (101, 0), (101, 1)]:
        path = base / f"state_{state}/decision_{decision}/summary.json"
        path.parent.mkdir(parents=True)
        (path.parent / "arrays.npz").write_bytes(b"synthetic serialized arrays for integrity test")
        native = args.output / "native_oracle/object_0/iteration_000025" / f"state_{state}/decision_{decision}/summary.json"
        native.parent.mkdir(parents=True)
        (native.parent / "arrays.npz").write_bytes(b"independent synthetic native output integrity fixture")
        historical = {"exact": True, "finite": True, "max_abs_difference": 0, "different_elements": 0}
        module.write(native, {"complete": True, "passed": True, "cache_hits_verified": True,
            "task": "object_0", "iteration": 25, "state_id": state, "decision_index": decision,
            "historical_recorded_replay": historical})
        module.write(path, {"complete": True, "passed": True, "task": "object_0", "iteration": 25,
            "state_id": state, "decision_index": decision,
            "checks": {name: {"exact": True, "finite": True, "max_abs_difference": 0} for name in module.EXPECTED_CHECKS},
            "arrays_sha256": module.sha(path.parent / "arrays.npz"),
            "historical_recorded_replay": historical,
            "oracle": {"summary_path": str(native), "summary_sha256": module.sha(native),
                       "arrays_path": str(native.parent / "arrays.npz"), "arrays_sha256": module.sha(native.parent / "arrays.npz")}})
        cases.append({"path": str(path), "sha256": module.sha(path), "passed": True})
    module.write(base / "summary.json", {"complete": True, "passed": True, "cases": cases,
                 "amendment": {"path": str(amendment), "sha256": module.sha(amendment)},
                 "source_input_sha256": {str(tmp_path / "methods/rldt/object_0/training/iteration_000025" / name): "c"*64
                                         for name in ("expert.npz", "rldt_pi05.json")},
                 "autotune_sha256": args.autotune_sha256})
    return args, base, cases


def test_partial_evidence_never_publishes_global_pass(tmp_path):
    args, _, _ = fixture_summary(tmp_path)
    result = module.aggregate(args)
    assert result["completed_checkpoints"] == 1
    assert result["complete"] is False and result["passed"] is None


def test_mutated_raw_arrays_are_rejected(tmp_path):
    args, _, cases = fixture_summary(tmp_path)
    (Path(cases[0]["path"]).parent / "arrays.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="arrays changed"):
        module.aggregate(args)


def test_completed_task_readiness_is_byte_stable(tmp_path, monkeypatch):
    args, _, _ = fixture_summary(tmp_path)
    monkeypatch.setattr(module, "ITERATIONS", (25,))
    module.aggregate(args)
    path = args.output / "task_readiness/object_0.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    monkeypatch.setattr(module, "utc", lambda: "different later timestamp")
    module.aggregate(args)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("cache_hit", [True, False])
def test_historical_delta_is_separate_but_native_cache_hit_is_required(tmp_path, cache_hit):
    args, base, cases = fixture_summary(tmp_path)
    path = Path(cases[0]["path"])
    case = module.read(path)
    native_path = Path(case["oracle"]["summary_path"])
    native = module.read(native_path)
    historical = {"exact": False, "finite": True, "max_abs_difference": .0007, "different_elements": 320}
    native.update(cache_hits_verified=cache_hit, historical_recorded_replay=historical)
    module.write(native_path, native)
    case["oracle"]["summary_sha256"] = module.sha(native_path)
    case["historical_recorded_replay"] = historical
    module.write(path, case)
    summary = module.read(base / "summary.json")
    summary["cases"][0]["sha256"] = module.sha(path)
    module.write(base / "summary.json", summary)
    if cache_hit:
        assert module.aggregate(args)["completed_checkpoints"] == 1
    else:
        with pytest.raises(ValueError, match="oracle is not certified"):
            module.aggregate(args)


@pytest.mark.parametrize("mutation", ["wrong_state", "missing_check", "nonexact"])
def test_resealed_invalid_evidence_is_rejected(tmp_path, mutation):
    args, base, cases = fixture_summary(tmp_path)
    path = Path(cases[0]["path"])
    case = module.read(path)
    if mutation == "wrong_state":
        case["state_id"] = 102
    elif mutation == "missing_check":
        case["checks"]["invented_check"] = case["checks"].pop("captured_path")
    else:
        case["checks"]["captured_actions"]["max_abs_difference"] = .001
    module.write(path, case)
    summary = module.read(base / "summary.json")
    summary["cases"][0]["sha256"] = module.sha(path)
    module.write(base / "summary.json", summary)
    with pytest.raises(ValueError):
        module.aggregate(args)


def test_environment_rejects_shared_cache_and_extra_precision_flag(tmp_path, monkeypatch):
    tuning = tmp_path / "autotune.textproto"
    tuning.write_text("version: 3\n")
    args = SimpleNamespace(run=tmp_path, autotune=tuning, autotune_sha256=module.sha(tuning), output=tmp_path / "proof")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "")
    monkeypatch.setenv("XLA_FLAGS", f"--xla_gpu_load_autotune_results_from={tuning}")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "shared_cache"))
    with pytest.raises(ValueError, match="isolated compilation cache"):
        module.check_inputs(args)
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(args.run / "compiler_caches" / tuning.parent.name / "fidelity_campaign"))
    module.check_inputs(args)
    monkeypatch.setenv("XLA_FLAGS", f"--xla_gpu_load_autotune_results_from={tuning} --xla_gpu_enable_triton_gemm=false")
    with pytest.raises(ValueError, match="Only the declared"):
        module.check_inputs(args)
