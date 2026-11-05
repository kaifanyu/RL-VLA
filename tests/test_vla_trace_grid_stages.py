"""Cross-process trace exchange cannot silently recalculate or certify partial work."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rl_vla.diagnostics.grid_stages import (
    STAGES, POLICIES, aggregate_task, atomic_json, load_context,
    publish_stage, read_stage, save_context, sha, verify_stage,
)


def test_context_roundtrip_preserves_tree_shapes_values_and_dtype(tmp_path):
    context = {"kv_cache": (np.arange(24, dtype=np.float32).reshape(2, 3, 4),),
               "mask": np.asarray([[True, False]]), "nested": [np.asarray([1, 2], np.int32), None]}
    save_context(tmp_path, context)
    restored = load_context(tmp_path)
    assert isinstance(restored["kv_cache"], tuple) and isinstance(restored["nested"], list)
    for original, loaded in ((context["kv_cache"][0], restored["kv_cache"][0]),
                             (context["mask"], restored["mask"]),
                             (context["nested"][0], restored["nested"][0])):
        assert original.dtype == loaded.dtype
        np.testing.assert_array_equal(original, loaded)
    assert restored["nested"][1] is None
    (tmp_path / "context.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="context hash"):
        load_context(tmp_path)


def test_bfloat16_context_exchange_is_exact_and_portable(tmp_path):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    values = np.asarray([-0.125, 0, 0.0078125, 1.015625, 32.25], dtype=ml_dtypes.bfloat16)
    save_context(tmp_path, {"features": values})
    with np.load(tmp_path / "context.npz", allow_pickle=False) as archive:
        assert archive["leaf_000"].dtype == np.float32
    restored = load_context(tmp_path, lambda array, dtype: np.asarray(array, dtype=ml_dtypes.bfloat16 if dtype == "bfloat16" else dtype))
    assert restored["features"].dtype == values.dtype
    assert restored["features"].tobytes() == values.tobytes()


def test_stage_artifacts_fail_closed_on_same_size_mutation_and_refuse_overwrite(tmp_path):
    path = tmp_path / "shared.npz"
    path.write_bytes(b"original")
    contract = publish_stage(tmp_path, "baseline", plan={"task": "object_0"}, owned_paths=[path])
    verify_stage(contract, full_hash=True)
    with pytest.raises(FileExistsError):
        publish_stage(tmp_path, "baseline", plan={}, owned_paths=[path])
    stamp = path.stat()
    path.write_bytes(b"mutated!")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match="hash changed"):
        verify_stage(contract, full_hash=True)


def populate_task(output, *, bad_equality=False):
    output.mkdir()
    method = output.parent / "method"
    method.mkdir()
    (method / "selection.json").write_text('{"locked": true}')
    (method / "checkpoint_audit.json").write_text('{"passed": true}')
    checkpoint = method / "expert.npz"
    checkpoint.write_bytes(b"checkpoint")
    atomic_json(output / "case_plan.json", {"checkpoint_audit_contract": {
        "audit_path": str(method / "checkpoint_audit.json"),
        "files": {str(checkpoint): {"sha256": sha(checkpoint)}}}})
    common = {"task": "object_0", "split": "test", "state_ids": [150], "steps": [0, 5, 9],
              "selection_sha256": sha(method / "selection.json"),
              "checkpoint_audit_sha256": sha(method / "checkpoint_audit.json"), "manifest_sha256": "frozen"}
    atomic_json(output / "shared/state_000150/metadata.json", {
        "bank_record": {"observation_sha256": "observation"}, "prefix_features_sha256": "features"})
    for policy in POLICIES:
        folder = output / policy / "state_000150"
        folder.mkdir(parents=True)
        atomic_json(folder.parent / "prefix_equality.json", {"complete": True, "passed": True,
                    "frozen_backbone_sha256": "frozen-backbone", "observations": [{"passed": True,
                    "prefix_max_abs_difference": 1 if bad_equality and policy == "rldt_000025" else 0,
                    "observation_sha256": "observation",
                    "baseline_feature_sha256": "features", "checkpoint_feature_sha256": "features"}]})
        names = ["actions.npz", "own_trajectory_attention.npz", "metadata.json", "open_loop_actions.npz",
                 "layer_knockout_summary.json", "layer_knockout_arrays.npz"]
        for step in common["steps"]:
            names += [f"{stem}_step{step:02d}_{condition}.npz" for stem in ("expert", "views") for condition in ("none", "no_image", "no_text")]
            if policy == "pi05_libero":
                names += [f"projection_ablation_step{step:02d}.{suffix}" for suffix in ("npz", "json")]
        for name in names:
            if name.startswith("projection") and name.endswith(".json"):
                atomic_json(folder / name, {"complete": True, "native_reconstruction_max_abs_difference": 0})
            else:
                (folder / name).write_bytes(b"test fixture")
        conditions = ["none", "no_image", "no_text"] + [f"{region}:{mode}"
                      for region in ("target", "distractor", "random", "robot", "background", "old_target")
                      for mode in ("background_fill", "mosaic")]
        atomic_json(folder / "open_loop_summary.json", {"complete": True, "conditions": [{"condition": c} for c in conditions]})
    raw_base = output / "pi05_base/state_000150"
    raw_base.mkdir(parents=True)
    for name in ("prefix.npz", "counterfactual_prefix.npz", "prefix_metadata.json", "counterfactual_prefix_metadata.json"):
        (raw_base / name).write_bytes(b"test fixture")
    baseline = publish_stage(output, "baseline", plan={**common, "profile_id": "native_baseline_v1"},
                             owned_paths=[output / "case_plan.json", output / "pi05_libero", output / "pi05_base", output / "shared"])
    control_rows = []
    for step in common["steps"]:
        path = output / "compiler_controls/state_000150" / f"step{step:02d}.json"
        atomic_json(path, {"complete": True})
        control_rows.append({"state_id": 150, "step": step, "path": str(path), "sha256": sha(path)})
    atomic_json(output / "compiler_controls/summary.json", {"complete": True, "states": [150],
                "steps": common["steps"], "cases": control_rows})
    trained = publish_stage(output, "trained", plan={**common, "profile_id": "native_trained_v2"},
                            owned_paths=[output / policy for policy in POLICIES[1:]] + [output / "compiler_controls"], dependencies=[baseline["path"]])
    return common, baseline, trained


def finish_projection(output, common, baseline, trained):
    files = []
    for policy in POLICIES[1:]:
        for step in common["steps"]:
            for suffix in ("npz", "json"):
                path = output / policy / "state_000150" / f"projection_ablation_step{step:02d}.{suffix}"
                if suffix == "json":
                    atomic_json(path, {"complete": True, "native_reconstruction_max_abs_difference": 0})
                else:
                    path.write_bytes(b"test fixture")
                files.append(path)
    return publish_stage(output, "baseline_projection", plan={**common, "profile_id": "native_baseline_v1"},
                         owned_paths=files, dependencies=[baseline["path"], trained["path"]])


def test_aggregate_requires_three_complete_stages_and_never_writes_global_marker(tmp_path):
    output = tmp_path / "object_0"
    common, baseline, trained = populate_task(output)
    with pytest.raises(FileNotFoundError):
        aggregate_task(output)
    assert not (output / "summary.json").exists()
    finish_projection(output, common, baseline, trained)
    summary = aggregate_task(output)
    assert summary["complete"] and summary["states"] == 1
    assert [stage["stage"] for stage in summary["stages"]] == list(STAGES)
    assert not (tmp_path / "summary.json").exists()
    # Appending the postpass files does not invalidate the immutable trained artifacts.
    verify_stage(trained, full_hash=True)
    (tmp_path / "method/selection.json").write_text('{"locked": false}')
    with pytest.raises(ValueError, match="selection changed"):
        aggregate_task(output)


def test_aggregate_rechecks_each_exact_prefix_audit_instead_of_trusting_top_level_pass(tmp_path):
    output = tmp_path / "object_0"
    common, baseline, trained = populate_task(output, bad_equality=True)
    finish_projection(output, common, baseline, trained)
    with pytest.raises(ValueError, match="Per-observation frozen-prefix"):
        aggregate_task(output)
    assert not (output / "summary.json").exists()


def test_grid_cli_help_and_cpu_aggregation_do_not_import_jax(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    script = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('grid', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
assert 'jax' not in sys.modules
sys.argv = ['grid', '--run-root', sys.argv[2], '--output', sys.argv[2], '--tasks', 'object_0', '--stage', 'aggregate']
try:
    module.main()
except FileNotFoundError:
    pass
assert 'jax' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(repository / "scripts/run_vla_trace_grid.py"), str(tmp_path)],
                   check=True, cwd=repository)


def test_projection_postpass_preserves_trained_artifacts_and_quantifies_compiler_effect(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[1] / "scripts/run_vla_trace_grid.py"
    spec = importlib.util.spec_from_file_location("test_grid_postpass", source)
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)
    output = tmp_path / "traces/object_0"
    output.mkdir(parents=True)
    method = tmp_path / "methods/rldt/object_0"
    method.mkdir(parents=True)
    atomic_json(method / "selection.json", {"locked": True})
    atomic_json(method / "checkpoint_audit.json", {"passed": True})
    common = {"task": "object_0", "split": "test", "state_ids": [150], "steps": [0, 5, 9],
              "selection_sha256": sha(method / "selection.json"),
              "checkpoint_audit_sha256": sha(method / "checkpoint_audit.json"), "manifest_sha256": "immutable"}
    base_file = output / "base_reference.npz"
    base_file.write_bytes(b"baseline")
    baseline = publish_stage(output, "baseline", plan={**common, "profile_id": "native_baseline_v1"}, owned_paths=[base_file])
    inputs = []
    for policy in POLICIES[1:]:
        destination = output / policy / "state_000150"
        destination.mkdir(parents=True)
        for step in common["steps"]:
            path = destination / f"projection_inputs_step{step:02d}.npz"
            np.savez(path, hidden_base=np.asarray([1.]), hidden_checkpoint=np.asarray([3.]),
                     baseH_baseW=np.asarray([2.]), baseH_checkpointW=np.asarray([4.]),
                     checkpointH_checkpointW=np.asarray([12.]), checkpoint_native_velocity=np.asarray([12.]),
                     baseH_baseW_under_trained_profile=np.asarray([2.01]),
                     checkpointH_baseW_under_trained_profile=np.asarray([6.02]))
            inputs.append(path)
    trained = publish_stage(output, "trained", plan={**common, "profile_id": "native_trained_v2"},
                            owned_paths=inputs, dependencies=[baseline["path"]])
    monkeypatch.setattr(grid, "_load_native", lambda: None)
    monkeypatch.setattr(grid, "verify_checkpoint_audit", lambda *a, **kw: None)
    monkeypatch.setattr(grid, "verify_effective_environment", lambda *a, **kw: None)
    monkeypatch.setattr(grid, "verify_task_fidelity", lambda *a, **kw: None)
    monkeypatch.setattr(grid, "project_suffix", lambda flow, params, hidden: 2 * hidden, raising=False)
    grid.baseline_projection_task(SimpleNamespace(params="baseline only"), tmp_path, "object_0", output,
                                  checkpoint_contract={"audit_sha256": common["checkpoint_audit_sha256"]},
                                  compiler_profile={}, compiler_contract={"profile_id": "native_baseline_v1"},
                                  fidelity_contract={}, source_guard=lambda: None)
    verify_stage(trained, full_hash=True)
    verify_stage(read_stage(output, "baseline_projection"), full_hash=True)
    destination = output / POLICIES[1] / "state_000150"
    with np.load(destination / "projection_ablation_step00.npz", allow_pickle=False) as archive:
        assert archive["checkpointH_baseW"][0] == 6.
        assert archive["checkpointH_checkpointW"][0] == 12.
        assert archive["checkpointH_baseW_under_trained_profile"][0] == 6.02
    metadata = json.loads((destination / "projection_ablation_step00.json").read_text())
    assert metadata["native_reconstruction_max_abs_difference"] == 0.
    assert metadata["profile_ownership"]["checkpointH_baseW"] == "native_baseline_v1"
    assert metadata["profile_ownership"]["checkpointH_checkpointW"] == "native_trained_v2"
    assert metadata["same_hidden_baseline_weights_compiler_effect"]["baseH_baseW_under_trained_profile"]["max_abs"] == pytest.approx(.01)
    assert metadata["same_hidden_baseline_weights_compiler_effect"]["checkpointH_baseW_under_trained_profile"]["l2"] == pytest.approx(.02)


def test_compiler_only_control_uses_unchanged_weights_exact_baseline_latent_and_no_sampler(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    source = Path(__file__).resolve().parents[1] / "scripts/run_vla_trace_grid.py"
    spec = importlib.util.spec_from_file_location("test_grid_compiler_control", source)
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)
    shared = tmp_path / "shared/state_000150"
    shared.mkdir(parents=True)
    baseline = tmp_path / "pi05_libero/state_000150"
    baseline.mkdir(parents=True)
    latent = np.asarray([[[7.]]], np.float32)
    np.savez(baseline / "expert_step00_none.npz", attention=np.asarray([1.]), velocity=np.asarray([2.]),
             x_t=latent[0], t=np.asarray([1.], np.float32))
    np.savez(shared / "projection_reference_step00.npz", hidden=np.asarray([2.]), velocity=np.asarray([2.]))
    np.savez(shared / "context.npz", value=np.asarray([4.]))
    np.savez(shared / "common_latents.npz", baseline_path=latent)
    flow = SimpleNamespace(params=object())
    calls = []

    def capture(flow_arg, params, context, x, t, metadata, return_hidden):
        assert flow_arg is flow and params is flow.params
        assert return_hidden and context == {"immutable": True}
        np.testing.assert_array_equal(x, latent[0])
        np.testing.assert_array_equal(t, [1.])
        calls.append(True)
        return np.asarray([3.]), np.asarray([.75]), np.asarray([3.])

    monkeypatch.setattr(grid, "native_timestep", lambda step: np.asarray([1.], np.float32))
    monkeypatch.setattr(grid, "capture_velocity", capture, raising=False)
    monkeypatch.setattr(grid, "native_velocity", lambda *a: np.asarray([3.]), raising=False)
    monkeypatch.setattr(grid, "project_suffix", lambda flow, params, hidden: hidden, raising=False)
    record = {"initial_state_id": 150, "task": "object_0", "observation_sha256": "immutable"}
    grid.compiler_precision_controls(flow, tmp_path, [(record, {}, {"immutable": True}, {}, None, latent, shared)], [0],
                                     baseline_stage={"summary": {"compiler_profile": {"profile_id": "native_baseline_v1"}}},
                                     compiler_contract={"profile_id": "native_trained_v2"})
    assert calls == [True]
    value = json.loads((tmp_path / "compiler_controls/state_000150/step00.json").read_text())
    assert value["differences"]["attention"]["max_abs"] == .25
    assert value["differences"]["velocity"]["l2"] == 1.
    assert value["same_profile_native_velocity_max_abs_difference"] == 0.
    assert value["baseline_profile"]["profile_id"] == "native_baseline_v1"
    assert value["control_profile"]["profile_id"] == "native_trained_v2"
