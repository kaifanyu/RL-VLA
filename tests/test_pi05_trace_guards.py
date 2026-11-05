"""Selection isolation and layer-group drift accounting for the native trace grid."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("openpi")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_vla_trace_grid as grid
from run_vla_trace_grid import parameter_drift, requested_steps
from run_vla_trace_interventions import checked_selection, masked_observation
from trace_pi05_rollout_bank import validate_phase_records


def audited_method(tmp_path):
    method = tmp_path / "methods/rldt/object_0"
    method.mkdir(parents=True)
    (method / "selection.json").write_text(json.dumps({
        "iteration": 25,
        "checkpoint": str(method / "training/iteration_000025"),
        "validation_counts": {"25": 20, "50": 20, "100": 19},
        "locked": "2026-10-31T00:00:00Z",
        "test_results_seen": False,
    }))
    checkpoints = {}
    for iteration in (25, 50, 100):
        folder = method / "training" / f"iteration_{iteration:06d}"
        folder.mkdir(parents=True)
        files = {}
        for name in ("expert.npz", "rldt_pi05.json"):
            payload = f"snapshot {iteration}/{name}".encode()
            (folder / name).write_bytes(payload)
            files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        checkpoints[str(iteration)] = {"files": files}
    return method, {"passed": True, "checkpoints": checkpoints}


def test_grid_audit_requires_passed_complete_audit_and_exact_snapshot_hashes(tmp_path):
    method, audit = audited_method(tmp_path)
    path = method / "checkpoint_audit.json"
    with pytest.raises(FileNotFoundError):
        grid.checkpoint_audit_contract(method)
    path.write_text(json.dumps({**audit, "passed": False}))
    with pytest.raises(ValueError, match="passed full"):
        grid.checkpoint_audit_contract(method)
    path.write_text(json.dumps({**audit, "checkpoints": {"25": audit["checkpoints"]["25"]}}))
    with pytest.raises(ValueError, match="passed full"):
        grid.checkpoint_audit_contract(method)
    path.write_text(json.dumps(audit))
    contract = grid.checkpoint_audit_contract(method)
    assert len(contract["files"]) == 6
    grid.verify_checkpoint_audit(contract, full_hash=True)
    snapshot = method / "training/iteration_000025/expert.npz"
    snapshot.write_bytes(b"X" * snapshot.stat().st_size)
    with pytest.raises(ValueError, match="differs from passed"):
        grid.checkpoint_audit_contract(method)
    with pytest.raises(ValueError, match="snapshot changed"):
        grid.verify_checkpoint_audit(contract)


def test_grid_audit_contract_rejects_mutated_completion_marker(tmp_path):
    method, audit = audited_method(tmp_path)
    path = method / "checkpoint_audit.json"
    path.write_text(json.dumps(audit))
    contract = grid.checkpoint_audit_contract(method)
    path.write_text(json.dumps({**audit, "passed": False}))
    with pytest.raises(ValueError, match="audit changed"):
        grid.verify_checkpoint_audit(contract)


def test_grid_completion_publication_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "prefix_equality.json"
    value = {"complete": True, "passed": True, "observations": [{"state": 150}]}
    original = Path.replace
    observed = []

    def replace(temporary, destination):
        assert not destination.exists()
        assert json.loads(temporary.read_text()) == value
        observed.append(True)
        return original(temporary, destination)

    monkeypatch.setattr(Path, "replace", replace)
    grid.write_summary(path, value)
    assert observed == [True]
    assert json.loads(path.read_text()) == value


def test_grid_waits_for_audit_and_validates_it_before_loading_any_model(tmp_path, monkeypatch):
    method, audit = audited_method(tmp_path)
    for name in ("observation_bank", "observation_bank_aux"):
        folder = tmp_path / name / "object_0/test"
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text('{"complete": true}')

    class Waiting(Exception):
        pass

    class Loading(Exception):
        pass

    def loading(*args, **kwargs):
        raise Loading

    def waiting(*args):
        raise Waiting

    monkeypatch.setattr(grid.Pi05Flow, "from_config", loading)
    monkeypatch.setattr(grid.time, "sleep", waiting)
    profile = tmp_path / "compiler_profile.json"
    contract = tmp_path / "compiler_contract.json"
    profile.write_text('{}')
    contract.write_text('{}')
    monkeypatch.setattr(grid, "load_profile", lambda path: {"manifest": {"profile_id": "native_baseline_v1"}})
    monkeypatch.setattr(grid, "verify_effective_environment", lambda *a, **kw: None)
    monkeypatch.setattr(grid, "policy_fidelity_contract", lambda *a: {"passed": True})

    def invoke(name):
        monkeypatch.setattr(sys, "argv", ["grid", "--run-root", str(tmp_path), "--output",
                                         str(tmp_path / name), "--tasks", "object_0",
                                         "--stage", "baseline",
                                         "--compiler-profile", str(profile), "--compiler-contract", str(contract)])
        grid.main()

    with pytest.raises(Waiting):
        invoke("missing_audit")
    (method / "checkpoint_audit.json").write_text(json.dumps({**audit, "passed": False}))
    with pytest.raises(ValueError, match="passed full"):
        invoke("failed_audit")
    (method / "checkpoint_audit.json").write_text(json.dumps(audit))
    monkeypatch.setattr(grid, "policy_fidelity_contract", lambda *a: None)
    with pytest.raises(Waiting):
        invoke("pending_fidelity")
    monkeypatch.setattr(grid, "policy_fidelity_contract", lambda *a: {"passed": True})
    with pytest.raises(Loading):
        invoke("passed_audit")


def test_ode_steps_preserve_native_schedule_and_reject_invalid_or_duplicate_indices():
    assert requested_steps("0,5,9") == [0, 5, 9]
    for value in ("-1,5", "0,10", "0,0,9", ""):
        with pytest.raises(ValueError):
            requested_steps(value)


def test_grid_timesteps_use_native_float32_index_expression():
    import jax.numpy as jnp

    for step in (0, 5, 9):
        expected = jnp.full((1,), 1.0 - jnp.asarray(step, jnp.float32) / 10, dtype=jnp.float32)
        np.testing.assert_array_equal(grid.native_timestep(step), expected)
    assert not np.array_equal(grid.native_timestep(9), np.asarray([1.0 - 9 / 10], np.float32))


def test_rollout_records_require_all_declared_phases_and_matching_policy_identity():
    plan = {
        "task": "object_0",
        "split": "test",
        "policy": "pi05_libero",
        "checkpoint": None,
        "initial_state_id": 150,
        "episode": "/recorded/episode_1",
        "phases": {
            "initial_decision": 0,
            "closest_requested_target_approach": 5,
            "last_decision": 5,
        },
    }
    summary = {
        "complete": True,
        "sensor_timed_masks_exact_policy_rgb": True,
        "phase_decisions": 2,
        "plan": plan,
    }
    records = [
        {**plan, "decision_index": 0, "phase_names": ["initial_decision"]},
        {
            **plan,
            "decision_index": 5,
            "phase_names": ["closest_requested_target_approach", "last_decision"],
        },
    ]
    assert validate_phase_records(summary, records) == plan
    with pytest.raises(ValueError, match="declared phase"):
        validate_phase_records(summary, [records[0], {**records[1], "decision_index": 6}])
    with pytest.raises(ValueError, match="identity"):
        validate_phase_records(summary, [records[0], {**records[1], "initial_state_id": 151}])
    with pytest.raises(ValueError, match="phase labels"):
        validate_phase_records(
            summary, [records[0], {**records[1], "phase_names": ["last_decision"]}]
        )


def test_selection_requires_validation_only_earliest_tie(tmp_path):
    path = tmp_path / "selection.json"
    selected = {
        "iteration": 25,
        "checkpoint": str(tmp_path / "training/iteration_000025"),
        "validation_counts": {"25": 20, "50": 20, "100": 19},
        "locked": "2026-10-31T00:00:00Z",
        "test_results_seen": False,
    }
    path.write_text(json.dumps(selected))
    assert checked_selection(path)["iteration"] == 25
    selected.update(iteration=50, checkpoint=str(tmp_path / "training/iteration_000050"))
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="earliest-tie"):
        checked_selection(path)


def test_selection_rejects_test_peek_and_foreign_run(tmp_path):
    path = tmp_path / "selection.json"
    selected = {
        "iteration": 100,
        "checkpoint": str(tmp_path / "training/iteration_000100"),
        "validation_counts": {"25": 10, "50": 20, "100": 30},
        "locked": "2026-10-31",
        "test_results_seen": True,
    }
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="validation-only"):
        checked_selection(path)
    selected.update(
        test_results_seen=False, checkpoint=str(tmp_path / "other_task/training/iteration_000100")
    )
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError, match="different task/run"):
        checked_selection(path)


def test_parameter_drift_preserves_scanned_layers_and_output_projection():
    class State:
        def __init__(self, arrays):
            self.arrays = {key: SimpleNamespace(value=value) for key, value in arrays.items()}

        def flat_state(self):
            return self.arrays

    key = ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w")
    projection = ("action_out_proj", "kernel")
    before = np.ones((18, 2, 3), np.float32)
    after = before.copy()
    after[4, 1, 2] += 3
    reference = State({key: before, projection: np.ones((2, 3), np.float32)})
    params = State({key: after, projection: np.full((2, 3), 2, np.float32)})
    groups = parameter_drift(reference, params)
    assert len(groups) == 19
    assert groups["expert_layer_04"]["l2"] == 3
    assert groups["expert_layer_03"]["l2"] == 0
    assert groups["action_out_proj"]["relative_l2"] == pytest.approx(1)


def test_old_target_control_uses_projected_region_not_distractor_instance():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    region = np.zeros((8, 8), bool)
    region[1:3, 1:3] = True
    target = np.zeros_like(region)
    target[5:7, 5:7] = True
    distractor = np.zeros_like(region)
    distractor[4:6, 1:3] = True
    masks = {}
    for camera in ("image", "wrist_image"):
        masks.update(
            {
                f"{camera}__target_1": target,
                f"{camera}__distractor_1": distractor,
                f"{camera}__old_target_prism": region,
                f"{camera}__unowned_background": ~(target | distractor),
            }
        )
    raw = {
        "openpi": {"observation/image": image, "observation/wrist_image": image.copy()},
        "diagnostic_masks": masks,
    }
    changed, severity = masked_observation(
        raw, "old_target:background_fill", target="target_1", distractor="distractor_1", seed=0
    )
    np.testing.assert_array_equal(changed["openpi"]["observation/image"][~region], image[~region])
    assert not np.array_equal(changed["openpi"]["observation/image"][region], image[region])
    assert severity["image"]["mask_pixels"] == 4
    np.testing.assert_array_equal(raw["openpi"]["observation/image"], image)


def test_grid_exports_all_fifteen_mask_conditions_from_wrapped_observation(tmp_path, monkeypatch):
    from rl_vla.diagnostics.interventions import action_delta

    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    target = np.zeros((8, 8), bool); target[1:3, 1:3] = True
    distractor = np.zeros_like(target); distractor[4:6, 4:6] = True
    robot = np.zeros_like(target); robot[7, :] = True
    background = ~(target | distractor | robot)
    source = tmp_path / "observation_bank/object_0/test/state_000150/observation.npz"
    source.parent.mkdir(parents=True)
    np.savez(source, image=image)
    masks = {f"{camera}__{name}": value for camera in ("image", "wrist_image")
             for name, value in (("target_1", target), ("distractor_1", distractor),
                                 ("robot_0", robot), ("unowned_background", background))}
    np.savez(source.parent / "masks.npz", **masks)
    aux = tmp_path / "observation_bank_aux/object_0/test/state_000150"
    aux.mkdir(parents=True)
    np.savez(aux / "old_target_region.npz", **{f"{camera}__old_target_prism": distractor
                                             for camera in ("image", "wrist_image")})
    observation_sha = grid.sha(source)
    (aux / "metadata.json").write_text(json.dumps({"artifact_sha256": grid.sha(aux / "old_target_region.npz"),
                                                  "observation_sha256": observation_sha}))
    raw = {"observation/image": image, "observation/wrist_image": image.copy(), "prompt": "target"}
    record = {"task": "object_0", "split": "test", "protocol": str(tmp_path / "tasks/object_0/protocol.json"),
              "observation_sha256": observation_sha}
    actions = np.zeros((1, 10, 32), np.float32)
    flow = SimpleNamespace(prepare=lambda values: {}, decode=lambda actions, observations: actions[..., :7])
    monkeypatch.setattr(grid, "_load_native", lambda: None)
    monkeypatch.setattr(grid, "masked_observation", masked_observation, raising=False)
    monkeypatch.setattr(grid, "action_delta", action_delta, raising=False)
    monkeypatch.setattr(grid, "sample_with_intervention", lambda *a, **kw: (actions, None), raising=False)
    monkeypatch.setattr(grid, "jax", SimpleNamespace(random=SimpleNamespace(key=lambda seed: seed)), raising=False)
    destination = tmp_path / "outputs"; destination.mkdir()
    grid.open_loop_interventions(flow, None, raw, {}, {"bank_record": record, "noise_sha256": "fixed"},
                                 np.zeros_like(actions), actions, source, "target_1", "distractor_1", destination,
                                 save_inputs=True)
    with np.load(destination / "masked_inputs.npz", allow_pickle=False) as archive:
        assert len(archive.files) == 30
        np.testing.assert_array_equal(archive["none__image"], image)
        np.testing.assert_array_equal(archive["target_background_fill__image"][~target], image[~target])
        assert not np.array_equal(archive["target_background_fill__image"][target], image[target])
    assert len(json.loads((destination / "open_loop_summary.json").read_text())["conditions"]) == 15
    np.testing.assert_array_equal(raw["observation/image"], image)
