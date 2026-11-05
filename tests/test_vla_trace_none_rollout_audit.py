import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / "scripts/audit_vla_trace_none_rollouts.py"
SPEC = importlib.util.spec_from_file_location("none_audit", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_unequal_lengths_do_not_pass_exact_common_prefix():
    result = MODULE.compare_sequences(np.ones((3, 2)), np.ones((4, 2)))
    assert result["common_prefix_exact_rows"] == 3
    assert result["different_common_rows"] == 0
    assert result["full_exact"] is False


def test_mismatch_position_and_finite_guard():
    a = np.ones((3, 2), dtype=np.float32)
    b = a.copy()
    b[1, 0] += 0.125
    result = MODULE.compare_sequences(a, b)
    assert result["common_prefix_exact_rows"] == 1
    assert result["different_common_rows"] == 1
    assert result["max_abs_difference_common"] == 0.125
    assert result["max_abs_difference_first_row"] == 0
    a[0, 0] = b[0, 0] = np.inf
    assert MODULE.compare_sequences(a, b)["finite"] is False


def test_selected_checkpoint_report_resolution(tmp_path):
    folder = tmp_path / "methods/rldt/object_0"
    folder.mkdir(parents=True)
    (folder / "selection.json").write_text(json.dumps({"iteration": 50}))
    assert MODULE.native_report(tmp_path, "object_0", "rldt_50").name == "selected_test.json"
    assert MODULE.native_report(tmp_path, "object_0", "rldt_25").name == "checkpoint_000025_test.json"


def test_episode_requires_all_primitive_states(tmp_path):
    causal = tmp_path / "causal"
    native = tmp_path / "native"
    causal.mkdir()
    native.mkdir()
    np.savez(causal / "episode_000_actions.npz", normalized=np.zeros((1, 10, 32), np.float32))
    telemetry = [{"primitive_steps": 1, "action": [0.] * 7, "simulator_state": [0.] * 3}]
    (causal / "episode_000_telemetry.json").write_text(json.dumps(telemetry))
    metadata = {"chunk": 1, "decision": {"decision": 0, "sampling_key": [1, 2]},
                "first_primitive_step": 0, "executed_steps": 1,
                "step_infos": [{"object_telemetry": {"simulator_state": [0.] * 3}}]}
    np.savez(native / "chunk_000001.npz", normalized_actions=np.zeros((10, 32), np.float32),
             executed_actions=np.zeros((1, 7), np.float32), metadata_json=json.dumps(metadata))
    reset = {"object_telemetry": {"simulator_state": [0.] * 3}}
    c = {"episode": 0, "primitive_steps": 1, "initial_state_id": 150, "seed": 100000,
         "reset": reset, "success": False}
    n = {"trajectory": str(native), "primitive_steps": 1, "reset": reset, "success": False}
    assert MODULE.compare_episode(c, n, causal)["passed"] is True
    n["primitive_steps"] = 2
    with pytest.raises(ValueError, match="full episode"):
        MODULE.compare_episode(c, n, causal)


def test_element_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="element shapes"):
        MODULE.compare_sequences(np.zeros((2, 7)), np.zeros((2, 6)))
