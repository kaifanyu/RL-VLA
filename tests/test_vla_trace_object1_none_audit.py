import importlib.util
import json
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[1] / "scripts/audit_vla_trace_object1_none_rollouts.py"
SPEC = importlib.util.spec_from_file_location("object1_none", SOURCE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_full_shared_telemetry_metadata_difference_fails_even_with_exact_actions_and_states(tmp_path):
    causal, native = tmp_path / "causal", tmp_path / "native"
    causal.mkdir()
    native.mkdir()
    telemetry = {"simulator_state": [0.] * 3, "objects": {"target": {"both_fingerpads_contact": False}},
                 "gripper_xyz": [0.] * 3, "grasp_candidate_ids": [], "definition": "synthetic test"}
    reset = {"object_telemetry": telemetry}
    c = {"episode": 0, "primitive_steps": 1, "initial_state_id": 150, "seed": 100000,
         "reset": reset, "success": False}
    n = {"trajectory": str(native), "primitive_steps": 1, "reset": reset, "success": False}
    np.savez(causal / "episode_000_actions.npz", normalized=np.zeros((1, 10, 32), np.float32))
    row = {**telemetry, "primitive_steps": 1, "action": [0.] * 7}
    causal_metadata = causal / "episode_000_telemetry.json"
    causal_metadata.write_text(json.dumps([row]))
    metadata = {"chunk": 1, "decision": {"decision": 0, "sampling_key": [1, 2]},
                "first_primitive_step": 0, "executed_steps": 1, "step_infos": [{"object_telemetry": telemetry}]}
    np.savez(native / "chunk_000001.npz", normalized_actions=np.zeros((10, 32), np.float32),
             executed_actions=np.zeros((1, 7), np.float32), metadata_json=json.dumps(metadata))
    assert AUDIT.compare_episode(c, n, causal)["passed"] is True
    changed = json.loads(causal_metadata.read_text())
    changed[0]["objects"]["target"]["both_fingerpads_contact"] = True
    causal_metadata.write_text(json.dumps(changed))
    result = AUDIT.compare_episode(c, n, causal)
    assert result["normalized_actions"]["full_exact"] is True
    assert result["simulator_states"]["full_exact"] is True
    assert result["step_metadata"]["full_exact"] is False
    assert result["step_metadata"]["first_differing_primitive_step"] == 1
    assert result["passed"] is False


def test_predecessor_source_and_scoped_pending_matrix(tmp_path):
    result = AUDIT.refresh(tmp_path, tmp_path / "evidence")
    assert result["complete"] is False
    assert result["passed"] is None
    assert result["expected_cells"] == 4
    assert len(result["pending"]) == 4
    assert {row["task"] for row in result["pending"]} == {"object_1"}
