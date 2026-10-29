"""Small guards for the experimental contract, independent of a GPU/model load."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from model_configs import algorithm_overrides, assert_native_model_settings, model_config
from model_contract import enforce_trainable_scope


def test_both_methods_preserve_original_policy_interface():
    for method, joint, entropy in (("flow_sde", False, 0), ("flow_noise", True, .005)):
        config = model_config("/tmp/pi05_libero", method)
        assert config["num_action_chunks"] == 5
        assert config["openpi"]["action_horizon"] == 10
        assert config["openpi"]["action_dim"] == 32
        assert config["openpi"]["action_env_dim"] == 7
        assert config["openpi"]["num_steps"] == 10
        assert config["openpi"]["discrete_state_input"] is False
        assert config["openpi"]["joint_logprob"] is joint
        assert algorithm_overrides(method)["entropy_bonus"] == entropy


def test_wrong_native_horizon_fails():
    with pytest.raises(ValueError, match="action_horizon"):
        assert_native_model_settings(SimpleNamespace(pi05=True, action_horizon=50))


class Parameter:
    def __init__(self, count, trainable=True):
        self.count, self.requires_grad = count, trainable

    def numel(self):
        return self.count

    def requires_grad_(self, value):
        self.requires_grad = value


def test_unused_language_head_is_excluded_from_expert_count():
    vlm, expert, unused, value = (
        Parameter(2_923_335_408), Parameter(430_098_464),
        Parameter(263_323_648), Parameter(100),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(train_expert_only=True, pi05=True),
        paligemma_with_expert=SimpleNamespace(gemma_expert=SimpleNamespace(
            lm_head=SimpleNamespace(parameters=lambda: [unused]))),
        freeze_vlm=lambda: vlm.requires_grad_(False),
        named_parameters=lambda: [
            ("paligemma_with_expert.paligemma.weight", vlm),
            ("paligemma_with_expert.gemma_expert.model.weight", expert),
            ("paligemma_with_expert.gemma_expert.lm_head.weight", unused),
            ("value_head.weight", value),
        ],
    )
    counts = enforce_trainable_scope(model)
    assert counts["action_expert"] == 430_098_464
    assert counts["total_trainable"] == 430_098_564
    assert unused.requires_grad is False
    assert vlm.requires_grad is False


def test_unknown_method_fails():
    with pytest.raises(ValueError):
        model_config("/tmp/pi05_libero", "residual_ppo")


def test_parity_rejects_finite_but_different_controls(tmp_path):
    import numpy as np
    from check_parity import compare

    arrays = {"first_velocity": np.zeros((1, 10, 32)),
              "normalized_actions": np.zeros((1, 10, 32)),
              "decoded_actions": np.zeros((1, 10, 7)), "state": np.zeros(32)}
    native, torch = tmp_path / "native.npz", tmp_path / "torch.npz"
    np.savez(native, **arrays)
    arrays["decoded_actions"][0, 0, 0] = .06
    np.savez(torch, **arrays)
    with pytest.raises(RuntimeError, match="declared tolerance"):
        compare(SimpleNamespace(native=str(native), torch=str(torch),
                                output=str(tmp_path / "report.json")))


def test_parity_provenance_binds_actual_artifacts_and_rejects_modified_weights(tmp_path):
    import json

    from check_parity import file_sha256, inference_provenance

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"converted-original-weights")
    manifest = checkpoint / "conversion_manifest.json"
    manifest.write_text(json.dumps({"source_checkpoint": "/original/pi05_libero",
                                    "weights_sha256": file_sha256(weights)}))
    fixture, output = tmp_path / "fixture.npz", tmp_path / "torch.npz"
    fixture.write_bytes(b"fixed-observation-and-noise")
    output.write_bytes(b"actual-inference-output")
    report = inference_provenance(checkpoint, fixture, output)
    assert report["fixture"] == {"path": str(fixture), "sha256": file_sha256(fixture)}
    assert report["output_artifact"] == {"path": str(output), "sha256": file_sha256(output)}
    assert report["checkpoint"]["conversion_manifest_sha256"] == file_sha256(manifest)
    assert report["checkpoint"]["model_sha256"] == file_sha256(weights)
    weights.write_bytes(b"different-weights")
    with pytest.raises(ValueError, match="conversion manifest"):
        inference_provenance(checkpoint, fixture, output)
