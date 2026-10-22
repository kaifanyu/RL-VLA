"""Check pi0.5 parameter boundaries and JAX backend contracts without weights."""

import json
from pathlib import Path

import numpy as np
import pytest

from rl_vla.rldt_pi05 import Pi05Flow, is_expert_parameter


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (("PaliGemma", "img", "Transformer", "encoderblock", "kernel"), False),
        (("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"), False),
        (("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"), True),
        (("PaliGemma", "llm", "layers", "pre_attention_norm_1", "Dense_0", "kernel"), True),
        (("PaliGemma", "llm", "final_norm_1", "Dense_0", "kernel"), True),
        (("PaliGemma", "llm", "layers", "mlp_1", "linear"), True),
        (("action_out_proj", "kernel"), True),
        (("time_mlp_in", "kernel"), True),
    ],
)
def test_expert_filter_preserves_vlm_and_includes_shared_attention_parent(path, expected):
    assert is_expert_parameter(path) is expected
    if path[0] == "PaliGemma":
        assert not is_expert_parameter(path, projection_only=True)


def test_detached_rollout_and_particle_batch_order():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    flow = Pi05Flow()
    flow.horizon, flow.action_dim = 2, 3
    flow.velocity = lambda params, context, x, t, *, frozen=None: jnp.ones_like(x) * params
    context = {
        "state": jnp.ones((2, 3)),
        "features": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "kv_cache": (jnp.zeros((2, 4, 5, 1, 3)), jnp.ones((2, 4, 5, 1, 3))),
    }
    repeated = flow.repeat_context(context, 3)
    np.testing.assert_array_equal(repeated["features"], np.repeat(context["features"], 3, 0))
    assert repeated["kv_cache"][0].shape == (6, 4, 5, 1, 3)
    key = jax.random.key(4)
    final, path = jax.jit(lambda p: flow.sample(p, context, key, 4))(jnp.array(2.0))
    assert path.shape == (4, 2, 2, 3)
    np.testing.assert_allclose(final, path[0] - 2.0, atol=1e-6)
    np.testing.assert_allclose(path[1], path[0] - 0.5, atol=1e-6)
    derivative = jax.grad(lambda p: jnp.sum(flow.sample(p, context, key, 4)[0]))(jnp.array(2.0))
    assert float(derivative) == 0.0


def test_prepare_passes_checkpoint_weights_as_dynamic_jit_arguments():
    """Regression: captured multi-GB weights cause enormous XLA compilation."""
    pytest.importorskip("openpi")
    jnp = pytest.importorskip("jax.numpy")
    flow = Pi05Flow()
    flow.params = {"expert": jnp.ones(2)}
    flow.frozen = {"vlm": jnp.ones(3)}
    flow._transform_observation = lambda _: {
        "image": {"base_0_rgb": np.zeros((2, 2, 3), np.uint8)},
        "image_mask": {"base_0_rgb": np.asarray(True)},
        "state": np.zeros(4, np.float32),
    }
    calls = []

    def encoder(params, frozen, observation):
        calls.append((params, frozen))
        return {"features": observation.state}

    flow._encode_jit = encoder
    result = flow.prepare([{}])
    assert calls[0][0] is flow.params
    assert calls[0][1] is flow.frozen
    assert result["features"].shape == (1, 4)


def test_expert_checkpoint_roundtrip_and_mismatched_base_rejection(tmp_path):
    pytest.importorskip("openpi")
    jnp = pytest.importorskip("jax.numpy")
    nnx = pytest.importorskip("flax.nnx")
    flow = Pi05Flow()
    flow.params = nnx.State(
        {"action_out_proj": {"kernel": nnx.VariableState(nnx.Param, jnp.ones((2, 3)))}}
    )
    flow.metadata = {
        "model_config": "pi05_libero",
        "projection_only": False,
        "checkpoint_metadata_sha256": "example",
        "checkpoint_manifest_sha256": "base-weight-content",
        "normalization_assets_sha256": "base-normalization",
    }
    path = flow.save(tmp_path / "checkpoint", flow.params)
    restored = flow.load_params(path)
    np.testing.assert_array_equal(restored["action_out_proj"]["kernel"].value, np.ones((2, 3)))
    with pytest.raises(FileExistsError):
        flow.save(tmp_path / "checkpoint", flow.params)
    flow.metadata["checkpoint_metadata_sha256"] = "different_base"
    with pytest.raises(ValueError, match="checkpoint_metadata_sha256"):
        flow.load_params(Path(path).parent)


@pytest.mark.parametrize("manifest_kind", ["missing", "generic", "different_vlm", "different_norm"])
def test_native_expert_load_rejects_unknown_base_before_restoring_weights(
    tmp_path, monkeypatch, manifest_kind
):
    model = pytest.importorskip("openpi.models.model")
    flow = Pi05Flow()
    flow.metadata = {
        "model_config": "pi05_libero",
        "projection_only": False,
        "checkpoint_metadata_sha256": "schema",
        "checkpoint_manifest_sha256": "base-weight-content",
        "normalization_assets_sha256": "base-normalization",
    }
    (tmp_path / "params").mkdir()
    saved = {**flow.metadata, "format": "openpi-native-rldt-v1"}
    if manifest_kind == "generic":
        saved["format"] = "unrelated-checkpoint"
    elif manifest_kind == "different_vlm":
        saved["checkpoint_manifest_sha256"] = "another-vlm"
    elif manifest_kind == "different_norm":
        saved["normalization_assets_sha256"] = "another-normalization"
    if manifest_kind != "missing":
        (tmp_path / "rldt_pi05.json").write_text(json.dumps(saved))

    def forbidden_restore(*args, **kwargs):
        pytest.fail("Weights were read before checkpoint provenance was validated")

    monkeypatch.setattr(model, "restore_params", forbidden_restore)
    with pytest.raises(ValueError, match="provenance|format|does not match"):
        flow.load_params(tmp_path)


def test_compact_expert_load_requires_manifest_before_reading_archive(tmp_path):
    pytest.importorskip("openpi")
    flow = Pi05Flow()
    # No archive exists: provenance must fail first, not an np.load file error.
    with pytest.raises(ValueError, match="requires RLDT provenance"):
        flow.load_params(tmp_path / "expert.npz")
