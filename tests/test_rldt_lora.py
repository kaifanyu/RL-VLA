"""Check LoRA identity, gradient isolation, scaling, and checkpoint contracts."""

import numpy as np
import pytest

from rl_vla.rldt_lora import (
    initialize_adapters,
    is_lora_target,
    lora_settings,
    materialize_adapters,
)


@pytest.mark.parametrize("settings", [
    {"lora_rank": -1}, {"lora_rank": True}, {"lora_rank": 1.5},
    {"lora_rank": 128, "projection_only": True},
    {"lora_rank": 128, "lora_alpha": float("nan")},
    {"lora_rank": 128, "lora_alpha": 0}, {"lora_seed": -1},
])
def test_invalid_adapter_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        lora_settings(settings)


@pytest.mark.parametrize("path, expected", [
    (("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"), True),
    (("PaliGemma", "llm", "layers", "mlp_1", "gating_einsum"), True),
    (("PaliGemma", "llm", "layers", "mlp_1", "linear"), True),
    (("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"), False),
    (("PaliGemma", "img", "kernel"), False),
    (("action_out_proj", "kernel"), False),
    (("PaliGemma", "llm", "layers", "pre_attention_norm_1", "Dense_0", "kernel"), False),
])
def test_only_action_expert_attention_and_mlp_are_adapted(path, expected):
    assert is_lora_target(path) is expected


def fixture_state():
    jnp = pytest.importorskip("jax.numpy")
    nnx = pytest.importorskip("flax.nnx")
    from flax import traverse_util

    target = ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w")
    other = ("action_out_proj", "kernel")
    return nnx.State(traverse_util.unflatten_dict({
        target: nnx.VariableState(nnx.Param, jnp.ones((2, 3, 4, 5))),
        other: nnx.VariableState(nnx.Param, jnp.ones((5, 2))),
    })), target, other


def test_zero_effect_initialization_and_update_change_only_effective_target():
    jax = pytest.importorskip("jax")
    frozen, target, other = fixture_state()
    params, paths = initialize_adapters(frozen, rank=2, seed=7)
    effective = materialize_adapters(params, frozen, paths, 3)
    for path, value in frozen.flat_state().items():
        np.testing.assert_array_equal(effective.flat_state()[path].value, value.value)
    repeated, _ = initialize_adapters(frozen, rank=2, seed=7)
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(repeated), strict=True):
        np.testing.assert_array_equal(a, b)

    def loss(p):
        return materialize_adapters(p, frozen, paths, 3).flat_state()[target].value.sum()

    gradients = jax.jit(jax.grad(loss))(params)
    assert np.linalg.norm(gradients.flat_state()[target + ("lora_b",)].value) > 0
    np.testing.assert_array_equal(gradients.flat_state()[target + ("lora_a",)].value, 0)
    updated = jax.tree.map(lambda p, g: p - 0.1 * g, params, gradients)
    result = materialize_adapters(updated, frozen, paths, 3).flat_state()
    assert not np.array_equal(result[target].value, frozen.flat_state()[target].value)
    np.testing.assert_array_equal(result[other].value, frozen.flat_state()[other].value)
    np.testing.assert_array_equal(frozen.flat_state()[target].value, 1)
    half = materialize_adapters(updated, frozen, paths, 1.5).flat_state()[target].value
    # Adding then subtracting the unit base incurs float32 cancellation.
    np.testing.assert_allclose(
        result[target].value - 1, (half - 1) * 2, atol=2 * np.finfo(np.float32).eps,
    )


def test_adapter_checkpoint_roundtrip_and_scale_mismatch(tmp_path):
    from rl_vla.rldt_pi05 import Pi05Flow

    frozen, _, _ = fixture_state()
    flow = Pi05Flow()
    flow.params, _ = initialize_adapters(frozen, rank=2, seed=7)
    flow.metadata = {
        "model_config": "pi05_base", "projection_only": False,
        "checkpoint_metadata_sha256": "schema", "checkpoint_manifest_sha256": "base",
        "normalization_assets_sha256": "libero", "lora_rank": 2, "lora_alpha": 6,
        "lora_seed": 7,
    }
    path = flow.save(tmp_path / "adapter", flow.params)
    loaded = flow.load_params(path)
    for key, value in flow.params.flat_state().items():
        np.testing.assert_array_equal(loaded.flat_state()[key].value, value.value)
    flow.metadata["lora_alpha"] = 4
    with pytest.raises(ValueError, match="lora_alpha"):
        flow.load_params(path)


def test_evaluation_split_override_uses_held_out_states(monkeypatch, tmp_path):
    pytest.importorskip("jax")
    import sys

    from rl_vla import rldt

    observed = []
    monkeypatch.setattr(rldt, "evaluate", lambda config, *args: observed.append(config) or {})
    monkeypatch.setattr(sys, "argv", [
        "rldt", "evaluate", "--config", "configs/rldt_pi05_base_object0_lora128.toml",
        "--env-split", "validation", "--output", str(tmp_path / "eval.json"),
    ])
    rldt.main()
    assert observed[0]["env"]["kwargs"]["split"] == "validation"
    original = rldt.load_config("configs/rldt_pi05_base_object0_lora128.toml")
    assert original["env"]["kwargs"]["split"] == "train"


def test_training_cannot_override_to_held_out_split(monkeypatch):
    pytest.importorskip("jax")
    import sys

    from rl_vla import rldt

    monkeypatch.setattr(sys, "argv", [
        "rldt", "train", "--config", "configs/rldt_pi05_base_object0_lora128.toml",
        "--env-split", "test", "--output", "unused",
    ])
    with pytest.raises(SystemExit) as error:
        rldt.main()
    assert error.value.code == 2
