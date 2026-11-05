"""VLM adapter boundaries, particle sharing, and RLDT prefix gradients."""
import numpy as np
import pytest

from rl_vla.rldt_lora import initialize_adapters, is_lora_target, lora_settings
from rl_vla.rldt_vlm_lora import VlmLoraFlow


@pytest.mark.parametrize("path, expected", [
    (("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"), True),
    (("PaliGemma", "llm", "layers", "attn", "kv_einsum", "w"), True),
    (("PaliGemma", "llm", "layers", "attn", "attn_vec_einsum", "w"), True),
    (("PaliGemma", "llm", "layers", "mlp", "gating_einsum"), True),
    (("PaliGemma", "llm", "layers", "mlp", "linear"), True),
    (("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"), False),
    (("PaliGemma", "llm", "layers", "mlp_1", "linear"), False),
    (("PaliGemma", "llm", "embedder", "input_embedding"), False),
    (("PaliGemma", "img", "Transformer", "kernel"), False),
    (("action_in_proj", "kernel"), False),
])
def test_vlm_targets_exclude_vision_encoder_and_motor_expert(path, expected):
    assert is_lora_target(path, "vlm") is expected
    assert not (is_lora_target(path, "vlm") and is_lora_target(path, "action_expert"))


def test_invalid_target_and_zero_rank_rejected():
    for config in ({"lora_target": "wrong"}, {"lora_target": "vlm", "lora_rank": 0}):
        with pytest.raises(ValueError):
            lora_settings(config)


def test_direct_factory_cannot_use_frozen_prefix_path_for_vlm(monkeypatch):
    from rl_vla.rldt_pi05 import Pi05Flow

    marker = object()
    monkeypatch.setattr(VlmLoraFlow, "from_config", classmethod(lambda cls, c: marker))
    assert Pi05Flow.from_config({"lora_target": "vlm", "lora_rank": 128}) is marker


def test_checkpoint_rejects_other_adapter_target_before_reading_weights(tmp_path):
    import json
    from rl_vla.rldt_pi05 import Pi05Flow

    flow = Pi05Flow()
    flow.metadata = dict(model_config="pi05_libero", projection_only=False,
                         checkpoint_metadata_sha256="schema", checkpoint_manifest_sha256="base",
                         normalization_assets_sha256="norm", lora_rank=128, lora_alpha=128.,
                         lora_seed=0, lora_targets="vlm_attention_mlp_last_two_axes")
    saved = dict(flow.metadata, format="rldt-pi05-expert-v1",
                 lora_targets="action_expert_attention_mlp_last_two_axes")
    (tmp_path / "rldt_pi05.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="lora_targets"):
        flow._validate_checkpoint_manifest(tmp_path, "rldt-pi05-expert-v1")


def test_vlm_adapter_initialization_only_contains_backbone_factors():
    jnp = pytest.importorskip("jax.numpy")
    from flax import nnx, traverse_util
    paths = [("PaliGemma", "llm", "layers", "mlp", "linear"),
             ("PaliGemma", "llm", "layers", "mlp_1", "linear")]
    frozen = nnx.State(traverse_util.unflatten_dict({
        p: nnx.VariableState(nnx.Param, jnp.ones((2, 4, 8))) for p in paths}))
    params, targets = initialize_adapters(frozen, 2, 0, "vlm")
    assert targets == (paths[0],)
    assert all(p[:len(paths[0])] == paths[0] for p in params.flat_state())


def test_particle_replication_reuses_unique_prefix_and_preserves_order():
    jnp = pytest.importorskip("jax.numpy")
    context = dict(vlm_tokens=jnp.ones((2, 7, 4)), vlm_mask=jnp.ones((2, 7), bool),
                   vlm_ar_mask=jnp.zeros((2, 7), bool), vlm_batch_index=jnp.arange(2),
                   features=jnp.array([[1., 2.], [3., 4.]]), state=jnp.ones((2, 3)))
    repeated = VlmLoraFlow.repeat_context(context, 3)
    assert repeated["vlm_tokens"] is context["vlm_tokens"]
    np.testing.assert_array_equal(repeated["vlm_batch_index"], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(repeated["features"], np.repeat(context["features"], 3, axis=0))
    twice = VlmLoraFlow.repeat_context(repeated, 2)
    np.testing.assert_array_equal(twice["vlm_batch_index"], [0]*6 + [1]*6)


def test_actor_update_differentiates_through_prefix_with_frozen_motor_expert():
    jnp = pytest.importorskip("jax.numpy")
    import optax
    from rl_vla.rldt import Learner

    class PrefixOnlyFlow:
        horizon, action_dim = 2, 3

        def policy_context(self, p, c, *, frozen):
            return {"encoded": p * c["tokens"]}

        def velocity(self, p, c, x, t, *, frozen):
            # Direct access to p is deliberately absent: only the prefix can
            # transmit the learning signal through this frozen motor expert.
            return jnp.ones_like(x) * c["encoded"][:, None, None]

    learner = Learner.__new__(Learner)
    learner.flow = PrefixOnlyFlow()
    learner.k, learner.steps = 2, 10
    learner.mask = jnp.ones((2, 3))
    learner.r = dict(consistency_weight=0., constraint_weight=0.)
    learner.actor_tx = optax.sgd(.1)
    p = jnp.array(.2)
    x = jnp.ones((2, 2, 3))
    result, _, metrics = learner._update_actor(
        p, learner.actor_tx.init(p), {"tokens": jnp.ones(2)}, x,
        jnp.ones(2)*.5, x, x, x, jnp.ones(2)*.5, x, {})
    assert float(metrics["actor_grad_norm"]) > 0
    assert float(result) != float(p)


def test_layerwise_adapter_prefill_matches_native_cache_and_gradients():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from flax import nnx, traverse_util
    from openpi.models import gemma
    from openpi.models.pi0 import make_attn_mask
    from rl_vla.rldt_lora import materialize_adapters
    from rl_vla.rldt_vlm_lora import layerwise_prefix_cache

    cfg = gemma.Config(width=16, depth=2, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)
    model = gemma.Module(configs=(cfg,), embed_dtype="float32")
    tokens = jax.random.normal(jax.random.key(1), (2, 5, 16))
    mask = jnp.ones((2, 5), bool)
    ar_mask = jnp.zeros_like(mask)
    positions = jnp.cumsum(mask, axis=1)-1
    _, variables = model.init_with_output(jax.random.key(0), [tokens],
                                          positions=positions, mask=make_attn_mask(mask, ar_mask))
    frozen = nnx.State({"PaliGemma": {"llm": traverse_util.unflatten_dict({
        p: nnx.VariableState(nnx.Param, v)
        for p, v in traverse_util.flatten_dict(variables["params"]).items()})}})
    params, paths = initialize_adapters(frozen, 2, 0, "vlm")
    # Exercise nonzero deltas too, not just the zero-adapter identity.
    params = params.map(lambda p, v: v.replace(v.value + (.01 if p[-1] == "lora_b" else 0)))

    def native(p):
        state = materialize_adapters(p, frozen, paths, 1.).to_pure_dict()["PaliGemma"]["llm"]
        return model.apply({"params": state}, [tokens], positions=positions,
                           mask=make_attn_mask(mask, ar_mask))[1]

    def bounded(p):
        return layerwise_prefix_cache(p, frozen, paths, 1., tokens, mask, ar_mask, cfg)

    for a, b in zip(jax.tree.leaves(native(params)), jax.tree.leaves(bounded(params)), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)
    def objective(function, p):
        return sum(jnp.square(a).mean() for a in function(p))
    g1 = jax.grad(lambda p: objective(native, p))(params)
    g2 = jax.grad(lambda p: objective(bounded, p))(params)
    for a, b in zip(jax.tree.leaves(g1), jax.tree.leaves(g2), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)
