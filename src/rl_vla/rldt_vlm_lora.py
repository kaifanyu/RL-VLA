"""RLDT adapters in PaliGemma's image/text transformer, with a frozen expert.

Only image/text embeddings and critic features may be detached and reused.
Policy KV caches are always recomputed from the supplied current/reference
adapters. Actor differentiation includes that prefill and the frozen expert.
The frozen SigLIP encoder, token embeddings, norms, and motor expert are not
LoRA targets. Critic features retain their original checkpoint coordinates.
"""

from rl_vla.rldt_pi05 import Pi05Flow


def layerwise_prefix_cache(params, frozen, paths, scale, tokens, mask, ar_mask, config):
    """Apply adapters inside the layer scan to bound backward temporaries.

    Materializing every layer's effective dense weights before the scan makes
    autodiff allocate multi-GB dense weight gradients simultaneously. Here only
    one layer's effective weights/gradients need to exist at a time. The native
    OpenPI Block supplies all attention, normalization, RoPE, and MLP operations.
    """
    import jax
    import jax.numpy as jnp
    from flax import traverse_util
    from openpi.models import gemma
    from openpi.models.pi0 import make_attn_mask

    root = ("PaliGemma", "llm", "layers")
    weights = {p[3:]: v.value for p, v in frozen.flat_state().items()
               if p[:3] == root and not any(str(n).endswith("_1") for n in p[3:])}
    factors = {p[3:]: v.value for p, v in params.flat_state().items()}
    targets = tuple(p[3:] for p in paths)
    positions = jnp.cumsum(mask, axis=1) - 1
    attention_mask = make_attn_mask(mask, ar_mask)[:, None, :, :]
    block = gemma.Block(configs=(config,))

    def layer(hidden, inputs):
        base, adapters = inputs
        effective = dict(base)
        for p in targets:
            effective[p] = base[p] + scale * jnp.matmul(
                adapters[p + ("lora_a",)], adapters[p + ("lora_b",)])
        output, cache = block.apply(
            {"params": traverse_util.unflatten_dict(effective)}, [hidden], None,
            positions, attention_mask, [None], deterministic=True)
        return output[0], cache

    _, cache = jax.lax.scan(jax.checkpoint(layer, prevent_cse=False), tokens, (weights, factors))
    return cache


class VlmLoraFlow(Pi05Flow):
    @classmethod
    def from_config(cls, config):
        if config.get("lora_target") != "vlm" or not config.get("lora_rank"):
            raise ValueError("VlmLoraFlow requires lora_target=vlm and positive rank")
        self = super().from_config(config)
        self.metadata.update(
            adapted_backbone="PaliGemma language/image-token transformer (expert 0)",
            vision_encoder_trainable=False,
            action_expert_trainable=False,
            actor_prefix="differentiable_current_adapters_shared_across_particles",
            adapter_execution="layerwise_materialization_with_rematerialized_backward",
            feature_extractor="frozen_base_vlm_hidden+normalized_padded_state",
        )
        return self

    def _encode_prefix(self, params, frozen, observation):
        import jax.numpy as jnp
        from flax import nnx
        from openpi.models import model as model_module
        from openpi.models.pi0 import make_attn_mask

        # Adapter-free prefix supplies stable critic features. Image and token
        # embeddings are frozen and can also be reused by the learned policy.
        model = nnx.merge(self.graphdef, frozen)
        observation = model_module.preprocess_observation(None, observation, train=False)
        tokens, mask, ar_mask = model.embed_prefix(observation)
        (prefix, _), _ = model.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(mask, ar_mask),
            positions=jnp.cumsum(mask, axis=1) - 1)
        weights = mask.astype(jnp.float32)[..., None]
        pooled = jnp.sum(prefix.astype(jnp.float32) * weights, axis=1) / jnp.maximum(
            jnp.sum(weights, axis=1), 1.)
        features = jnp.concatenate([pooled, observation.state.astype(jnp.float32)], axis=-1)
        return dict(state=observation.state, features=features, critic_features=features,
                    vlm_tokens=tokens, vlm_mask=mask,
                    vlm_ar_mask=jnp.broadcast_to(ar_mask, mask.shape),
                    vlm_batch_index=jnp.arange(tokens.shape[0]))

    @staticmethod
    def repeat_context(context, repeats):
        import jax.numpy as jnp

        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if "vlm_tokens" not in context:
            return Pi05Flow.repeat_context(context, repeats)
        # Encode each distinct observation once, then gather its KV cache for
        # particles. Replicating a 2B-transformer prefill K times wastes memory.
        unique = {"vlm_tokens", "vlm_mask", "vlm_ar_mask"}
        return {key: value if key in unique else jnp.repeat(value, repeats, axis=0)
                for key, value in context.items()}

    def policy_context(self, params, context, *, frozen=None):
        import jax
        import jax.numpy as jnp
        from openpi.models import gemma

        if "vlm_tokens" not in context:
            return context
        mask = context["vlm_mask"]
        cache = layerwise_prefix_cache(
            params, self.frozen if frozen is None else frozen, self.lora_paths,
            self.lora_scale, context["vlm_tokens"], mask, context["vlm_ar_mask"],
            gemma.get_config("gemma_2b"))
        indices = context["vlm_batch_index"]
        return dict(state=context["state"], features=context["features"],
                    critic_features=context["critic_features"], prefix_mask=mask[indices],
                    kv_cache=jax.tree.map(lambda a: jnp.swapaxes(a, 0, 1)[indices], cache))

    def velocity(self, params, context, x, t, *, frozen=None):
        context = self.policy_context(params, context, frozen=frozen)
        return super().velocity(params, context, x, t, frozen=frozen)

    def sample(self, params, context, key, steps=10, *, frozen=None):
        # One current-policy prefill per rollout, reused for all ODE steps.
        context = self.policy_context(params, context, frozen=frozen)
        return super().sample(params, context, key, steps, frozen=frozen)
