# Copyright 2024 Big Vision Authors.
# SPDX-License-Identifier: Apache-2.0
"""Isolated native Gemma layer-window key-mask interventions.

The scoped Linen interceptor preserves the pinned Gemma setup/call arithmetic,
parameters, rematerialization and scan body. Only the scan mask input changes
from broadcast to a layer axis. Prefix encoding remains outside this scope.
Native no-op fidelity must be checked before interpreting any intervention.
"""

from contextlib import contextmanager

import numpy as np

from rl_vla.diagnostics.pi05_attention import key_keep_mask


def layer_keep_mask(metadata, knockout="none", layers=None, keys=None):
    depth = int(metadata["layers"])
    if depth < 1:
        raise ValueError("Transformer depth must be positive")
    selected = list(range(depth)) if layers is None else list(layers)
    if len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < depth for i in selected):
        raise ValueError("Layer indices must be distinct zero-based integers within the native depth")
    keys_kept = key_keep_mask(metadata, knockout, keys)
    keep = np.ones((depth, len(keys_kept)), dtype=bool)
    keep[selected] = keys_kept
    return keep


class LayerMask:
    """Scoped setup/call override; no monkeypatch or upstream file modification."""

    def __init__(self, keep):
        self.keep = keep

    def setup(self, module):
        import jax
        from flax import linen as nn
        from openpi.models import gemma

        if not all(config.depth == module.configs[0].depth for config in module.configs):
            raise ValueError("Native experts must have the same depth")
        if self.keep.shape[0] != module.configs[0].depth:
            raise ValueError("Layer-mask depth differs from the native transformer")
        # Same modules, names, variable axes, RNG splits and remat policy as the
        # pinned gemma.Module.setup. The third scan input is the sole change.
        module.embedder = gemma.Embedder(vocab_size=gemma.PALIGEMMA_VOCAB_SIZE,
                                        embed_dim=module.configs[0].width, name="embedder")
        block_cls = nn.remat(gemma.Block, prevent_cse=False, static_argnums=(5,),
                            policy=jax.checkpoint_policies.nothing_saveable)
        module.layers = nn.scan(block_cls, variable_axes={"params": 0},
                                split_rngs={"params": True, "dropout": True},
                                in_axes=(0, nn.broadcast, 0, nn.broadcast, nn.broadcast),
                                length=module.configs[0].depth)(
                                    configs=module.configs, dropout=module.dropout,
                                    dropout_bdims=module.dropout_bdims)
        module.final_norms = [gemma.RMSNorm(name=gemma._name("final_norm", i))
                              for i in range(len(module.configs))]

    def call(self, module, embedded, positions, mask, adarms_cond=None, *, kv_cache=None,
             deterministic=True):
        import jax
        import jax.numpy as jnp

        if len(embedded) != 2 or embedded[0] is not None or embedded[1] is None or kv_cache is None:
            raise ValueError("Layer-window intervention is restricted to cached-prefix expert-1 decoding")
        embedded = jax.tree.map(lambda e: e.astype(module.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if mask.shape[-1] != self.keep.shape[-1]:
            raise ValueError("Layer-mask keys differ from the native prefix+suffix key sequence")
        # [layer,batch,1,action_query,all_keys], scanned to the exact native
        # Attention argument shape [batch,1,action_query,all_keys].
        mask = mask[None] & jnp.asarray(self.keep, dtype=bool)[:, None, None, None, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(module.configs)
        embedded, kv_cache = module.layers(embedded, kv_cache, positions, mask,
                                           adarms_cond, deterministic)
        assert all(e.dtype == jnp.dtype(module.embed_dtype) for e in embedded if e is not None)
        return [f(e, a)[0] if e is not None else e
                for f, e, a in zip(module.final_norms, embedded, adarms_cond, strict=True)], kv_cache

    def interceptor(self, next_fun, args, kwargs, context):
        from openpi.models import gemma

        if not isinstance(context.module, gemma.Module):
            return next_fun(*args, **kwargs)
        if context.method_name == "setup":
            return self.setup(context.module)
        if context.method_name == "__call__":
            return self.call(context.module, *args, **kwargs)
        return next_fun(*args, **kwargs)

    @contextmanager
    def intercept(self):
        from flax import linen as nn

        with nn.intercept_methods(self.interceptor):
            yield


def velocity_with_layer_knockout(flow, params, context, x, t, metadata, *, knockout="none",
                                 layers=None, keys=None):
    import jax
    import jax.numpy as jnp

    keep = layer_keep_mask(metadata, knockout, layers, keys)
    if not hasattr(flow, "_layer_knockout_velocity_jit"):
        def velocity(p, frozen, c, x, t, layer_keep):
            with LayerMask(layer_keep).intercept():
                return flow.velocity(p, c, x, t, frozen=frozen)
        flow._layer_knockout_velocity_jit = jax.jit(velocity)
    return flow._layer_knockout_velocity_jit(params, flow.frozen, context, x, t, jnp.asarray(keep))


def sample_with_layer_knockout(flow, params, context, key, metadata, *, steps=10, knockout="none",
                               layers=None, keys=None, initial_noise=None):
    """Same native sampler/noise API as sample_with_intervention, plus layers."""
    import jax
    import jax.numpy as jnp

    from rl_vla.diagnostics.pi05_attention import _sample_from_noise

    keep = layer_keep_mask(metadata, knockout, layers, keys)
    if not hasattr(flow, "_layer_knockout_sample_jits"):
        flow._layer_knockout_sample_jits = {}
    cache_key = (steps, initial_noise is not None)
    if cache_key not in flow._layer_knockout_sample_jits:
        def sample(p, frozen, c, argument, layer_keep):
            with LayerMask(layer_keep).intercept():
                if initial_noise is None:
                    return flow.sample(p, c, argument, steps, frozen=frozen)
                return _sample_from_noise(flow, p, frozen, c, argument, steps)
        flow._layer_knockout_sample_jits[cache_key] = jax.jit(sample)
    argument = key if initial_noise is None else jnp.asarray(initial_noise, jnp.float32)
    return flow._layer_knockout_sample_jits[cache_key](params, flow.frozen, context, argument,
                                                     jnp.asarray(keep))


def capture_layer_velocity(flow, params, context, x, t, metadata, *, knockout="none", layers=None,
                            keys=None):
    """Capture native per-layer probabilities for auditing the applied mask."""
    import jax
    import jax.numpy as jnp

    from rl_vla.diagnostics.pi05_attention import _Capture

    keep = layer_keep_mask(metadata, knockout, layers, keys)
    capture = _Capture(capture_features=False)
    if not hasattr(flow, "_layer_knockout_capture_velocity"):
        def velocity(p, frozen, c, x, t, layer_keep):
            with LayerMask(layer_keep).intercept(), capture.intercept():
                return flow.velocity(p, c, x, t, frozen=frozen)
        flow._layer_knockout_capture_velocity = (jax.jit(velocity), capture)
    function, capture = flow._layer_knockout_capture_velocity
    capture.layers.clear()
    result = function(params, flow.frozen, context, x, t, jnp.asarray(keep))
    jax.block_until_ready(result)
    jax.effects_barrier()
    return np.asarray(result), capture.stacked(int(metadata["layers"]))
