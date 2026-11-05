"""Small actual upstream Flax attention contract test, including KV caching."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("openpi")
import jax.numpy as jnp
from flax import linen as nn
from openpi.models import gemma

from rl_vla.diagnostics.pi05_attention import _Capture


def test_scanned_native_cached_attention_is_unchanged_and_masked():
    cfg = gemma.Config(width=16, depth=2, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)
    model = gemma.Module(configs=[cfg, cfg], embed_dtype="bfloat16", adarms=True)
    variables = nn.Module.init(
        model, jax.random.key(0), use_adarms=[False, True], method=model.init
    )
    prefix = jax.random.normal(jax.random.key(1), (1, 4, 16)).astype(jnp.bfloat16)
    suffix = jax.random.normal(jax.random.key(2), (1, 2, 16)).astype(jnp.bfloat16)
    _, cache = model.apply(
        variables, [prefix, None], positions=jnp.arange(4)[None], mask=jnp.ones((1, 4, 4), bool)
    )
    mask = jnp.ones((1, 2, 6), bool).at[..., 3].set(False)
    cond = [None, jnp.zeros((1, 16), jnp.bfloat16)]

    def run(variables, cache):
        return model.apply(
            variables,
            [None, suffix],
            positions=jnp.array([[4, 5]]),
            mask=mask,
            kv_cache=cache,
            adarms_cond=cond,
        )[0][1]

    original = jax.jit(run)(variables, cache)
    cap = _Capture(key_keep=np.ones(6, bool))

    def captured(variables, cache):
        with cap.intercept():
            return run(variables, cache)

    actual = jax.jit(captured)(variables, cache)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    attention = cap.stacked(2)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(original))
    assert attention.shape == (2, 1, 2, 2, 6)
    np.testing.assert_allclose(attention.sum(-1), 1, atol=0.004)
    assert np.all(attention[..., 3] == 0)


def test_native_knockout_changes_attention_and_output():
    cfg = gemma.Config(width=16, depth=2, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)
    model = gemma.Module(configs=[cfg], embed_dtype="bfloat16")
    variables = nn.Module.init(model, jax.random.key(3), use_adarms=[False], method=model.init)
    x = jax.random.normal(jax.random.key(4), (1, 4, 16)).astype(jnp.bfloat16)
    kwargs = {"positions": jnp.arange(4)[None], "mask": jnp.ones((1, 4, 4), bool)}
    original = model.apply(variables, [x], **kwargs)[0][0]
    capture = _Capture(key_keep=np.array([False, False, True, True]))

    def run(variables, x):
        with capture.intercept():
            return model.apply(variables, [x], **kwargs)[0][0]

    result = jax.jit(run)(variables, x)
    jax.block_until_ready(result)
    jax.effects_barrier()
    attention = capture.stacked(2)
    assert (attention[..., :2] == 0).all()
    assert (
        float(np.max(np.abs(np.asarray(result, np.float32) - np.asarray(original, np.float32)))) > 0
    )


def test_exported_attention_reconstructs_native_value_aggregation():
    cfg = gemma.Config(width=16, depth=1, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)
    layer = gemma.Attention(configs=[cfg])
    x = jax.random.normal(jax.random.key(9), (1, 4, 16)).astype(jnp.bfloat16)
    positions = jnp.arange(4)[None]
    mask = jnp.ones((1, 1, 4, 4), bool).at[..., 3].set(False)
    variables = layer.init(jax.random.key(8), [x], positions, mask, None)
    capture = _Capture()

    def run(variables, x):
        with capture.intercept():
            return layer.apply(variables, [x], positions, mask, None)

    (actual,), (_, v) = jax.jit(run)(variables, x)
    jax.block_until_ready(actual)
    jax.effects_barrier()
    probs = jnp.asarray(capture.stacked(1)[0], dtype=jnp.bfloat16)
    # Native grouped-query attention shares this one value head across two Q heads.
    encoded = jnp.einsum("BHTS,BSKD->BTHD", probs, v)
    weight = variables["params"]["attn_vec_einsum"]["w"].astype(jnp.bfloat16)
    reconstructed = jnp.einsum("BTNH,NHD->BTD", encoded, weight)
    np.testing.assert_allclose(
        np.asarray(reconstructed, np.float32), np.asarray(actual, np.float32), atol=0.008, rtol=0.01
    )
