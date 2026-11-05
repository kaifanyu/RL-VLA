"""LoRA reparameterization of OpenPI action or vision-language transformers.

Each scanned layer/head keeps its own rank-r factors on the last two weight
axes, matching OpenPI's attention/MLP tensor layout. The effective weight is
W + (alpha / rank) A B. No upstream modules or pretrained weights are changed.
"""

from __future__ import annotations

import math

import numpy as np


def lora_settings(config):
    rank = config.get("lora_rank", 0)
    if type(rank) is not int or rank < 0:
        raise ValueError("policy.lora_rank must be a nonnegative integer")
    alpha = config.get("lora_alpha", rank if rank else 1)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or (
        not math.isfinite(alpha) or alpha <= 0
    ):
        raise ValueError("policy.lora_alpha must be finite and positive")
    seed = config.get("lora_seed", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("policy.lora_seed must be a nonnegative integer")
    if rank and config.get("projection_only", False):
        raise ValueError("LoRA and projection_only cannot be combined")
    target = config.get("lora_target", "action_expert")
    if target not in {"action_expert", "vlm"}:
        raise ValueError("policy.lora_target must be action_expert or vlm")
    if target == "vlm" and not rank:
        raise ValueError("VLM LoRA requires a positive lora_rank")
    return rank, float(alpha), seed


def is_lora_target(path, target="action_expert"):
    names = tuple(map(str, path))
    if target not in {"action_expert", "vlm"}:
        raise ValueError(f"Unknown LoRA target: {target}")
    expert = any(name.endswith("_1") for name in names[2:])
    return (
        names[:2] == ("PaliGemma", "llm")
        and expert == (target == "action_expert")
        and names[-1] in {"w", "gating_einsum", "linear"}
        and ("attn" in names or ("mlp_1" if expert else "mlp") in names)
    )


def initialize_adapters(frozen, rank, seed, target="action_expert"):
    import jax.numpy as jnp
    from flax import nnx, traverse_util

    rng = np.random.default_rng(seed)
    factors = {}
    paths = []
    for path, value in sorted(frozen.flat_state().items()):
        if not is_lora_target(path, target):
            continue
        shape = value.value.shape
        if rank > min(shape[-2:]):
            raise ValueError(f"LoRA rank {rank} exceeds matrix dimensions at {path}: {shape}")
        a = rng.normal(0, 0.01, (*shape[:-1], rank)).astype(np.float32)
        b = np.zeros((*shape[:-2], rank, shape[-1]), dtype=np.float32)
        for name, array in (("lora_a", a), ("lora_b", b)):
            factors[path + (name,)] = nnx.VariableState(nnx.Param, jnp.asarray(array))
        paths.append(path)
    if not paths:
        raise ValueError(f"No recognized {target} attention/MLP weights for LoRA")
    return nnx.State(traverse_util.unflatten_dict(factors)), tuple(paths)


def materialize_adapters(params, frozen, paths, scale):
    """Differentiable effective weights; the optimizer only receives factors."""
    import jax.numpy as jnp
    from flax import nnx, traverse_util

    factors = params.flat_state()
    weights = dict(frozen.flat_state())
    for path in paths:
        a = factors[path + ("lora_a",)].value
        b = factors[path + ("lora_b",)].value
        base = weights[path]
        delta = jnp.matmul(a, b) * scale
        weights[path] = base.replace(base.value + delta)
    return nnx.State(traverse_util.unflatten_dict(weights))
