"""Native OpenPI attention export without changing training or upstream code.

The scoped Linen interceptor calls the original attention implementation. It
temporarily taps the original softmax while an Attention call is tracing,
exposing its native full-denominator probabilities without reimplementing Q/K.
Ordered debug callbacks retain the scanned transformer layer order. Capture is
for offline diagnostics only and is deliberately separate from RLDT training.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def array_sha256(array: Any) -> str:
    a = np.asarray(array)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()


def key_keep_mask(metadata: dict, mode: str = "none", keys: list[int] | None = None) -> np.ndarray:
    """Return a key mask in actual prefix+suffix coordinates.

    An empty intervention is exactly a no-op. The dummy camera still has a
    span; its original padding mask remains in force inside native attention.
    """
    total = int(metadata["prefix_tokens"]) + int(metadata["action_tokens"])
    keep = np.ones(total, dtype=bool)
    if mode == "no_image":
        for span in metadata["camera_spans"]:
            keep[span["start"] : span["end"]] = False
    elif mode == "no_text":
        lo, hi = metadata["text_span"]
        keep[lo:hi] = False
    elif mode == "keys":
        if keys is None or any(k < 0 or k >= total for k in keys):
            raise ValueError("keys intervention requires valid absolute key indices")
        keep[keys] = False
    elif mode != "none":
        raise ValueError(f"Unknown native attention knockout: {mode}")
    return keep


class _Capture:
    def __init__(self, *, query_indices=None, key_keep=None, capture_features=True):
        self.query_indices = query_indices
        self.key_keep = key_keep
        self.layers: list[np.ndarray] = []
        self.active_dtype = None
        self.capture_features = capture_features
        self.prefix_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self.suffix_features: list[np.ndarray] = []

    def _append(self, attention):
        self.layers.append(np.asarray(attention, dtype=np.float32))

    def _append_features(self, tokens, features):
        self.prefix_pairs.append(
            (np.asarray(tokens, dtype=np.float32), np.asarray(features, dtype=np.float32))
        )

    def _append_suffix(self, features):
        self.suffix_features.append(np.asarray(features, dtype=np.float32))

    def interceptor(self, next_fun, args, kwargs, context):
        import jax
        import jax.numpy as jnp
        from openpi.models import gemma

        module = context.module
        if isinstance(module, gemma.Module) and context.method_name == "__call__":
            result = next_fun(*args, **kwargs)
            embedded = args[0]
            if self.capture_features and embedded[0] is not None and result[0][0] is not None:
                jax.debug.callback(self._append_features, embedded[0], result[0][0], ordered=True)
            if (
                self.capture_features
                and len(embedded) > 1
                and embedded[1] is not None
                and result[0][1] is not None
            ):
                jax.debug.callback(self._append_suffix, result[0][1], ordered=True)
            return result
        if not isinstance(module, gemma.Attention) or context.method_name != "__call__":
            return next_fun(*args, **kwargs)
        xs, positions, mask, cache = args
        if self.key_keep is not None:
            mask = mask & jnp.asarray(self.key_keep)[None, None, None, :]
        previous = self.active_dtype
        self.active_dtype = next(x.dtype for x in xs if x is not None)
        try:
            return next_fun(xs, positions, mask, cache, **kwargs)
        finally:
            self.active_dtype = previous

    @contextmanager
    def intercept(self, *, attention=True):
        """Scoped trace-time tap; restored even if tracing raises.

        This diagnostic process must be single-threaded during JAX tracing.
        No files or loaded weights are modified. The original softmax return
        value is returned untouched to the original native Attention method.
        """
        import jax
        import jax.numpy as jnp
        from flax import linen as nn

        original = jax.nn.softmax

        def tapped_softmax(*args, **kwargs):
            result = original(*args, **kwargs)
            if attention and self.active_dtype is not None:
                probs = result.astype(self.active_dtype)
                b, k, g, q, s = probs.shape
                probs = probs.reshape(b, k * g, q, s)
                if self.query_indices is not None:
                    probs = probs[:, :, jnp.asarray(self.query_indices)]
                jax.debug.callback(self._append, probs.astype(jnp.float32), ordered=True)
            return result

        jax.nn.softmax = tapped_softmax
        try:
            with nn.intercept_methods(self.interceptor):
                yield
        finally:
            jax.nn.softmax = original

    def stacked(self, expected_layers: int) -> np.ndarray:
        if len(self.layers) != expected_layers:
            raise RuntimeError(
                f"Expected {expected_layers} ordered layer captures, got {len(self.layers)}"
            )
        return np.stack(self.layers)


def transformed_observation(flow, raw: dict):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation

    transformed = flow._transform_observation(raw)
    return Observation.from_dict(
        jax.tree.map(lambda x: jnp.asarray(np.asarray(x)[None]), transformed)
    )


def _token_metadata(observation, image_tokens: int, horizon: int) -> dict:
    # derive token count from actual encoder outputs, never toolkit defaults
    camera_count = len(observation.images)
    if image_tokens % camera_count:
        raise ValueError("Image tokens do not divide evenly across configured cameras")
    count = image_tokens // camera_count
    side = int(np.sqrt(count))
    if side * side != count:
        raise ValueError("Only square patch grids are supported")
    camera_spans = []
    for i, name in enumerate(observation.images):
        camera_spans.append(
            {
                "name": name,
                "start": i * count,
                "end": (i + 1) * count,
                "grid": [side, side],
                "valid": bool(np.asarray(observation.image_masks[name])[0]),
            }
        )
    ids = np.asarray(observation.tokenized_prompt)[0]
    valid = np.asarray(observation.tokenized_prompt_mask)[0]
    return {
        "camera_spans": camera_spans,
        "text_span": [image_tokens, image_tokens + len(ids)],
        "prefix_tokens": image_tokens + len(ids),
        "action_tokens": horizon,
        "action_key_span": [image_tokens + len(ids), image_tokens + len(ids) + horizon],
        "action_query_span": [0, horizon],
        "prompt_ids": ids.tolist(),
        "prompt_mask": valid.tolist(),
        "prefill_query_indices": (image_tokens + np.flatnonzero(valid)).tolist(),
        "image_transform": "input RGB is already rotated 180 degrees by LIBERO; OpenPI ResizeImages 224x224, uint8 to [-1,1]; no extra rotation",
        "attention_axes": ["layer", "batch", "head", "query", "key"],
        "attention_values": "native bfloat16 softmax, exported float32; full prefix+suffix denominator; no per-image renormalization",
    }


def capture_prefix(
    flow, raw: dict, *, capture_attention: bool = True, canonical_context: bool = True
):
    """Return frozen context, final spatial features, and text-query attention."""
    import jax
    from flax import nnx
    from openpi.models import model as model_module

    observation = transformed_observation(flow, raw)

    def embed(params, frozen, obs):
        model = nnx.merge(flow.graphdef, params, frozen)
        obs = model_module.preprocess_observation(None, obs, train=False)
        tokens, mask, ar = model.embed_prefix(obs)
        return tokens, mask, ar, obs

    text_count = observation.tokenized_prompt.shape[1]
    if not hasattr(flow, "_diagnostic_image_token_count"):
        tokens, _, _, _ = jax.jit(embed)(flow.params, flow.frozen, observation)
        flow._diagnostic_image_token_count = int(tokens.shape[1] - text_count)
    metadata = _token_metadata(observation, flow._diagnostic_image_token_count, flow.horizon)
    capture = _Capture(query_indices=metadata["prefill_query_indices"])

    def encode(params, frozen, observation):
        with capture.intercept(attention=capture_attention):
            # Retain the complete original compilation boundary. Splitting
            # image embedding from prefix encoding changes BF16 fusion/rounding.
            return flow._encode_prefix(params, frozen, observation)

    cache_key = (capture_attention, tuple(metadata["prefill_query_indices"]))
    if not hasattr(flow, "_diagnostic_encode_jits"):
        flow._diagnostic_encode_jits = {}
    if cache_key not in flow._diagnostic_encode_jits:
        flow._diagnostic_encode_jits[cache_key] = (jax.jit(encode), capture)
    encode_jit, capture = flow._diagnostic_encode_jits[cache_key]
    capture.layers.clear()
    capture.prefix_pairs.clear()
    context = encode_jit(flow.params, flow.frozen, observation)
    jax.block_until_ready(context)
    jax.effects_barrier()
    if len(capture.prefix_pairs) != 1:
        raise RuntimeError(f"Expected one prefix feature capture, got {len(capture.prefix_pairs)}")
    actual_tokens, prefix = capture.prefix_pairs[0]
    layers = int(context["kv_cache"][0].shape[1])
    arrays = {
        "image_features": actual_tokens[:, : metadata["text_span"][0]],
        "prefix_features": np.asarray(prefix, dtype=np.float32),
        "prefix_mask": np.asarray(context["prefix_mask"]),
        "normalized_state": np.asarray(observation.state),
        "prompt_ids": np.asarray(observation.tokenized_prompt),
        "prompt_mask": np.asarray(observation.tokenized_prompt_mask),
    }
    if capture_attention:
        arrays["prefill_attention"] = capture.stacked(layers)
    metadata["layers"] = layers
    if not hasattr(flow, "_diagnostic_tokenizer"):
        from openpi.models.tokenizer import PaligemmaTokenizer

        flow._diagnostic_tokenizer = PaligemmaTokenizer(max_len=text_count)._tokenizer
    metadata["prompt_token_pieces"] = [
        flow._diagnostic_tokenizer.id_to_piece(i) for i in metadata["prompt_ids"]
    ]
    nouns = [
        "alphabet soup",
        "cream cheese",
        "salad dressing",
        "butter",
        "milk",
        "tomato sauce",
        "orange juice",
        "basket",
    ]
    metadata["object_noun_tokens"] = {}
    for noun in nouns:
        noun_ids = list(flow._diagnostic_tokenizer.encode(noun))
        matches = [
            i
            for i in range(text_count - len(noun_ids) + 1)
            if metadata["prompt_ids"][i : i + len(noun_ids)] == noun_ids
        ]
        indices = [i + j for i in matches for j in range(len(noun_ids))]
        if indices:
            metadata["object_noun_tokens"][noun] = {
                "prompt_indices": indices,
                "prefix_positions": [metadata["text_span"][0] + i for i in indices],
                "aggregation": "report individual subwords and arithmetic mean of their full-denominator attention rows",
            }
    if not canonical_context:
        metadata["expert_context"] = (
            "diagnostic context only; prohibited for action analysis without canonical recomputation"
        )
        metadata["prefill_capture_fidelity"] = (
            "not rechecked on this feature-only bank frame; see first-frame audit"
        )
        metadata["prefix_features_sha256"] = array_sha256(arrays["prefix_features"])
        return context, arrays, metadata
    if not hasattr(flow, "_diagnostic_canonical_encode_jit"):
        flow._diagnostic_canonical_encode_jit = jax.jit(flow._encode_prefix)
    canonical = flow._diagnostic_canonical_encode_jit(flow.params, flow.frozen, observation)
    jax.block_until_ready(canonical)
    fidelity = {}
    for name in context:
        old_leaves = jax.tree.leaves(context[name])
        new_leaves = jax.tree.leaves(canonical[name])
        diff = max(
            float(np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))
            for a, b in zip(old_leaves, new_leaves, strict=True)
        )
        scale = max(float(np.max(np.abs(np.asarray(b, np.float32)))) for b in new_leaves)
        fidelity[name] = {
            "max_abs_difference": diff,
            "relative_to_max_abs": diff / max(scale, 1e-12),
        }
    valid = np.asarray(context["prefix_mask"])[0]
    fidelity["valid_kv_cache"] = {
        "max_abs_difference": max(
            float(
                np.max(
                    np.abs(
                        np.asarray(a, np.float32)[:, :, valid]
                        - np.asarray(b, np.float32)[:, :, valid]
                    )
                )
            )
            for a, b in zip(context["kv_cache"], canonical["kv_cache"], strict=True)
        )
    }
    metadata["prefill_capture_fidelity"] = fidelity
    metadata["prefill_capture_bit_exact"] = all(
        v["max_abs_difference"] == 0 for v in fidelity.values()
    )
    metadata["expert_context"] = (
        "canonical untouched native _encode_prefix, never diagnostic-capture cache"
    )
    context = canonical
    metadata["kv_cache_sha256"] = [array_sha256(a) for a in context["kv_cache"]]
    metadata["prefix_features_sha256"] = array_sha256(arrays["prefix_features"])
    return context, arrays, metadata


def camera_views(attention: np.ndarray, metadata: dict) -> dict[str, np.ndarray]:
    """VLA-Trace plot/overlay vectors, one camera at a time, never re-softmaxed."""
    a = np.asarray(attention, dtype=np.float32)
    if a.ndim != 5 or a.shape[1] != 1:
        raise ValueError("Expected [layers,1,heads,queries,keys]")
    result = {}
    for span in metadata["camera_spans"]:
        if not span["valid"]:
            continue
        view = a[:, 0, :, :, span["start"] : span["end"]]
        result[span["name"]] = view.mean(axis=(0, 1, 2))
        result[span["name"] + "__per_layer"] = view.mean(axis=(1, 2))
        result[span["name"] + "__per_head"] = view.mean(axis=2)
    return result


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def capture_velocity(
    flow,
    params,
    context: dict,
    x,
    t,
    metadata: dict,
    *,
    knockout="none",
    keys=None,
    return_hidden=False,
):
    """Capture every expert layer/head/action query at one common x_t and t.

    Knockout changes the native pre-softmax boolean mask in all layers. The
    original suffix self-attention remains valid, avoiding all-masked rows.
    """
    import jax

    keep = key_keep_mask(metadata, knockout, keys)
    capture = _Capture(key_keep=keep)

    def velocity(params, frozen, context, x, t):
        with capture.intercept():
            return flow.velocity(params, context, x, t, frozen=frozen)

    cache_key = keep.tobytes()
    if not hasattr(flow, "_diagnostic_velocity_jits"):
        flow._diagnostic_velocity_jits = {}
    if cache_key not in flow._diagnostic_velocity_jits:
        flow._diagnostic_velocity_jits[cache_key] = (jax.jit(velocity), capture)
    velocity_jit, capture = flow._diagnostic_velocity_jits[cache_key]
    capture.layers.clear()
    capture.suffix_features.clear()
    velocity = velocity_jit(params, flow.frozen, context, x, t)
    jax.block_until_ready(velocity)
    jax.effects_barrier()
    if return_hidden:
        if len(capture.suffix_features) != 1:
            raise RuntimeError("Expected one final normalized expert suffix hidden-state capture")
        return np.asarray(velocity), capture.stacked(metadata["layers"]), capture.suffix_features[0]
    return np.asarray(velocity), capture.stacked(metadata["layers"])


def project_suffix(flow, params, hidden):
    """Apply actual native NNX action_out_proj to captured final BF16 suffix states.

    Hidden values are exported as portable float32; they are exactly representable
    in their original BF16 dtype. Cast back before calling the native projection.
    """
    import jax
    import jax.numpy as jnp
    from flax import nnx

    def project(p, frozen, h):
        model = nnx.merge(flow.graphdef, p, frozen)
        return model.action_out_proj(h[:, -flow.horizon :]).astype(jnp.float32)

    if not hasattr(flow, "_diagnostic_project_suffix_jit"):
        flow._diagnostic_project_suffix_jit = jax.jit(project)
    return np.asarray(
        flow._diagnostic_project_suffix_jit(
            params, flow.frozen, jnp.asarray(hidden, dtype=jnp.bfloat16)
        )
    )


def native_velocity(flow, params, context, x, t):
    import jax

    if not hasattr(flow, "_diagnostic_native_velocity_jit"):
        flow._diagnostic_native_velocity_jit = jax.jit(
            lambda p, f, c, x, t: flow.velocity(p, c, x, t, frozen=f)
        )
    return np.asarray(flow._diagnostic_native_velocity_jit(params, flow.frozen, context, x, t))


def _sample_from_noise(flow, params, frozen, context, noise, steps):
    """Same native Euler recurrence, with explicit persisted Gaussian latent."""
    import jax
    import jax.numpy as jnp

    def step(x, index):
        t = jnp.full((x.shape[0],), 1.0 - index / steps, dtype=jnp.float32)
        velocity = flow.velocity(params, context, x, t, frozen=frozen)
        return x - velocity / steps, x

    final, path = jax.lax.scan(step, noise, jnp.arange(steps, dtype=jnp.float32))
    return jax.lax.stop_gradient(final), jax.lax.stop_gradient(path)


def native_sample(flow, params, context, seed: int, steps: int = 10, *, initial_noise=None):
    import jax
    import jax.numpy as jnp

    if not hasattr(flow, "_diagnostic_native_sample_jits"):
        flow._diagnostic_native_sample_jits = {}
    cache_key = (steps, initial_noise is not None)
    if cache_key not in flow._diagnostic_native_sample_jits:
        if initial_noise is None:
            fun = lambda p, f, c, key: flow.sample(p, c, key, steps, frozen=f)
        else:
            fun = lambda p, f, c, noise: _sample_from_noise(flow, p, f, c, noise, steps)
        flow._diagnostic_native_sample_jits[cache_key] = jax.jit(fun)
    argument = (
        jax.random.key(seed) if initial_noise is None else jnp.asarray(initial_noise, jnp.float32)
    )
    return flow._diagnostic_native_sample_jits[cache_key](params, flow.frozen, context, argument)


def capture_sample(
    flow,
    params,
    context,
    seed: int,
    metadata: dict,
    *,
    steps=10,
    knockout="none",
    keys=None,
    initial_noise=None,
):
    """Native ten-step sampler with attention [step,layer,batch,head,query,key]."""
    import jax
    import jax.numpy as jnp

    capture = _Capture(key_keep=key_keep_mask(metadata, knockout, keys))

    def sample(p, frozen, context, key):
        with capture.intercept():
            if initial_noise is None:
                return flow.sample(p, context, key, steps, frozen=frozen)
            return _sample_from_noise(flow, p, frozen, context, key, steps)

    cache_key = (
        steps,
        key_keep_mask(metadata, knockout, keys).tobytes(),
        initial_noise is not None,
    )
    if not hasattr(flow, "_diagnostic_sample_jits"):
        flow._diagnostic_sample_jits = {}
    if cache_key not in flow._diagnostic_sample_jits:
        flow._diagnostic_sample_jits[cache_key] = (jax.jit(sample), capture)
    sample_jit, capture = flow._diagnostic_sample_jits[cache_key]
    capture.layers.clear()
    capture.suffix_features.clear()
    argument = (
        jax.random.key(seed) if initial_noise is None else jnp.asarray(initial_noise, jnp.float32)
    )
    actions, path = sample_jit(params, flow.frozen, context, argument)
    jax.block_until_ready(actions)
    jax.effects_barrier()
    attention = capture.stacked(steps * metadata["layers"])
    attention = attention.reshape((steps, metadata["layers"], *attention.shape[1:]))
    return np.asarray(actions), np.asarray(path), attention


def sample_with_intervention(
    flow,
    params,
    context,
    key,
    metadata: dict,
    *,
    steps=10,
    knockout="none",
    keys=None,
    initial_noise=None,
):
    """Efficient native sampler with an actual mask intervention and no export.

    ``key`` comes from the caller, preserving native evaluation's per-episode
    PRNGKey(seed+episode) followed by one split per decision. No implicit seed
    reset or alternate latent-noise generator is used here.
    """
    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    keep = key_keep_mask(metadata, knockout, keys)
    capture = _Capture(key_keep=keep, capture_features=False)

    def sample(p, frozen, context, sampling_key):
        # No softmax tap or debug callback is enabled in a closed-loop policy.
        with nn.intercept_methods(capture.interceptor):
            if initial_noise is None:
                return flow.sample(p, context, sampling_key, steps, frozen=frozen)
            return _sample_from_noise(flow, p, frozen, context, sampling_key, steps)

    cache_key = (steps, keep.tobytes(), initial_noise is not None)
    if not hasattr(flow, "_diagnostic_intervention_jits"):
        flow._diagnostic_intervention_jits = {}
    if cache_key not in flow._diagnostic_intervention_jits:
        flow._diagnostic_intervention_jits[cache_key] = jax.jit(sample)
    argument = key if initial_noise is None else jnp.asarray(initial_noise, jnp.float32)
    return flow._diagnostic_intervention_jits[cache_key](params, flow.frozen, context, argument)


def load_observation(path: str | Path) -> dict:
    """Load safe NPZ bank schema; images must already have policy orientation."""
    with np.load(path, allow_pickle=False) as a:
        prompt = (
            str(a["prompt"].item())
            if "prompt" in a
            else "pick up the alphabet soup and place it in the basket"
        )
        return {
            "observation/image": a["image"],
            "observation/wrist_image": a["wrist_image"],
            "observation/state": a["state"],
            "prompt": prompt,
        }


def synthetic_observation(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observation/image": rng.integers(0, 256, (224, 224, 3), np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), np.uint8),
        "observation/state": np.zeros(8, np.float32),
        "prompt": "pick up the alphabet soup and place it in the basket",
    }


class Pi05VLMReference:
    """Raw-base image/text-only probe, with NO LIBERO action/state adapter.

    Images use native OpenPI resolution/range/camera names. Text uses the
    checkpoint's PaliGemma vocabulary with BOS/newline and no discrete robot
    state. Omitting state is a declared representation intervention: there is
    no defensible mapping of LIBERO proprioception into the raw-base robot
    contract. This class intentionally has no velocity/sample/decode methods.
    """

    @classmethod
    def from_checkpoint(cls, checkpoint):
        import jax.numpy as jnp
        from flax import nnx
        from openpi.models import model as model_module
        from openpi.models.pi0_config import Pi0Config
        from openpi.models.tokenizer import PaligemmaTokenizer

        from rl_vla.rldt_pi05 import is_expert_parameter

        checkpoint = Path(checkpoint).resolve()
        config = Pi0Config(pi05=True)
        model = config.load(model_module.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
        model.eval()
        graphdef, params, frozen = nnx.split(
            model, nnx.All(nnx.Param, lambda path, _: is_expert_parameter(path)), ...
        )
        self = cls()
        self.graphdef, self.params, self.frozen = graphdef, params, frozen
        self.horizon = 0  # no action contract is exposed
        self.checkpoint = checkpoint
        self.tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)
        self.metadata = {
            "model": "pi05_base",
            "role": "image_text_only_representation_reference",
            "action_contract": "N/A — incompatible action contract",
            "checkpoint": str(checkpoint),
            "state_tokens": "omitted: no valid LIBERO-to-raw-base embodiment normalization",
            "tokenizer": "native PaliGemma BOS + cleaned instruction + newline; no discrete state",
            "checkpoint_metadata_sha256": hashlib.sha256(
                (checkpoint / "params/_METADATA").read_bytes()
            ).hexdigest(),
            "checkpoint_manifest_sha256": hashlib.sha256(
                (checkpoint / "params/manifest.ocdbt").read_bytes()
            ).hexdigest(),
        }
        return self

    def _transform_observation(self, raw):
        from openpi.transforms import ResizeImages

        payload = raw.get("openpi", raw)
        image = np.asarray(payload["observation/image"])
        wrist = np.asarray(payload["observation/wrist_image"])
        if image.dtype != np.uint8 or wrist.dtype != np.uint8:
            raise ValueError("Raw-base reference requires explicit uint8 RGB images")
        ids, mask = self.tokenizer.tokenize(str(payload["prompt"]), state=None)
        result = {
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": np.zeros(32, np.float32),
            "tokenized_prompt": ids,
            "tokenized_prompt_mask": mask,
        }
        return ResizeImages(224, 224)(result)

    def _encode_prefix(self, params, frozen, observation):
        # Native prefix contains no action execution and ignores the placeholder
        # numeric state; the pooled critic tail is not used by reference probes.
        from rl_vla.rldt_pi05 import Pi05Flow

        return Pi05Flow._encode_prefix(self, params, frozen, observation)
