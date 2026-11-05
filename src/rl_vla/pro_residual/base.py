"""Frozen OpenPI pi05_libero base policy, loaded in-process.

Actions come from OpenPI's own ``create_trained_policy(...).infer(obs, noise=...)``,
the same call ``scripts/serve_baseline.py`` makes for the robustness queue, with the
same explicit noise protocol (``numpy-pcg64-float32-standard-normal-v1``: a
[action_horizon, 32] float32 standard normal from ``default_rng(seed)``). The
checkpoint's input transforms normalize state/images and its output transforms
unnormalize actions to LIBERO controller coordinates, so returned chunks are
already in environment units ([10, 7]).

Frozen features for the residual actor are a separate jitted pass that reuses the
same model object and input transform: masked mean of the final VLM prefix
hidden states (PaliGemma width 2048). Nothing here is differentiated or updated;
JAX arrays are converted to NumPy before reaching PyTorch.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

NOISE_PROTOCOL = "numpy-pcg64-float32-standard-normal-v1"
JAX_NOISE_PROTOCOL = "jax-prngkey-split-normal-v1"  # The stream of rl_vla.rldt.evaluate.


class FrozenPi05Base:
    def __init__(self, checkpoint: str | Path, model_config: str = "pi05_libero", with_features: bool = True,
                 noise_protocol: str = NOISE_PROTOCOL):
        import jax
        from flax import nnx
        from openpi import transforms
        from openpi.policies.policy_config import create_trained_policy
        from openpi.training.config import get_config

        self.noise_protocol = noise_protocol
        self.checkpoint = Path(checkpoint).resolve()
        if not (self.checkpoint / "params").is_dir() or not (self.checkpoint / "assets").is_dir():
            raise FileNotFoundError(f"Expected OpenPI JAX checkpoint with params/ and assets/: {self.checkpoint}")
        started = time.monotonic()
        self.train_config = get_config(model_config)
        self.model_config_name = model_config
        self.policy = create_trained_policy(self.train_config, self.checkpoint)
        self.load_seconds = time.monotonic() - started
        self.noise_shape = (self.train_config.model.action_horizon, self.train_config.model.action_dim)
        model = self.policy._model
        self._graphdef, self._state = nnx.split(model)
        self.with_features = with_features
        self._features_jit = jax.jit(self._features)
        self._set_normalization(self.policy._input_transform, self.policy._output_transform)
        from openpi.models import gemma

        width = gemma.get_config(self.train_config.model.paligemma_variant).width
        self.feature_dim = int(width) if with_features else 0
        self.calls = 0

    def _set_normalization(self, input_transform, output_transform) -> None:
        from openpi import transforms

        self.normalize = [t for t in input_transform.transforms if isinstance(t, transforms.Normalize)]
        self.unnormalize = [t for t in output_transform.transforms if isinstance(t, transforms.Unnormalize)]
        if len(self.normalize) != 1 or len(self.unnormalize) != 1:
            raise RuntimeError("Expected exactly one Normalize and one Unnormalize transform")
        stats = self.normalize[0].norm_stats["state"]
        self.use_quantiles = bool(self.normalize[0].use_quantiles)
        if not self.use_quantiles:
            raise RuntimeError("pi05_libero is expected to use quantile normalization")
        self.state_q01 = np.asarray(stats.q01, np.float32).copy()
        self.state_q99 = np.asarray(stats.q99, np.float32).copy()

    # -- inference ---------------------------------------------------------------------
    def noise(self, seed: int) -> np.ndarray:
        return np.random.default_rng(int(seed)).standard_normal(self.noise_shape).astype(np.float32)

    def noise_stream(self, seed: int):
        """Per-episode noise chain matching rl_vla.rldt.evaluate, or None for keyed NumPy noise."""
        if self.noise_protocol != JAX_NOISE_PROTOCOL:
            return None
        import jax

        key = jax.random.PRNGKey(int(seed))

        def next_noise() -> tuple[np.ndarray, int]:
            nonlocal key
            key, subkey = jax.random.split(key)
            noise = np.asarray(jax.random.normal(subkey, (1, *self.noise_shape)), np.float32)[0]
            return noise, int.from_bytes(np.asarray(subkey, np.uint32).tobytes(), "little")

        return next_noise

    def infer(self, openpi_obs: dict, noise_seed: int,
              noise: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Return (actions [H,7] in env units, features [2048] or None, timing)."""
        started = time.perf_counter()
        result = self.policy.infer(dict(openpi_obs), noise=self.noise(noise_seed) if noise is None else noise)
        actions = np.array(result["actions"], dtype=np.float32, copy=True)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError(f"Invalid base action chunk {actions.shape}")
        action_ms = (time.perf_counter() - started) * 1000
        features = self.features(openpi_obs) if self.with_features else None
        self.calls += 1
        return actions, features, {"action_ms": action_ms, "total_ms": (time.perf_counter() - started) * 1000,
                                   "model_infer_ms": float(result.get("policy_timing", {}).get("infer_ms", np.nan))}

    def _features(self, state, observation):
        import jax.numpy as jnp
        from flax import nnx
        from openpi.models import model as model_module
        from openpi.models.pi0 import make_attn_mask

        model = nnx.merge(self._graphdef, state)
        observation = model_module.preprocess_observation(None, observation, train=False)
        tokens, mask, autoregressive = model.embed_prefix(observation)
        positions = jnp.cumsum(mask, axis=1) - 1
        (prefix, _), _ = model.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, autoregressive),
                                             positions=positions)
        weights = mask.astype(jnp.float32)[..., None]
        return jnp.sum(prefix.astype(jnp.float32) * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)

    def features(self, openpi_obs: dict) -> np.ndarray:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_module

        inputs = self.policy._input_transform(jax.tree.map(lambda x: x, dict(openpi_obs)))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        pooled = self._features_jit(self._state, model_module.Observation.from_dict(inputs))
        features = np.asarray(pooled, np.float32)[0]
        if not np.isfinite(features).all():
            raise ValueError("Non-finite frozen features")
        return features

    # -- frozen-ness evidence ----------------------------------------------------------
    def parameter_fingerprint(self) -> dict:
        """SHA-256 over every model array (bfloat16 bytes), in sorted path order."""
        from flax import nnx

        return self._fingerprint(nnx.state(self.policy._model).flat_state())

    @staticmethod
    def _fingerprint(flat) -> dict:
        digest, count, leaves = hashlib.sha256(), 0, 0
        for path in sorted(flat, key=lambda p: tuple(map(str, p))):
            value = np.asarray(flat[path].value)
            digest.update("/".join(map(str, path)).encode())
            digest.update(str(value.dtype).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
            count += value.size
            leaves += 1
        return {"sha256": digest.hexdigest(), "elements": int(count), "leaves": leaves}

    def norm_stats_fingerprint(self) -> dict:
        def hash_stats(stats):
            digest = hashlib.sha256()
            for key in sorted(stats):
                for field in ("mean", "std", "q01", "q99"):
                    value = getattr(stats[key], field, None)
                    if value is not None:
                        digest.update(f"{key}.{field}".encode())
                        digest.update(np.ascontiguousarray(np.asarray(value, np.float64)).tobytes())
            return digest.hexdigest()

        assets = self.checkpoint / "assets"
        files = [(str(p.relative_to(assets)), hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in sorted(assets.rglob("*")) if p.is_file()]
        return {"input_normalize": hash_stats(self.normalize[0].norm_stats),
                "output_unnormalize": hash_stats(self.unnormalize[0].norm_stats),
                "asset_files": hashlib.sha256(repr(files).encode()).hexdigest()}

    def metadata(self) -> dict:
        return {"checkpoint": str(self.checkpoint), "model_config": self.model_config_name,
                "noise_protocol": self.noise_protocol, "noise_shape": list(self.noise_shape),
                "action_horizon": int(self.train_config.model.action_horizon),
                "inference_api": "openpi.policies.policy_config.create_trained_policy(...).infer(obs, noise=...)",
                "feature_extractor": ("masked mean final VLM prefix hidden state (same model, same input "
                                      "transforms), computed once per base inference" if self.with_features else None),
                "feature_dim": self.feature_dim, "load_seconds": self.load_seconds,
                "state_quantile_norm": {"q01": self.state_q01[:8].tolist(), "q99": self.state_q99[:8].tolist()}}


class FrozenPi05Flow(FrozenPi05Base):
    """The frozen base through rl_vla.rldt_pi05.Pi05Flow, the inference path of RLDT and DICE-RL.

    Pi05Flow computes the action expert in float32 where ``policy.infer`` uses bfloat16, so the
    two paths differ numerically. ``noise`` passed to ``infer`` is a JAX key, consumed exactly as
    in rl_vla.rldt.evaluate. Features are the same masked-mean VLM prefix hidden state.
    """

    def __init__(self, checkpoint: str | Path, model_config: str = "pi05_libero", with_features: bool = True,
                 noise_protocol: str = JAX_NOISE_PROTOCOL, ode_steps: int = 10):
        import jax

        from rl_vla.rldt_pi05 import Pi05Flow

        if noise_protocol != JAX_NOISE_PROTOCOL:
            raise ValueError(f"The pi05_flow backend requires noise_protocol = {JAX_NOISE_PROTOCOL!r}")
        self.noise_protocol = noise_protocol
        self.checkpoint = Path(checkpoint).resolve()
        started = time.monotonic()
        self.flow = flow = Pi05Flow.from_config({"checkpoint": str(self.checkpoint), "model_config": model_config,
                                                 "projection_only": False})
        self.load_seconds = time.monotonic() - started
        self.model_config_name = model_config
        self.with_features = with_features
        self.ode_steps = int(ode_steps)
        self.noise_shape = (flow.horizon, flow.action_dim)
        self._sample = jax.jit(lambda params, context, key, frozen: flow.sample(
            params, context, key, self.ode_steps, frozen=frozen)[0])
        self._sample_explicit = jax.jit(lambda params, context, noise, frozen: flow.sample_with_noise(
            params, context, noise, self.ode_steps, frozen=frozen)[0])
        self._set_normalization(flow._input_transform, flow._output_transform)
        self.feature_dim = flow.critic_feature_dim - flow.action_dim if with_features else 0
        self.calls = 0

    def noise_stream(self, seed: int):
        import jax

        key = jax.random.PRNGKey(int(seed))

        def next_noise():
            nonlocal key
            key, subkey = jax.random.split(key)
            return subkey, int.from_bytes(np.asarray(subkey, np.uint32).tobytes(), "little")

        return next_noise

    def infer(self, openpi_obs: dict, noise_seed: int, noise=None) -> tuple[np.ndarray, np.ndarray | None, dict]:
        if noise is None:
            raise ValueError("The pi05_flow backend needs a JAX key from noise_stream")
        started = time.perf_counter()
        flow = self.flow
        context = flow.prepare([openpi_obs])
        sample = self._sample_explicit if getattr(noise, "ndim", 1) == 3 else self._sample
        actions = flow.decode(sample(flow.params, context, noise, flow.frozen), [openpi_obs])[0]
        features = None
        if self.with_features:
            features = np.asarray(context["features"][0, :self.feature_dim], np.float32)
            if not np.isfinite(features).all():
                raise ValueError("Non-finite frozen features")
        self.calls += 1
        elapsed = (time.perf_counter() - started) * 1000
        return actions, features, {"action_ms": elapsed, "total_ms": elapsed, "model_infer_ms": float("nan")}

    def features(self, openpi_obs: dict) -> np.ndarray:
        return np.asarray(self.flow.prepare([openpi_obs])["features"][0, :self.feature_dim], np.float32)

    def parameter_fingerprint(self) -> dict:
        """Same convention as the base class: float32 expert weights return to their bfloat16 bytes."""
        import jax.numpy as jnp
        from flax import nnx

        flat = nnx.state(self.flow._merge_model(self.flow.params, self.flow.frozen)).flat_state()
        return self._fingerprint({path: type("Leaf", (), {"value": leaf.value.astype(jnp.bfloat16)})
                                  for path, leaf in flat.items()})

    def metadata(self) -> dict:
        return {"checkpoint": str(self.checkpoint), "model_config": self.model_config_name,
                "backend": "pi05_flow", "noise_protocol": self.noise_protocol,
                "noise_shape": list(self.noise_shape), "ode_steps": self.ode_steps,
                "action_horizon": int(self.flow.horizon),
                "inference_api": "rl_vla.rldt_pi05.Pi05Flow.prepare/sample/decode",
                "feature_dim": self.feature_dim, "load_seconds": self.load_seconds}
