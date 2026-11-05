"""Frozen pi0.5 adapter for DICE; reuses Kai's tested OpenPI transforms.

Only construction initializes JAX. Full Gaussian noise is returned alongside
the corresponding prior action, so the residual uses the same latent sample.
"""

import numpy as np
import hashlib
from collections import OrderedDict


class FrozenPi05:
    def __init__(self, policy, settings):
        import jax

        from rl_vla.rldt_pi05 import Pi05Flow

        self.flow = Pi05Flow.from_config(policy)
        self.settings = settings
        self.execute_steps = settings["execute_steps"]
        if not 1 <= self.execute_steps <= self.flow.horizon:
            raise ValueError("execute_steps must fit the pretrained horizon")
        self.feature_dim = self.flow.feature_dim
        self.noise_dim = self.flow.horizon * self.flow.action_dim
        self.action_dim = self.execute_steps * self.flow.motor_dim
        self.key = jax.random.key(settings.get("seed", 0))
        steps = settings["ode_steps"]
        flow = self.flow
        self._sample = jax.jit(
            lambda params, frozen, context, key: flow.sample(
                params, context, key, steps, frozen=frozen
            )
        )
        self._sample_noise = jax.jit(lambda params, frozen, context, noise: flow.sample_with_noise(
            params, context, noise, steps, frozen=frozen))
        self._prefix_cache = OrderedDict()
        self._prefix_cache_bytes = 0
        self.execution_counts = dict(logical_prior_calls=0, candidates=0, sampler_batches=0,
                                     encoder_calls=0, cache_hits=0)
        self.metadata = {
            **self.flow.metadata,
            "method": "dice-pi05-residual",
            "prior_frozen": True,
            "action_coordinates": "normalized-executed-prefix",
            "execute_steps": self.execute_steps,
        }

    def propose(self, raw, count):
        import jax

        context = self._context(raw)
        context = self.flow.repeat_context(context, count)
        self.key, key = jax.random.split(self.key)
        actions, path = self._sample(self.flow.params, self.flow.frozen, context, key)
        self.execution_counts["logical_prior_calls"] += 1
        self.execution_counts["candidates"] += count
        self.execution_counts["sampler_batches"] += 1
        full = np.asarray(actions, np.float32)
        noise = np.asarray(path[0], np.float32).reshape(count, -1)
        features = np.asarray(context["features"], np.float32)
        # Fixed per-state normalization, no running statistics or target leakage.
        features = (features - features.mean(-1, keepdims=True)) / np.sqrt(
            features.var(-1, keepdims=True) + 1e-5
        )
        prior = full[:, : self.execute_steps, :7].reshape(count, -1).copy()
        if not all(np.isfinite(x).all() for x in (features, noise, prior)):
            raise RuntimeError("Non-finite frozen policy output")
        return features, noise, prior, full

    def propose_explicit(self, raw, noise):
        """Diagnostic supplied-noise path; never advances the collection RNG."""
        context = self.flow.repeat_context(self.flow.prepare([raw]), noise.shape[0])
        full, _ = self._sample_noise(self.flow.params, self.flow.frozen, context, noise)
        full = np.asarray(full, np.float32)
        features = np.asarray(context["features"], np.float32)
        features = (features-features.mean(-1, keepdims=True))/np.sqrt(features.var(-1, keepdims=True)+1e-5)
        return features, np.asarray(noise).reshape(noise.shape[0],-1), full[:, :self.execute_steps, :7].reshape(noise.shape[0],-1), full

    def _context(self, raw):
        import jax
        limit = int(self.settings.get("prefix_cache_max_mb", 0)) * 1024**2
        if not limit:
            self.execution_counts["encoder_calls"] += 1
            return self.flow.prepare([raw])
        observation = raw.get("openpi", raw)
        digest = hashlib.sha256()
        for name, value in sorted(observation.items()):
            array = np.asarray(value)
            digest.update(name.encode()); digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode()); digest.update(array.tobytes())
        key = digest.digest()
        if key in self._prefix_cache:
            self.execution_counts["cache_hits"] += 1
            self._prefix_cache.move_to_end(key)
            return self._prefix_cache[key]
        self.execution_counts["encoder_calls"] += 1
        # Store exact frozen context on host, not proposals/noise or learned values.
        context = jax.tree.map(lambda x: np.asarray(x).copy(), self.flow.prepare([raw]))
        size = sum(x.nbytes for x in jax.tree.leaves(context))
        while self._prefix_cache and self._prefix_cache_bytes + size > limit:
            _, previous = self._prefix_cache.popitem(last=False)
            self._prefix_cache_bytes -= sum(x.nbytes for x in jax.tree.leaves(previous))
        if size <= limit:
            self._prefix_cache[key] = context
            self._prefix_cache_bytes += size
        return context

    def propose_many(self, observations, count):
        """Microbatch only frozen inference; retain one original RNG split per row.

        Fresh latent arrays have exactly the serial [K,H,D] key/shape convention.
        Losses, gradients, optimizer updates and reductions remain full batch256.
        """
        import jax
        import jax.numpy as jnp
        rows = int(self.settings.get("proposal_microbatch_rows", 1))
        output = [[], [], []]
        for start in range(0, len(observations), rows):
            contexts, noises = [], []
            for raw in observations[start:start + rows]:
                contexts.append(self._context(raw))
                self.key, key = jax.random.split(self.key)
                noises.append(jax.random.normal(key, (count, self.flow.horizon, self.flow.action_dim)))
            context = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *contexts)
            context = self.flow.repeat_context(context, count)
            noise = jnp.concatenate(noises, axis=0)
            full, _ = self._sample_noise(self.flow.params, self.flow.frozen, context, noise)
            batch = len(contexts)
            full = np.asarray(full, np.float32)
            features = np.asarray(context["features"], np.float32)
            features = (features-features.mean(-1, keepdims=True))/np.sqrt(features.var(-1, keepdims=True)+1e-5)
            output[0].append(features.reshape(batch, count, -1))
            output[1].append(np.asarray(noise, np.float32).reshape(batch, count, -1))
            output[2].append(full[:, :self.execute_steps, :7].reshape(batch, count, -1).copy())
            self.execution_counts["logical_prior_calls"] += batch
            self.execution_counts["candidates"] += batch*count
            self.execution_counts["sampler_batches"] += 1
        return tuple(np.concatenate(x, axis=0) for x in output)

    def decode(self, corrected, full, raw):
        padded = full[None].copy()
        padded[0, : self.execute_steps, :7] = np.asarray(corrected).reshape(self.execute_steps, 7)
        return self.flow.decode(padded, [raw])[0, : self.execute_steps]
