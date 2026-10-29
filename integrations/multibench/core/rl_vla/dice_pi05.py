"""Frozen pi0.5 adapter for DICE; reuses Kai's tested OpenPI transforms.

Only construction initializes JAX. Full Gaussian noise is returned alongside
the corresponding prior action, so the residual uses the same latent sample.
"""

import numpy as np


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
        self.metadata = {
            **self.flow.metadata,
            "method": "dice-pi05-residual",
            "prior_frozen": True,
            "action_coordinates": "normalized-executed-prefix",
            "execute_steps": self.execute_steps,
        }

    def propose(self, raw, count):
        import jax

        context = self.flow.prepare([raw])
        context = self.flow.repeat_context(context, count)
        self.key, key = jax.random.split(self.key)
        actions, path = self._sample(self.flow.params, self.flow.frozen, context, key)
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

    def decode(self, corrected, full, raw):
        padded = full[None].copy()
        padded[0, : self.execute_steps, :7] = np.asarray(corrected).reshape(self.execute_steps, 7)
        return self.flow.decode(padded, [raw])[0, : self.execute_steps]
