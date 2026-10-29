"""Small differentiable flow for CPU pipeline checks; not a pretrained VLA."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class ToyFlow:
    def __init__(self, config):
        self.horizon = int(config.get("horizon", 4))
        self.action_dim = self.motor_dim = 2
        self.frozen = {}
        key = jax.random.PRNGKey(int(config.get("seed", 0)))
        self.params = {
            "w": jax.random.normal(key, (4, self.horizon * 2)) * .01,
            "b": jnp.zeros((self.horizon * 2,)),
        }

    def prepare(self, observations):
        state = jnp.asarray(np.stack([o["state"] for o in observations]))
        return {"state": state, "features": state}

    def repeat_context(self, context, count):
        return jax.tree.map(lambda x: jnp.repeat(x, count, axis=0), context)

    def velocity(self, params, context, x, t, *, frozen=None):
        del frozen
        state = context["state"]
        goal = jnp.tile(.65 * (state[:, 2:] - state[:, :2]), (1, self.horizon))
        goal = goal + state @ params["w"] + params["b"]
        # A smooth affine Gaussian flow, using pi0.5's t=1 noise -> t=0 data.
        goal = goal.reshape(x.shape)
        t = t[:, None, None]
        return (.8 * x - goal) / (.2 + .8 * t)

    def sample(self, params, context, key, steps, *, frozen=None):
        x = jax.random.normal(key, (context["state"].shape[0], self.horizon, 2))

        def step(x, i):
            t = jnp.full((x.shape[0],), 1 - i / steps)
            return x - self.velocity(params, context, x, t, frozen=frozen) / steps, x

        final, path = jax.lax.scan(step, x, jnp.arange(steps))
        return jax.lax.stop_gradient(final), jax.lax.stop_gradient(path)

    def decode(self, actions, observations):
        del observations
        return np.asarray(actions)

    def save(self, output, params):
        path = Path(output)
        path.mkdir(parents=True, exist_ok=False)
        np.savez(path / "actor.npz", **jax.device_get(params))

    def load_params(self, path):
        path = Path(path)
        with np.load(path / "actor.npz" if path.is_dir() else path, allow_pickle=False) as data:
            return {k: jnp.asarray(data[k]) for k in self.params}
