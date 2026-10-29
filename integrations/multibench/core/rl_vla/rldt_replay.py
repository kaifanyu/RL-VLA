"""Trajectory replay and finite-window TD(lambda), preserving reset boundaries."""

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np


def lambda_returns(rewards, discounts, next_values, trace_continues, td_lambda=.5):
    """discount=gamma**executed_steps*(1-terminated); traces stop on any boundary.

    A truncated trajectory bootstraps its actual last observation but never uses
    the following reset episode's return. The last row is always a trace boundary.
    """
    trace_continues = jnp.asarray(trace_continues).at[-1].set(0)

    def step(future, inputs):
        reward, discount, value, continues = inputs
        mixing = td_lambda * continues
        result = reward + discount * ((1 - mixing) * value + mixing * future)
        return result, result

    _, result = jax.lax.scan(
        step, jnp.zeros_like(rewards[0]),
        (rewards, discounts, next_values, trace_continues), reverse=True,
    )
    return result


class TrajectoryReplay:
    """Store raw uint8 observations on CPU, never full VLM KV caches.

    Capacity is in decisions; eviction is by complete collected segment. Segments
    end at reset or collection-window boundaries. Balance nonzero/zero-return
    segments, falling back to the available pool when either is empty.
    """

    def __init__(self, capacity, seed=0):
        self.capacity = int(capacity)
        self.segments = deque()
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, segment):
        if not segment:
            return
        if len(segment) > self.capacity:
            raise ValueError("Replay capacity must fit a collection segment")
        self.segments.append(segment)
        self.size += len(segment)
        while self.size > self.capacity:
            self.size -= len(self.segments.popleft())

    def sample(self, count, *, balanced=True, fresh=(), fresh_fraction=.33):
        if not self.segments:
            raise ValueError("Cannot sample empty replay")
        selected = []
        if count < 1 or not 0 <= fresh_fraction <= 1:
            raise ValueError("count must be positive and fresh_fraction must be in [0,1]")
        if fresh:
            n = min(count, max(int(fresh_fraction > 0), round(count * fresh_fraction)))
            selected.extend(fresh[i] for i in self.rng.integers(len(fresh), size=n))
        fresh_ids = {id(segment) for segment in fresh}
        pool = [segment for segment in self.segments if id(segment) not in fresh_ids]
        pool = pool or list(self.segments)
        positive = [s for s in pool if any(row["reward"] != 0 for row in s)]
        zero = [s for s in pool if not any(row["reward"] != 0 for row in s)]
        # Randomize the unmatched stratum for odd counts (default: one old
        # segment). Always starting positive would exclude old failures forever.
        offset = int(self.rng.integers(2))
        for i in range(count - len(selected)):
            source = ((positive if (i + offset) % 2 == 0 else zero) or pool) if balanced else pool
            selected.append(source[int(self.rng.integers(len(source)))])
        return selected
