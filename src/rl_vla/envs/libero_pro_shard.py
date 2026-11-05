"""Enumerate an immutable contiguous evaluation slice of a frozen PRO split."""

from rl_vla.envs.libero_pro import make_env as make_frozen_env


def make_env(*, protocol, task_key, split, shard_start, shard_count, technical_max_steps=None):
    if split not in {"validation", "test", "source"}:
        raise ValueError("Sharding is evaluation-only; training cannot be sharded")
    if type(shard_start) is not int or type(shard_count) is not int or shard_start < 0 or shard_count < 1:
        raise ValueError("Shard start/count must be nonnegative/positive integers")
    env = make_frozen_env(protocol=protocol, task_key=task_key, split=split)
    try:
        base = env.unwrapped
        if shard_start + shard_count > len(base.state_ids):
            raise ValueError("Shard exceeds its original frozen split")
        # Only this fresh process's enumeration cursor changes. The protocol,
        # initial-state matrix, reset seeds, controls and simulator are unchanged.
        base.state_ids = tuple(base.state_ids[shard_start:shard_start + shard_count])
        base.evaluation_index = 0
        if technical_max_steps is not None:
            if type(technical_max_steps) is not int or not 1 <= technical_max_steps < base.max_steps:
                raise ValueError("Technical smoke limit must shorten the episode")
            base.max_steps = technical_max_steps
        return env
    except BaseException:
        env.close()
        raise
