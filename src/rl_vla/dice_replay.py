"""Explicit chunk-time returns (paper Eq. 4) and bounded episode n-step replay.

No reward is repeated through overlapping windows. This deliberately fixes the
official online buffer's sliding-window n-step summation; see fidelity report.
"""

import numpy as np


def chunk_reward(rewards, gamma, mode="primitive", reward_scale=1.0):
    if mode not in ("primitive", "chunk"):
        raise ValueError("discount_mode must be primitive or chunk")
    return float(
        sum((gamma**i if mode == "primitive" else 1.0) * r for i, r in enumerate(rewards))
        * reward_scale
    )


def chunk_discount(gamma, duration, ended, mode="primitive"):
    if duration < 1:
        raise ValueError("Empty chunks are invalid")
    return 0.0 if ended else gamma ** (duration if mode == "primitive" else 1)


def finish_episode(records, n_step=1):
    """Preserve one-chunk MC calibration, then aggregate up to n chunks for TD.

    Records belong to exactly one completed finite episode. Budget stops must
    be discarded by the caller, not assigned false zero-return MC labels.
    """
    from rl_vla.dice_core import discounted_returns

    if not records or records[-1]["discounts"] != 0:
        raise ValueError("MC labels require a completed finite episode")
    if n_step < 1:
        raise ValueError("n_step must be positive")
    returns = discounted_returns([r["rewards"] for r in records], [r["discounts"] for r in records])
    output = []
    for i, record in enumerate(records):
        reward, discount, last = 0.0, 1.0, record
        for last in records[i : i + n_step]:
            reward += discount * last["rewards"]
            discount *= last["discounts"]
            if discount == 0:
                break
        output.append(
            {
                **record,
                "rewards": reward,
                "discounts": discount,
                "next_observation": last["next_observation"],
                "returns": returns[i],
            }
        )
    return output


def sample_rows(demos, online, batch_size, ratio, rng, *, fixed_split=False):
    """Official mixed replay floors B*r; legacy Bernoulli mode remains explicit."""
    if fixed_split:
        demo_count = int(batch_size * ratio) if demos else 0
        pools = [demos] * demo_count + [online] * (batch_size - demo_count)
    else:
        pools = [demos if demos and rng.random() < ratio else online for _ in range(batch_size)]
    if any(not len(pool) for pool in pools):
        raise ValueError("Requested replay source is empty")
    return [pool[int(rng.integers(len(pool)))] for pool in pools]


def action_mask(action_dim, executed_steps, execute_steps):
    mask = np.zeros(action_dim, np.float32)
    mask[: executed_steps * (action_dim // execute_steps)] = 1
    return mask
