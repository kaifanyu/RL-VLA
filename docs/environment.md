# Connect your task

For the included LIBERO reference adapter, see [LIBERO setup](libero.md). The contract below is for your own task.

Provide a single Gymnasium environment through an importable function:

```toml
[env]
kind = "external"
factory = "my_robot.adapter:make_env"
[env.kwargs]
task_id = "my_task"
domain_split = "train"
```

Install your task package in the learner environment. `make_env(**kwargs)` returns a `gymnasium.Env` with a finite floating `Box(shape=(D,))` action space. Each `step(action)` executes **one controller command** (possibly several physics substeps). The RL wrapper owns action chunks and episode resets; do not add an autoreset wrapper.

## Observation contract

`reset(seed=..., options=...)` returns `(observation, info)`. `step` returns `(observation, scalar_reward, terminated, truncated, info)`. Observations are dictionaries:

```python
observation = {
    "state": proprioception.astype("float32"),  # flat, deployable
    "features": frozen_features,               # optional flat deployable features
    "critic_state": simulator_state,            # optional flat privileged state
    "openpi": {                                # checkpoint-specific payload
        "observation/image": external_rgb,
        "observation/wrist_image": wrist_rgb,
        "observation/state": checkpoint_state,
        "prompt": task_instruction,
    },
}
info = {"is_success": verified_task_completion}
```

Keep dimensions fixed across resets and domain splits. `features` must include any additional deployment information the residual needs (for example a fixed language embedding). Include relevant domain parameters in `critic_state` if the privileged critic needs them. Never put privileged object poses into actor `state` unless available at deployment.

For the `pi05_libero` example, use the [exact observation and action mapping](openpi.md#4-match-the-checkpoints-observation-and-action-contract). The generic adapter does not rotate, resize, normalize, or reinterpret your robot automatically. The server applies checkpoint normalization; `chunk.state_mean/state_std` only scale the residual's proprioceptive input.

## Actions and rewards

OpenPI output and environment actions must share units, coordinate frame, motor order and control rate. `chunk.residual_scale` specifies maximum additive corrections in those units. Quaternion or absolute-pose actions need a task-specific composition adapter; simple elementwise addition is intended for compatible scalar/joint/delta controller coordinates. Validate the base at zero residual first.

`terminated=True` means an absorbing success/failure. `truncated=True` means a collection time limit with a valid continuing-state bootstrap. Return the **last real observation before reset**, including at a time limit. If time expiry is defined as task failure instead, terminate and expose remaining time to the policy.

Use simulator-verified completion for `info['is_success']` and a first-success reward. For manipulation, verify all task conditions such as correct object, release and stability. Optional potential shaping must use the same primitive `gamma` as the learner and zero potential at true terminal states. The task defines all rewards; the chunk wrapper only discounts and aggregates them.

## Adapter check

```python
from rl_vla.config import load_config
from rl_vla.runtime import build_env
import numpy as np

env = build_env(load_config("configs/my_task.toml"))
try:
    decision, info = env.reset(seed=0)
    decision, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_dim, dtype=np.float32)
    )
    print(info["executed_steps"], info["discount"], info["executed_actions"])
finally:
    env.close()
```

The wrapper returns exact executed commands, per-command rewards, requested normalized residual, projection frequency and the real next decision context. Record these plus raw observations in a task-level episode recorder when collecting data for later distillation. Run frozen-base evaluation before starting either learner.

## MuJoCo Warp extension

Put task assets, resets, camera rendering, controller, randomization and reward checks in your task package. This baseline can call a single-world adapter immediately. A high-throughput Warp implementation additionally needs batched base inference and a vector collector; the current single-example websocket path is not that optimization. Compare controller/physics/rendering behavior against a known task implementation before treating a port as the original benchmark.
