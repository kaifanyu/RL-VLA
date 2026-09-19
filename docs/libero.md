# Optional LIBERO reference task

Use this adapter with **`pi05_libero`** when you want an existing robot task instead of writing your own environment. `configs/libero_sac.toml` and `configs/libero_ppo.toml` select LIBERO Spatial task 0. The generic `configs/openpi_*.example.toml` files remain templates for other tasks.

The adapter and mock-backend tests are implemented. **The actual LIBERO simulator, renderer, task assets and GPU policy have not been installed or run here.** This does not establish robot performance or reproduce the official benchmark.

## Simulator installation boundary

Use Linux for the simulator and follow the [upstream OpenPI LIBERO setup](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md) and [LIBERO installation instructions](https://github.com/Lifelong-Robot-Learning/LIBERO#installtion). The upstream Docker workflow is useful for first validating its simulator and policy example independently.

There is a dependency conflict to resolve before using this in-process adapter: the upstream reference simulator recipe uses Python 3.8, Torch 1.11 and NumPy 1.22, while this learner uses Python 3.11+, Torch 2.7+ and NumPy 1.26. **Do not sync that old requirements file into the learner `.venv`.** The simulator adapter runs inside the learner process, so it requires a LIBERO/robosuite/MuJoCo installation that has been validated with the learner's Python and dependencies. A compatible simulator lock is not supplied or verified here; a separate legacy simulator environment alone does not provide that compatibility. The OpenPI policy server remains a separate process as documented in [openpi.md](openpi.md).

The reference source/assets location is the submodule `third_party/openpi/third_party/libero`. Initialize it from the repository root when preparing the simulator:

```bash
git -C third_party/openpi submodule update --init third_party/libero
# After installing compatible simulator dependencies in the learner environment:
uv pip install -e third_party/openpi/third_party/libero
uv pip install -e third_party/openpi/packages/openpi-client
uv pip install 'websockets>=14,<16'
```

LIBERO's initial import may ask where to store its paths. Configure its `config.yaml` (`~/.libero` by default, or `LIBERO_CONFIG_PATH`) so `bddl_files`, `init_states`, and `assets` resolve to the checked-out task resources. Demonstration datasets are separate and are not needed for online rollouts. [LIBERO path configuration source](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/__init__.py).

The adapter reads official initial-state files with a restricted PyTorch loader and scoped NumPy-array support, avoiding the old upstream `torch.load` default incompatibility. It does not download any files.

## Configuration and use

These are the concrete environment settings already used by the LIBERO configs:

```toml
[env]
kind = "external"
factory = "rl_vla.envs.libero:make_env"

[env.kwargs]
task_suite = "libero_spatial"
task_id = 0
max_steps = 220
camera_resolution = 256
resize_size = 224
settle_steps = 10
# initial_state_id = 0  # Optional fixed official reset; omit to sample resets.
```

First confirm the installed simulator can reset and step without a policy:

```bash
python -c "from rl_vla.envs.libero import make_env; e=make_env(); o,i=e.reset(seed=7); print(o['state'].shape, i); print(e.step([0,0,0,0,0,0,-1])[1:4]); e.close()"
```

Then start the `pi05_libero` server using [openpi.md](openpi.md), and run from the repository root:

```bash
python -m rl_vla evaluate --config configs/libero_sac.toml --episodes 5 --output runs/libero_base.json
python -m rl_vla train --config configs/libero_sac.toml --output runs/libero_sac
python -m rl_vla train --config configs/libero_ppo.toml --output runs/libero_ppo
```

These train one task. Use separate runs for other `task_id` values. Match the task horizon to the suite: Spatial 220, Object 280, Goal 300, LIBERO-10 520, LIBERO-90 400, following the [official example](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py). Random reset selection is seeded but samples with replacement. For a full benchmark, explicitly enumerate every official initial-state ID and task; five sampled episodes are only a connection check.

## Observation and reward contract

- Actions are seven normalized OSC pose/gripper commands in `[-1,1]`; they are not meters or radians. Residual scales use these controller coordinates.
- The frozen VLA receives front and wrist RGB images rotated 180 degrees and resized with the official OpenPI helper, an eight-dimensional state, and the task's language prompt.
- State is end-effector position (3), quaternion converted from **xyzw** to axis-angle (3), and gripper joint positions (2). The small residual actor receives this deployable state plus the cached base prefix. It does not independently encode camera pixels or consume object ground truth.
- Reset restores an official initial state and executes ten `[0,0,0,0,0,0,-1]` settling commands. These are reset work, outside the RL reward/step budget; reset reports their count. Success during settling is rejected instead of rewarded.
- The [simulator success predicate](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py) produces a single `+1` and true termination. The selected step limit truncates the episode with its final observation intact. An unexpected native `done` without verified success is also a cutoff. There is no reward shaping.
- The outer residual wrapper handles chunk execution and `gamma**actual_steps` discounts. The adapter never auto-resets.

Run `python -m pytest tests/test_libero_adapter.py` to check these mappings with an injected fake backend. Those tests validate the contract, including modern initial-state deserialization, **not** MuJoCo rendering or a real checkpoint.

The preprocessing protocol follows the Apache-2.0 OpenPI example; its license is retained at `third_party/openpi/LICENSE`. This Gymnasium adapter adds reset, validation and termination handling and is not copied as an upstream benchmark runner.
