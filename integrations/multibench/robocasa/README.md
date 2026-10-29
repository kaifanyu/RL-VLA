# RoboCasa365: counter to cabinet

This integration runs the real `robocasa/PickPlaceCounterToCabinet` task with
PandaOmron in the official target kitchen split. The policy starts from the raw
`pi05_base` weights. Only normalization statistics are taken from the official
RoboCasa pi05 release; no RoboCasa SFT policy weights are used.

## Reproducible setup

`setup.sh` installs the simulator in `.venv-robocasa365`, checks the pinned source
revisions, and downloads the six official asset bundles. `requirements.lock`,
`assets/manifest.json`, and `assets/norm_stats_provenance.json` record the runtime,
asset hashes, and normalization provenance. The adapter source and environment
are separate from the existing LIBERO experiments.

On this host, glibc 2.28 cannot install the optional LeRobot dataset converter's
`rerun-sdk` wheel. The runtime packaging shim therefore installs the unmodified
official simulator source without that converter or the optional Tianshou speed
benchmark. It also avoids recursively packaging the asset directory. These
optional tools are not needed for simulator reset, control, rendering, task
success, or policy training. The IPC client is installed without its conflicting
NumPy version pin; actual NumPy 2 transport was checked through `ProcessEnv`.

Official sources:

- [RoboCasa installation](https://robocasa.ai/docs/build/html/introduction/installation.html)
- [Policy learning algorithms](https://robocasa.ai/docs/build/html/benchmarking/policy_learning_algorithms.html)
- [RoboCasa simulator](https://github.com/robocasa/robocasa)
- [RoboCasa OpenPI policy reference](https://github.com/robocasa-benchmark/openpi)

## Environment and policy contract

`task_contract.json` records the full contract. The 12 normalized commands are
arm position delta (3), rotation delta (3), gripper (1), base motion (3), torso
(1), and controller mode (1). The native Gym wrapper performs the official
discrete gripper/mode conversion. The state has 16 native proprioceptive values;
quaternions remain quaternions.

All three cameras are used: left agent view, wrist, and right agent view. Native
256-pixel images receive the official single vertical flip and OpenPI bilinear
resize to 224 pixels. The simulator runs at 20 control steps per second with a
750-step task horizon. Success is the native object-inside-cabinet condition with
the gripper away from the object. The prompt names the sampled object, for
example, “Pick the wine from the counter and place it in the cabinet.”

`policy.py` loads the pinned official RoboCasa input/output transforms. Shared
campaign preprocessing then applies mean/std normalization and the raw pi05 base
model contract: 50 predicted actions, 32 padded dimensions, discrete state input.
The simulator uses its 12 native action dimensions.

## Validation

Run on an allocated CPU node with a compatible OSMesa library:

```bash
source integrations/multibench/robocasa/env.sh
export CUDA_VISIBLE_DEVICES=''
.venv-robocasa365/bin/python -m integrations.multibench.robocasa.smoke
```

The completed smoke evidence is in `evidence/smoke/report.json`: two distinct
seeds, a repeated seed with identical initial state/images/prompt, real camera
renders, eight controller steps per reset, and a check of horizon truncation.
`evidence/ipc_smoke.json` additionally validates the actual campaign's isolated
simulator process with a frozen validation seed. It records a 16-value state,
12-value action, all three images, one native control step, and binary reward.

The smoke actions are only integration checks. They are not a measured baseline
success rate. Policy inference, training updates, and task-success evaluation
have separate campaign validation gates.
