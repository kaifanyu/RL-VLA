# RoboCasa365: counter to cabinet

This integration runs the real `robocasa/PickPlaceCounterToCabinet` task with
PandaOmron in the official target kitchen split. The policy starts from the raw
`pi05_base` weights. Only normalization statistics are taken from the official
RoboCasa pi05 release; no RoboCasa SFT policy weights are used.

## Reproducible setup

`setup.sh` installs the simulator in `.venv-robocasa365`, checks the pinned source
revisions, and downloads the official simulator assets. The installer now uses pinned
Hugging Face publisher revisions because the historical Box links return 404.
`runs/setup/robocasa365/assets_hf_manifest.json` records every downloaded archive
and SHA256. The default downloads all fixtures, Lightwheel objects, Objaverse
objects, and ordinary textures used by native kitchen tasks. Set
`ROBOCASA_OPTIONAL_ASSETS=1` to additionally download AI-generated objects and
textures for optional registry/texture augmentation settings. The historical
`requirements.lock` remains unchanged; the actual installation is recorded in `runs/setup/robocasa365/requirements-installed.txt`.
`assets/norm_stats_provenance.json` records normalization provenance. The adapter
source and environment are separate from the existing LIBERO experiments.

The runtime packaging shim installs the unmodified official simulator source
without the optional LeRobot dataset converter or Tianshou speed benchmark. It
also avoids recursively packaging the asset directory. Policy inference runs in
the separate OpenPI environment. The pinned OpenPI client helpers are loaded
from source through `PYTHONPATH`, because their package metadata requires
NumPy<2 while this simulator requires NumPy 2. The client distribution is not
installed with modified metadata.

All environments, Python interpreters, source checkouts, assets, and caches live
under `/workspace`. Only `/workspace` persists across container reboots. This
host uses OSMesa/llvmpipe software rendering (`ROBOCASA_RENDER_THREADS=4`,
exported as `LP_NUM_THREADS`); the policy can still use a CUDA GPU. The launcher defaults to GPU 2 and CPU cores 16–23.
The real kitchen preflight measured about 0.49 s/control step with four renderer
threads versus 0.79 s/control step with one; reset cost is separate. Set
`ROBOCASA_RENDER_THREADS=1` when running many parallel environments.

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
.venv-robocasa365/bin/python -m integrations.multibench.robocasa.smoke \
  --output runs/local_setup_20261031/robocasa/smoke
```

The smoke report is written to `runs/local_setup_20261031/robocasa/smoke/report.json`.
It checks two distinct seeds, a repeated seed with identical initial
state/images/prompt, real camera renders, eight controller steps per reset, and
native horizon truncation. `runs/local_setup_20261031/robocasa/runtime_inventory.json`
records source revisions, installed packages, the real Mesa renderer probe,
and the pinned client source's NumPy 2 array transport check.
`runs/local_setup_20261031/robocasa/ipc_smoke_native.json` verifies the actual
OpenPI interpreter communicating with the isolated NumPy 2 simulator through
`ProcessEnv`: reset, all three cameras, state/action dimensions, and a native
control step. The baseline recorder below uses that protocol for the full episode.

The smoke actions are only integration checks. They are not a measured baseline
success rate. Policy inference, training updates, and task-success evaluation
have separate campaign validation gates.

## Frozen native baseline video

After the OpenPI environment and native checkpoint are installed, record one
complete task episode with all three policy cameras:

```bash
bash integrations/multibench/robocasa/record_baseline.sh \
  --output runs/local_setup_20261031/robocasa/baseline --seed 1
```

Use a new output directory for each run to preserve the original evidence.
The default checkpoint is `checkpoints/openpi/pi05_base`. The recorder uses
10 flow ODE steps, executes five predicted actions per policy call, and runs to
native success or the 750-controller-step horizon. It writes `baseline.mp4`
(20 FPS, one frame per controller step), initial/final camera mosaics,
`actions_states.npz`, per-decision timing, and `rollout.json` with seeds, native
success, clipping counts, and checkpoint provenance. A single base-policy
rollout checks integration; it is not a task success-rate estimate. Raw pi05
base weights are not RoboCasa task fine-tuned weights.

## Completed local baseline (2026-10-31)

The frozen pretrained `pi05_base` policy completed the full 750-step wine-to-cabinet
attempt with environment seed 1 and policy seed 100000. **Native task success
was false**: the wine remained on the counter. This is a complete baseline
attempt, not a successful task demonstration or a benchmark success-rate estimate.

The verified [video](../../../runs/local_setup_20261031/robocasa/baseline/baseline.mp4)
contains 751 frames at 20 FPS (672×288, 37.55 seconds). Twenty hertz is the virtual
control/video rate; actual wall-clock time was 502.9 seconds. The run recorded
150 real policy calls, finite 750×12 actions and 750×16 states, and 36 clipped
actuator values out of 9,000 (0.4%). Start, midpoint, and final video frames were
visually inspected. Simulation took 335.0 seconds; warm policy inference averaged
0.139 seconds per five-action chunk. The first policy call included 81.3 seconds
of compilation.

The [rollout report](../../../runs/local_setup_20261031/robocasa/baseline/rollout.json)
records checkpoint/source hashes, normalization provenance, seeds, native
termination, and timing. The [validation report](../../../runs/local_setup_20261031/robocasa/baseline/validation.json)
records decoded video frame count/duration, finite action/state checks, SHA256,
and visual review. The same directory contains `actions_states.npz`,
`decisions.jsonl`, initial/final mosaics, and extracted video review frames.

## Wine-only RLDT queue on GPU 7

The wine-specific campaign restricts the native target object sampler with
`obj_groups="wine"` and rejects any reset whose native instruction differs from
“Pick the wine from the counter and place it in the cabinet.” Kitchen and wine
instance variation remain native. The generic integration still defaults to its
original all-object task.

[`wine_campaign.py`](wine_campaign.py) prepares a source snapshot and runs a
sequential, failure-stopping queue: native simulator/sampler/optimizer checks,
frozen raw-pi05-base validation and test, 100 continuous full-action-expert RLDT
iterations, validation at 25/50/100, and validation-selected final test. Each
held-out split has five paired episodes. The native reward and 750-step horizon
are unchanged. The queue binds GPU 7 by UUID and uses CPU OSMesa rendering.

The local launch is recorded in
[`runs/rldt_robocasa_wine_gpu7_20261031/RUN.md`](../../../runs/rldt_robocasa_wine_gpu7_20261031/RUN.md).
Its `queue_status.json` reports the actual stage and any failure; existence of a
queue is not evidence that training or task evaluation has completed. No task
success is claimed until the episode reports are available.
