# RoboTwin 2.0: handover_block

This adapter uses the official `RoboTwin-Platform/RoboTwin` `RLinf_support` branch,
pinned to `dca9ec682688821c42944eacd7f5dd9bfb15396f`. It runs the original
`handover_block` task in the official `demo_clean` Aloha-AgileX embodiment.
The fork's supported `mplib` planner avoids CuRobo CUDA compilation while retaining
the native qpos controller and TOPP interpolation. The simulator lives in
`.venv-robotwin` (Python 3.10), separate from all existing experiment runtimes.

Task prompt: “Use the left arm to grasp the red block on the table, hand it to the
right arm, and place it on the blue pad.” The horizon is the official task's
800 controller actions. Success is the original task predicate: the block's
functional point must be within 3 cm in x/y and 1 cm in z of the pad's functional
point and the right gripper must be open. The predicate itself does not check the
history of the handover.

`adapter.make_env()` exposes a Gymnasium environment. Reset uses the supplied
seed exactly, with no expert solvability filtering and no silent seed substitution.
The campaign owns the frozen training/validation/test seed lists. Invalid reset
seeds raise. The environment does not auto-reset.
Official RoboTwin evaluation normally runs the scripted expert first and skips
failed planning/execution seeds. This campaign deliberately uses fixed,
unfiltered seeds shared by all three methods. A valid reset does not establish
expert solvability, so its scores are custom paired Aloha-clean evaluations,
not a reproduction of the official expert-filtered benchmark score. The
training archive's 50 successful demonstration seeds are not evaluation seeds.

The 14-dimensional native state/action order is left six arm joints, left
gripper, right six arm joints, right gripper. Joint values are radians; grippers
use 0 closed, 1 open. Upstream proprioception contains joint drive targets.
Actions are absolute joint targets passed to the unmodified `take_action(qpos)`.
Action bounds come from the robot URDF, plus [0, 1] for grippers. One controller
action can take multiple 1/250-second physics steps under native TOPP interpolation.
It must not be equated to one simulator physics step or one LIBERO step.

Head, left wrist, and right wrist images are native 320×240 RGB uint8, with no
image flip. Observation keys use the campaign's `state` + `openpi` schema.
`openpi` contains `observation/state`, `observation/image`,
`observation/wrist_image`, `observation/right_wrist_image`, and `prompt`.
Physics runs on CPU; SAPIEN rendering requires GPU/Vulkan access.
The bundled OIDN CUDA denoiser does not support this Blackwell GPU. The adapter
disables denoising while preserving upstream ray tracing, 32 samples per pixel,
and path depth 8. This renderer compatibility change applies identically to all
three methods and leaves the physical task and success predicate unchanged.
The bright white table and pale reflected robot silhouettes also appear in
the official demonstration images; they are not missing robot geometry.
`smoke/official_demo_{head,left_wrist,right_wrist}.png` are decoded first frames
from the official training archive for visual comparison. The observed
compatibility difference is Monte Carlo noise without the OIDN pass.

`policy.data_transforms()` loads the exact Aloha transform implementation from
the pinned existing RLinf checkout. It uses `adapt_to_pi=True`, matching RLinf's
`pi05_aloha_robotwin` config: joint sign flips, the separate official state/action
gripper conversions, and delta actions for the twelve arm joints with absolute
gripper actions. Model tokenization/quantile normalization are supplied by the
campaign policy wrapper.

`norm_stats.json` is computed from the official task's 50 clean Aloha-AgileX
training demonstrations, without loading or changing any model weights.
It contains exact 1%/99% quantiles for the transformed states and 50-action
chunks, with terminal action padding. `normalization_manifest.json` records
the dataset revision, hash, frame count, demonstrated seeds, transform source
hash, and action transform roundtrip error. These statistics calibrate the
unchanged `pi05_base` policy and are shared by RLDT and Flow-SDE. This is
demonstration-based normalization, not SFT and not a task-specific checkpoint.

Reproduce installation/calibration with `bash integrations/multibench/robotwin/setup.sh`.
The complete Python package lock is `requirements.lock`. Only the Aloha-AgileX
files are extracted from the verified 220 MB official embodiment archive.
Two object-index JSON files are also extracted from the verified 3.74 GB object
archive because upstream imports them unconditionally. The selected clean task
uses procedural block/table/wall geometry; object meshes and randomized
background assets are not required. Other tasks or randomized conditions would
need their additional assets.

Run the real reset/render/controller/truncation/predicate smoke under Slurm:

```bash
export PYTHONPATH="$PWD/integrations:$PWD/third_party/robotwin"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-robotwin/bin/python -m multibench.robotwin.smoke \
  --output integrations/multibench/robotwin/smoke
```

The report's `goal_predicate_fixture` is a simulator predicate unit check with a
manually placed block, not a successful policy rollout. Measured policy success
must come from the campaign baseline/training evaluations.

Sources: [official task](https://robotwin-platform.github.io/doc/tasks/handover_block.html),
[official installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html),
[RoboTwin source](https://github.com/RoboTwin-Platform/RoboTwin/tree/dca9ec682688821c42944eacd7f5dd9bfb15396f),
[official assets and training demonstrations](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/3dc3b798668feb99ac61cc9086d84cbcc3d79186).
