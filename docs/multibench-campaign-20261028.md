# Raw π0.5 base: RoboTwin 2.0, RoboCasa365, and ManiSkill3

This campaign tests the original `pi05_base` policy and adapts the same weights with RLDT or the official RLinf Flow-SDE implementation. It is a raw-base adaptation experiment. Published task-specific SFT policy weights are not used. Benchmark normalization statistics and robot/camera transforms are applied consistently to all methods within each task.

| Benchmark | Native task | Robot / control | Policy observations | Native episode limit |
|---|---|---|---|---:|
| RoboTwin 2.0 | `handover_block` | Aloha-AgileX; 14 absolute joint/gripper targets | 14D state; head and both wrist cameras | 800 controller calls |
| RoboCasa365 | `robocasa/PickPlaceCounterToCabinet` | PandaOmron; 12D end-effector/mobile-base commands | 16D state; left, right, and wrist cameras | 750 controller calls |
| ManiSkill3 + official RLinf task extension | `PutOnPlateInScene25Main-v3` | WidowX; 7D Cartesian/gripper commands | 8D state; third-person camera | 80 controller calls |

Task prompts are passed directly into the policy. RoboTwin uses “Use the left arm to grasp the red block on the table, hand it to the right arm, and place it on the blue pad.” RoboCasa supplies a native prompt for each sampled object and kitchen, such as “Pick the wine from the counter and place it in the cabinet.” ManiSkill supplies the selected object's native instruction, such as “put fast food cup on plate.”

A RoboTwin controller call can perform multiple internal physics steps. Controller calls, policy decisions, and physics steps are different quantities; the reported interaction budgets count controller calls. RoboCasa uses 20 Hz control; ManiSkill uses 5 Hz control with 500 Hz physics.

## Experimental protocol

RoboTwin uses the official `demo_clean` Aloha-AgileX scene with unfiltered fixed reset seeds. Its official evaluation script first filters seeds by expert success; this campaign does not apply that filter. Report these as custom paired task evaluations, not official leaderboard scores. The benchmark's bundled denoiser is unsupported on Blackwell, so rendering retains the official ray-tracing settings with denoising disabled.

Each task has 100 frozen training reset seeds, 50 validation seeds, and 50 test seeds. All methods share the same seeds and native initial-state IDs. These are held-out resets within the same task distribution: RoboCasa uses its official `target` distribution, and ManiSkill uses the official training object pool with disjoint native episode IDs spread across its 802,816 combinations. This does not claim held-out object-category generalization.

Each RL method runs continuously for 100 iterations. Validation occurs at **25, 50, and 100**, with no iteration-75 validation. An iteration collects 32 policy decisions; each decision executes at most five controller calls. The maximum training budget is 3,200 decisions / 16,000 controller calls per run. Early episode termination may reduce the controller total. This is an interaction budget, not 100 episodes.

The original base model architecture is preserved: 50-action prediction horizon, 32 internal action dimensions, discrete state tokens, and 10 ODE denoising steps. Only the first five actions are executed. RLDT and Flow-SDE train the action expert while freezing the vision/language backbone. Both use the same per-task input and output transforms.

RLDT uses the native JAX checkpoint. Flow-SDE uses a verified tensor-by-tensor conversion of that checkpoint into the pinned RLinf Torch model. JAX and Torch random number generators differ, so each backend has its own frozen ODE baseline, evaluated on the same reset states as its adapted policy. Their success rates must be labeled separately.

Each experiment runs baseline validation and baseline test, trains once with synchronous checkpoint validation, then evaluates the checkpoint with the highest validation success count on the test split. Ties choose the earlier iteration. Test results are not used to choose a checkpoint.

Training counters, per-episode outcomes, losses, checkpoints, validation costs, model initialization costs, inference time, simulation time, and Slurm job identity are saved. Full raw RGB training traces are disabled to keep the six-run campaign within available storage. Simulator setup screenshots and test evidence are retained separately.

## Calibration provenance

- RoboTwin: normalization computed from 50 official `handover_block` Aloha-AgileX training demonstrations, with the official Aloha sign/gripper transforms and 50-step delta-action chunks. Demonstrations provide scaling only; no SFT updates are performed.
- RoboCasa: official RoboCasa365 π0.5 preprocessing and mean/std normalization assets. Only normalization assets are taken from the published checkpoint bundle.
- ManiSkill: official RLinf preprocessing and quantile normalization assets for the requested task family. Only normalization assets are taken from the published SFT bundle.

Exact commits, dependency locks, asset checksums, and normalization provenance are in `integrations/multibench/{robotwin,robocasa,maniskill}`. The experiment protocol and frozen seeds are in `runs/multibench_20261028/campaign/tasks`.

## Runtime isolation and execution

The simulator interpreters are `.venv-robotwin`, `.venv-robocasa365`, and `.venv-maniskill3`. They communicate with policy processes through local binary IPC. SAPIEN rendering retains the assigned GPU; RoboCasa uses OSMesa. The adaptation implementation is an isolated snapshot under `integrations/multibench/core` so the existing Object 1/2 runs can continue with their frozen source.

The new runner is `integrations/multibench/run_experiment.py`. It verifies a frozen readiness/source contract before every stage and writes each experiment's `status.json` and stage logs. `integrations/multibench/submit.py` submits six sequential GPU jobs; an individual failure does not prevent the next independent experiment from starting. A failed experiment is never treated as a completed result or automatically overwritten.

Readiness requires actual reset/render/controller/success-predicate checks, preprocessing agreement, shared-noise native/Torch inference checks, and real learner-update smoke tests. Setup smoke tests are software validation and are not baseline success estimates.

## Submitted campaign

Submitted October 28, 2026 (America/New_York). All three environment checks, preprocessing and numerical parity checks, and all six learner-update checks passed before submission. The readiness contract freezes 2,486 files, including the base checkpoint weights and simulator runtime inputs.

| Task | RLDT job | Flow-SDE job |
|---|---:|---:|
| RoboTwin `handover_block` | 1477960 | 1477961 |
| RoboCasa `PickPlaceCounterToCabinet` | 1477962 | 1477963 |
| ManiSkill `PutOnPlateInScene25Main-v3` | 1477964 | 1477965 |

Each job includes its backend's raw-base baseline, continuous training, validation at 25/50/100, and the selected checkpoint's test evaluation. Jobs run in table order, RLDT then Flow-SDE. They request one Blackwell GPU, eight CPUs, and 128 GB RAM, with a four-day wall-time limit per job; this limit is not a runtime estimate.

The temporary setup allocation 1477821 was released after submission. The existing Object 1/2 allocation 1470624 was preserved. The submission receipt is `runs/multibench_20261028/campaign/submission.json`; per-job progress is in `campaign/experiments/<benchmark>_<method>/status.json` and stage logs. Completed experiment metrics are collected in `campaign/results.json` and `campaign/results.csv`.

No task success rates are claimed from the setup checks. Production evaluations supply those results.
