# test-box-1: pi0.5 training and simulation

This installation lives under `/workspace/upenn`. Python interpreters, virtual
environments, upstream source, model files, datasets, caches, and outputs are on
the persistent `/workspace` mount. The host driver and native graphics libraries
come from the container image. Installation and validation evidence is in
`runs/local_setup_20261031/` and `/workspace/upenn/setup/trainer/`.
All five interpreter paths were resolved and checked to remain under
`/workspace`; see `runs/local_setup_20261031/persistence_and_host.json`.
The container image must retain FFmpeg, OSMesa, EGL, the Vulkan loader, and
the NVIDIA container driver integration. Their installed package versions are
recorded in that report; no login-time activation of `/opt/pi05` is required.

## Hardware and process layout

The machine exposes eight NVIDIA CMP 170HX / GA100 GPUs with 64 GiB each
(512 GiB total). The driver reports them as `Graphics Device`; they have compute
capability 8.0. These are compute GPUs with slow PCIe 2.0 x4 connections and no
NVLink. The container has a 336 GiB memory limit, 68 CPU equivalents, and 64 GiB
of shared memory. The host has two Xeon E5-2699 v4 CPUs.

Use GPUs for policy inference and training. Use OSMesa/llvmpipe CPU rendering for
LIBERO and RoboCasa365. Their MuJoCo OpenGL renderer otherwise performs poorly
on these cards. ManiSkill3 uses SAPIEN/Vulkan and needs its own renderer check;
`MUJOCO_GL=osmesa` does not configure SAPIEN. On the real ManiSkill plate task,
GPU Vulkan averaged 0.101 seconds per controller step versus 0.184 seconds for
CPU lavapipe (nine steps each). Use CPU physics with GPU Vulkan rendering there;
the persistent CPU fallback has also been tested.

LIBERO's recorder uses four llvmpipe threads within eight assigned CPU cores.
Five steps averaged 0.38 seconds each, versus 0.61 seconds with one renderer
thread; the initial pixels were identical. Evidence is in
`runs/local_setup_20261031/libero/render_threads4.json` and `libero/ready.json`.
Set `LP_NUM_THREADS` explicitly to override the recorder's default.

RoboCasa uses four llvmpipe threads within eight assigned CPU cores. In the
native kitchen preflight this averaged about 0.49 seconds per controller step,
versus 0.79 seconds with one renderer thread. This is still slower than real
time; the saved video plays at the simulator's 20 Hz control rate.

Keep the following environments separate. Their PyTorch, NumPy, MuJoCo, and
Robosuite requirements differ.

| Purpose | Python executable | Main software |
| --- | --- | --- |
| Supervised expert/LoRA fine-tuning | `/workspace/upenn/prism-pi-trainer/.venv/bin/python` | Python 3.12, LeRobot 0.6, PyTorch 2.11 |
| Native RLDT / frozen policy inference | `third_party/openpi/.venv/bin/python` | Python 3.11, JAX 0.5.3, Flax 0.10.2, pinned OpenPI |
| LIBERO simulator | `.venv/bin/python` | CPU PyTorch 2.7.0, Robosuite 1.4.1, MuJoCo 3.2.3 |
| RoboCasa365 simulator | `.venv-robocasa365/bin/python` | pinned RoboCasa365/Robosuite sources, MuJoCo 3.3.1 |
| ManiSkill3 simulator | `.venv-maniskill3/bin/python` | pinned ManiSkill/RLinf sources, SAPIEN, CUDA PyTorch 2.7.1 |

Paths without a leading slash are relative to `/workspace/upenn/RL-VLA`.
The simulator runs in a child process and communicates with the policy through
local pipes. For the initial diagnostics, GPU 2 is assigned to RoboCasa,
GPU 3 to ManiSkill, and GPU 5 to LIBERO. These assignments are launch defaults,
not permanent reservations.

## Checkpoints and training modes

There are two implementations of pi0.5 here:

* `/workspace/upenn/checkpoints/lerobot/pi05_base` is the official LeRobot
  PyTorch checkpoint used by the copied trainer.
* `checkpoints/openpi/pi05_base` and `checkpoints/openpi/pi05_libero` are the
  original native JAX/Orbax checkpoints used by this repository's RLDT code.

The PyTorch checkpoint is verified against the publisher's SHA256; the native
checkpoints are verified against GCS object sizes and CRC32C checksums.
The local PaliGemma tokenizer is the official public tokenizer, with its digest
checked against the Hugging Face tokenizer metadata. Runtime use does not need
a Hugging Face login or a repeated model download.

The copied trainer is `/workspace/upenn/prism-pi-trainer`. Its supplied training,
data preparation, communication, and precision documentation is preserved in
`docs/`. Read its local setup guide for the completed validation reports.

* **Expert only:** trains the 430,098,464-parameter action expert, including
  action/time projections, while freezing the vision/language backbone. This
  is the supported interpretation of "action head"; it is larger than the final
  output linear layer.
* **LoRA, rank 128:** keeps the expert trainable and adds 156,893,184 trainable
  adapter parameters to 126 Gemma language projections. The original backbone
  weights stay frozen. Total trainable parameters: 586,991,648.

Entry points are `scripts/run_expert.sh` and `scripts/run_lora.sh` in the copied
trainer. Set the dataset, output directory, GPU count, and training budget as
described there. The launcher enables the required FP32/TF32 expert precision.
The small validation dataset demonstrates a working optimizer update; it is not
a sufficient dataset for learning a useful benchmark policy.

Both modes completed three optimizer steps from the unchanged baseline on one
real LIBERO demonstration. Every one of the 813 loaded tensors matched the
pretrained source before training, and every saved tensor was finite. Expert-only
changed 208 expert tensors while all 605 frozen tensors stayed identical. LoRA
changed the expert and merged adapter projections; all 479 tensors outside that
allowed scope stayed identical. Saved action normalization also passed its
round-trip check. The complete comparisons are in
`/workspace/upenn/prism-pi-trainer/outputs/setup_expert_20261031/verification.json`
and `outputs/setup_lora128_20261031/verification.json` in the same trainer folder.
Observed GPU allocation for these batch-one checks was about 23.6 GB for
expert-only and 26.6 GB for LoRA; larger batches require their own sizing.

Trainer outputs use LeRobot/PyTorch checkpoint format. They cannot be passed
directly to the native JAX RLDT loader. Cross-framework checkpoint conversion or
a PyTorch policy integration would be separate work. Native RLDT already has its
own action-expert update path; see [RLDT](rldt.md) and the multibench integrations.

To exercise that native RLDT path with a short budget, from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 RLDT_CPUSET=0-7 LP_NUM_THREADS=4 \
  bash scripts/run_rldt.sh train --config configs/rldt_pi05_libero.toml \
  --output runs/my-rldt-smoke --smoke --particles 8
```

Choose a free GPU and a new output directory. This uses the native LIBERO
checkpoint and trains its full action expert. The standalone trainer launchers
above use the separate PyTorch baseline.

## Record a frozen policy episode

From `/workspace/upenn/RL-VLA`, use a new output directory each time:

```bash
CUDA_VISIBLE_DEVICES=5 bash scripts/record_libero_baseline.sh \
  --output runs/my-libero-baseline

CUDA_VISIBLE_DEVICES=5 bash scripts/record_libero_baseline.sh \
  --policy pi05_base --output runs/my-libero-raw-base

CUDA_VISIBLE_DEVICES=2 bash integrations/multibench/robocasa/record_baseline.sh \
  --output runs/my-robocasa-baseline

CUDA_VISIBLE_DEVICES=3 bash integrations/multibench/maniskill/run_baseline.sh \
  --output runs/my-maniskill-baseline
```

LIBERO defaults to the official task-adapted `pi05_libero` checkpoint. Pass
`--policy pi05_base` to record the unchanged general base weights instead, using
the official LIBERO observation/action transforms and normalization statistics.
This preserves the base model's 50-action horizon and discrete state input and
uses upstream OpenPI's frozen inference API. It does not use LIBERO SFT weights.
RoboCasa365 and ManiSkill3 use `pi05_base`, plus the repository's benchmark-specific
observation, action, and normalization adapters. The general base policy is not
a benchmark-specific fine-tuned policy.

Each recorder saves the video, initial/final images, action trace, and a JSON
report. Videos contain actual policy actions and one frame per executed
controller command. Reports identify the checkpoint, task, seed, horizon,
renderer, action clipping, and simulator-reported success. One episode is a
pipeline diagnostic, not a measured benchmark success rate.

The existing RLDT numerical and CPU pipeline tests passed: 31 tests covering
transport math, replay, optimizer updates, checkpointing, and evaluation/video
behavior. See `runs/local_setup_20261031/rldt_core_tests.log`. These tests
complement the separate real-simulator and pretrained-model checks.
The LIBERO observation/environment adapter tests also passed (17 tests).
The native pi0.5 parameter-boundary, sampler, and checkpoint-provenance tests
passed in the installed OpenPI environment (16 tests), recorded in
`runs/local_setup_20261031/pi05_contract_tests.log`. The native readiness report
also verifies a JAX matrix multiplication on a GPU and the exact base/LIBERO
model configurations used by the rollout code.

The real native RLDT smoke on GPU 4 also completed: ten LIBERO controller steps,
eight particles, one critic update, and one full action-expert update. Actor
gradient norm was 0.0318673 and update norm was 0.00900929; all reported losses
and gradients were finite. It saved the native expert and learner state under
`runs/local_setup_20261031/rldt_native_smoke/final/`. Its short training budget
does not measure policy improvement.
The saved checkpoint comparison checked all 430,098,464 expert parameters:
all were finite, the whole-expert delta from the original checkpoint was
0.00900923648, and the frozen backbone was excluded from the saved expert.
Both saved critics were finite. The full comparison is in
`runs/local_setup_20261031/rldt_native_smoke/verification.json`.
The consolidated installation report is
`runs/local_setup_20261031/setup_report.json`.

### Completed rollout evidence

| Simulator | Task | Policy | Outcome | Steps / video frames |
| --- | --- | --- | --- | --- |
| LIBERO Spatial 0 | Bowl between plate and ramekin onto plate | `pi05_base` | Native timeout; no success | 220 / 221 |
| LIBERO Spatial 0 | Bowl between plate and ramekin onto plate | `pi05_libero` | Native success | 78 / 79 |
| RoboCasa365 | Wine from counter into cabinet | `pi05_base` | Native timeout; no success | 750 / 751 |
| ManiSkill3 plate-25 | Fast food cup onto plate | `pi05_base` | Native timeout; no success | 80 / 81 |

The raw-base LIBERO video is
`runs/local_setup_20261031/libero/base_baseline/baseline.mp4` (11.05 seconds,
20 FPS). Its `verification.json` checks all 220 applied actions, the native
timeout, the complete video decode, and its checksum. The task-adapted LIBERO
video is `runs/local_setup_20261031/libero/baseline/baseline.mp4`
(3.95 seconds, 20 FPS). The ManiSkill video is
`runs/local_setup_20261031/maniskill/baseline/baseline.mp4` (16.2 seconds, 5 FPS).
The RoboCasa video is
`runs/local_setup_20261031/robocasa/baseline/baseline.mp4` (37.55 seconds, 20 FPS).
Start/middle/end frames were inspected. Frame counts and video durations match
the executed actions, and the action traces are finite and respect the native
applied-action bounds. Detailed outcome and validation reports sit beside each
video. These are the first completed episodes at the fixed seeds, not selected
successful attempts.

The three raw-base failures are recorded task outcomes, with functioning simulator,
camera, policy-inference and controller interfaces. Task learning and performance
measurement remain separate experiments. No
task-fine-tuned model was substituted for the requested raw-base recordings.
The separately labeled `pi05_libero` success video uses task-adapted weights.

The completion recheck is recorded under `/workspace/upenn/setup/recheck_20261031/`.
It independently verified the full LeRobot SHA256 and every GCS CRC32C object in
both native checkpoints, checked dependencies in all five environments, exercised
the trainer's CUDA and offline tokenizer, and decoded all four videos while
checking their action traces. The raw-base LIBERO recording was added in this
recheck. The earlier real-data training smoke checkpoints remain present; their
optimizer runs were not repeated.

## Rebuild and download

`scripts/setup_local_core.sh` builds the core OpenPI/LIBERO environments.
The two additional simulator setup scripts accept `--local` for this host.
Source revisions and compatibility patches remain pinned by the existing
multibench dependency lock. Official asset mirrors are recorded in the asset
manifests because some older RoboCasa download links no longer resolve.

This container's network intermittently fails DNS queries and stalls long
transfers. The setup includes resumable, checksum-verified download helpers.
`scripts/download_pi05.py --chunk-workers 32 --chunk-mib 1 --ipv4` enables
parallel ranges for native checkpoints. `scripts/download_verified_file.py`
handles files with published SHA256 digests. Partial downloads remain reusable;
the final path is published only after whole-file verification.

Keep generated checkpoints, assets, and environments out of Git. A copy of the
`uv` executable is also available at `/workspace/upenn/bin/uv`.
