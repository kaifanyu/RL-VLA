# ManiSkill3 plate-25 environment

The requested identifier resolves to **`PutOnPlateInScene25Main-v3`**, an RLinf
extension running on ManiSkill3. It is not registered by stock ManiSkill alone.
The adapter imports exactly that extension from the existing pinned RLinf checkout;
it does not substitute the simpler single-carrot task.

## Standalone Docker setup (2026-10-31)

From the repository root, run `bash integrations/multibench/maniskill/setup.sh --local`.
The explicit flag permits this standalone container; the Slurm guard remains in
place for the existing cluster workflow. Python 3.11.13, `.venv-maniskill3`, wheel
caches, assets, and the optional lavapipe driver live under `/workspace` and persist
with the workspace. The committed dependency lock stays unchanged; setup rewrites
its old editable source path only into `runs/setup/maniskill/requirements-local.txt`.

Use **CPU physics plus GPU Vulkan rendering** for this machine. The real plate-25
smoke passed on physical GPU 3 (PCI `0000:07:00.0`, verified against its CUDA UUID).
Nine ordinary controller steps averaged **0.101 s** with GPU rendering and
**0.184 s** with lavapipe on eight CPU threads. Each includes 100 physics substeps,
task checks and RGB readback. These are short diagnostic timings for one scene,
not a broad simulator benchmark. Evidence is in
`runs/local_setup_20261031/maniskill/render_comparison.json` and `smoke_{gpu,cpu}/`.
Both tests check repeated resets, finite states, nonblank images and the native
success oracle. The synthetic goal check is separate from policy evaluation.
`policy_contract.json` also validates the real reset observation through the
shared native OpenPI transforms: 32 padded state coordinates, one valid camera,
two masked absent wrists, a 50-action prediction horizon and seven decoded motor
coordinates. `ipc_smoke.json` checks the isolated worker communication.

The container exposes its NVIDIA Vulkan ICD under `/etc/vulkan/icd.d`; SAPIEN
3.0.3 does not discover that location automatically. The rollout worker sets
`VK_DRIVER_FILES` and `VK_ICD_FILENAMES` explicitly. CPU fallback uses the isolated
`.cache/native-maniskill/lvp_icd.json` prepared by `prepare_software_vulkan.py`.
It does not replace the system driver. OSMesa settings used by MuJoCo do not select
SAPIEN's Vulkan renderer.

After the shared native OpenPI environment and checkpoint are installed, record
a complete frozen base-policy episode with:

```bash
CUDA_VISIBLE_DEVICES=3 MANISKILL_CPUSET=24-31 \
  bash integrations/multibench/maniskill/run_baseline.sh \
  --render-backend gpu --output runs/my_maniskill_baseline
```

The output directory must be new. The default checkpoint is
`checkpoints/openpi/pi05_base`; seed 20261029 selects `put fast food cup on plate`.
The recorder uses the original base architecture and task calibration statistics,
performs no parameter updates, executes five actions per policy decision, and
records every controller step through native success or the 80-step limit.
`baseline.mp4`, `report.json`, `steps.jsonl`, `actions_states.npz`, and first/last
PNGs preserve the outcome, clipping, backend and timing evidence. A failed episode
is still reported as failed; a diagnostic video is not a success-rate estimate.

## Recorded baseline on this container

The completed video is
[`runs/local_setup_20261031/maniskill/baseline/baseline.mp4`](../../../runs/local_setup_20261031/maniskill/baseline/baseline.mp4).
The unadapted native `pi05_base` policy attempted **put fast food cup on plate**
with environment seed 20261029 and policy seed 100000. It executed all **80 control
steps / 16 policy decisions** and timed out with **success=false**. The cup remained
off the plate. This is an actual policy rollout, with no synthetic placement or
parameter updates and no selection of a successful retry.

The MP4 contains **81 frames at 5 fps**, 640×480, H.264, duration **16.2 s**.
Full video decoding, frame counts, episode boundaries, finite actions/states,
clipping, and native gripper conversion were checked against the JSONL and NPZ
traces. Initial, midpoint and final frames were visually inspected. Ten of 560
proposed action coordinates were clipped to native bounds. Total wall time was
176.74 s, including 23.93 s model loading, 91.20 s simulator initialization,
53.31 s policy inference (including first compilation), and 7.66 s stepping.

The video directory contains `report.json`, `validation.json`,
`decision_timing.json`, `steps.jsonl`, `actions_states.npz`, and the inspected PNGs.
The full installation evidence is in the parent `maniskill/` run directory,
including `environment.json`, `policy_contract.json`, `ipc_smoke.json`,
`render_comparison.json`, and `code_validation.json`.

## Sources and revisions

- [Official RLinf task guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/maniskill.html)
- [Official RLinf task implementation](https://github.com/RLinf/RLinf/blob/0505431899574619da86f551bad70b71e0ea2177/rlinf/envs/maniskill/tasks/put_on_in_scene_multi.py)
- [ManiSkill v3.0.0b22](https://github.com/haosulab/ManiSkill/tree/33967b9e3ead1f841eec57cc9f31d0d8b8cf0907), the version specified by this RLinf checkout
- [Task assets](https://huggingface.co/datasets/RLinf/maniskill_assets/tree/c23fc1880ed7861686d4f995360101eaee4d18a0), carrot assets only
- [Normalization metadata](https://huggingface.co/RLinf/RLinf-Pi05-ManiSkill-25Main-SFT/tree/eb0e2c90726f7f0bdf99e8c9b6656917f0eb5030), assets only; **no SFT weights are downloaded or used**

`requirements.lock` records the complete isolated `.venv-maniskill3` environment.
`assets_manifest.json` records downloaded file hashes. `setup.sh` runs inside
an allocated compute step, or with `--local` in this Docker container, with CUDA
hidden during installation/downloads.
The upstream downloader verifies the Bridge scene archive against SHA256
`618512a205b4528cafecdad14b1788ed1130879f3064deb406516ed5b9c5ba92`.
The WidowX model is the upstream `ManiSkill-WidowX250S` v0.2.0 asset.

## Task and controller contract

| Property | Setting |
|---|---|
| Robot | `widowx250s_bridgedataset_flat_table`, 6 arm joints + two gripper joints |
| Observation | Native third-person RGB 640×480; eight joint positions |
| Policy cameras | `base_0_rgb` valid; both absent wrist views padded and masked false |
| Instruction | Native `put {object name} on {plate name}` |
| Task family | 25 object models; `obj_set=train` contains 16 objects/16 backgrounds; `test` contains 9 objects/5 backgrounds |
| Plate | One plate, following the published Pi05 experiment configuration |
| Action | Seven decoded coordinates: Cartesian translation, rotation delta, gripper |
| Controller | `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` |
| Timing | 500 Hz physics, 5 Hz controller, maximum 80 controller steps |
| Success | Native object/plate geometry and contact test; never equated with timeout |
| Adaptation reward | Binary native task success, identical for both learning methods |

The policy receives metres/radians after task action denormalization. The adapter
keeps native finite action bounds and implements the official RLinf WidowX
gripper rule (`value > 0.5` becomes +1/open, otherwise −1/close). The published
normalization file has gripper quantiles −1 and +1 despite an upstream wrapper
comment describing a 0–1 interval; the executable threshold is used consistently.
Images are not flipped. Resizing and normalization belong to the policy transform.

The raw observation follows the shared project contract:
`{'state': qpos8, 'openpi': {'observation/state': qpos8,
'observation/image': RGB, 'observation/wrist_image': zeros,
'observation/right_wrist_image': zeros, 'prompt': instruction}}`.
Missing wrist images must remain **masked false**, not treated as real cameras.

`make_env(episode_ids=None, obj_set='train', sim_backend='cpu',
render_backend='gpu', max_episode_steps=80)` returns a scalar Gymnasium environment.
Its `reset(seed=...)` accepts an explicit `episode_id` option when no frozen
episode schedule is supplied. Otherwise seeds map reproducibly to native episode
IDs with SHA256, avoiding the bias of using tiny seeds directly as episode IDs.
The returned info reports both `episode_id` and `initial_state_id`. Frozen train,
validation, and test schedules should contain disjoint native episode IDs.

## Policy initialization distinction

The upstream RLinf Pi05 benchmark uses `RLinf-Pi05-ManiSkill-25Main-SFT` weights,
a stateless configuration, and a shorter action horizon. This campaign instead
requests unchanged **raw `pi05_base`** weights. Dataset action/state calibration
is useful for the robot interface, but it is not SFT. Preserve the raw base
architecture, including its action horizon and state-token setting. Results from
this campaign cannot be described as reproducing the paper's SFT baseline.

## Historical cluster validation

The evidence in this section predates the standalone container setup above.

`smoke.py` builds the actual scene, checks two distinct task resets and a repeated
reset, verifies nonblank RGB and finite eight-dimensional proprioception,
executes three controller steps per reset, and checks native success and boundary
signals. It writes RGB PNGs and `smoke.json` only after all assertions pass.

The completed GPU-rendering check is recorded in `validation_gpu/smoke.json`:
two native episode IDs (527917 and 403586), nine ordinary controller steps, and
an exact repeated reset image hash. A synthetic object-on-plate placement also
verified that native success becomes true and produces reward 1/termination;
this diagnostic is not policy evaluation or training data. The observed prompts
were `put fast food cup on plate` and `put pipe on plate`.

`device_audit.json` independently verifies that SAPIEN's CUDA renderer uses PCI
`0000:07:00.0`, matching CUDA device 0's UUID in allocation 1477821 (Slurm physical
GPU 1). The adapter now checks the selected renderer PCI against CUDA device 0
on every construction and rejects silent CPU fallback or an unexpected GPU.
`policy_contract_check.json` verifies one valid camera/two masked absent cameras,
eight state coordinates, seven motor coordinates, and the unchanged base model's
50-action horizon without loading task-trained weights.

The actual native RLDT update smoke also passed, recorded in
`runs/multibench_20261028/readiness/maniskill_rldt_verification.json`: the unchanged
base initialized the run, eight target particles were used, and ten controller
steps produced one critic and one actor update. All 430,098,464 saved action-expert
parameters were finite, the global update norm was positive, and direct comparison
of four complete projection tensors confirmed changed weights. The 2,923,335,408
frozen parameters remain outside the optimizer and are restored from the original
base checkpoint. This short readiness check does not estimate task success rate.

Example CPU test, inside an allocation:

```bash
export CUDA_VISIBLE_DEVICES=""
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
.venv-maniskill3/bin/python integrations/multibench/maniskill/smoke.py \
  --sim-backend cpu --render-backend cpu \
  --output integrations/multibench/maniskill/validation_cpu
```

Successful CPU simulation alone does not establish that GPU rendering or the
Pi05 model works. The queued policy campaign must also pass its own weight,
normalization, inference, memory, and training smoke gates.
