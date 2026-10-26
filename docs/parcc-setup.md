# PARCC installation

This guide describes the installation made on 2026-10-21 in
`/vast/projects/jgu32/lab/kai/RL-VLA`. Historical results under
`/workspace/upenn/RL-VLA` are from another machine.

## Use the installation

```bash
cd /vast/projects/jgu32/lab/kai/RL-VLA
source scripts/env.sh

# Real LIBERO reset, camera rendering and one controller step; no GPU required.
python docker/smoke.py --libero

# Two real policy episodes, then the RLDT/OpenPI sampler parity check.
# Submit from the repository root; each job uses a new result directory.
sbatch scripts/check_install.slurm

# First real RLDT update: eight particles, full expert, one small iteration.
sbatch scripts/run_rldt_smoke.slurm
```

The installation-check job requests one B200 GPU, 28 CPUs and 224 GB RAM for
at most 15 minutes under `jgu32-lab`, matching PARCC's allocation rules.
GPU inference belongs on an allocated compute node; `login02` has no GPU.
The environment scripts preserve Slurm's
`CUDA_VISIBLE_DEVICES`, including MIG UUIDs. The installation check runs one
Spatial and one Object episode, with zero optimizer updates. It records
success, images, first action chunks, timings and package versions. This is an
installation check, not a success-rate benchmark.

For subsequent RLDT work, use `bash scripts/run_rldt.sh ...` inside a suitable
GPU allocation; see [the RLDT guide](rldt.md). The older baseline, DICE and
robustness supervisors still contain assumptions about the previous server and
need their own cluster resource configuration before use.

## Monitor RLDT

The RLDT smoke launcher runs only RLDT, with `--smoke --particles 8`. It uses
the full 430,098,464-parameter action expert and freezes the 2,923,335,408 VLM
parameters. It collects two action-chunk decisions (up to ten controller steps),
then executes one critic and one actor optimizer update. The training seed is
zero and the task is ordinary LIBERO Spatial task 0. This short run validates
the training path; it cannot establish improvement or generalization.

The launcher requests one full B200 for at most 20 minutes and records GPU
utilization/memory every five seconds. Replace `9289643` with the job ID printed
by `sbatch` for a later run:

```bash
cd /vast/projects/jgu32/lab/kai/RL-VLA
tail -F runs/rldt_pi05_smoke_9289643.log
# In separate terminals:
tail -F runs/rldt_pi05_smoke_9289643.gpu.csv
watch -n 2 'squeue -j 9289643'
tail -F runs/rldt_pi05_smoke_9289643/metrics.jsonl
```

The training log reports model loading, simulator startup, rollout collection,
critic targets and the actor update. JAX compilation can take minutes on the
first invocation. Structured losses, gradient/update norms and step counts are
written to `metrics.jsonl` after each completed iteration. `final/` holds the
expert and learner state; `summary.json` appears after the checkpoint is saved.
After a job leaves `squeue`, inspect its final state with:

```bash
sacct -j 9289643 --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

The first run, **9289643**, completed successfully on `dgx011` in **4 minutes
19 seconds**, including startup, compilation and checkpoint writing. It executed
ten controller steps, one critic update and one actor update with eight
particles. The actor gradient norm was **0.031777** and update norm **0.009001**;
all logged metrics were finite. Sampled peak GPU memory was **33,582 MiB**
(about 32.8 GiB). The saved expert and learner state are in
`runs/rldt_pi05_smoke_9289643/final/`, with a compact verification record at
`runs/setup/rldt_training_check.json`. The ten-step budget did not complete an
episode, so this run provides no success-rate estimate.

## Verified installation results

Slurm job **9287728** completed with exit code **0** on `dgx001`, using an
NVIDIA B200 and driver 580.95.05. Its complete runtime was 5 minutes 21 seconds.

| Check | Result | Local artifact |
|---|---|---|
| Official checkpoint integrity | 16/16 object sizes and CRC32C checksums match | `runs/setup/pi05_download.json` |
| Both Python dependency environments | `uv pip check` passes | `runs/setup/*_pip_check.log` |
| Standard LIBERO suite task/state files | Spatial 10, Object 10, Goal 10, LIBERO-90 90, LIBERO-10 10 | `runs/setup/libero_assets.json` |
| Headless LIBERO reset/render/step | Passed, nonblank 224 × 224 image | `runs/setup/libero_smoke.log` |
| Frozen π0.5 rollout, Spatial task 0 | Success in 82 controller steps | `runs/setup/gpu-9287728/report.json` |
| Frozen π0.5 rollout, Object task 0 | Success in 151 controller steps | `runs/setup/gpu-9287728/report.json` |
| RLDT/OpenPI sampler parity | Max normalized error 0.003153; decoded error 0.002610, within 0.025 tolerances | `runs/setup/rldt-parity-9287728.json` |
| Root unit/integration suite | 117 passed, 18 optional-dependency skips | `runs/setup/pytest_root.log` |
| Source lint | Passed | `runs/setup/ruff.log` |

The GPU rollout directory also contains front/wrist camera images and the
initial action chunks. Neither these two episodes nor sampler parity estimates
a benchmark success rate. Package snapshots are in `runs/setup/libero_freeze.txt`
and `runs/setup/openpi_freeze.txt`; source revisions and input hashes are in
`runs/setup/installation.json`.

## Installed components

| Component | Location / version |
|---|---|
| Python | `.cache/python/cpython-3.11.13-linux-x86_64-gnu/` |
| uv | `.cache/bootstrap/bin/uv`, version 0.13.4 |
| Simulator and root package | `.venv/`, Python 3.11.13 |
| Native OpenPI JAX environment | `third_party/openpi/.venv/`, Python 3.11.13 |
| OpenPI source | `third_party/openpi/`, commit `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LIBERO source and assets | `third_party/openpi/third_party/libero/`, commit `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| Official native JAX checkpoint | `checkpoints/openpi/pi05_libero/`, both `params/` and `assets/` |
| PaliGemma tokenizer | `.cache/openpi/big_vision/paligemma_tokenizer.model` |
| LIBERO path configuration | `.cache/libero/config.yaml` |
| Local native utilities | `.cache/native/usr/`: Git LFS 3.4.1, Mesa OSMesa 25.1.7 |
| Downloads and package caches | `.cache/` |
| Setup logs and validation results | `runs/setup/` |

The checkpoint has **16 objects totaling 12,439,085,481 bytes**. Every object
was verified against its GCS byte length and CRC32C checksum. The download
manifest records immutable object generations. The official source is
`gs://openpi-assets/checkpoints/pi05_libero/`, listed in the
[pinned upstream checkpoint table](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md#model-checkpoints).

The simulator environment uses MuJoCo 3.2.3, robosuite 1.4.1, NumPy 1.26.4,
Gymnasium 1.3.0 and CPU PyTorch 2.7.0. OpenPI uses its frozen upstream lock:
JAX/JAXlib 0.5.3, Flax 0.10.2, Optax 0.2.4, Orbax 0.11.13, PyTorch 2.7.1,
Transformers 4.53.2 and Gymnasium 0.29.1. Keep these environments separate;
the RLDT learner accesses the root source using `PYTHONPATH=src`, and starts
the simulator with `.venv/bin/python`.

OSMesa is extracted locally from Ubuntu packages and loaded using
`LD_LIBRARY_PATH`; system LLVM 20 supplies its remaining native dependency.
Headless rendering uses `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`.
`NUMBA_DISABLE_JIT=1` retains the repository's existing Mesa/Numba workaround.
No system package installation or shell startup-file modification is required.

## Reproduce or repair the environments

The source and submodule pins are recorded above. For cloning instructions see
[OpenPI setup](openpi.md). From an existing installation:

```bash
source scripts/env.sh

# Simulator dependencies are hash-verified against the repository locks.
uv pip install --python .venv/bin/python --require-hashes --torch-backend cpu \
  -r docker/requirements-libero.lock -r docker/requirements-test.lock
uv pip install --python .venv/bin/python \
  editables==0.5 imageio==2.37.0 imageio-ffmpeg==0.6.0 google-crc32c==1.9.0
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  -e . -e third_party/openpi/packages/openpi-client
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  --config-setting editable_mode=compat -e third_party/openpi/third_party/libero

# UV_LINK_MODE=copy protects the wheel cache from the required patch below.
GIT_LFS_SKIP_SMUDGE=1 uv sync --project third_party/openpi --frozen --python 3.11.13
uv pip install --python third_party/openpi/.venv/bin/python --no-deps -e third_party/openpi

# Required upstream Transformers replacement; reapply after any reinstall.
third_party/openpi/.venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
import shutil
source = Path('third_party/openpi/src/openpi/models_pytorch/transformers_replace')
target = Path(importlib.util.find_spec('transformers').origin).parent
shutil.copytree(source, target, dirs_exist_ok=True)
PY

python scripts/download_pi05.py
uv pip check --python .venv/bin/python
uv pip check --python third_party/openpi/.venv/bin/python
```

On a clean clone, bootstrap uv into `.cache/bootstrap/` and set
`UV_PYTHON_INSTALL_DIR=$PWD/.cache/python` and `UV_CACHE_DIR=$PWD/.cache/uv`
before creating environments. `uv python install 3.11.13 --no-bin` keeps the
interpreter entirely inside the repository. The existing `scripts/env.sh`
requires `.venv` to exist before it can be sourced.

## Relation to the latest proposal

`proposal.pdf` supersedes the broader original roadmap. Its main experiment
updates the π0.5 action expert with RLDT, freezes the VLM, and compares with a
frozen policy and SFT from the same starting checkpoint. It also calls for
source-task retention and a source-demonstration replay ablation. Hardware
availability determines real-to-real Franka adaptation versus 2–4 LIBERO
Spatial/Object tasks with disjoint LIBERO-PRO position training and test
conditions. A fallback SFT warm start must be reported separately if initial
success is near zero.

This setup supplies ordinary LIBERO task definitions, assets and initial
states, plus the native checkpoint needed by the existing RLDT implementation.
The PRO training adapter, frozen train/test split, matched SFT comparison and
source replay experiment remain experiment work. Demonstration archives have
not been downloaded; they are unnecessary for the installed simulator and
frozen-policy rollout check, but will be needed for SFT/source replay.
