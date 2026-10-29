# GRASP installation

This checkout is at `/mnt/projects/jg/kaifany/RL-VLA`. The login host runs
Rocky Linux 8.10; the `gu-compute` GPU node runs Ubuntu 24.04. Use the GPU
node for the OpenPI environment and model execution. The older PARCC Slurm
scripts target a different account and partition.

## Environments and checkpoints

```bash
cd /mnt/projects/jg/kaifany/RL-VLA
source scripts/env.sh
```

The root `.venv` contains Python 3.11.13, CPU PyTorch and LIBERO/MuJoCo.
`third_party/openpi/.venv` contains the native JAX learner and the pinned
OpenPI dependencies. RLDT imports this repository through `PYTHONPATH=src`
and starts the simulator using `.venv/bin/python`. Keep these environments
separate.

Both official native JAX checkpoints include their complete `params/` and
`assets/` trees:

| Checkpoint | Directory | Bytes |
|---|---|---:|
| LIBERO-adapted π0.5 | `checkpoints/openpi/pi05_libero/` | 12,439,085,481 |
| Base π0.5 | `checkpoints/openpi/pi05_base/` | 12,441,749,581 |

The downloader verifies GCS sizes and CRC32C checksums and records object
generations. It resumes incomplete downloads and reuses verified files:

```bash
python scripts/download_pi05.py --checkpoint pi05_libero
python scripts/download_pi05.py --checkpoint pi05_base
```

Manifests are `runs/setup/pi05_download.json` and
`runs/setup/pi05_base_download.json`. The PaliGemma tokenizer is cached at
`.cache/openpi/big_vision/paligemma_tokenizer.model`.

The existing RLDT adapter supports the LIBERO-adapted checkpoint and its
normalization assets. The base checkpoint is available for future adaptation;
it is not a drop-in replacement in `configs/rldt_pi05_libero.toml`.

## First RLDT run

Submit the small validation run from the repository root:

```bash
sbatch scripts/run_rldt_smoke_grasp.slurm
```

This requests one GPU, eight CPUs and 96 GiB host RAM in `gu-compute`, using
account `gu-account` and QOS `gu-low`, for at most 20 minutes. It collects two
decisions (up to ten controller commands), performs one critic update and one
full action-expert update with eight particles, then saves a checkpoint.
It checks execution, not policy improvement.

For interactive work, obtain an allocation first:

```bash
srun --partition=gu-compute --account=gu-account --qos=gu-low \
  --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=01:00:00 --pty bash
cd /mnt/projects/jg/kaifany/RL-VLA
source scripts/env.sh
bash scripts/run_rldt.sh preflight --config configs/rldt_pi05_libero.toml
```

Choose a new output directory for every run. To start the normal 100-iteration
budget inside a sufficiently long allocation:

```bash
bash scripts/run_rldt.sh train --config configs/rldt_pi05_libero.toml \
  --output runs/rldt_pi05_libero_trial01
```

The normal configuration collects 32 decisions and makes 64 critic updates
per iteration. Iterations 1 and 2 leave the actor unchanged. Iteration 3 makes
the first four actor updates. Each update draws eight denoising trajectories
for one observation, obtains action gradients from the twin critics, and trains
local flow velocities using transport, endpoint consistency and reference
velocity losses. See [the full RLDT walkthrough](rldt.md).

## Full RLDT training and analysis

The first full run was submitted on 2026-10-26 at approximately 17:06
America/New_York as job **1470466**, using `scripts/run_rldt_train_grasp.slurm`.
Its experiment directory is `runs/rldt_pi05_libero_1470466/`. Check its
`status.json` and Slurm for current state; this document is not a live status.

The run starts from the original `pi05_libero` checkpoint, not the smoke-test
weights. It uses ordinary LIBERO Spatial task 0, training seed 0, and the
unchanged full RLDT config. There is no residual learner, SFT or demonstration
replay. One GPU, eight CPUs and 96 GiB host RAM are requested for at most eight
hours. The time limit is a scheduling allowance, not a runtime prediction.

| Budget | Value |
|---|---:|
| Outer iterations | 100 |
| New chunk decisions per iteration | 32 |
| Total chunk decisions | 3,200 |
| Controller steps per chunk | Up to 5 |
| Total controller interaction budget | Up to 16,000, excluding reset settling |
| Critic updates | 64 per iteration, 6,400 total |
| Actor updates | 4 per iteration after iterations 1–2, 392 total |
| Actor batch / particles / denoising steps | 1 observation / 8 particles / 10 steps |
| Checkpoints | Iterations 25, 50, 75, 100, plus `final` |

The job trains first, then evaluates the original frozen policy and final RLDT
policy for **50 episodes each**, using the same reset/noise seeds 100000–100049.
These are matched evaluations on the training task. Initial states can repeat
and overlap the training states; this is not a held-out-state or generalization
experiment. Both evaluations use the same native RLDT sampler and precision.

| Artifact within the experiment directory | Meaning |
|---|---|
| `status.json` | Current/last stage, state, exit code and host |
| `config.toml`, `source_snapshot.tar.gz`, `source_commit.txt`, `source_changes.patch` | Launch configuration and source provenance |
| `train.log` | Model loading, collection, updates and errors |
| `training/config.json`, `training/manifest.json` | Resolved learner settings and model/device metadata |
| `training/metrics.jsonl` | One record per completed iteration |
| `training/iteration_000025/`, etc., `training/final/` | Expert and learner snapshots; about 8.1 GiB each |
| `training/summary.json` | Successful completion of training and checkpoint saving |
| `gpu.csv` | GPU use and memory sampled every ten seconds |
| `evaluation_frozen.json`, `evaluation_trained.json` | Complete matched evaluation outcomes |
| `evaluation_*.episodes.jsonl` | Episode outcomes streamed during each evaluation |
| `analysis.json`, `progress.png` | RLDT summary/plots, plus evaluation comparison when both reports exist |

Monitor or regenerate plots from the login host:

```bash
squeue -j 1470466
tail -F runs/rldt_pi05_libero_1470466/train.log
tail -F runs/rldt_pi05_libero_1470466/training/metrics.jsonl
# After at least one iteration has completed:
.venv/bin/python scripts/analyze_rldt.py runs/rldt_pi05_libero_1470466
# After the job leaves the queue:
sacct -j 1470466 --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

The analysis command produces a snapshot; rerun it to refresh during training.
The job also generates reports after training and after both evaluations.

Read results in this order:

1. Check Slurm/status, finite metrics, nonzero actor updates after iteration 2,
   and expected checkpoint files. A missing `summary.json` means training has
   not recorded successful completion. Checkpoints support warm starts, not
   exact recovery of replay and simulator state; the job does not auto-requeue.
2. Plot cumulative training success and recent success using differences of
   cumulative episode/success counters. Aggregate episode counts, not an
   unweighted average of per-iteration success rates. The provided plot uses
   the last ten iterations. These are changing-policy training outcomes.
3. Inspect critic loss, actor gradient/update norms, the three actor objective
   terms and GPU use. Actor objectives need not decrease monotonically as the
   critic and transport targets change. Logged losses are the **last update**
   in each iteration, not iteration averages. Finite losses alone do not prove
   better behavior.
4. Use matched evaluation success as the primary comparison. `analysis.json`
   reports success-rate change, newly successful episodes, lost successes and
   step-count change on episodes where both policies succeed. It includes
   episode-level Wilson intervals, with their independent-trial assumption;
   repeated states and one training seed limit the inference.
5. If the frozen baseline is near perfect, use harder, separately defined test
   conditions to assess improvement. Repeat training with multiple seeds and
   assess other tasks for retention before making broader claims. The current
   run does not establish generalization or a full LIBERO benchmark score.

## PRO Spatial task 0 adaptation

The experiment `runs/pro_spatial0_rldt_20261026` starts from the original
`pi05_libero` and adapts only Spatial task 0 under the published PRO Position
condition (`libero_spatial_swap`). No SFT or demonstration replay is used.
It uses the reusable GPU allocation **1470624**, with one RTX PRO 6000 GPU,
eight CPUs, and 96 GiB host memory. Its detached `srun` step has a 24-hour
limit; the allocation itself remains available after the experiment completes.
Consult `launch.json`, `status.json`, and Slurm for actual execution status.

The PRO simulator lives in `.venv-pro`, with source at `third_party/libero_pro`,
commit `eafdb809426b13153aa1e4c42d6601844217dfec`. Its pinned dataset revision
is `c86fc3b8293185a6f373677018ff3e37f8391602`. The learner continues to use
`third_party/openpi/.venv`; `LIBERO_CONFIG_PATH=.cache/libero-pro` selects PRO
assets without modifying the ordinary LIBERO environment.

The protocol contains 100 generated training states, 50 generated validation
states, all 50 published PRO states for testing, and all 50 original Spatial
task 0 states for retention. Generation matches the pinned upstream generator's
sampling point immediately after a freshly constructed environment. Seeds,
rejections, state hashes, and reset checks are recorded. Generated states are
checked for duplicate physical configurations, including overlap with published
test states, and are rejected if invalid or already successful during settling.
Filtering uses no policy outcomes. All 250 resets must pass before launch.

This tests disjoint initial states within one PRO Position condition. It does
not establish generalization across disjoint perturbation ranges, other tasks,
the full Spatial suite, or multiple training seeds.

Automatic stages:

1. Frozen-policy validation, 50 episodes.
2. One continuous 100-iteration RLDT run, training seed 0. The normal budget is
   3,200 decisions, at most 16,000 controller steps, 6,400 critic updates, and
   392 actor updates. Replay and optimizers remain continuous across saves.
3. Evaluate saved iterations 25, 50, 75, and 100 on the same 50 validation
   states. This happens after training, without restarting training at saves.
4. Select the RLDT checkpoint with most validation successes; break ties by
   earliest iteration. The baseline is not a selection candidate, so a poor
   RLDT result is retained rather than hidden by returning the baseline.
5. Persist selection, then evaluate frozen and selected policies on the same
   50 published test states and 50 original-task retention states.

Every evaluation enumerates distinct states once, with reset/noise seeds
100000–100049 and identical sampler/precision. The first five episodes of each
evaluation include side-by-side camera videos sampled at action-chunk boundaries.
This is 450 evaluation episodes total. Checkpoint selection never reads final
test outcomes. The experiment does not silently add demonstrations or change
training settings if initial performance is low.

Important artifacts:

| File | Contents |
|---|---|
| `protocol.json`, `assets.json`, `generation.json`, `pro_states.npy` | Immutable task, splits, provenance and reset states |
| `setup_check.json`, `initial_*.png`, `reset_checks/*.png` | Simulator reset checks and sample views |
| `launch_contract.json`, `source_snapshot.tar.gz`, `launch.json` | Source/config hashes, snapshot and Slurm step launch |
| `status.json`, `launcher.log`, `train.log`, `gpu.csv` | Execution stage, logs and telemetry |
| `training/metrics.jsonl`, `training/episodes.jsonl` | Iteration metrics and completed training episode identities/outcomes |
| `baseline_validation.json`, `validation_*.json`, `validation.png` | Baseline and checkpoint validation results |
| `selection.json` | Predeclared selection rule and selected checkpoint |
| `baseline_test.json`, `selected_test.json` | Matched final test results |
| `baseline_source.json`, `selected_source.json` | Matched original-task retention |
| `*.episodes.jsonl`, `*.videos/` | Streamed evaluation outcomes and selected videos |
| `comparison.json`, `analysis.json`, `progress.png` | Paired outcomes, uncertainty and training diagnostics |

Use new directories for later experiments. After cloning/installing the PRO
simulator, the preparation and launch commands are:

```bash
source scripts/env.sh
python scripts/prepare_pro_spatial_grasp.py assets --output runs/NEW_EXPERIMENT
# Run generation on the compute node with the PRO interpreter:
export LIBERO_CONFIG_PATH="$RL_VLA_ROOT/.cache/libero-pro"
.venv-pro/bin/python scripts/prepare_pro_spatial_grasp.py generate --output runs/NEW_EXPERIMENT
# Run tests/checks before freezing the source/config contract and launching:
python scripts/launch_pro_spatial_grasp.py --experiment runs/NEW_EXPERIMENT --allocation ALLOCATION_ID
```

The launch command refuses already launched experiments. Source or config changes
after launch stop subsequent stages; preserve the frozen inputs until completion.
Checkpoints support warm starts, not exact replay/simulator recovery.

## Dependency repair

OpenPI is pinned to `215abfb217dbac7d5f1273282331b9b1866c0479`; its LIBERO
submodule is pinned to `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`.
Use the source bootstrap in [OpenPI setup](openpi.md) for a fresh checkout.

Run the OpenPI lock installation **on the Ubuntu compute node**. Its pinned
`rerun-sdk==0.23.1` wheel requires glibc 2.31 or newer, so the entire lock cannot
be installed on the Rocky login host. There is no dependency version override:

```bash
source scripts/env.sh
GIT_LFS_SKIP_SMUDGE=1 uv sync --project third_party/openpi --frozen --python 3.11.13
third_party/openpi/.venv/bin/python - <<'PY'
import importlib.util
import shutil
from pathlib import Path
source = Path('third_party/openpi/src/openpi/models_pytorch/transformers_replace')
target = Path(importlib.util.find_spec('transformers').origin).parent
shutil.copytree(source, target, dirs_exist_ok=True)
PY
uv pip check --python third_party/openpi/.venv/bin/python
```

`scripts/env.sh` sets `UV_LINK_MODE=copy` to protect cached wheels from the
required Transformers replacement. Reapply that replacement after reinstalling
Transformers. Do not install the root project's extras into this environment.

The simulator uses the repository's hash-locked requirements:

```bash
uv pip install --python .venv/bin/python --require-hashes --torch-backend cpu \
  -r docker/requirements-libero.lock -r docker/requirements-test.lock
uv pip install --python .venv/bin/python \
  editables==0.5 imageio==2.37.0 imageio-ffmpeg==0.6.0 google-crc32c==1.9.0
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  -e . -e third_party/openpi/packages/openpi-client
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  --config-setting editable_mode=compat -e third_party/openpi/third_party/libero
uv pip check --python .venv/bin/python
python docker/smoke.py --libero
```

LIBERO paths are in `.cache/libero/config.yaml`. Native OSMesa libraries are
extracted locally from distro packages: Rocky's `mesa-libOSMesa` under
`.cache/native/usr/lib64/`, and Ubuntu's `libosmesa6` under
`.cache/native/usr/lib/x86_64-linux-gnu/`. The launchers select the appropriate
directory for each operating system. No system packages or shell startup files
were modified. The Rocky activation also selects its system TLS certificate
bundle for the standalone Python interpreter.

Setup logs, package snapshots and verification results live in `runs/setup/`.

## Verified on 2026-10-26

GPU allocation **1470420** used one NVIDIA RTX PRO 6000 Blackwell Server
Edition (97,887 MiB reported memory, driver 580.95.05) on
`jg-b6000-0.grasp.maas`. The allocation was released after validation.
The machine-readable record is `runs/setup/grasp_installation.json`.

| Check | Result |
|---|---|
| Checkpoint sizes and CRC32C | LIBERO 16/16 objects; base 29/29 objects verified |
| Dependency compatibility | Both environments pass `uv pip check`; OpenPI's lock is unchanged |
| Standard LIBERO task/state assets | All 130 tasks verified across Spatial, Object, Goal, 90 and 10 |
| Real simulator reset/render/step | Passed on the Rocky login host and Ubuntu compute node |
| Root test suite | 120 passed, 18 optional-dependency skips |
| RLDT numerical/integration suite | 45 passed, no skips |
| Python lint and shell syntax | Passed |
| Real GPU checkpoint load | 430,098,464 trainable expert parameters; 2,923,335,408 frozen parameters |
| Frozen-policy Spatial task 0, initial state 0 | Success in 81 controller steps, 17 policy calls |
| RLDT/OpenPI sampler parity | Max normalized error 0.001945; decoded error 0.001604; tolerance 0.025 |
| Eight-particle full-expert smoke | Ten controller steps, one critic update, one actor update; finite metrics |

The smoke actor gradient norm was **0.040229**, and its parameter-update norm
was **0.009498**. Sampled peak GPU memory was **33,439 MiB** (about 32.7 GiB).
The reported training time including checkpoint writing was 120.81 seconds,
excluding initial model loading and simulator-process creation.

The saved expert and learner state are in
`runs/rldt_pi05_smoke_1470420/final/`; the resolved configuration, metrics and
summary are in its parent directory. The smoke run completed no episodes and
does not measure policy improvement. The separate frozen-policy check contains
one episode and is not a success-rate benchmark.
