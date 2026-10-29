# DICE pipeline on PrismaX

Implemented on local branch `dice/prismax-pipeline`, based on RL-VLA commit
`9d5c9c1694e42a398722e7eaddd79e387ab6f475`. Kai's unpublished RLDT changes are
not required: this uses the already committed `Pi05Flow` adapter and simulator
IPC. Keep RLDT work separate and reconcile changes to those shared adapters
when Kai pushes. Nothing in this setup pushes or starts a training job.

This is a project adaptation, not a validated reproduction: frozen pi0.5,
pooled frozen features, CPU residual actor and double critic, normalized
executed prefixes, primitive-discounted finite-episode returns. The prior is
queried again during actor/critic updates; JAX inference remains the major GPU
cost. This does not update pi0.5 weights or train a new visual representation.

## 1. Prepare the host

Use a Linux CUDA host with Git, uv, Python 3.11 and a working NVIDIA driver.
Headless simulator rendering needs OSMesa (for example `libosmesa6` on Ubuntu)
and its native dependencies. Use the existing prepared container if available;
setup does not install OS packages or start Docker. From the repo root:

```bash
bash scripts/setup_dice.sh
```

This downloads pinned OpenPI/DICE sources, creates separate OpenPI and simulator
environments, installs locked simulator packages, applies the required upstream
Transformers replacement, and creates LIBERO's local path config if absent.
Existing tracked source edits or a different revision stop setup. It does not
fetch model weights or demonstration data. Do not install the authors' DICE
package stack into OpenPI.

Stage the full official `pi05_libero` checkpoint at
`checkpoints/openpi/pi05_libero` (both `params/` and `assets/`), or edit a copy of
the TOML to point to an existing copy. Verify the LIBERO task assets and initial
states under `third_party/openpi/third_party/libero`; source code alone is not
proof that assets are complete. See [OpenPI setup](openpi.md) and
[server dependency details](parcc-setup.md).

```bash
bash scripts/run_dice.sh check
bash scripts/run_dice.sh validate-policy --output runs/dice_validate_01
```

`check` lists missing paths/revision mismatches and separately reports data
readiness. Missing setup returns a nonzero status. `validate-policy` loads the
checkpoint, resets a simulator and checks zero-residual decode equivalence;
it takes zero environment steps and performs zero optimizer updates.

`DICE_PYTHON` overrides the OpenPI interpreter. `DICE_CPUSET` optionally limits
CPU affinity. `CUDA_VISIBLE_DEVICES` is preserved, defaulting to device 0 when
unset. The launcher resolves paths from the repo root and uses local caches.
Resource guard defaults: at least 2048 MiB GPU free, temperature below 85 C,
8192 MiB host RAM free, 10 GB disk free. Override using `DICE_MIN_GPU_FREE_MB`,
`DICE_MAX_GPU_TEMP`, `DICE_MIN_RAM_MB`, `DICE_MIN_DISK_GB` for the allocated host.
These are periodic checks, not memory caps or an estimate of model requirements.

## 2. Verify collection and learning

For an explicit two-episode diagnostic requiring no demonstration file:

```bash
bash scripts/run_dice.sh train --config configs/dice_pi05_online_smoke.toml \
  --output runs/dice_smoke_01
```

This is the **online-only ablation**, not the demonstration-assisted recipe.
Check finite losses, positive update count, terminal episode records and
`final.pt`. A short successful run establishes plumbing, not improvement.

The default `configs/dice_pi05_libero.toml` requires verified demonstration
replay. Use [the episode export schema and converter](dice.md#demonstration-data-preparation)
to prepare `datasets/dice/libero_spatial_0.mpk`; it must match the task, checkpoint,
normalization, execution horizon and discount. Raw HDF5 cannot be passed to this
runner. Full episode exports require a real final observation; replay the
recorded simulator trajectory if the dataset lacks one. No missing observation
or task success is fabricated. The converter is:

```bash
PYTHONPATH=src third_party/openpi/.venv/bin/python -m rl_vla.dice_data \
  --config configs/dice_pi05_libero.toml \
  --input datasets/dice/verified_episodes.mpk \
  --output datasets/dice/libero_spatial_0.mpk
```

Run the converter with the same rendering/cache environment as the launcher
(e.g. source `scripts/env.sh` once the simulator environment exists).
Demonstration preprocessing is still a required data step. Kai's unpublished
code is not a substitute for that file.

## 3. Baseline, train, evaluate

Use the same task/config for all three commands. Choose task-specific budgets
and a task on which the base policy has room to improve before a full run.
The shipped 100-episode/small-batch configuration is a starting budget, not a
claim that it is adequate for convergence.

```bash
bash scripts/run_dice.sh evaluate --baseline --episodes 50 --seed 100000 \
  --output runs/dice_base_eval_01
bash scripts/run_dice.sh train --output runs/dice_train_01
bash scripts/run_dice.sh evaluate --checkpoint runs/dice_train_01/final.pt \
  --episodes 50 --seed 100000 --output runs/dice_eval_01
```

Training emits `metrics.jsonl`, `metadata.json`, `latest.pt`, `final.pt` and
`resume.pt`. Logs include initial-state IDs, primitive steps, clipping count,
MC return, critic/actor losses and regularization fraction. Offline replay mix
decays from 0.5 to 0.1 over 10000 collection decisions; both schedule units and
warmup units are chunk decisions. Critic targets average minimum target-Q over
`particles` independent next-state candidates. The selective gate follows the
released upstream replay-action calibration, which differs from the paper's
candidate-action notation; expert samples retain regularization.

Evaluation never performs optimizer updates. It writes per-episode JSONL and
success rate with a Wilson 95% interval in `summary.json`. Matching reset seeds
matches simulator initial-state sampling; policy randomness differs when DICE
uses several candidates. Use additional training seeds for comparative results.
The independent evaluation seed range is a convention, not proof of disjoint
LIBERO initial states: the standard suite has a finite state pool. A real held-out
state split/PRO perturbation protocol must be explicitly configured separately.

## 4. Restart

Increase `train.episodes` in a copied config, then use a new output directory:

```bash
bash scripts/run_dice.sh train --config configs/dice_longer.toml \
  --resume runs/dice_train_01/resume.pt --output runs/dice_train_02
```

Only the episode budget and save frequency may change on resume. Replay,
optimizers and Torch/NumPy/prior RNG are restored; the next environment reset
uses the original episode's seed. State is saved at `save_every` episode
boundaries and at completion, so a crash loses work since the last save. Native
GPU/simulator nondeterminism can prevent bitwise replay. `resume.pt` includes
NumPy replay through pickle: load only trusted locally generated files.
Inference checkpoints use restricted `weights_only=True` loading instead.

## Local verification and remaining GPU checks

Local verification: **123 passed, 18 skipped** across the repository suite
(Python 3.12, CPU Torch 2.15); the dedicated DICE suite has 12 passing tests.
Skipped checks require unavailable optional dependencies. Production setup
continues to use the repository's Python 3.11 pins. Lint, formatting, shell
syntax and expected nonzero fresh-host preflight also pass.

The CPU suite tests numerical updates, the exact multi-candidate target,
collection, serialization, uninterrupted-versus-resumed synthetic training and
baseline/residual evaluation with optimizer updates forbidden. Fresh-host
preflight and shell syntax are also checked. This work environment has no staged
OpenPI weights/simulator or assigned PrismaX GPU: dependency installation on
that host, real checkpoint validation, GPU smoke and training remain unverified.
Historical server validation in `docs/dice.md` predates this branch.
