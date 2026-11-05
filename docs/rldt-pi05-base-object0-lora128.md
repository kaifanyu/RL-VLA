# General π0.5 base → Object 0 with RLDT and LoRA rank 128

This recipe starts from the local **general `pi05_base` JAX weights**, trains
action-expert attention and MLP adapters, and collects online LIBERO-PRO
Object 0 rollouts: **pick up the alphabet soup and place it in the basket**.
It uses the same fixed 100 training, 50 validation, and 50 test initial states
as `runs/vla_trace_objects_20261031/tasks/object_0/protocol.json`.

The installed environments already provide the required dependencies:

- Learner: `third_party/openpi/.venv/bin/python` (OpenPI, JAX, Flax, Optax).
- Simulator: `.venv-pro/bin/python` (LIBERO-PRO, MuJoCo, OSMesa).
- Weights: `checkpoints/openpi/pi05_base/params`.
- Robot calibration: `checkpoints/openpi/pi05_libero/assets`.

The calibration supplies LIBERO normalization statistics and the camera/state/
action mapping. **No `pi05_libero` model weights are loaded.** This is general
base initialization with an explicit LIBERO adapter; it is not an off-the-shelf
LIBERO policy. It preserves the base model's 50-action horizon and discrete
state tokens. Each decision executes the first five actions. The earlier
`pi05_libero` experiments used horizon 10 and no discrete state tokens, so this
is a different experimental condition.

## Training configuration

The recipe is [rldt_pi05_base_object0_lora128.toml](../configs/rldt_pi05_base_object0_lora128.toml).

| Setting | Value |
| --- | --- |
| LoRA rank / alpha | 128 / 128 (scale 1) |
| Trainable policy parameters | 88,473,600 |
| Frozen original policy parameters | 3,353,433,872 |
| Targets | Action-expert Q, KV, output attention projections; gated and linear MLP weights |
| Factorization | Last two matrix axes; independent factors for each layer/head/gate |
| Initialization | Random A, zero B; exactly zero initial weight delta |
| Frozen modules | VLM, original action-expert weights, action/time projections, normalization layers |
| RLDT budget | 100 iterations; 32 decisions, 64 critic updates and 4 actor updates per iteration |
| Actor learning rate | 2e-5; 32-update warmup |
| Critic warmup | First 2 iterations |
| SVGD particles / denoising steps | 8 / 10 |
| Checkpoints | Every 25 iterations and `final/` |

The two critics train separately. LoRA uses a differentiable
`W_effective = W_frozen + (alpha / rank) A @ B` reconstruction in the existing
OpenPI graph. Only A/B receive policy optimizer states. This reduces the
trainable state; it does not promise a proportional speedup because the effective
dense weights are materialized. The VLM prefix stays frozen and cached.

This is the repository's π0.5 adaptation of
[RLDT](https://github.com/RPFey/rldt), with an additional local LoRA
parameterization. The supervised PyTorch launcher in `prism-pi-trainer` is a
separate training path and cannot be substituted for these commands.

## Run on this workstation

From `/workspace/upenn/RL-VLA`:

```bash
# Inspect checkpoint, device, adapter settings, and trainable parameter counts.
bash scripts/run_rldt_base_object0_lora128.sh preflight

# Real simulator rollout, critic update, actor update, and checkpoint save.
# Use a fresh output directory for every command.
bash scripts/run_rldt_base_object0_lora128.sh train --smoke --particles 8 \
  --output runs/pi05_base_object0_lora128_smoke

# Full configured budget, freshly initialized from general pi05_base.
bash scripts/run_rldt_base_object0_lora128.sh train \
  --output runs/pi05_base_object0_lora128_train
```

The launcher defaults to GPU 3, CPUs 80–87, and a separate compilation cache.
Override these for another available allocation:

```bash
CUDA_VISIBLE_DEVICES=4 RLDT_CPUSET=64-71 \
  bash scripts/run_rldt_base_object0_lora128.sh train \
  --output runs/pi05_base_object0_lora128_seed0_alternate
```

The launcher preserves an existing `CUDA_VISIBLE_DEVICES`, including an empty
one. `RLDT_CPUSET` must name CPUs available on the current host. Do not reinstall
the root package's dependencies into the OpenPI environment.

Monitor `metrics.jsonl` for finite losses, nonzero `actor_update_norm` after
warmup, completed episodes and success counts. `summary.json` marks completion.
A successful smoke run proves optimizer/environment integration, not task
learning. Starting from general base with sparse success rewards may require
more data or a demonstration warm start; measure validation success before
claiming improvement.

## Evaluate on separate states

Use matching seeds and the validation split for baseline and trained adapters:

```bash
bash scripts/run_rldt_base_object0_lora128.sh evaluate --env-split validation \
  --episodes 50 --seed 100000 --output runs/pi05_base_object0_validation.json

bash scripts/run_rldt_base_object0_lora128.sh evaluate --env-split validation \
  --checkpoint runs/pi05_base_object0_lora128_train/final \
  --episodes 50 --seed 100000 --output runs/pi05_base_object0_lora128_validation.json
```

Repeat validation for saved iterations when selecting a checkpoint; use the
held-out `test` split only after choosing it. Evaluating without a checkpoint
uses zero-delta adapters on the general base, with identical precision, robot
calibration, state input and action horizon to the trained comparison.

`final/expert.npz` stores adapters, `final/rldt_pi05.json` records their base,
normalization and LoRA settings, and `final/learner.npz` holds learner state.
The base checkpoint remains required. `--init-from .../final` starts a new run
with restored learner state but fresh replay/counters; it is not an exact resume.
Do not warm-start a full run from `--smoke`, which changes RLDT settings.

Native exports merge adapter deltas into full weights and retain an adapter
sidecar for reloading. Serving a base-model export needs the same horizon-50,
discrete-state-enabled LIBERO model configuration; the standard `pi05_libero`
serving configuration has different input and horizon settings.

## Local verification and launched run

The [setup evidence](../runs/pi05_base_object0_lora128_setup/validation.json)
records 50 passing RLDT/LoRA tests in the learner environment and 2 passing
warm-start regression tests in the root environment, plus Ruff and shell checks.
The [eight-particle smoke](../runs/pi05_base_object0_lora128_smoke_v1/summary.json)
completed 10 primitive simulator steps, one critic update and one actor update.
Its actor gradient norm was 0.00890124 and update norm was 0.00359521; all losses
were finite. The [checkpoint audit](../runs/pi05_base_object0_lora128_smoke_v1/checkpoint_verification.json)
verified all 88,473,600 saved adapter values were finite and all five B-factor
tensors had nonzero updates. No complete episode fits that smoke budget, so it
provides no success-rate estimate.

A fresh 100-iteration run was launched on GPU 3 at 10:37 p.m. EDT on
October 31, 2026, with [launch metadata](../runs/pi05_base_object0_lora128_setup/training_launch.json).
Its output is `runs/pi05_base_object0_lora128_train_v1`; the launch alone does
not establish completion. Monitor it with:

```bash
tail -f runs/pi05_base_object0_lora128_train_v1.log
```

When the budget finishes, `runs/pi05_base_object0_lora128_train_v1/summary.json`
and `final/` contain the completion counters and trained adapters. Use that run
path in the evaluation commands above.
