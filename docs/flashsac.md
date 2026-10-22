# Residual FlashSAC

`algorithm = "flashsac"` trains the same residual around frozen OpenPI as `sac`, but with
FlashSAC's critic and actor networks ([paper](https://arxiv.org/abs/2604.04539),
[code](https://github.com/Holiday-Robot/FlashSAC), MIT).

## Setup

```
scripts/fetch_flashsac.sh            # clones third_party/FlashSAC at the tested commit
python -m rl_vla train --config configs/flashsac_toy.toml --output runs/flashsac_trial
python -m rl_vla train --config configs/libero_flashsac.toml --output runs/libero_flashsac
```

No FlashSAC dependencies are installed (no JAX, Hydra or simulators). `rl_vla/flashsac_vendor.py`
loads only the upstream `layer.py`, `network.py` and `utils/distribution.py` from the checkout.
`FLASHSAC_ROOT` points to a checkout elsewhere. Docker images do not include the checkout yet.

The LIBERO example currently selects Spatial task 5, a CPU learner, and an
OpenPI endpoint at `localhost:8002`. Set `RL_VLA_OPENPI_HOST` and
`RL_VLA_OPENPI_PORT` explicitly for the server you intend to use: these variables
override the TOML, and `source scripts/env.sh` selects the robustness endpoint
at port 8001. An inference server alone is not a training run. The local
`runs/libero_flashsac_s0/` has partial metrics but no final checkpoint or summary;
it should not be reported as a completed experiment.

## What is upstream and what changed

Unmodified upstream: inverted residual blocks, pre-activation BatchNorm, post-RMSNorm, unit weight
normalization after every optimizer step, the ensembled double critic with a 101-atom categorical
value head, cross-batch (current + next) BatchNorm passes, delayed actor updates, EMA target,
cosine learning-rate decay and return-based reward scaling.

Adapted for RL-VLA's decision semantics:

| Upstream | Here | Why |
|---|---|---|
| scalar `gamma**n` in the categorical target | per-transition `discount = gamma**k` | a decision executes k <= K commands |
| `NormalTanhPolicy` head (unit-norm projections) | same trunk; head projections scaled by zero-initialized gains | residual must start at mean 0, std exp(-3) |
| entropy target from sigma = 0.15 | sigma = 0.05 (`target_sigma`) | normalized residual, not full motor commands |
| noise repetition, AMP, torch.compile | off | a decision already spans K commands; speedups need big parallel batches |

Batch size and update-to-data ratio come from `[train]` (`batch_size`, `updates_per_decision`).
Upstream's single-environment setting is batch 512 with one update per transition.

Note: the networks are larger than the default SAC MLPs; set `train.device = "cuda"` for long runs.
