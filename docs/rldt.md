# RLDT fine-tuning of π0.5

This pipeline updates the **π0.5 action expert** using RLDT. It loads the local
OpenPI JAX `pi05_libero` checkpoint, freezes the vision/language backbone, and
trains the action expert and action/time projections with RLDT's transport,
consistency, and reference-velocity objectives. Two learned critics provide the
action gradients. The existing SAC/PPO residual-policy commands are separate
experiments.

The implementation follows the supplied [16-page paper](../rldt.pdf), including
Algorithm 1 and the appendices, cross-checked against the
[official implementation at commit `3ab73e7`](https://github.com/RPFey/rldt/tree/3ab73e775efd0f041ad28170c616146be86bb6ea).
The paper evaluates smaller policies on Gym, FurnitureBench, and Robomimic. Its
experiments do not establish results for π0.5 or LIBERO. This is a π0.5 adaptation
with explicit differences described below.

## Run the pipeline

This workspace already has the two Python environments, the local π0.5
checkpoint, and LIBERO configured. **No installation is needed here.** Run these
commands from `/workspace/upenn/RL-VLA`:

```bash
# Validate the complete training loop on a small CPU flow policy.
JAX_PLATFORMS=cpu bash scripts/run_rldt.sh train --config configs/rldt_toy.toml \
  --output runs/rldt_toy_smoke --smoke

# Load the real checkpoint and inspect device/shape/trainable-parameter selection.
bash scripts/run_rldt.sh preflight --config configs/rldt_pi05_libero.toml

# Exercise a short real-checkpoint run before allocating a full training budget.
bash scripts/run_rldt.sh train --config configs/rldt_pi05_libero.toml \
  --output runs/my_rldt_pi05_smoke --smoke

# Keep the short budget while using the paper's eight SVGD particles.
bash scripts/run_rldt.sh train --config configs/rldt_pi05_libero.toml \
  --output runs/rldt_pi05_smoke_8particles --smoke --particles 8

# Train with the configured rollout and update budget.
bash scripts/run_rldt.sh train --config configs/rldt_pi05_libero.toml \
  --output runs/rldt_pi05_libero
```

Use a new output directory for each run. The CPU toy checks the algorithm and
software interfaces; it does not load a VLA or measure LIBERO performance. A short
π0.5 run checks checkpoint integration and optimizer execution, not task learning.
`preflight` only loads and inspects the model; it does not start LIBERO, sample
actions, run backpropagation, or establish that training fits GPU memory. The
full-expert parameter selection has been checked: **430,098,464 parameters** are
trainable, while the vision/language backbone remains frozen.

`--smoke` overrides the budget to one iteration with two collected decisions,
one critic update, one actor update, two particles, one bootstrap action, actor
batch size one, and no critic warmup. It preserves the configured denoising steps
and trainable parameter subset. CPU selection is explicit above because the
launcher otherwise uses an available GPU even for a toy policy. `--particles 8`
is applied after the smoke overrides and restores eight particles without
changing that short budget.

The launcher uses `third_party/openpi/.venv` for the JAX learner and the existing
root `.venv` for the LIBERO simulator child process. The simulator exchanges raw
observations and primitive actions with the learner over local pipes. This keeps
the two dependency environments separate. The learner accesses model parameters
directly; a frozen OpenPI websocket policy server is not part of this training
path.

The configured environments use these principal package versions:

| Environment | Versions |
|---|---|
| JAX learner: `third_party/openpi/.venv` | JAX/JAXlib 0.5.3, Flax 0.10.2, Optax 0.2.4, Orbax Checkpoint 0.11.13, NumPy 1.26.4, Gymnasium 0.29.1 |
| Simulator: `.venv` | MuJoCo 3.2.3, robosuite 1.4.1, LIBERO 0.1.0, Gymnasium 1.3.0, NumPy 1.26.4, PyTorch 2.7.0+cpu, BDDL 1.0.1 |

Both include `openpi-client` 0.1.0 for the NumPy message serialization. The
launcher uses `PYTHONPATH` to expose this repository to the learner. Do not install
the root project's editable package/extras into the locked OpenPI environment:
its Torch/Gymnasium requirements belong to the separate simulator environment.

The π0.5 config expects the complete local JAX checkpoint at
`checkpoints/openpi/pi05_libero/`, including **both `params/` and `assets/`**.
Normalization assets and the matching `pi05_libero` configuration are required.
A converted `model.safetensors` checkpoint uses a different backend and is not
accepted here. Installation details for the pinned source are in
[OpenPI setup](openpi.md); simulator installation and observations are described
in [the LIBERO adapter guide](libero.md).

For another Linux host, recreate the two environments from their respective
pinned installation recipes, including LIBERO assets and the OSMesa native
libraries. Keep OpenPI's dependency lock separate from the root project's
Torch/Gymnasium dependencies. The launcher selects four available CPUs,
preferring `4-7` when permitted, and places default data caches under `.cache/`
beside the repository directory. Set `RLDT_CPUSET` to override that selection,
`RLDT_PYTHON` to its OpenPI interpreter, `OPENPI_DATA_HOME` to its OpenPI cache,
and `LIBERO_CONFIG_PATH` to its configured LIBERO path directory as needed.
The simulator interpreter is separately selected by `[env].python`.

Evaluate the original and trained policies on the same reset-seed sequence:

```bash
bash scripts/run_rldt.sh evaluate --config configs/rldt_pi05_libero.toml \
  --episodes 5 --seed 100000 --output runs/rldt_base_eval.json
bash scripts/run_rldt.sh evaluate --config configs/rldt_pi05_libero.toml \
  --checkpoint runs/rldt_pi05_libero/final \
  --episodes 5 --seed 100000 --output runs/rldt_trained_eval.json

# Reload actor, reference, critics, optimizer states, and learner RNG into a new run.
bash scripts/run_rldt.sh train --config configs/rldt_pi05_libero.toml \
  --init-from runs/rldt_pi05_libero/final --output runs/rldt_pi05_continued

# Produce a self-contained checkpoint for the upstream OpenPI policy server.
bash scripts/run_rldt.sh export --config configs/rldt_pi05_libero.toml \
  --checkpoint runs/rldt_pi05_libero/final \
  --output checkpoints/openpi/pi05_libero_rldt
```

The five-episode evaluation is a connection and behavior check; it is too small
to establish a reliable performance gain. `--init-from` is a **warm start**:
replay, simulator episodes, counters, and the iteration-based critic warmup start
again. It is not an exact interrupted-run resume. Keep the `[policy]`, `[env]`, and
`[rldt]` sections identical to the saved run; `[train]` can specify the new budget.
An ordinary config cannot warm-start a smoke run because `--smoke` changes its
RLDT settings; use the same smoke settings for such a diagnostic continuation.

Regular π0.5 actor saves contain `expert.npz` plus `rldt_pi05.json`, retaining a
dependency on the original frozen checkpoint. Toy actors use `actor.npz`. The
learner also saves its training state. The `export` subcommand combines a trained
expert with the frozen backbone and original normalization assets into an OpenPI
`params/` + `assets/` checkpoint suitable for native serving. That full export
duplicates the backbone and requires substantially more disk space than an
expert-only save. An actor save alone is not a complete replay/optimizer restart.
Full-expert learner checkpoints also include the reference weights and Adam
moments: the measured snapshot occupies about **8.1 GiB**, including roughly
1.7 GiB for `expert.npz` and 6.5 GiB for `learner.npz` (rounded individually).
Upstream's standard
serving loader converts weights to bfloat16; exported-policy evaluation can
therefore differ from the learner's float32 expert and should be checked
separately before comparing deployment results.

Each run records its resolved configuration in `config.json`, device/source/model
information in `manifest.json`, one metrics record per iteration in
`metrics.jsonl`, and final counters in `summary.json`. Periodic checkpoints are
named `iteration_000025/`, etc.; the final checkpoint is `final/`. Each checkpoint
contains the actor files and `learner.npz` for the warm start described above.

## Training script walkthrough

The entry point is [`src/rl_vla/rldt.py`](../src/rl_vla/rldt.py). Its numerical
objective is in [`rldt_math.py`](../src/rl_vla/rldt_math.py), trajectory replay and
returns are in [`rldt_replay.py`](../src/rl_vla/rldt_replay.py), and the model adapter
is [`rldt_pi05.py`](../src/rl_vla/rldt_pi05.py).

1. **Load the checkpoint and initialize the learner.** Split π0.5 into frozen
   vision/language parameters and trainable action-expert parameters. Store
   trainable weights in float32 and frozen weights in bfloat16. Initialize two
   critics, target critics, actor/critic optimizers, and a frozen reference actor.
   Full expert updates include its attention and MLP parameters as well as
   action/time projections. A projection-only setting is a memory diagnostic;
   it changes which parameters are trained.

2. **Collect action chunks.** Encode the current cameras, instruction, and state
   using the checkpoint's input transforms. Sample Gaussian noise and integrate
   the action expert for `T` denoising steps. Decode the normalized model actions
   using the original output transforms, execute the configured prefix, and stop
   early at termination or truncation. Save the raw observations, sampled
   normalized action, discounted chunk reward, actual execution length, and
   boundary flags.

3. **Sample trajectories from replay.** Mix recent segments with replay segments;
   balance nonzero-reward and zero-reward segments when both exist. Keep their
   temporal order for TD(λ). Raw image observations stay on CPU; full VLM key/value
   caches are created for the current batch rather than stored for the entire
   replay buffer.

4. **Fit the critics.** Sample the trajectory set and its current-policy bootstrap
   actions once per outer iteration. At every critic gradient step, re-evaluate
   those actions using the updated target critics, average their minimum values
   over the bootstrap action samples, and recompute finite-window TD(λ) returns.
   Sample transition indices from that trajectory set and regress both critics
   to the corresponding detached targets. Update target critics by exponential
   averaging. The actor waits through the configured critic warmup before its
   first update.

5. **Construct the transport field.** For each actor observation, sample `K`
   independent current-policy trajectories and choose one intermediate step per
   particle. Estimate the clean endpoint at that step. Differentiate the mean of
   the online critics with respect to the **clean terminal actions**. Combine
   those gradients with RBF kernel repulsion to evaluate the SVGD field at the
   estimated endpoints. All of this becomes a detached training target.

6. **Update the actor.** Re-evaluate velocity at each chosen detached intermediate
   state with gradients enabled. Add the endpoint-consistency loss and a velocity
   constraint against the frozen reference on independent noise/clean-action
   interpolations. Backpropagate through these local velocity evaluations. The
   sampled multi-step denoising trajectory itself is detached, so the update does
   not backpropagate through the entire ODE.

7. **Refresh the reference and save.** Keep the reference fixed throughout an
   outer training iteration. With the default moving-reference setting, copy the
   trained actor into it after the iteration's updates. Record metrics and save
   model/learner state. A permanently pretrained reference is an optional
   experimental change to this default.

The critic sees a masked mean of frozen final vision/language hidden states,
concatenated with normalized proprioception. Its MLP is trained independently of
the actor. Language and both camera views therefore enter the value features,
while the pretrained visual encoder remains frozen.

## Pseudocode

```text
load π0.5 parameters θ and freeze the vision/language backbone
initialize Q1, Q2; target critics Q̄ ← Q; reference expert θref ← θ
initialize trajectory replay D and optimizers

for each outer iteration:
    collect current-policy rollouts:
        encode observation with frozen backbone
        sample normalized action chunk with T-step Euler denoising
        decode to controller actions; execute prefix until an episode boundary
        store transition with true final observation and actual chunk discount
    add collected trajectory segments to D

    sample ordered segments, mixing fresh and balanced replay
    sample bootstrap actions a′ from the current policy once for this trajectory set
    for each critic update:
        Vnext ← average_over_actions(min(Q̄1(s′, a′), Q̄2(s′, a′)))
        optionally clip Vnext to the configured task-value interval
        y ← reverse TD(λ) recursion, stopping traces at episode/window boundaries
        draw a minibatch of transition indices from the selected trajectories
        minimize mean_over_minibatch_and_heads((Q − stopgrad(y))²)
        Q̄ ← ρ Q̄ + (1 − ρ) Q

    if critic warmup is complete:
        for each actor update:
            sample observations from that trajectory set; encode frozen context once
            without actor gradients:
                sample K full T-step trajectories per observation
                clean[j] ← terminal action of trajectory j
                choose independent step i[j] uniformly in {0, …, T−1}
                t[j] ← 1 − i[j]/T                       # native π0.5 time
                x[j] ← trajectory j at selected step
                query[j] ← x[j] − t[j] * vθ(x[j], t[j])
                g[j] ← ∇clean[j] mean(Q1, Q2)(s, clean[j])
                φ[j] ← SVGD(query[j], clean[1:K], g[1:K])
                draw independent grid time u[j] and noise ε[j]
                z[j] ← u[j] ε[j] + (1 − u[j]) clean[j]
                reference[j] ← vθref(z[j], u[j])
            with actor gradients:
                v[j] ← vθ(stopgrad(x[j]), t[j])
                predicted_clean[j] ← x[j] − t[j] * v[j]
                LRL ← mean(dot(v[j], stopgrad(φ[j])) / (T*t[j]))
                LC ← mean(||predicted_clean[j] − clean[j]||² / (T*t[j])²)
                LF ← mean(||vθ(z[j], u[j]) − reference[j]||²)
                minimize LRL + λ1*LC + λ2*LF
    θref ← θ                                           # default reference mode
    log metrics and save the configured checkpoints
```

All norms and dot products above **sum over active action coordinates** and then
average across observations and particles. The implementation masks unexecuted
horizon positions and padded action coordinates in all three actor losses.

## Time convention, action units, and targets

The paper starts with noise at `τ=0` and reaches actions at `τ=1`. OpenPI starts
with noise at `t=1` and reaches actions at `t=0`, so `t=1−τ` and
`v_openpi=−v_paper`. Its Euler step is `x_next=x−v/T`; its expected clean action
is `x−t*v`. Consequently, gradient descent uses **positive**
`dot(v_openpi, field)/(T*t)`. Changing this sign moves actions against the intended
transport direction. Sampled update times are `1, (T−1)/T, …, 1/T`; `t=0` is
excluded.

The checkpoint predicts `10 × 32` normalized internal action values. LIBERO uses
seven physical controller coordinates, and the default executes the first five
actions. The critic and kernel operate on the `5 × 7` active normalized prefix.
Unused horizon positions and padded coordinates cannot contribute to the kernel
distance, Q gradient, or actor losses. Original OpenPI transforms decode actions
only at the environment boundary. Controller-bound clipping remains an
environment projection; replay records the sampled normalized policy action whose
execution produced the transition.

The standard official-code SVGD setting uses clean sources `a_j`, endpoint
queries `b_i`, and `g_j=∇a mean(Q1,Q2)(s,a_j)`:

```text
κ = κ0 * mean_j(||g_j||)
h_i² = max(lower_median_j(||b_i − a_j||²) / (2*log(K+1)), 1e−8)
k_ij = exp(−||b_i − a_j||² / (2*h_i²))
φ_i = mean_j(k_ij*g_j + κ*k_ij*(b_i − a_j)/h_i²)
```

The lower median matches PyTorch's median when `K` is even. Repulsion is the
derivative with respect to the **source** action. The complete field is detached.
This formula is κ times the usual unnormalized Boltzmann-target SVGD direction;
changing to `g/κ` without adjusting the objective changes its scale relative to
the regularizers.

For chunk length `m`, the collector stores `R=Σ_l γ^l r_l` and
`d=γ^m*(1−terminated)`. With `c=1` only when the next row belongs to the same
episode and sampled segment, the target recursion is:

```text
G_t = R_t + d_t * [(1 − λ*c_t)*Vnext_t + λ*c_t*G_(t+1)]
```

At a timeout, `c=0` while `d` remains positive: bootstrap from the actual final
observation, never the next reset observation. At true termination, `d=0`.
The final row of each sampled window has `c=0`. LIBERO success emits one `+1`
and terminates, so there are no later repeated-success rewards to remove.
Critic targets have no policy log-probability or entropy term.

## Defaults and deliberate adaptations

The checked-in configurations are the executable source of truth. Core π0.5
defaults are:

| Setting | Default | Reason |
|---|---:|---|
| Checkpoint | `pi05_libero` | Matching language, cameras, controller, and normalization |
| Trainable actor | Full action expert + action/time projections | Direct flow-policy fine-tuning |
| Denoising steps `T` | 10 | π0.5 sampling convention; paper visual experiments use 8 |
| Particles `K` | 8 | Paper and official code default |
| Executed prefix | 5 of 10 actions, 7 active dimensions | Existing LIBERO chunk interface |
| Temperature coefficient `κ0` | 0.001 | Adaptive repulsion scale |
| Consistency / constraint weights | 0.5 / 0.2 | Official actor objective |
| Actor / critic learning rate | `2e-5` / `1e-3` | Paper starting values |
| Actor learning-rate warmup | 32 optimizer steps | Paper default |
| Critic-only warmup | 2 outer iterations | Value initialization before actor updates |
| Bootstrap actions | 2 | Average sampled current-policy target values |
| TD trace coefficient | 0.5 | Robotics default |
| Target critic momentum `ρ` | 0.995 | Visual-task official config |
| Reference actor | Moving outer-iteration snapshot | Algorithm 1 and released code |
| Primitive-step discount `γ` | 0.999 | Use `γ**actual_executed_steps` at chunk boundaries |
| Critic width | 256 | Three SiLU hidden layers in each independent critic |
| Actor gradient-norm cap | 1.0 | Clip the complete trainable actor gradient before Adam |

The default training budget in [`rldt_pi05_libero.toml`](../configs/rldt_pi05_libero.toml)
is 100 outer iterations, each collecting 32 chunk decisions, with 64 critic
updates and 4 actor updates after the first two critic-only iterations. Critic
minibatches contain 64 transitions; actor minibatches contain one observation and
therefore eight particles. Each outer iteration selects two trajectory segments
for its critic targets and actor observations. Replay holds up to 4,096 decisions
and evicts whole segments. The config saves every 25 iterations and at the end.
At most 16,000 primitive interaction steps are collected when every decision
executes all five actions; episode endings can reduce that total.

`fresh_fraction=0.33` is rounded to a whole segment, so with two selected segments
the default explicitly draws **one fresh segment and one replay segment**. That
small-batch ratio is 50:50, rather than exactly 33:67. Increase the trajectory
count to represent a finer mixing ratio. With one replay segment, the reward
stratum is randomized, giving 50:50 nonzero/zero sampling in expectation. Replay
falls back to the available pool when a reward stratum or older trajectories are
absent.

[`rldt_toy.toml`](../configs/rldt_toy.toml) uses three iterations, 16 decisions per
iteration, four critic and two actor updates, critic/actor batch sizes 8/2, four
denoising steps, and a four-action prefix. Pass `JAX_PLATFORMS=cpu` to exercise it
without a GPU. `--smoke` applies the smaller overrides described above to either
config.

The collector is sequential and the rollout/update budgets are reduced for an
initial runnable integration. Both critics train on the same sampled batch;
upstream trains ensemble members on disjoint trajectory partitions. The critic
uses frozen π0.5 visual/language features rather than the paper's separately
trainable small ViT. Restricting all losses to the active prefix, freezing the
VLM, and using primitive-step chunk discounting are explicit adaptation choices.
These defaults are not a reproduction of the paper's 64/100/1000-environment
benchmark budgets or a tuned π0.5 recipe.

## Paper/code audit

| Detail | Paper | Released code and implementation choice |
|---|---|---|
| Transport-loss sign | Eq. 13 prints a positive dot product in paper time | Released minimization uses a negative dot product; native π0.5's reversed velocity makes the implemented sign positive |
| Temperature scaling | Eq. 6 uses `∇Q/κ` in the target score | Released code uses `∇Q + κ*repulsion`; use its scaling by default |
| Kernel bandwidth | Section 4.1 describes median pairwise distance among clean samples | Released code uses a per-query lower median of squared query/source distances divided by `2 log(K+1)`; use that explicit mode |
| Reference policy | Prose describes staying close to the pretrained policy | Algorithm 1 and code refresh the reference after each outer iteration; moving snapshot is the default |
| Constraint samples | Eq. 15 describes sampled flow-trajectory states | Code draws a separate Gaussian/clean-action interpolation; follow code |
| Critic momentum | Algorithm 1 puts `ρ` on the online critic | Code and appendix momentum values imply `ρ*target + (1−ρ)*online`; use that convention |
| TD targets | Delegated to the TD(λ) reference | Released recursion uses clipped target-critic bootstrap values and ordinary rewards; implement that recursion with explicit boundary handling |
| Appendix `λ` | Text calls it a consistency weight; table has 0 for Gym and 0.5 for robotics | Configs use those values for TD λ and retain 0.5 consistency; keep separate names |
| Timeout handling | Not specified separately | Released code merges termination and truncation; this adapter preserves timeout bootstrapping and stops traces |
| Image gradient accumulation | Not specified | Released caller's split scaling is inconsistent; average accumulated gradients correctly |
| Short executed prefix | Not detailed | Released image padding concatenates on the wrong axis; explicit masks handle prefix/padding here |

Sources for the audit are the
[actor/field implementation](https://github.com/RPFey/rldt/blob/3ab73e775efd0f041ad28170c616146be86bb6ea/agent/finetune/train_fm_svgd_agent.py),
[visual actor](https://github.com/RPFey/rldt/blob/3ab73e775efd0f041ad28170c616146be86bb6ea/agent/finetune/train_fm_svgd_img_agent.py),
[visual critic loop](https://github.com/RPFey/rldt/blob/3ab73e775efd0f041ad28170c616146be86bb6ea/agent/finetune/train_qfm_img_agent.py),
[critic ensemble](https://github.com/RPFey/rldt/blob/3ab73e775efd0f041ad28170c616146be86bb6ea/model/common/critic.py),
and [visual configuration](https://github.com/RPFey/rldt/blob/3ab73e775efd0f041ad28170c616146be86bb6ea/cfg/robomimic/finetune/square/ft_svgd_flowmatching_mlp_img_vit.yaml).

## Verification

The final implementation also passed the **eight-particle** full-expert check:
`K=8`, `T=10`, ten primitive LIBERO steps, one critic update, and one actor
update. The actor gradient norm was **0.0323868** and the parameter-update norm
was **0.00901871**; every reported metric was finite. See its local
`runs/rldt_pi05_k8_validation/summary.json`,
`runs/rldt_pi05_k8_validation/metrics.jsonl`, and saved expert/learner
under `runs/rldt_pi05_k8_validation/final/`. This run used `--smoke --particles 8`,
so its bootstrap-sample count was one and its update budget was deliberately small.

The saved K=8 expert was then reloaded through the evaluation command and completed
one LIBERO Spatial task-0 episode successfully in **80 primitive steps** (seed
100000). The local `runs/rldt_pi05_k8_evaluation.json` report records the
checkpoint and full configuration. This verifies the saved-model rollout path;
one successful episode does not establish a training improvement.

The real π0.5 full-expert smoke completed with `K=2`, `T=10`, ten primitive
LIBERO steps, one critic update, and one actor update. It saved and reloaded the
430,098,464-parameter action expert with the VLM frozen. The recorded actor
gradient norm was **0.0381648**, and its optimizer-update norm was **0.00860636**.
See `runs/rldt_pi05_smoke_v2/summary.json` and
`runs/rldt_pi05_smoke_v2/metrics.jsonl`. No episode completed
in this short run; these results establish execution and nonzero finite updates,
not a success-rate improvement.

A native export at `runs/rldt_pi05_native_validation/` successfully loaded through
upstream OpenPI's policy factory, with action horizon 10 and internal action
dimension 32. The local `runs/rldt_pi05_native_load.json` report records
that compatibility check. It is a loading check, not a rollout or a numerical
equivalence claim after bfloat16 conversion.

A separate GPU sampling-parity check (`runs/rldt_pi05_parity.json`) compared
the adapter with ordinary upstream `Policy.infer`, using the same exported
checkpoint, mixed-precision model, synthetic observation, and Gaussian noise.
The maximum absolute discrepancy was **0.00173307** in normalized coordinates
and **0.00173256** after decoding. The check used `atol=rtol=0.025`; Euler
time-grid rounding through bfloat16 layers means the two samplers are not
bitwise identical. This parity run skipped its optional gradient check because
the real RLDT training checks above already exercised full expert gradients.

The CPU numerical/integration suite contains 45 checks across
[`test_rldt_math.py`](../tests/test_rldt_math.py),
[`test_rldt_pi05.py`](../tests/test_rldt_pi05.py), and
[`test_rldt_pipeline.py`](../tests/test_rldt_pipeline.py). The earlier baseline
regression run reported 116 passed and 3 skipped. Test results and verification
details are also tracked in [the verification record](verification.md).

Numerical checks establish transport direction, repulsion sign, bandwidth
semantics, coordinate masking, detached targets, regularizer scaling, and
termination/truncation returns. The toy smoke exercises collection, replay,
critic updates, actor updates, and checkpoint serialization. Real-checkpoint
checks additionally establish unchanged-weight sampling equivalence, correct
expert selection, and finite gradients through the selected parameters.

Successful software checks do not establish improved LIBERO success rates. That
requires matched pre/post-training evaluation over the intended task and official
initial states, with multiple seeds for a performance claim.

All `runs/` evidence paths above refer to the prepared server. Raw artifacts are
ignored by Git; the measured values are retained here and in the verification record.

To rerun the RLDT tests in the existing JAX environment:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 4-7 \
  third_party/openpi/.venv/bin/python -m pytest \
  tests/test_rldt_math.py tests/test_rldt_pi05.py tests/test_rldt_pipeline.py -q
```
