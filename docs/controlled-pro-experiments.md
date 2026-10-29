# Controlled LIBERO-PRO adaptation and benchmark screening

Historical protocol. The active replacements are the
[LIBERO object campaign](object-campaign-20261028.md) and
[multi-benchmark campaign](multibench-campaign-20261028.md).

The experiment asks whether adapting the original `pi05_libero` checkpoint
improves success on unused initial states of a fixed perturbation condition.
It does not assume that the general `pi05_base` checkpoint can execute LIBERO
without an embodiment-specific adapter and fine-tuning.

## Current experiments

Run root: `runs/pro_multimethod_20261027`.

| Task | Fixed LIBERO-PRO condition | Episode limit |
| --- | --- | ---: |
| Spatial 1: bowl next to ramekin onto plate | Position / swap | 220 |
| Spatial 2: bowl from table center onto plate | Position / swap | 220 |
| Object 0: alphabet soup into basket | Position / swap | 280 |

Each task has 100 generated training states, 50 separately generated validation
states, and 50 published PRO states reserved for final testing. Another 50
original LIBERO states measure retention. State generation rejects successful
resets and duplicate physical configurations. Every task passed 250 reset checks.
The protocol, simulator sources, state files and configuration are hashed before
each experiment. A changed input stops that experiment.

Methods are native RLDT, official pi-RL Flow-SDE PPO, and official pi-RL
Flow-Noise PPO. Pi-RL is implemented in RLinf; these are two methods in that
framework, rather than two unrelated PPO repositories. The pinned RLinf release
is v0.3 (`0505431899574619da86f551bad70b71e0ea2177`); its OpenPI fork is
`c5dc4b9296a1a4739bf52828f28a579f12dce763`.

Both PPO methods introduce stochastic denoising transitions with computable
conditional probabilities. Those probabilities support the PPO likelihood
ratio; they are not an exact density evaluation of the original deterministic
flow sampler. Binary task success supplies the reward, and the learned value
head and GAE estimate advantages. Sparse rewards remain an exploration and
credit-assignment challenge, but do not prevent PPO from being defined.

All methods start from the same native checkpoint weights. PPO uses a recorded
conversion into Torch safetensors, including the original normalization assets.
The action expert has **430,098,464** parameters in this checkpoint. The
**2,923,335,408** VLM parameters stay frozen. PPO additionally learns a value
head, and Flow-Noise learns a noise head. An unused Torch vocabulary projection
is frozen and is not counted as an action-expert parameter.

Each method runs one continuous 100-iteration training trajectory; iterations
25, 50, 75 and 100 are checkpoints from that trajectory. Each iteration collects
32 policy decisions. Each decision samples a 10-action chunk with 10 denoising
steps and executes at most its first five actions. Thus the training allowance is
3,200 decisions and at most 16,000 controlled environment steps. Early success
shortens an executed chunk, so **actual primitive interactions can differ**
between methods. Plots use measured primitive interactions, not an assertion
that equal iterations imply equal physical steps or equal optimizer work.
Reset settling and evaluation are recorded separately from controlled training
interactions.

RLDT uses 64 critic updates and four actor updates per iteration, with two
warmup iterations. PPO uses the official clipped actor-critic objective, one
update epoch, global batch eight, microbatch one, actor learning rate 5e-6 and
value learning rate 1e-4. PPO critic warmup is two optimizer steps. These update
schedules are method-specific; this is an interaction-limited comparison, not
a matched compute comparison or an exact reproduction of the paper's training
scale. Flow-SDE uses noise level 0.5 and no entropy bonus. Flow-Noise uses learned
noise bounds [0.08, 0.16] and entropy coefficient 0.005. The config files are the
authoritative source for every setting.

## Evaluation and interpretation

1. Evaluate each frozen starting policy on the fixed 50 validation states.
2. Train once to iteration 100 and evaluate all four checkpoints on those same
   validation states with paired seeds.
3. Select the checkpoint with the most validation successes; break ties by the
   earliest iteration. Write the selection before opening the final test split.
4. Evaluate the selected checkpoint and its frozen baseline on the identical
   50 PRO test cases and 50 original LIBERO retention cases.
5. Report successes, newly solved cases, lost successes, paired confidence
   intervals, steps to completion, and replay links. A compelling replay is an
   illustration; the paired held-out aggregate is the evidence for improvement.

The selected checkpoint is selected among the trained checkpoints, even if all
score below baseline. This prevents a fallback baseline from concealing harmful
adaptation. JAX and Torch do not generate identical noise from the same integer
seed. PPO therefore gets its own frozen converted-Torch baseline. Cross-backend
checks use explicitly shared initial noise and exact preprocessing comparisons.
Use each method's paired before/after change when comparing backends.
Native RLDT promotes the restored expert to FP32 for optimization, while the
upstream Torch PPO model retains its mixed BF16/FP32 parameter storage. This is
another implementation difference to report. The PPO smoke checks that an
optimizer step actually changes expert weights at the configured learning rate.

These are single-seed pilot runs in three fixed perturbation conditions. A
positive pilot should be followed by multiple independent training seeds and
predeclared perturbation generalization tests. Validation success is not final
test success; do not choose checkpoints using test replays.

## Results and full trajectories

Each method writes `methods/METHOD/TASK/` under the run root:

- `baseline_validation.json` and `validation_000025.json` through
  `validation_000100.json` contain per-case validation outcomes.
- `selection.json`, `comparison.json`, and `validation_progress.json` record
  selection and the paired final comparison.
- `validation_interactions.png` plots success against actual controlled steps.
- `training/` contains diagnostics, checkpoints and training trajectories.
- Evaluation reports have adjacent trajectory directories and video files.
- `status.json`, `input_contract.json` and source snapshots preserve execution
  status and the exact experiment inputs.

RLDT and PPO use different native checkpoint formats. Keep the model config,
normalization, source manifest and checkpoint together when exporting a policy.
PPO's `actor_critic_diagnostics.png` plots policy loss, value loss, approximate
KL, clipping fraction, explained variance and gradient norm when reported.
Undefined explained variance is stored as null; nonfinite loss or gradient
diagnostics stop training.

The trajectory recorder stores both policy cameras and proprioception at every
controlled primitive step, normalized planned actions when supplied by the
policy, decoded planned actions, executed clipped commands, rewards, terminal
flags and reset identity. RGB images are the rotated/resized policy inputs,
not unprocessed simulator camera buffers. A training rollout cut off by the
budget is marked incomplete. Reconstruct contiguous sequences while removing
the duplicated observation at adjacent chunk boundaries.

For a later SFT/distillation experiment, use successful **training** episodes
from the specialists, retain actual executed actions and timing, and balance
spatial and object tasks. Keep validation/test trajectories out of training.
Train the student's actual flow-matching objective with the correct action
normalization and chunk horizon. One can separately compare successful-only
SFT, joint specialist distillation, and an all-weights VLM+expert ablation;
those experiments are not part of the current three-method launch.

## Finding harder conditions

`runs/benchmark_screen_20261027` screens the frozen native policy on 40 PRO
conditions (ten Spatial and ten Object tasks, each with Position and Object
perturbations) and six LIBERO-Plus conditions. The Plus pilot covers object
layout, camera viewpoint and robot initial state changes for Spatial 0 and
Object 0. Each condition uses five freshly generated development states,
excluding published states and the current adaptation splits.

`runs/benchmark_screen_libero90_20261027` follows that screen with four original
LIBERO-90 tasks (IDs 12, 13, 46, 47), five new states each and the official
400-step limit. These test front/back bowl selection and different target
objects. They are outside the documented 40-task LIBERO fine-tuning dataset;
this does not establish their absence from base-model pretraining. LIBERO-90
is another suite within LIBERO, not an independent robot domain. Its queue
produces the same ranking and replay artifacts. Together the screens contain
50 conditions and 250 development episodes.

`ranking.json` ranks completed conditions and links failed replays. A simulator
or execution error is listed separately and is not counted as a policy failure.
Five episodes are a screening pilot, not a reliable benchmark score. Confirm
promising conditions with a larger, newly frozen development sample, then
create unused training/validation/test splits before adaptation. Keep several
conditions with nonzero baseline successes so sparse-reward RL has positive
examples; investigate zero-success conditions for exploration or demonstration
requirements instead of assuming more gradient steps will solve them.

LIBERO-Plus retains the same Panda embodiment and action interface. Benchmarks
using another robot/controller require a matching policy and normalization or
an explicitly evaluated adaptation stage; a low score from an incompatible
action interface is not evidence of weak spatial reasoning.

## Execution

All full GPU experiment stages share allocation `1470624` and the lock
`runs/gpu_allocation_1470624/experiment.lock`. Each stage releases the GPU before
the next stage. CPU preparation can overlap. Queues persist independently of
the chat, but stop if the Slurm allocation expires; queue files distinguish
pending, running, failed and completed work.

PPO's long-run queue requires successful installation, quantitative conversion
parity, and a real GPU update smoke test for each method. A smoke must show a
changed action expert, an unchanged VLM, and finite loss/gradient diagnostics.
Readiness is written only after these checks pass. A submitted smoke or queued
experiment is not a completed training run.

The repaired implementation's two functional smoke tests can run sequentially
in a short separate L40 allocation using `scripts/run_pi_rl_smokes_grasp.slurm`.
These checks do not contribute training data or checkpoints to the full runs;
the comparative experiments still use the original RTX PRO 6000 allocation.
The helper probes assigned GPUs before loading the model and explicitly places
all three worker roles on the selected GPU. This matters when reserving more
than one device to bypass a hardware ECC fault: RLinf's NVML inventory still
sees the other assigned device even when `CUDA_VISIBLE_DEVICES` selects one.

The repaired Torch environment pins Hydra 1.3.2 and OmegaConf 2.3.0: the
RLinf v0.3 entry point requests compatibility behavior removed in Hydra 1.4.
Unused TensorFlow/Keras prerelease packages are removed because their NumPy 2
C API conflicts with OpenPI's NumPy 1 environment. The installer records these
choices; readiness checks the installed dependencies and saves their versions.
The local Ray cluster disables its dashboard and distributed log collector;
stage logs, iteration diagnostics and aggregate TensorBoard remain enabled.
Expandable CUDA allocations are disabled for pi-RL because its interprocess
weight transfer otherwise requires `pidfd_getfd`, which the cluster disallows.
The actor remains resident on the 96 GB GPU. For this expert-only policy, the
frozen VLM and unused expert vocabulary head are excluded from FSDP's parameter
management through `ignored_modules`; they remain in the model and full
checkpoints. This avoids a reproduced Torch 2.7 `NO_SHARD` / `use_orig_params`
failure on the second microbatch when a frozen root contains trainable children.
The exclusion does not change gradients, sampling, losses or the trainable scope.

Official references: [pi-RL paper](https://arxiv.org/abs/2510.25889),
[RLinf OpenPI recipe](https://rlinf.readthedocs.io/en/release-v0.3/rst_source/examples/embodied/pi0.html),
[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus).
