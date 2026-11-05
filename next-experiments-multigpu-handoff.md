# RL-VLA next-experiment handoff: parallel GPU campaign

Prepared 2026-10-31. Source workspace: `/mnt/projects/jg/kaifany/RL-VLA`.

**First production run: run RLDT on Object 2 using the destination server's existing pi_0.5 LoRA setup, for up to 100 total RL iterations. Start this before the other production experiments. Use all GPUs available to this campaign for independent experiments, seeds, and evaluations.**

This file is an execution brief for another server or coding agent. It does not claim that the destination's LoRA code, checkpoints, hardware, or data have been inspected. No jobs were submitted by writing this handoff. The companion `next-experiments-multigpu-matrix.csv` lists experiment families; it is not an executable scheduler manifest.

## 1. Assignment and first-run contract

Discover the destination's working LoRA implementation and reuse its current model configuration, adapter rank/scale, target modules, checkpoint loader, precision, and proven launcher. Do not replace it with an invented LoRA recipe. The task defaults to **Object 2: pick the salad dressing and place it in the basket, LIBERO-PRO position swap**, because this is the zero-success failure motivating the new research direction.

Record two separate facts:

1. **Initialization:** the original base checkpoint and any existing supervised LoRA adapter/checkpoint loaded before RL.
2. **RL trainable scope:** which backbone adapters, action-expert adapters, full action-expert weights, and projections actually receive RL gradients.

An existing LoRA-fine-tuned checkpoint followed by action-expert-only RL is a valid first run if that is the destination's current setup. Label it accurately. It is not evidence that RL trained the VLM adapters. If the current setup is intended to train LoRA adapters during RL, demonstrate nonzero gradients and parameter changes for those adapters before its production launch. Do not silently fall back to the original action-only implementation if the intended LoRA path is broken.

### E00 — mandatory first run

| Setting | First-run requirement |
| --- | --- |
| Task | Object 2, published position-swap condition |
| Initialization / LoRA | Destination's existing configured checkpoint and LoRA recipe; preserve and record them |
| Training seed | 0 |
| RL budget | 100 total iterations; do not extend beyond 100 merely because scores remain zero |
| Collection | 32 policy decisions/iteration, at most 5 controller steps/decision, unless the existing LoRA recipe explicitly differs; record differences |
| Nominal maximum interactions | 3,200 decisions / 16,000 controlled steps under the above settings; episode counts depend on termination |
| Reward | Existing binary task success; add no new shaping or teacher data in E00 |
| Checkpoints | Initialization and iterations 25, 50, 75, 100 |
| Primary validation | Iterations 0, 25, 50, 100; 50 fixed validation states each |
| Selection | Best validation success count, ties earliest; initialization is eligible |
| Final testing | Baseline and selected checkpoint on the same 50 test states; 50 original-layout states for retention |
| Recording | Per-episode outcomes and stage telemetry; baseline/selected videos, including failures; checkpoints sufficient to replay other iterations |
| Reporting | Plot success versus iteration and controlled steps; distinguish pre-LoRA baseline, post-LoRA/pre-RL baseline, and post-RL |

Preserve the existing RLDT optimization settings for E00 unless an actual compatibility issue requires a documented change. Historical defaults are listed in section 4. If the existing LoRA recipe uses a different base model or pretraining dataset, its scores are a new initialization condition, not an isolated test of LoRA versus the old runs.

Run to 100 despite a validation decline: that decline is part of the research question. Stop only for an execution failure, nonfinite updates, invalid simulator contract, exhausted allocation, or a user-requested stop; save partial state and the precise reason. Do not call a stopped or queued experiment completed. A 100-iteration all-zero result is a completed experiment.

### Launch sequence

1. Locate the repository, GPU allocation, existing LoRA configuration, model/adapter checkpoint, Python environments, and Object 2 assets on the destination.
2. Resolve paths and capture a manifest. Check model loading, normalization, controller semantics, simulator success, and the claimed trainable scope with a short isolated smoke run. Do not reuse smoke-updated weights for E00.
3. Start E00 on the first suitable GPU. Its initialization validation is part of this first pipeline. Do not wait for the complete campaign tooling, external datasets, or all historical evaluations before starting it.
4. Once E00's configuration and smoke checks are valid, use the remaining GPUs for E01, E02, E03, E04, and E15 as dependencies allow. E00 need not finish before those start.
5. Keep every available campaign GPU assigned to a ready job. Populate the dependency-aware queue from the families below, then execute all ready core families. Record a concrete dependency for every deferred family.

## 2. What the existing evidence establishes

| Experiment | Baseline to selected held-out success | Relevant observation |
| --- | --- | --- |
| Object 0, action-expert RLDT | 20/50 to 39/50, iteration 50 | Sampled frames from all 11 residual failures show wrong-butter selection |
| Object 1-hard, action-expert RLDT | 38/50 to 46/50, iteration 25 | Validation later falls from 44/50 to 27/50 at iteration 100 |
| Object 2, action-expert RLDT | 0/50 to 0/50 | 0 successes / 57 completed training episodes; sampled test failures select tomato sauce at the old location |
| ManiSkill, action-expert RLDT | 0/50 to 0/50 | 0 successes / 200 training episodes; production videos were not saved |
| Object 0 same-task layout pilots | 4/10 to 7/10; 1/10 to 3/10; 5/10 to 8/10 | Soup instruction unchanged: these are not cross-object transfer measurements |

The action-only runs trained 430,098,464 action-expert/projection parameters and froze 2,923,335,408 parameters. Object 0's matched 20 successful episodes became faster, averaging 165 to 134.95 controller steps. Object 1-hard's lower all-episode mean mostly reflects fewer timeouts; do not report a universal speed gain.

Object 0 validation declines from 39/50 at iteration 50 to 33/50 at 100. Late failures include both wrong-object selection and transport/placement errors. The current actor learning rate stays constant after warmup, the reference policy moves every iteration, and many critic updates use two sampled replay segments. Drift, critic error, and limited coverage are hypotheses, not established causes. Baseline success is not sufficient for improvement: Spatial 1 went from 20% to 18%.

These are single-training-seed results. A score of 39/50 is not a demonstrated 78% algorithmic ceiling. Historical test episodes have already informed diagnosis; use an additional fresh, locked test set for final new scientific claims.

## 3. Use all allocated GPUs through independent jobs

Prefer one independent training process per GPU when the model fits. Parallelize seeds and ablations before adding distributed training: this produces more informative comparisons without changing a run's batch size or interaction budget. If the existing LoRA recipe requires multiple GPUs, allocate the smallest proven group to it and use the remaining devices for other jobs. Do not claim identical optimization if distributed batch/accumulation settings change.

Use **all available GPUs assigned to this campaign**, including devices that become free later. Do not kill unrelated processes or take GPUs allocated to other users. Respect Slurm or other scheduler assignments; a process-local CUDA device 0 can refer to a different physical GPU in each job.

Implement a bounded worker queue on the destination with these properties:

- One atomic job claim per worker; immutable config and source snapshot per job.
- A GPU is released and immediately assigned the highest-priority ready job after completion or failure.
- Independent CPU/NUMA affinity and thread limits. The source `run_rldt.sh` otherwise defaults to the same four preferred CPU cores, which would bottleneck parallel jobs.
- Separate output, temporary, renderer, and compilation-cache paths, or a verified concurrency-safe shared cache. Share model downloads read-only.
- JAX/Torch and simulator rendering remain within each job's allocated device set. Assert renderer device identity for GPU-rendered environments.
- No two evaluators append to the same JSONL or modify the same experiment status file. Parallel evaluation shards write separate outputs and merge with episode-ID checks.
- Serialize a run's checkpoint publication; evaluate only complete immutable checkpoints. Background evaluation must not mutate training RNG, optimizer, or replay state.
- Retry a transient execution failure once after recording its cause. Repeated failures block that job, not the entire queue. Do not keep launching identical failing jobs.
- Monitor memory, utilization, CPU saturation, disk usage, job progress, and evaluation backlog. Low GPU utilization may be a simulator/CPU bottleneck; diagnose before oversubscribing.

Example initial assignment for eight single-GPU slots, after E00 readiness:

| GPU slot | Work |
| --- | --- |
| 0 | E00: current LoRA + RLDT, Object 2, seed 0 |
| 1–2 | E01: identical seeds 1 and 2 |
| 3 | E02: matched-initialization action-expert-only control, seed 0 |
| 4 | E04: Object 0 stability reference, seed 0 |
| 5 | E04: Object 1-hard stability reference, seed 0 |
| 6 | E03: saved-checkpoint cross-task evaluation |
| 7 | E15: grounding/ManiSkill diagnostics, then next ready ablation |

For more GPUs, expand remaining seeds and ready variants. For fewer GPUs, preserve priority and dependencies. No GPU should wait for PrismaX access while a local experiment is ready. This is an example mapping, not an assumption about the destination's GPU count.

## 4. Shared experimental protocol

### Initialization, seeds, splits, and budgets

- Use training seeds **0, 1, 2** for primary comparisons. E00 is seed 0 of the first LoRA condition, not an extra fourth replicate. Record supervised-adaptation seeds separately from RL seeds; shared SFT weights only measure RL-seed variation.
- Compare variants from the same effective starting policy whenever isolating training scope, reward, or optimizer changes. If introducing zero-initialized adapters, verify action equivalence to the unadapted model before training.
- Reuse a completed control only when initialization, code, trainability, dataset, reset schedule, RNG, reward, and budget match. Historical runs are contextual evidence until this equality is established.
- Historical LIBERO splits are 100 train, 50 validation, 50 test states, plus 50 original-layout retention states. Relocate paths without changing asset hashes or state IDs. Keep all frames, permutations, and instructions derived from a reset family in one split.
- New grounding/factorial datasets require fresh train/validation/test reset families. Lock final test families before training and never select demonstrations or hyperparameters using their outcomes.
- Main RL conditions: 100 iterations, 32 decisions/iteration, at most 5 controller steps/decision. Report actual decisions, controlled steps, episodes, successful episodes, teacher steps/queries, simulator settling, and evaluation interactions separately.
- Main validation: 0/25/50/100. E04 additionally evaluates 75 to resolve late decline; prespecify this schedule for that family. Save iteration 75 for all runs even when it is not a selection candidate.
- Select by unshaped binary task success on validation; ties earliest, including initialization. A new comparator must not silently inherit the old helper's rule excluding iteration 0.
- Supervised adaptation: reuse the destination's finite existing SFT update budget across matched arms; record examples seen, batch size, gradient accumulation, updates, and checkpoint selection. If none exists, use a declared pilot cap of 10,000 optimizer updates with validation every 1,000, then freeze this budget before comparisons. Do not turn E00 into a new SFT sweep.
- Final confirmation: 3 training seeds, at least 50 fresh test families per task, and 3 prespecified inference-noise seeds per family. Aggregate uncertainty by reset family and training seed; repeated noise samples are not independent scenes.

Historical RLDT defaults, for reproduction/control arms rather than an instruction to overwrite E00's destination recipe:

```text
critic_steps=64              actor_steps=4
critic_batch_size=64         actor_batch_size=1
critic_trajectories=2        replay_capacity=4096
ode_steps=10                execute_steps=5
particles=8                 kappa0=0.001
actor_lr=2e-5               critic_lr=1e-3
critic_width=256            critic_warmup_iterations=2
actor_warmup_steps=32       target_momentum=0.995
gamma=0.999                 td_lambda=0.5
bootstrap_samples=2         reference=moving
consistency_weight=0.5      constraint_weight=0.2
balanced_replay=true        fresh_fraction=0.33
clip_bootstrap=true         save_every=25
```

### Required outcomes

Log binary task success and identity of the first grasped object, plus correct-target approach, grasp, lift, transport, release/placement, wrong-object attempts, clipping/saturation, and timeouts. Use simulator identity/contact telemetry where possible; label visual-only classifications as such. For ManiSkill, distinguish cumulative ever-grasped flags from current grasp state.

Report success-conditional controller steps and **paired common-success** steps separately from all-episode means. Within the same simulator, report simulated completion seconds using its control frequency. Report wall time, training GPU-hours, validation GPU-hours, and teacher cost separately; 100 iterations and equal controller steps need not be equal compute across methods.

For RL diagnostics, record critic targets/predictions, measured rollout returns, actor/critic gradients, update norms, policy drift, reward components, replay success/progress composition, and actual trainable-group changes. Loss magnitudes alone are not comparable across objectives. Use a fixed validation state/action bank for critic calibration; changing replay distributions cannot establish overestimation.

## 5. Full experiment list and dependencies

All families below belong to this research plan. P0 starts first; P1 is the highest-value local work; P2 expands the mechanism and transfer studies; P3 is an exploratory extension after its prerequisites. Do not launch every dependent combination blindly. Resolve dependencies, deduplicate identical conditions, and generate explicit bounded jobs. Use idle GPUs for other ready families.

### E00 / E01 — current LoRA + RLDT, Object 2

**Question:** Does the destination's current LoRA-initialized/configured policy improve under RLDT up to 100 iterations?

E00 is seed 0 under section 1; E01 adds seeds 1 and 2. Evaluate the original pre-LoRA policy, existing LoRA checkpoint before RL, and selected post-RL policy through the same adapter. If the pre-LoRA checkpoint is unavailable, report that limitation rather than claiming an isolated LoRA gain. Preserve zero-success outcomes and all-negative validation curves.

**Dependency:** working destination LoRA path and relocated Object 2 assets. **Priority:** P0/P1. **Runs:** 3 total, not 3 plus E00.

### E02 — initialization and parameter-scope controls

**Question:** Is any gain due to supervised LoRA initialization, RL adaptation of backbone adapters, or action-expert updates?

Compare (A) the original `pi05_libero` with action-expert-only RL, (B) the destination's LoRA-initialized policy with backbone/adapters frozen and action-expert-only RL, and (C) that same effective LoRA initialization with backbone LoRA and the agreed action scope trainable during RL. Add frozen evaluations of each initialization. E00/E01 can supply one condition only if exactly equivalent.

Use zero-initialized adapters or merge/load paths that preserve effective starting weights when necessary; verify equivalence on fixed inputs/noise. Keep reward, data, resets, RL budget and decoder identical. If a condition is unsupported, implement and verify it; do not relabel another condition.

**Dependency:** E00 readiness, scope checks. **Priority:** P1. **Runs:** up to 3 conditions × 3 seeds, less exact reuse.

### E03 — transfer matrix using existing checkpoints, no new policy training

**Question:** Does Object 0 adaptation help or hurt Object 1-hard and Object 2?

Evaluate original `pi05_libero`, Object 0 iteration 50, and Object 1-hard iteration 25 on all three tasks, with original and shifted layouts. Add E00-selected and its matched initialization when available. Use paired reset/inference seeds. Produce a checkpoint-by-task transfer matrix, first-grasp confusion matrices, retention, and matched-success timing.

Keep the soup-task layout pilots separate; they did not change the requested object. New models can be added asynchronously without blocking evaluation of existing ones.

**Dependency:** saved checkpoints and compatible LIBERO adapter. **Priority:** P1. **Jobs:** 3 initial checkpoints × 3 tasks × 2 layout conditions, then 2 additional checkpoints; split safely across GPUs.

### E04 — why performance declines after an early peak

**Question:** Are constant updates, moving-reference drift, or narrow replay coverage responsible?

Run Object 0 and Object 1-hard from the same original initialization with five variants:

1. Historical reference settings.
2. Actor learning rate reduced to one quarter after iteration 25; all else unchanged.
3. Fixed reference at the original initialization; retain the same constraint coefficient.
4. Sample 8 replay segments rather than 2; keep critic steps and batch sizes fixed.
5. Combine variants 2–4 as a practical stabilization recipe, analyzed after the single-factor arms.

Evaluate 0/25/50/75/100 and retain all checkpoints. Measure drift, stage failures, and Q-versus-realized-return calibration. Replay volume and unique reset exposure must be reported. Do not interpret decreasing TD loss as improved policy quality. If resuming from a historical best checkpoint, label it a **warm-start branch** unless replay, simulator state, optimizer, RNG and counters are restored exactly; the source `--init-from` creates fresh replay/environments. Full matched runs from iteration 0 are the primary comparison.

**Dependency:** historical reproduction path; no new LoRA dependency. **Priority:** P1. **Runs:** 2 tasks × 5 variants × 3 seeds = 30, less exact reuse.

### E05 — reward and successful-experience bootstrap

**Question:** What overcomes all-zero sparse-reward experience?

On Object 2, compare (A) sparse RLDT, (B) target-aware progress shaping, (C) 10 verified successful training demonstrations plus supervised anchoring and demonstration replay, and (D) shaping plus demonstrations. Fix parameter scope and effective starting weights across these arms. Repeat on ManiSkill once E15 exposes its failure stages and a valid teacher/reward exists.

Progress must refer to the instructed object: approach, grasp, lift, and destination progress. Prefer a bounded potential-difference reward with the correct per-step discount and terminal/truncation handling. Preserve original binary success for selection and test. Success criteria must not be loosened to count wrong objects.

Dense rewards require revising the current `[0,1]` bootstrap clipping when incompatible with shaped returns. Replay currently treats any nonzero reward as positive; use explicit success/progress labels instead. Record original reward and each shaping component. Demonstrations must use the actual executed action, next observation, and correct action coordinates.

**Dependency:** stage telemetry, reward validation, demonstration importer/teacher. **Priority:** P1 on Object 2; P2 on ManiSkill. **Runs:** 4 variants × 3 seeds per task, less exact reuse.

### E06 — offline demonstrations versus on-policy distillation (OPD)

**Question:** Does labeling the student's own visited states teach better correction/recovery?

Compare supervised adaptation without RL, offline demonstrations plus RLDT, and teacher labels on student-visited states plus RLDT. Use the same teacher, task distribution, starting weights, and a matched total teacher-label budget (default pilot: 1,000 action-chunk queries per condition). Count teacher-executed rollouts separately. Query states for the offline condition from a fixed pretraining policy rather than secretly using the evolving student.

Use flow-matching targets for coherent action chunks, not the mean of incompatible action modes. A teacher must demonstrate correct target selection in changed layouts. Teacher labels for unexecuted actions belong in supervised data, not fabricated Q-learning transitions with the student's next state. Evaluate the supervised checkpoint before RL and the final mixed procedure separately. Keep some original behavior examples for retention, equally across matched arms.

**Dependency:** competent teacher and valid chunk labels. **Priority:** P2. **Runs:** 3 variants × 3 seeds on Object 2.

### E07 — action scope × layout-diverse supervision

**Question:** Is the limitation representation capacity, training-data shortcuts, or both?

Use a 2×2 factorial: action expert only versus action expert + VLM LoRA; fixed layouts versus positions independently varied from object identity. Each data arm has the same three requested objects, demonstration count, teacher, update budget and target instructions. A pilot budget is 30 successful trajectories, balanced at 10 per requested object; the diverse condition distributes each object's trajectories over several valid positions. Do not compare one-object data with three-object data and attribute the difference solely to layout variation.

Train with the same supervised flow-matching objective. Evaluate before RL, then apply identical 100-iteration RLDT refinement to each resulting checkpoint, initially freezing the adapted VLM for this refinement so SFT and RL effects remain separable. Joint VLM-RL scope is measured in E02. Test new layouts, instructions on identical scenes, and objects excluded from adaptation.

**Dependency:** multi-object counterbalanced data and working VLM LoRA. **Priority:** P1 once data ready. **Runs:** 4 SFT→RL pipelines × 3 seeds.

### E08 — explicit grounding supervision and preservation

**Question:** Does instruction-to-object supervision improve behavior beyond action labels?

At fixed LoRA scope and diverse data, compare flow-matching alone, flow-matching plus target localization, and those losses plus preservation of original behavior/representations. Use simulator identity/segmentation to label visible target regions or positions. Labels supervise predictions from native observations; do not silently provide ground-truth target coordinates to the deployed policy.

Evaluate localization and first-grasp identity independently from placement. For representation-preservation loss, avoid anchoring all task-relevant representations so strongly that correction is impossible; tune its weight only on validation. Backbone/grounding heads that are unused at inference must be documented. An oracle target-coordinate input is a separate upper-bound diagnostic, not the same RGB-policy method.

**Dependency:** E07 data, reliable labels. **Priority:** P2. **Runs:** 3 conditions × 3 seeds; reuse the matching E07 control.

### E09 — demonstration data-efficiency curve

**Question:** Is ten demonstrations enough, and does generalization improve with coverage?

Use nested training-only subsets of 0, 10, 50, and 100 verified successful demonstrations with a fixed parameter scope and reward setting. Balance target identity/layout coverage as far as each budget permits; disclose the coverage of small subsets. Use identical subset membership across methods and the same supervised update budget so data quantity is not confounded with extra updates. Report overfitting/epochs for small subsets. Then apply the same 100-iteration RLDT budget.

**Dependency:** a validated recipe and demonstration bank. **Priority:** P2. **Runs:** 4 budgets × 3 seeds, less exact reuse.

### E10 — curriculum and exploration

**Question:** Can useful experience be reached with less teacher dependence?

Compare the selected sparse/progress-reward reference, a curriculum of valid resets progressively farther from success, and a structured exploration variant. For Object 2, independently vary target identity/location rather than progressively teaching a new fixed location. For ManiSkill, a near-placement or already-grasped state can precede full resets; retain a prespecified fraction of full-task starts throughout.

The current SVGD repulsion is scaled by the critic-gradient norm. Test a documented nonzero temperature floor as the exploration variant after reward scale is fixed; choose its value on a small training/validation pilot, then lock it. If testing 16 instead of 8 particles, make it a separate arm and report compute. Particles are candidate actions, not additional executed trajectories. Never choose curriculum/reset difficulty from test outcomes.

**Dependency:** simulator state reset support and meaningful stage telemetry. **Priority:** P2/P3. **Runs:** 3 primary arms × 3 seeds on Object 2; ManiSkill extension if supported.

### E11 — external data, including PrismaX

**Question:** Does external manipulation or grounding data improve target selection and policy transfer?

Compare target-simulator data only; target data plus PrismaX robot demonstrations; target data plus externally annotated grounding examples; and combined action/grounding data. Add an equal-budget extra-simulator-data comparator. Keep target-data exposure and training compute matched or report both explicitly. Apply the same RLDT refinement only after measuring SFT-only effects.

Inspect an actual sample package before implementing the pipeline: actions, executed versus desired controls, timestamps, instructions, camera calibration, robot identity, controller frequency, gripper encoding, and coordinate conventions. PrismaX MCAP/multiview video packaging alone does not establish compatible action targets or grounding/reasoning annotations. Video-only examples require representation/grounding objectives, not invented action labels. Use an embodiment-specific adapter/head or validated common action representation for robot demonstrations; do not apply LIBERO normalization to incompatible YAM joint commands.

PrismaX reports a relevant pi_0.5 expert-only versus LoRA comparison on its custom pick/place task. That motivates this experiment, not an expected LIBERO improvement or semantic guarantee. Define any claimed reasoning task separately from object recognition. Split by scene/episode/object instance; never put neighboring frames in different splits. Use the destination's available dataset agreement/access; do not block other experiments while access is unresolved.

**Dependency:** accessible data, labels and validated adapter. **Priority:** P2 after local grounding ablations. **Runs:** 5 conditions × 3 seeds, less matched-control reuse.

### E12 — joint multi-object training and retention

**Question:** Can shared skills improve without specializing to one object?

Compare separate Object 0 and Object 2 adaptation, balanced joint Object 0/Object 1/Object 2 training, and sequential adaptation with versus without rehearsal of earlier tasks. Begin with a fixed total interaction budget: joint training divides the same 100-iteration-equivalent budget among tasks; sequential conditions use Object 0 for the first 50 iterations and Object 2 for the next 50. Rehearsal divides that second-stage budget between current and earlier tasks rather than adding interactions. Report per-task exposure. If adding equal-per-task exposure, label it a separate larger-compute condition.

Evaluate the complete task matrix and original-layout retention after each stage. Keep the instruction and simulator success target synchronized when switching tasks. This studies interference as well as positive transfer.

**Dependency:** E03 evaluation matrix and multi-task collector. **Priority:** P2. **Runs:** 5 conditions × 3 seeds, less exact reuse.

### E13 — cross-simulator transfer and target adaptation

**Question:** Does LIBERO adaptation help RoboCasa immediately or reduce target adaptation cost?

First evaluate the same `pi05_libero` effective checkpoint before/after source adaptation with one validated RoboCasa input/output adapter and target normalization. Keep raw `pi05_base` as a separate comparator, not the before-policy. The current RoboCasa PandaOmron contract has 12D arm/gripper/base/torso/mode controls, versus LIBERO's fixed-arm 7D actions. Validate this contract before interpreting zero success. The model's padded output size is not evidence of mobile-base competence.

Measure zero-shot transfer before target training. Then run matched 100-iteration target adaptation from the unadapted-source, source-adapted, and raw-base initializations. Add a mixed-source/target rehearsal arm if target training harms retention. Keep target reward and teacher access equal. Use per-stage metrics and a strict simulator RPC watchdog; previous RoboCasa runs spent most of their allocation evaluating long failures. Budget evaluation time from a short measured pilot before requesting wall time.

A cleaner visual-domain transfer subexperiment holds robot/action semantics constant and changes scenes/cameras. Treat true simulator/robot transfer separately. Export complete effective weights and use an explicit target adapter; do not bypass expert-checkpoint base/normalization compatibility checks.

**Dependency:** validated target adapter, sufficient simulator budget, E03 selected checkpoints. **Priority:** P2. **Runs:** frozen evaluations plus 3 target-training conditions × 3 seeds; one rehearsal extension × 3 if needed.

### E14 — algorithm comparison under useful feedback

**Question:** Is the bottleneck RLDT-specific after correcting the learning signal?

Compare RLDT and Flow-SDE with matched effective initialization, reward, demonstrations, executed action prefix, reset distribution, and controlled-step budget. Run Object 2 and Object 0 using the best prespecified useful-feedback setup. Verify checkpoint conversion parity before attributing differences to the algorithm; JAX and Torch need their own measured frozen baselines. Report compute and policy scope, not just iteration counts.

**Dependency:** reward/data path works in both implementations and conversion checks pass. **Priority:** P3. **Runs:** 2 tasks × 2 methods × 3 seeds, less exact reuse.

### E15 — grounding and failure-stage diagnostics

**Question:** Is identity/location information absent, incorrectly bound to the instruction, or ignored by action generation?

Use `docs/grounding-diagnostics-handoff.md` as the detailed specification for frozen-checkpoint language probes, attention measurements, frozen-feature probes and controlled interventions. These are diagnostic jobs, not evidence of semantic improvement by themselves. Do not let their completion block E00.

For ManiSkill, record 20 fixed validation rollouts with baseline and available selected policy before choosing a reward intervention. Include images, current/cumulative grasp flags, object/plate distances, action clipping, and contact/placement status. No videos from the earlier production run means its physical failure stage is not yet established. For LIBERO, collect object-contact identity telemetry to verify the sampled-frame audit.

For instruction interventions, change the simulator's designated target/goal consistently with the requested instruction while restoring the same physical state. Changing only the prompt while leaving the success checker tied to soup is not a valid cross-object success test. Retain the original checker in ordinary benchmark evaluation.

**Dependency:** recorded inputs and simulator support. **Priority:** P1, parallel with E00. **Jobs:** bounded diagnostic batches, not policy-training seeds.

### E16 — fresh held-out confirmation

**Question:** Which findings persist beyond the layouts and seeds used for development?

Choose finalists using validation only: original baseline, action-expert-only reference, best stabilized RLDT, best LoRA/grounding condition, and best transfer-oriented condition, deduplicating identical models. Evaluate fresh locked reset families, held-out layout combinations, and held-out object instances where supported. Report seen-instance/new-layout separately from genuinely unseen-instance/category results.

Use three training seeds and three inference-noise seeds per reset family as specified above. Produce confidence intervals, full failure counts, task transfer matrices, retention, and cost. No further hyperparameter tuning on this set; another round requires a new locked test set.

**Dependency:** finalists and locked confirmation dataset. **Priority:** P2 final stage. **Jobs:** checkpoint × task × split evaluation shards.

### E17 — hindsight goal relabeling, exploratory

**Question:** Can otherwise wrong-object rollouts provide reusable manipulation experience?

Compare the selected goal-conditioned reference with a hindsight-relabeling extension. Relabel instruction/goal, reward, success/termination semantics, and all goal-dependent policy/critic features consistently. A tomato-sauce placement may become experience for a tomato-sauce goal; it remains failure for salad dressing. Reset-state and trajectory feasibility must be preserved. Account for early success under the relabeled goal rather than treating arbitrary later transitions as valid continuation.

This requires implementation and may help generic skills without fixing requested-object selection. Keep it behind direct grounding/teacher experiments and judge it on original-goal success and transfer, not relabeled reward.

**Dependency:** validated goal-conditioned replay and recomputation. **Priority:** P3. **Runs:** 2 conditions × 3 seeds, less exact reuse.

## 6. Implementation checks before claiming LoRA + RLDT works

The source checkout's `src/rl_vla/rldt_pi05.py` is action-expert-only. Its prepared prefix context is detached. The destination may already have solved this; inspect its implementation rather than assuming the source path supports LoRA.

Verify:

1. The intended adapters are in the optimizer parameter tree and saved checkpoints. Report named-group parameter counts, gradient norms, and before/after deltas from a real update.
2. For joint backbone RL, encoding is recomputed inside the differentiated actor loss from current learner parameters. Removing a `stop_gradient` outside the differentiated computation is not enough.
3. Collection, critic preparation, evaluation, and restored-checkpoint inference all encode observations with the correct **current effective backbone weights**. A cached baseline prefix or stale `flow.params` would invalidate a purported LoRA evaluation.
4. Replay does not mix old stored VLM features with newly encoded ones unnoticed. Recompute from stored observations or use a deliberately frozen critic encoder with a documented contract.
5. Reference-policy constraints use the corresponding reference backbone and action weights. If comparing policies with different prefixes, do not reuse one policy's cache as though both representations were equal.
6. Save/load round trips preserve actions on identical observations/noise. Effective base+adapter and any merged export agree within a documented tolerance. Never attach an expert checkpoint to a different base/normalization by bypassing hash checks.
7. A successful simulator trajectory produces the expected original binary success and a wrong-object placement does not. Dense shaping remains a separate training channel.
8. Prefix/vision freezing is explicit: VLM transformer LoRA is not synonymous with training the image encoder. Preserve the existing image-encoder scope in E00; add visual-encoder adaptation only as a separately recorded extension if diagnostics motivate it.

These are necessary correctness checks for the experiment, not a new research sweep that should delay a working destination setup.

## 7. Source assets and relocation

The destination must discover its own absolute repository root and record an old-to-new path map. Do not assume `/mnt/projects/jg/kaifany` exists there. Copy or locate the exact assets below; do not move/delete the originals. The source working tree contains substantial uncommitted and untracked work, so a bare Git clone is insufficient to reproduce these experiments. Capture tracked changes, required untracked files, and upstream revisions.

| Source-relative path under RL-VLA | Use |
| --- | --- |
| `runs/object_adaptation_20261028/tasks/object_2/` | Object 2 protocol, train/validation/test/source configs, states, hashes |
| `runs/pro_multimethod_20261027/tasks/object_0/` | Object 0 equivalent inputs |
| `runs/object_adaptation_20261028/tasks/object_1_hard/` | Object 1-hard inputs |
| `checkpoints/openpi/pi05_libero/` | Original LIBERO initialization and normalization |
| `checkpoints/openpi/pi05_base/` | Separate general-base comparator, where available |
| `runs/pro_multimethod_20261027/methods/rldt/object_0/training/iteration_000050/` | Saved Object 0 policy for transfer |
| `runs/object_adaptation_20261028/methods/rldt/object_1_hard/training/iteration_000025/` | Saved Object 1 policy |
| `runs/object_adaptation_20261028/methods/rldt/object_2/training/iteration_000025/` | Historical zero-success comparator |
| `datasets/benchmark_assets/libero_pro/` and referenced upstream LIBERO assets | Exact BDDLs, textures and initial states |
| `runs/analysis_20261031/experiment_review/` | Audited results, plots, guide, videos and manifests |
| `runs/analysis_20261031/question_analysis/` | Residual and late-failure contact sheets |
| `integrations/multibench/` | ManiSkill/RoboCasa adapters and frozen core snapshot |
| `docs/grounding-diagnostics-handoff.md` | Detailed diagnostic specification |

Rewrite copied path references through a new relocation manifest while preserving data hashes. Do not edit the original frozen launch contracts in place. Create new contracts for the new campaign and preserve provenance to old ones. Rebuild environment paths and native libraries appropriately; copying a virtualenv directory alone is not a portable installation.

The older grounding handoff prohibits policy training **within its diagnostic assignment**. This new handoff explicitly adds the user-requested LoRA/RL training campaign. Likewise, the October 28 campaign policy is historical scope, not an instruction to silently block the new experiment families. Keep old records immutable and create a new campaign configuration for this plan.

## 8. Launcher discovery and verified command interfaces

Read-only discovery examples on the destination:

```bash
pwd
git status --short
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader
rg -n 'lora|LoRA|trainable|freeze|stop_gradient' configs src scripts
```

Also inspect the scheduler allocation and local training documentation. Resolve the actual LoRA launch command there. Record it as an argument array in `jobs.jsonl`, with absolute config/checkpoint/output paths. Do not invent a `--lora` switch: the source CLI does not expose one.

The source CLI supports the following interfaces. These are **templates only after the destination's LoRA-capable implementation and configs have passed the checks above**; they do not make the original source implementation LoRA-capable:

```bash
# Set these to verified destination paths before executing.
: "${EXP_REPO:?Set the destination repository path}"
: "${EXP_TRAIN_CONFIG:?Set the resolved LoRA-capable training config}"
: "${EXP_VALID_CONFIG:?Set its validation-split config}"
: "${EXP_RUN_DIR:?Set a new, unique experiment output directory}"

cd "$EXP_REPO"
bash scripts/run_rldt.sh evaluate \
  --config "$EXP_VALID_CONFIG" \
  --output "$EXP_RUN_DIR/baseline_validation.json" \
  --episodes 50 --seed 100000 --record-episodes 50

bash scripts/run_rldt.sh train \
  --config "$EXP_TRAIN_CONFIG" \
  --output "$EXP_RUN_DIR/training"
```

The training config, not the `--seed` evaluation flag, sets the training seed and iteration budget. A task runner must arrange checkpoint validation and selection; the bare training command does not automatically establish a complete campaign protocol. In this checkout the online-validation helper relies on experiment metadata beside the training directory. Ensure that metadata is valid for the new destination and LoRA loader, or use a verified destination orchestrator. Follow the destination's actual launcher if its interface differs.

Do not run the source `configs/rldt_pi05_libero.toml` unmodified: it defaults to a Spatial task, not the requested Object 2. Do not launch the historical sequential Object campaign wrapper as the parallel dispatcher. Source task runners also contain seed/path/selection assumptions that must be checked before reuse.

## 9. Queue records, deliverables, and stopping conditions

Generate a resolved `jobs.jsonl` with one record per actual job:

```text
job_id, family_id, priority, task, variant, training_seed, sft_seed,
initialization_hash, adapter_hash, trainable_scope, protocol_hash,
config_path, command_argv, output_directory, dependencies,
gpu_count, estimated_memory_gb, max_iterations, max_steps,
teacher_budget, state, scheduler_id, started_at, ended_at, exit_code
```

Use states `blocked_dependency`, `ready`, `running`, `completed`, `failed`, and `cancelled`. A scientific score of zero is not a failed job. Running processes, scheduler state and completed output manifests must reconcile; stale `status.json` alone is not proof a run is active. Record estimated runtimes from pilots and request enough allocation for baseline, training, validation and final evaluation.

Required campaign outputs:

- `environment_inventory.json`, resolved LoRA scope report, source snapshot and relocation map.
- `jobs.jsonl`, scheduler logs and failure/resume instructions.
- Per-run configuration, checkpoint lineage, interaction/compute accounting and evaluation episodes.
- `results.csv` with raw success counts/denominators, task/split/seed, scope, initialization and selected iteration.
- Learning curves with seed variation; first-grasp confusion matrices; critic calibration; retention and cross-task/simulator transfer matrices.
- Demonstration-count curves and comparisons of SFT-only versus additional RL gain.
- Videos of baseline versus selected checkpoint on matched resets, including representative recovered, persistent-failure and newly-lost cases. Select examples by a documented rule; do not show only successes.
- A final report separating measured effects, unresolved mechanisms, dependencies not completed, and the exact commands needed to continue.

Do not launch unbounded sweeps simply to keep devices busy. Finish the finite ready queue, retain all artifacts, release campaign resources, and report unresolved dependencies. Do not start new external-data collection, paid services, or indefinite reservations as a substitute for a blocked experiment.

## 10. Research references

- [RLDT](https://arxiv.org/abs/2606.08602)
- [pi_0.5 heterogeneous supervision](https://arxiv.org/abs/2504.16054)
- [DAgger / supervision at student-visited states](https://proceedings.mlr.press/v15/ross11a.html)
- [Demonstration-assisted exploration](https://arxiv.org/abs/1709.10089)
- [Potential-based shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- [Reverse curriculum](https://arxiv.org/abs/1707.05300)
- [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
- [PrismaX pi_0.5 report](https://prismax.ai/blog/article.html?slug=openpi-pi05-fine-tuning)
- [PrismaX recording/package schema](https://github.com/PrismaXAI/sdk-vla-foundry)

The papers motivate experiments; their results are not measurements of this campaign.
