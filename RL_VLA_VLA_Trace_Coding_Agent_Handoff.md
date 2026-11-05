# Coding-agent handoff: RL-VLA Object 0/1/2 RLDT training + VLA-Trace diagnosis

**Date:** 2026-10-31  
**Project:** [kaifanyu/RL-VLA](https://github.com/kaifanyu/RL-VLA)  
**Research question:** When `pi05_libero` selects a distractor at a familiar target location after an object-position swap, is the target identity absent from the vision–language representation, poorly grounded to the instruction, or available but underused by the action expert? Can RLDT improve object-conditioned action selection without changing the VLM?

## 0. Mission and non-negotiable rules

You are a coding agent **with local access to the full RL-VLA repository, GPU hardware, simulator, checkpoints, and raw `runs/` directory**. Carry this project end to end: prepare Object 0, 1, and 2; run three **independent** RLDT training trajectories starting from the unmodified `pi05_libero` checkpoint; preserve checkpoints at iterations **25, 50, and 100**; benchmark five requested policy conditions per task; implement VLA-Trace-compatible diagnostics; and produce a research-grade report with figures, raw measurements, and runnable commands. The checkpoints, training runs, and analysis are not yet claimed to exist or to have succeeded.

**Required policy conditions on each object task:**

1. `pi05_base` (raw pretrained, *not* LIBERO-fine-tuned). This is a representation reference. **Do not invent valid LIBERO action results for this model:** existing `Pi05Flow` explicitly only supports `model_config="pi05_libero"`; implement and validate a distinct embodiment/normalization/action adapter before any closed-loop comparison. If the raw-base interface cannot be made valid, mark its closed-loop and action-attention comparisons `N/A — incompatible action contract`, but complete its compatible perception/representation diagnostics. Never pretend using `pi05_libero` transforms on raw base is its native policy.
2. `pi05_libero` (frozen native JAX LIBERO-fine-tuned baseline, iteration 0).
3. Task-specific RLDT checkpoint `iteration_000025`.
4. Task-specific RLDT checkpoint `iteration_000050`.
5. Task-specific RLDT checkpoint `iteration_000100`.

There are **three independently trained RLDT policies** (one per task), not one shared mult-task fine-tune. Training starts anew from the same untouched `pi05_libero` checkpoint, with no demonstrations/SFT warm start and no warm start from previous RLDT checkpoints.

**Experimental safeguards:** Do not overwrite/modify any existing run or frozen experiment inputs. Use a new dated run root. Training, validation, held-out test, and diagnostic OOD states are disjoint. Never select a checkpoint by test-set success. Preserve checkpoints, RNG seeds, transforms, task definitions, and source hashes. A completed smoke test is not a research result. Do not silently substitute a different LIBERO condition or report local custom tasks as official LIBERO-PRO scores. No fabricated metrics.

## 1. Repository facts verified from GitHub `main`

Repository revision inspected: `d0ef690c4f0d38463c7fdc1eee7aacae2194fcc3` (2026-10-30 UTC). The agent MUST first inspect `git rev-parse HEAD`, uncommitted local changes, cluster environment, and the actual contents of `runs/`. The local checkout may be newer than public GitHub; treat on-disk code, manifests, and logs as authoritative once verified, and record differences.

Core files:

| Path | Role / caution |
|---|---|
| `docs/grasp-setup.md`, `scripts/env.sh` | GRASP setup, pinned Python environments, GPU launch; do not assume an old Slurm allocation still exists. |
| `docs/object-campaign-20261028.md` | **Current** Object 2 PRO swap and Object 1-*hard* six-item cycle; 100 train/50 validation/50 test/50 original retention, validation at 25/50/100. |
| `docs/controlled-pro-experiments.md` | Older Object 0 PRO Position/swap protocol; 100/50/50 and original-task retention. |
| `scripts/prepare_pro_suite_grasp.py` | Existing frozen preparation for **Object 0** PRO swap (`object_0`) plus spatial tasks. |
| `scripts/prepare_object_campaign_grasp.py` | Existing preparation for **Object 2** PRO swap (`object_2`) and **Object 1 hard** (`object_1_hard`). |
| `scripts/run_pro_task_grasp.py` | One-task RLDT experiment: baseline, training, validation, selection, final paired test/retention. |
| `src/rl_vla/checkpoint_validation.py` | Synchronous independent checkpoint validation using the frozen validation split. |
| `configs/rldt_pi05_libero.toml`, `scripts/run_rldt.sh`, `src/rl_vla/rldt.py` | Native JAX RLDT learner, 100-iteration settings, checkpoint save/evaluation. |
| `src/rl_vla/rldt_pi05.py` | `Pi05Flow` frozen VLM + trainable action expert and projections, input transforms, prefix cache, `velocity`, `sample`, expert save/reload. |
| `src/rl_vla/envs/libero_pro.py` | Immutable PRO/custom task resets and split enforcement. |
| `third_party/dependencies.lock.json` | Pinned OpenPI, LIBERO-PRO, etc.; do not upgrade arbitrarily. |
| `runs/README.md` | Existing run naming, status semantics, retained historical runs, gitignored bulk outputs. |

Pinned upstreams listed in the checked repo: OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`; LIBERO-PRO `eafdb809426b13153aa1e4c42d6601844217dfec`. Benchmark dataset: `zhouxueyang/LIBERO-Pro` with the revision and verified assets described by the preparation scripts. Recheck all locally.

**Existing RLDT config:** seed 0; 100 iterations; 32 new policy decisions/iteration; 64 critic updates/iteration; four actor updates after two critic-warmup iterations; eight RLDT particles; ten flow denoising steps; a 10-action predicted chunk, first five executed; up to 16,000 controlled primitive steps per task (actual count may be lower). The full ~430-million-parameter action expert **and** action/time projections are trained; VLM frozen. `save_every=25` produces 25/50/75/100 by default. User-required *analysis* checkpoints are **25/50/100 only**; a 75 recovery snapshot may exist, but is not to be used for validation/selection/comparison. Preserve `final/` too.

`Pi05Flow.save()` writes `expert.npz` + `rldt_pi05.json`; `Learner.save()` also writes `learner.npz`. Do **not** export duplicate multi-GB native full-model checkpoints for every iteration unless strictly required: load expert NPZ with its original frozen backbone via `Pi05Flow.load_params()`.

## 2. Task setup: exactly define Object 0, 1, and 2

| Main task key | Instruction | Primary perturbation | Repository provenance |
|---|---|---|---|
| `object_0` | Pick up the **alphabet soup** and place it in the basket | Official LIBERO-PRO Object Position/swap. In the known failure, soup and butter exchange their usual target regions. | `scripts/prepare_pro_suite_grasp.py` already supports `object_0`. |
| `object_1` | Pick up the **cream cheese** and place it in the basket | Prefer official LIBERO-PRO Object Position/swap to make 0/1/2 directly comparable. | **New preparation wrapper needed**; public repo currently prepares `object_1_hard`, not standard `object_1`. Reuse existing downloader/generator/split code without mutating old protocols. |
| `object_2` | Pick up the **salad dressing** and place it in the basket | Official LIBERO-PRO Object Position/swap. | `scripts/prepare_object_campaign_grasp.py` already supports `object_2`. |

**Separate secondary stress test:** `object_1_hard` = locally defined six-object position cycle (cream cheese/alphabet soup/milk/tomato sauce/butter/orange juice), from `prepare_object_campaign_grasp.py`. Keep `object_1` official swap and `object_1_hard` custom in separately named directories, charts, and denominators. If time/resources prevent making the official Object 1 condition, you may complete Object 1-*hard* but label it clearly and list the missing standard Object 1 experiment as incomplete; do not relabel it as the official swap.

Preparation procedure:

1. Inspect and inventory existing local `runs/pro_multimethod_20261027/`, `runs/object_adaptation_20261028/`, `runs/benchmark_screen_20261027/`, local benchmark assets, and any existing checkpoints. They are ignored by Git and may not appear in the GitHub clone. Reuse verified immutable task assets where appropriate, but put **new training/evaluation** under the new run root. Never destroy or corrupt historical runs.
2. If missing, restore the two Python environments, JAX checkpoints `checkpoints/openpi/pi05_libero/{params,assets}` and `checkpoints/openpi/pi05_base/{params,assets}`, pinned PRO source, and 50 published PRO states/task using the project’s documented download/verification code. Validate hashes.
3. Run/reuse Object 0 official-swap preparation (`scripts/prepare_pro_suite_grasp.py`), Object 2 official-swap preparation (`scripts/prepare_object_campaign_grasp.py`), and create a small **new** `scripts/prepare_vla_trace_objects.py` (or equivalent isolated wrapper) to prepare Object 1 official swap with the same state generation, pinned BDDL, source-state loading, split validation, hash/freeze rules, and `train.toml`, `validation.toml`, `test.toml`, `source.toml` generation as the existing object experiment code. Avoid altering the behavior of old launchers or their hash-frozen inputs.
4. Use **100 newly generated training states, 50 distinct validation states, 50 untouched published PRO final-test states, 50 original LIBERO source/retention states** for each official-swap task. For `object_1_hard`, there are 200 locally generated states split 100/50/50; **no official PRO test score**. Exclude published, original, and development-screen states from generated train/validation sets; no policy-success-based rejection.
5. For each of the four splits, prove no duplicated state IDs or simulator state vectors across train/validation/test, and no initially completed goals. Verify every reset produces finite state, nonblank external/wrist 224×224 camera images, proper instruction, correct target/distractor placements and basket, and a valid simulator goal predicate. Produce a one-page montage of initial frames with object IDs/locations.
6. Freeze task metadata in `protocol.json` and asset hashes. For these new experiments use `validation_iterations=[25,50,100]` and `online_validation=true`; do not include iteration 75 in selection.

**Reference commands verified in existing code** (not a claim that all assets are currently present):

```bash
cd /mnt/projects/jg/kaifany/RL-VLA  # confirm actual local root
source scripts/env.sh
git status --short && git rev-parse HEAD
python scripts/download_pi05.py --checkpoint pi05_libero
python scripts/download_pi05.py --checkpoint pi05_base
bash scripts/run_rldt.sh preflight --config configs/rldt_pi05_libero.toml

# Existing Object 0 generator (Object 0 only):
python scripts/prepare_pro_suite_grasp.py all \
  --output runs/vla_trace_objects_20261031/tasks \
  --tasks object_0 --workers 4

# Existing Object 2 / Object 1-hard generator (creates both):
python scripts/prepare_object_campaign_grasp.py \
  --output runs/vla_trace_objects_20261031/legacy_task_assets --workers 4

# Implement separately for the standardized official Object 1 swap:
# python scripts/prepare_vla_trace_objects.py --tasks object_1 \
#   --output runs/vla_trace_objects_20261031/tasks --workers 4
```

If generated Object 2 assets reside in `legacy_task_assets/object_2`, reference or copy them immutably with source hashes into the new official-task map. If assets already exist in earlier run roots, avoid unnecessary regeneration, but record exactly which protocol and test IDs are used. **Do not run Object 1 and Object 2 as if one already standardized utility existed; it does not.** If the supplied cluster root path differs, use the actual `pwd` instead. The previous Slurm GPU allocation documented in October 28 runs was time-limited and must not be assumed active on October 31; obtain a fresh authorized allocation before GPU work.

## 3. Train three separate RLDT action experts and save all requested checkpoints

For each `object_0`, `object_1`, `object_2`:

1. Complete a small **real-model** smoke run using the proper task config; confirm a full expert optimizer step changes `_1` expert parameters and **does not** change frozen backbone parameters. This smoke is not a training result.
2. Evaluate `pi05_libero` on 50 validation states using exactly the preprocessing and sampler used by subsequent RLDT evaluations. Write `baseline_validation.json` with per-episode state IDs, RNG seeds, success, steps, and videos for selected examples.
3. Run **one continuous** training trajectory from the original `pi05_libero` checkpoint to iteration 100. Do not restart from 25 or 50. Save full expert/optimizer snapshots at `iteration_000025`, `iteration_000050`, `iteration_000100` (plus `final` and optional unused iteration-75 recovery checkpoint). At 25, 50, 100, run a separate-process validation using the 50 validation states; preserve the learner state while paused and continue it unchanged. Do not feed validation/test data to replay or critic training.
4. Freeze the selection record before inspecting test results: highest validation success **count** at 25, 50, 100; ties choose the earliest iteration. Baseline is not a checkpoint candidate, and iteration 75 is not an eligible candidate.
5. Evaluate the selected RLDT checkpoint and `pi05_libero` baseline on paired 50-state untouched PRO test and 50 original-layout retention sets. To satisfy this project’s five-policy comparison, *also* evaluate the other two RLDT checkpoints on the **same predetermined** test cases **once after selection is locked**, and label these extra checkpoint-wise test diagnostics exploratory; do not repeatedly peek/reselect. Use the same states and per-episode noise protocol for paired comparisons.
6. Save `training/summary.json`, `metrics.jsonl`, validation/test results, status, checkpoint manifests, hashes, source snapshots, replay/video paths, and actual controlled simulator steps/optimizer counts. Confirm all three saved checkpoints reload and produce finite actions; store checkpoint SHA256/manifest IDs and the frozen base identity.

Existing per-task runner interface (the new `object_1` protocol must be compatible):

```bash
# Run separately per task; output directory must NOT already exist.
python scripts/run_pro_task_grasp.py \
  --task runs/vla_trace_objects_20261031/tasks/object_0 \
  --output runs/vla_trace_objects_20261031/methods/rldt/object_0

# Analogously for tasks/object_1 and tasks/object_2, only after the
# new official-swap task protocols are frozen and validated.
```

Expected expert paths for each task:

```text
methods/rldt/<task>/training/iteration_000025/{expert.npz,rldt_pi05.json,learner.npz}
methods/rldt/<task>/training/iteration_000050/{expert.npz,rldt_pi05.json,learner.npz}
methods/rldt/<task>/training/iteration_000100/{expert.npz,rldt_pi05.json,learner.npz}
methods/rldt/<task>/training/final/
```

`run_pro_task_grasp.py` also performs baseline/validation/selection/test flow. Verify the local version’s `protocol.json` reports **only** `[25,50,100]` for validation; do not modify a frozen old campaign in place. Add automated checks that `iteration_000025/000050/000100` are from the same uninterrupted training run, and that validation time is not incorrectly counted as training interaction time.

## 4. Five-policy comparison: protect the raw `pi05_base` condition

The existing JAX RLDT adapter `src/rl_vla/rldt_pi05.py::Pi05Flow.from_config()` raises if `model_config != "pi05_libero"`. The project README says the raw `pi05_base` policy belongs to a different experiment family and is **not a drop-in LIBERO task controller**. A valid comparison therefore requires a **separate base checkpoint loader, observation/state normalization, embodiment-specific control mapping, action horizon/execution rules, and confirmed target robot interface**. If an adapter is developed, validate action shape/range/unit, state slots, normalization provenance, camera orientation, sample/ODE steps, controller timestep and a simulator smoke rollout before scoring. Preserve its native semantic identity; do not secretly fine-tune it or swap in LIBERO-learned weights. Call results `raw-base transferred to LIBERO with adapter X`, not the ordinary matched task baseline.

If no valid actuator mapping is available, still run `pi05_base` through compatible **VLM-only** probes with documented model-specific image/text transforms, and compare its semantic representations to `pi05_libero`; set action attention, closed-loop success and causal action effects to `N/A` with a specific reason. Check representation tensor dimensions and tokenization before attempting cross-checkpoint CKA.

For `pi05_libero` and all three RLDT checkpoints, use the same native JAX `Pi05Flow` preprocessing/normalization and the *same* frozen VLM. Check that backbone state hashes are identical and that only expert/projection tensors changed.

## 5. Implement VLA-Trace integration — do not assume the public repo directly captures JAX attention

Official toolkit: [VLA-Trace/VLA-Trace](https://github.com/VLA-Trace/VLA-Trace). The public toolkit provides **attention-export**, **attention-overlay**, **plot-attention-map**, attention localization metrics, PatchMask, knockout-mask builders, CKA, and adapter interfaces. It **does not ship a drop-in exporter for our exact frozen JAX RLDT checkpoint/PRO tasks**. Build a small RL-VLA-specific extraction/rollout adapter and export transparent NPZ/JSON artifacts. Pin the toolkit’s Git SHA. Keep it in an isolated environment if its Torch extras would conflict with the pinned JAX environment; exchange files, not packages. `pip install -e '.[test,plot]'` suffices for plotting tools; extra LIBERO/model dependencies only where required.

**OpenPI code path to instrument:** `src/rl_vla/rldt_pi05.py::Pi05Flow.prepare/_encode_prefix` builds frozen VLM image/language tokens and KV cache. `Pi05Flow.velocity` then calls `model.PaliGemma.llm([None, tokens], ...)` using action-expert queries and the cached prefix. The pinned OpenPI `third_party/openpi/src/openpi/models/gemma.py` computes

```python
probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
```

inside attention. Capture these probabilities **before** their weighted-value contraction, along with per-layer expert-1 queries, keys/values where feasible, masks and token-group offsets. Use a reversible, opt-in diagnostic path, e.g. a small local patched module or wrapper, not an unconditional change to training. Verify `diagnostics=False` produces bitwise/tolerance-matched actions to the original implementation for fixed RNG and fixed observations. If a source patch is unavoidable, record diff, source commit and test that source immutability checks of existing experiments remain undisturbed. Do not insert large host-side conversions into the training JIT.

**Attention to collect:**

- **Prefill text query → image patch keys:** for target word(s), distractor words, spatial terms; distinguish per-camera patch ranges and preserve token/subtoken offsets.
- **Generation action query → image patch keys:** the **action expert's actual attention** (not a made-up visualization of the last linear projection). Record layer, head, action-token index, camera, ODE timestep, denoising latent `x_t`, and original token masks; aggregate only after saving metadata.
- **Action query → instruction token keys**, and layer-wise modality mass; optionally attention-entropy and head-level specialization.

For the native OpenPI prefix, **do not assume VLA-Trace's sample span order** `text,visual,action`. OpenPI's `embed_prefix()` concatenates image patches first (one span per camera), **then language tokens**; the suffix contains action tokens. Derive actual spans from `embed_prefix`, image encoder patch counts, prompt tokenizer IDs, mask and the attention query/key layout. The generation softmax is over prefix *and* suffix keys, so preserve the full denominator before extracting target/distractor mass. The prefix cache has layer/batch/token dimensions different from our public batch-first context: verify reshape logic.

**Controlled comparisons are mandatory:** for all RLDT checkpoints and the LIBERO baseline, replay the **identical saved RGB images, exact robot state, exact instruction, identical initial action noise, and fixed flow time(s)**. Otherwise different closed-loop frames and different sampled action latents confound checkpoint effects. Collect *two* kinds of attention: (A) fixed common `x_t` and `t` for causal checkpoint comparison, ideally using one baseline denoising path; (B) each model's own actual ODE trajectory and rollout attention to describe experienced behavior. Use `t` corresponding to the actual ODE 10-step schedule; examples can show early/middle/late denoising steps. Do not attribute the fixed-input plots to actual rollouts.

Produce one normalized NPZ/JSON record per task/policy/state/decision/ODE time with:

```text
source sha; model/checkpoint sha; task/variant/split/state ID; seed;
preprocessing and tokenizer version; actual instruction token IDs/spans;
agentview and wrist RGB files; state; initial noise; x_t; flow timestep;
layer/head/action-token attention [layers, heads, action_tokens, keys];
query/key modality span table, image patch grid and transform metadata;
per-camera image-ground-truth object/robot/distractor masks;
normalized output velocity, final action chunk, decoded executed actions;
observed target/distractor approach/grasp/placement labels where obtainable.
```

Export both **raw unnormalized attention** and **area-adjusted region scores**. Common metrics for each layer/head: target-region attention mass, distractor-region mass, mass ratio with epsilon, target/total image mass, action-to-text mass, attention entropy, normalized heatmap difference (e.g. Jensen–Shannon divergence), peak-hit accuracy and top-10% patch IoU. The target/distractor objects have different pixel areas, so report area-normalized densities and mask sizes as well as raw mass. Do not aggregate external/wrist camera grids together before localization.

**How to see what the action expert learned:** baseline-vs-iteration-25/50/100 attention panels with identical inputs, per-layer action-to-target and action-to-distractor curves, per-head shift matrices, differences in action velocity/decoded gripper approach, and change in attention-routing sensitivity after causal attention knockouts. Also measure **parameter drift per expert layer/projection** (L2 and relative L2 of `*_1`, action/time projections) and optional representation similarity on identical suffix hidden states. Prefix VLM representations should be invariant across these four checkpoints for the same input. Attention change is not itself proof of semantic understanding.

## 6. Exactly five required tests, across all five policy conditions and all three tasks

### Test 1 — Frozen-backbone object localization probe

**Question:** Is the identity and spatial location of the requested target recoverable from the VLM's image-conditioned features, even when action execution selects the distractor?

Export patch-level **spatially indexed** VLM image features and contextualized multimodal features from frozen `pi05_base`, `pi05_libero`, and 25/50/100 checkpoints on exactly matched saved scenes. Derive ground-truth object masks/centroids/IDs from simulator instance segmentation, not manually drawn attention maps. For each task, train a small **diagnostic-only linear/low-capacity probe** to predict target/distractor per patch or object-region pooled feature. Use *training* states for probe fitting and untouched validation/test states for scoring; show confusion matrices and target centroid/localization error. Test both prompts requesting target and valid distractor; if the model/tokenizer cannot form a verbal answer, use features, not an invented chat interface. Probe weights are NOT applied to the VLA policy and must not affect RLDT training. If a diagnostic probe succeeds, information is *decodable*; that does not prove the action expert uses it. If it fails, this does not prove information is absent.

**Outputs:** probe ROC/accuracy and localization/IoU per model/task/condition, target-vs-distractor scatter or heatmaps, cross-layout evaluation, a control probe without target-language features; show that π0.5-LIBERO backbone features are identical for RLDT checkpoints given identical input.

### Test 2 — Instruction-token → image attention grounding

**Question:** Where does the text instruction direct attention: the true target, distractor, or memorized original target region?

Capture prefill attention for target noun phrase, distractor noun phrase and any relevant spatial tokens. Handle multi-subword tokens by reporting each token and a documented aggregation. For static paired frames, use instructions `pick up [target] ...` versus a **counterfactual valid distractor instruction** while holding pixels and state fixed. Do not score the changed instruction with the old environment's target-only success predicate: either report offline requested-object approach or construct a matching BDDL/success-condition variant. Plot external-camera and wrist-camera overlays; target/distractor mask attention mass, peak-hit, IoU, area-normalized selectivity; early/middle/late backbone layers; compare raw-base (when compatible), LIBERO and RLDT. Because the backbone is frozen, do not expect prefill attention to change between LIBERO and RLDT. A static text-to-image map does not prove correct recognition.

**Outputs:** baseline/25/50/100 (plus raw-base if valid) token-over-image grids, original-vs-counterfactual instruction overlay, attention-mass CSV, non-target/random-token controls.

### Test 3 — **Action expert** → image attention and learning trajectory (priority)

**Question:** When RLDT changes the action policy, does it change which object/patch the action expert reads, or does it change actions while attention stays similar?

At matched frames, same prompt, state, initial flow noise, `x_t`, timestep and action query indices, extract expert-1 action-query attention. Visualize attention for **all expert transformer layers** (and optionally each head), for both cameras, with separate `t` columns for selected flow steps. Record output velocity and decoded action chunk at that same input. Contrast `pi05_libero` vs RLDT 25, 50, 100. Display true target and distractor masks and the prior target-location polygon.

Compute target/distractor mass and mass-density ratio, layer/head attention drift, modality-mass curves, entropy, and whether action endpoints/approach direction follow the instructed target. Pair qualitative panels of **wrong-object baseline episodes subsequently recovered** with unsolved episodes; use saved identical initial observations and RNG, not screenshots from independently diverged rollouts. Quantify how much of expert-1 query/projection weights changed, keeping frozen prefix K/V checksums fixed. Distinguish changes in action attention from changes in output projection or flow dynamics. To see `action_out_proj` specifically, measure its weight drift and use controlled hidden-state swap or expert-output ablations as a **separate secondary analysis**; the final linear projection has no attention map of its own.

**Outputs:** Figure A fixed-observation camera overlays for the five conditions; Figure B layer × checkpoint target/distractor mass heatmaps; Figure C flow-time evolution; Figure D per-layer weight drift and action-vector changes; a CSV with per-example comparisons.

### Test 4 — Causal masking and attention-pathway knockout

**Question:** Are the visually salient regions / pathways *causally necessary* for choosing the target?

For the same frozen observations and seed, apply simulator-derived instance masks to target, distractor, old-target position, robot/gripper, background, and a matched random-area control. Use at least `background_fill`/neutral replacement and another mode such as mosaic; black masks can create severe OOD artifacts. Apply consistently to all applicable cameras at the policy-input boundary only. Compute **open-loop** action-chunk change (normalized and physical coordinate norms), predicted target-choice changes, and **closed-loop** success on a predeclared set where valid. Separately perform transformer attention knockouts: action queries cannot attend to image keys; action queries cannot attend to text keys; optionally block prefill vision-language communication; and localized layer-window/target-key-specific blocking. Knockouts must operate in the **actual native OpenPI attention mask**, not just darkening pixels. Include no-op intervention verifying unchanged actions.

VLA-Trace provides mask builders/metrics/plotters (`knockout`, `knockout-sweep`, `patchmask`, `attention-metrics`), but its `eval-libero` online runner does not automatically know our PRO/custom-reset protocol or JAX RLDT expert weights. Bridge to `src/rl_vla/envs/libero_pro.py`, the native sampler, and frozen state IDs instead of accidentally evaluating standard LIBERO-Object. Preserve valid baselines and fixed seed/state/noise pairings. Measure region-specific action sensitivity **before/after RLDT**: if masking actual soup changes RLDT's action substantially more than baseline's while masking butter does the reverse, that is evidence the expert's causal reliance shifted. If localization/attention is correct but a knockout does not affect outputs, do not call the region causal. Report both success and intervention severity.

**Outputs:** mask-demo image grids, paired open-loop action-delta curves, closed-loop per-episode tables and uncertainty, layer-wise knockout response, target/distractor causal sensitivity ratios, image-vs-text ablation table.

### Test 5 — Unseen-layout and counterfactual generalization

**Question:** Did RLDT learn to follow *object identity*, or merely memorize the **new swapped location**?

Freeze **new, never-trained** layouts before testing. Include: (i) original layout; (ii) standard held-out PRO swap; (iii) target and distractor at two **new** feasible regions away from both familiar locations; (iv) several seeded continuous offsets or position permutations absent from training/validation; (v) distractor moved while true target fixed; (vi) target moved while distractor fixed; and optionally task instruction edits with a **matching goal predicate**. Ensure positions are collision-free, physically reachable, visible where intended, and not accidentally initially successful. Equalize reset/settling and action budgets. Prefer 50 predetermined states per condition/task for the primary analysis and multiple independent evaluation noise seeds as resources allow. Report known vs truly unseen **spatial regions**, not merely new reset IDs within the same distribution. Make failure-mode labels `wrong object`, `correct target but failed grasp`, `correct grasp but placement failure`, `timeout`, `other/ambiguous`, with reviewable video and the rule used to classify.

**Outputs:** per-layout success and target-choice matrices, paired recovered/regressed cases across 25/50/100, wrong-object fraction, original-task retention, sample-efficiency vs controlled environment steps, and qualitative episodes showing whether the learned policy follows target identity or learned a second position shortcut.

## 7. Evaluation and analysis protocol

Use **all five policy conditions** for every feasible test/task cell; explicitly record `N/A` when raw base is not a valid action policy. The four directly comparable action-generating conditions are frozen `pi05_libero` and RLDT 25/50/100. When auditing 50 frozen test states, store one row per `(task, variant, state ID, policy, checkpoint, seed)` including task BDDL hash, fixed instruction, requested target, distractor positions, success, grasped object ID, failure type, controlled steps, selected action chunk, attention artifacts and video. Maintain a **single shared observation bank** for the fixed-input tests rather than using different scenes from each rollout as if matched.

Statistics:

- Success: successes / episodes, Wilson confidence intervals; paired success difference and paired bootstrap CIs or exact paired discordance test (McNemar) against `pi05_libero`. Provide recovery (`base fail → new success`) and regression (`base success → new fail`) counts separately. A 50-episode single-seed study supports preliminary conclusions, not broad population claims.
- Wrong-object selection: object-ID classification from gripper target/grasp contact plus frame review; distinguish identification from placement mechanics; no assumption all failures are semantic.
- Object grounding: target vs distractor mask attention mass, area-normalized density, IoU/top-10% hit, localization error and response to counterfactual instructions. Use per-camera and per-layer disaggregations rather than cherry-picked heads.
- Causality: intervention-induced action displacement and target-choice shift, coupled to paired success/placement effects; report black/mosaic artifacts and matched-area controls.
- Representation/weight drift: checkpoint-vs-baseline frozen VLM feature equality, expert layer and final output projection drift; if attention does not shift but actions improve, test whether action-token hidden states and output projection explain the improvement.
- Learning curve: validation success and objective metrics at 0/25/50/100 vs **actual primitive training steps** and RLDT actor/critic update counts; not just iteration indices.
- Predeclare selection and plot aggregation choices before opening final test. Save raw NPZs, not only presentation heatmaps. Do not infer that attention equals belief or that base semantics were preserved from attention alone.

**Hypotheses to distinguish (not conclusions to assume):**

| Evidence | Interpretation / remaining caveat |
|---|---|
| VLM probe localizes soup, but baseline action attends to butter/old position; RLDT action attends to soup | Supports downstream routing/use-of-information explanation. Needs intervention to establish causality. |
| Both VLM probe and text-image grounding fail on swapped input | Supports upstream grounding/representation limitation; a negative probe is not proof of absence. |
| Action attention unchanged but decoded trajectories shift | RLDT may change action-token processing or output projections rather than where it attends. |
| RLDT success only on trained swap, not third positions | Likely learned another layout shortcut; cannot claim general target-identity grounding. |
| RLDT generalizes across truly unseen object locations, target-region masking strongly affects RLDT behavior, frozen VLM features unchanged | Stronger evidence that RLDT improved downstream use of existing representations. Still report limits of attention/causal identification. |

## 8. Suggested implementation files (new, isolated paths)

Implement small, testable modules; adapt names to existing conventions:

```text
scripts/prepare_vla_trace_objects.py
scripts/trace_pi05_attention.py
scripts/run_vla_trace_objects.py
scripts/analyze_vla_trace_objects.py
src/rl_vla/diagnostics/__init__.py
src/rl_vla/diagnostics/pi05_attention.py       # instrumentation, query/key spans, camera patch maps
src/rl_vla/diagnostics/observation_bank.py      # deterministic matched samples, masks, IDs
src/rl_vla/diagnostics/probes.py                # diagnostic-only localization probe
src/rl_vla/diagnostics/interventions.py        # input masks and native attention knockout
src/rl_vla/diagnostics/metrics.py               # grounding/causal/paired statistics
integrations/vla_trace/README.md               # exact adapter, versions, CLI instructions
tests/test_pi05_attention_capture.py
tests/test_pi05_token_spans.py
tests/test_pi05_attention_knockout.py
tests/test_vla_trace_observation_bank.py
tests/test_vla_trace_split_isolation.py
```

Keep a read-only or explicitly isolated VLA-Trace checkout under `third_party/vla_trace` or a separate tools directory; pin its SHA. Do not install broad Torch extras into `third_party/openpi/.venv`. Train without capture enabled for memory/performance; profile capture on a small sample in analysis jobs. Use streaming/chunked NPZ or sharded Zarr if every head×layer×ODE-step tensor is too large. Predefine representative analysis frames (initial decision; approach; grasp; placement; failure and recovery), then optionally sample more. Benchmark both GPU memory and attention-export fidelity with one 5-policy matched frame first.

Recommended reproducible output tree:

```text
runs/vla_trace_objects_20261031/
  manifest.json                    # all source/tool/dependency/checkpoint hashes
  tasks/{object_0,object_1,object_2}/
      {assets.json,protocol.json,train.toml,validation.toml,test.toml,source.toml}
      reset_checks/               # image montages and validation logs
  methods/rldt/{object_0,object_1,object_2}/
      baseline_validation.json
      training/{iteration_000025,iteration_000050,iteration_000100,final}/
      validation_000025.json, validation_000050.json, validation_000100.json
      selection.json, baseline_test.json, selected_test.json, baseline_source.json
  reference/pi05_base/            # loader/adapter validity, feature-only results if necessary
  evaluations/<task>/<policy>/<split>/{episodes.jsonl,videos,summary.json}
  observation_bank/<task>/<split>/{manifest.jsonl,images,masks,states,noises}
  traces/<task>/<policy>/<state_id>/{metadata.json,attention_maps.npz,views.npz}
  interventions/<task>/<policy>/<setting>/{episodes.jsonl,actions.npz,summary.json}
  probes/<task>/<policy>/{metrics.json,probe_weights.*,predictions.csv}
  figures/{task_comparison,attention_fixed,attention_rollout,layer_heads,causal,ood}/
  tables/{success.csv,wrong_object.csv,grounding.csv,causal.csv,checkpoint_drift.csv}
  report.md                      # full scientific interpretation; failures and limitations
  runbook.md                     # exact executable commands, environments, schedules, known blockers
```

## 9. VLA-Trace plotting and knockout API examples

Use these commands **only after producing correctly formatted artifacts and confirming the installed version's `--help`**. All paths are examples, not artifacts known to exist. The `--visual-span` etc. are **placeholders** and MUST be replaced using the actual OpenPI token span metadata; OpenPI has multiple image spans followed by text (unlike VLA-Trace example order).

```bash
# Run isolated tools environment, e.g. third_party/vla_trace_tools/.venv:
# git clone https://github.com/VLA-Trace/VLA-Trace.git third_party/vla_trace
# <tools-python> -m pip install -e 'third_party/vla_trace[test,plot]'

vla-trace attention-export \
  --attention <RAW_ATTENTION_NPZ> --key <KEY> \
  --visual-span <ACTUAL_VISUAL_TOKEN_SPAN> \
  --text-span <ACTUAL_TEXT_TOKEN_SPAN> \
  --action-span <ACTUAL_ACTION_TOKEN_SPAN> \
  --output <QUALITATIVE_VIEW_NPZ>

vla-trace attention-overlay \
  --image <EXTERNAL_CAMERA_OBS_NPY> \
  --attention <ACTION_TO_IMAGE_NPZ> --attention-key <STEP_KEY> \
  --masks <INSTANCE_MASKS_NPZ> --mask-key <TARGET_MASK_KEY> \
  --grid-size <ACTUAL_GRID_SIZE> --output <FIGURE_PNG>

vla-trace plot-attention-map <TOKEN_TO_IMAGE_NPZ> \
  --key <STEP_KEY> --kind heatmap --normalize row \
  --output <TOKEN_PATCH_HEATMAP_PNG>

vla-trace attention-metrics \
  --attention <ATTENTION_MAPS_NPZ> --masks <STEP_MASKS_NPZ> \
  --metadata <METADATA_JSON> --objects <STEP_OBJECTS_JSON> \
  --output <ATTENTION_METRICS_CSV> \
  --summary <ATTENTION_SUMMARY_JSON>

# Build a mask *specification*, not automatically an applied intervention:
vla-trace knockout --model pi0.5 --dataset libero_object \
  --phase generation --mode no_image --layers all \
  --output <NO_IMAGE_SPEC_JSON>
```

`attention-export` handles a single contiguous visual span; for multi-camera OpenPI explicitly export each camera slice separately or build an adapter that combines/reshapes the expected keys while retaining distinct camera metadata. Never concatenate token groups incorrectly or normalize region mass after dropping unknown key positions. For generation attention, an action query-to-image key slice may be extracted directly from a raw per-layer/per-head tensor rather than relying on a packed all-to-all matrix where the query subset is not present.

## 10. Run order and acceptance gates

**Gate A — Inventory and readiness.** Source revisions, envs, assets/checkpoints verified; GPUs/quotas checked; fresh run root; Object 0/1/2 PRO conditions clearly labeled; all train/val/test/source resets validated; task prompt/BDDL inspectable. If `object_1` official assets are missing, state it and generate or use separate `object_1_hard` with honest label.

**Gate B — One observation, one checkpoint.** Run `pi05_libero` fixed-input inference and capture action-to-image attention without changing numerical outputs. Verify token span/camera mapping with a synthetic or controlled object mask, per-head softmax and unmasked attention mass; use deterministic noise. Verify no-op intervention has no effect.

**Gate C — Three training runs.** For each task, training reaches 100 uninterrupted iterations; all 25/50/100 expert+learner snapshots reload and have correct frozen base; all three validation reports have exactly 50 fixed IDs; checkpoint selection locked; records include actual interactions and loss diagnostics. If training fails, keep partial files and clearly report failed gate rather than marking it complete.

**Gate D — Five policy conditions.** `pi05_base` evaluated for every compatible diagnostic, or incompatibility clearly documented; LIBERO and each of three RLDT checkpoint policies run through the same observation bank and frozen split with reproducible actions. Do not label raw base as having meaningful LIBERO accuracy without a validated adapter.

**Gate E — Five tests.** Localization probes, text→image maps, expert action→image maps, causal intervention/knockout, and unseen-layout evaluations have data files, figures, and metrics for each task/policy cell or explicit reason unavailable. At least one panel shows soup/butter with their **ground-truth regions and expert attention** across LIBERO/25/50/100 at the same fixed observation and flow latent.

**Gate F — Scientific report.** Deliver a result table with successes/episodes, uncertainty and paired recovered/regressed cases; attention and causal metrics per layer/time, failure taxonomy, new-layout generalization, original-layout retention, compute/storage costs, and frank explanation of what supports or weakens the hypothesis that RLDT improved action routing. Include runnable README commands and reproducible paths. Highlight all limitations (single RL seed, distribution of generated states, baseline adapter validity, attention is not causal explanation). **Do not claim 'the problem is in the action head' from an attention heatmap alone.**

## 11. First actions for the coding agent

1. Print the actual repository revision and `git status`, and read `README.md`, `docs/grasp-setup.md`, `docs/object-campaign-20261028.md`, `docs/controlled-pro-experiments.md`, `scripts/prepare_pro_suite_grasp.py`, `scripts/prepare_object_campaign_grasp.py`, `scripts/run_pro_task_grasp.py`, `src/rl_vla/rldt_pi05.py`, `src/rl_vla/checkpoint_validation.py`, and the pinned OpenPI `models/{pi0,gemma}.py`.
2. Inspect the on-disk older Object 0/2/1-hard protocols and checkpoint trees; detect existing results without relying on historical statements for current job status.
3. Write `runbook.md` with the exact host/GPU/environment availability, verified train/test split hashes, chosen official `object_1` setup, and the raw-base compatibility verdict.
4. Prove a small deterministic expert attention capture and an unchanged-inference sanity test **before** starting the three expensive RLDT runs.
5. Prepare/freeze the three task datasets; run the three task-specific RLDT trainings; validate/reload all checkpoints; then run all five diagnostic tests and publish the reports.

## 12. References

- [RL-VLA repository](https://github.com/kaifanyu/RL-VLA)
- [VLA-Trace official implementation and README](https://github.com/VLA-Trace/VLA-Trace)
- [Pinned OpenPI repository](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)
- [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO)
- [VLA-Trace paper](https://arxiv.org/abs/2605.30117)
- [RLDT method reference from RL-VLA docs](https://github.com/RPFey/rldt/tree/3ab73e775efd0f041ad28170c616146be86bb6ea)

**Final mandate:** The objective is not merely to generate attractive attention overlays. The objective is to determine whether a frozen-backbone, RLDT-updated action expert begins to **use the correct object information** more effectively, and whether that causal use generalizes beyond the specific training swap.
