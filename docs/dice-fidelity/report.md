# DICE fidelity audit and launch specification — 2026-11-05

The new profiles implement the paper's algorithm with explicit pi0.5/LIBERO domain adaptations. They are **not an exact reproduction or a production result**. The historical pilot is preserved and remains an online-only, small-network, small-K ablation. Adding critics alone would not have repaired it.

The source entry point was the [official project website](https://zhanyisun.github.io/dice.rl.2026/). Its paper link leads to [paper v2, 1 July 2026](https://arxiv.org/pdf/2603.10263v2); Section 4, Algorithm 1 and Appendices A–D were read. Its simulation link redirects from `zhanyisun/dice-rl` to `real-stanford/dice-rl`. The audited pin is **`98d936e52523878f0172e606ea32403176497100`**, also confirmed by `git ls-remote origin HEAD` on this date. The local reference is `third_party/dice_rl_upstream`, clean and detached at that revision. The dependency lock and setup script already specify this pin; the previously empty local checkout was fetched and materialized. No upstream source was changed.

Function-level pinned permalinks and file hashes are in [source-trace.json](source-trace.json).

Every finetuning YAML matching `*distill_residual_flow*.yaml`, including PushT, is preserved with its SHA256 and parsed values in [pinned-configs.json](pinned-configs.json). Hydra interpolation strings are retained, rather than represented as falsely resolved values. The starting recipe for our profiles is **Can pixel**, not the README and not the real-robot schedule.

## Fidelity matrix

Official paths below are relative to the pinned [repository](https://github.com/real-stanford/dice-rl/tree/98d936e52523878f0172e606ea32403176497100). Abbreviations:

- **M**: `model/rl/distill_residual_rl.py`, `DistillResidualRLModel`, `DistilledActor`, `DistilledCritic`.
- **I**: `model/rl/distill_residual_rl_img.py`, image subclass and its overridden actor loss.
- **A**: `agent/finetune/train_distill_residual_flow_agent.py`; image agent inherits the loop/update machinery.
- **B**: `util/hybrid_replay_buffer.py`, `HybridReplayBuffer`.
- **D**: `agent/dataset/sequence.py`, `StitchedSequenceQLearningDataset`.
- **C**: `cfg/robomimic/finetune/can/ft_distill_residual_flow_unet_img.yaml`.

Tests are in [test_dice_fidelity.py](../../tests/test_dice_fidelity.py), [test_dice.py](../../tests/test_dice.py), and [test_rldt_pi05.py](../../tests/test_rldt_pi05.py). “Real verification” means the bounded smoke described below, not a held-out performance experiment.

| paper equation/section | official code function/config | our implementation | discrepancy | test evidence |
|---|---|---|---|---|
| §4, Eq. 1–2: frozen prior and corresponding latent | M `_load_pretrained_policy`, `get_action`; I freezes encoder and uses same supplied noise | `FrozenPi05.propose`; `DICE.action` detaches prior, features and full latent; only Torch residual/critics have optimizers | Native pi0.5 replaces trained Robomimic flow backbone; complete prior, including its action expert, is frozen | `test_initial_residual_exactly_zero_and_prior_has_no_gradient`; real same-key before/after arrays bitwise equal |
| §4 encoder freeze; Appendix A VLA latent | I `_load_pretrained_policy`; image agent merges frozen visual/proprio features | `Pi05Flow.prepare/_encode_prefix`, then fixed per-state normalization in `FrozenPi05.propose` | 2048 pooled VLM features + 32 normalized padded state = 2080; Appendix A uses pi0 and pooled latent, not our pi0.5 adapter | real verification finite shapes and invariance; dynamic-JIT argument test |
| Eq. 2 residual conditioning | M/I `get_action`, C `condition_residual_on_base_action=false` | `DICE.action(features, noise, prior)` | Full 10×32 latent retained even though correction covers only 5×7; no independent latent or demonstration action substituted | zero-residual test; selection association test; real best-of-16 oracle |
| Eq. 4 chunk TD, Algorithm 1 replay | B `_process_completed_episode`; D `__getitem__` | `dice_replay.chunk_reward/chunk_discount/finish_episode`; `run`; `convert_episode` | New profiles sum rewards inside chunks and use one gamma per chunk; legacy configs retain primitive-time discounting | `test_reward_units_nstep_no_duplicate_rewards_or_cross_episode`; demonstration chunking test |
| Eq. 4 next-state expectation | M `loss`, C `multi_sample_next_noise=true`, `num_next_noise_samples=4` | `DICE.update`: current residual actor, target critics, min over critics **then** mean over independent next-state proposals | Paper profile explicitly uses 16 bootstrap samples, following the paper's common K; pin uses 4 | `test_min_precedes_expectation_and_terminal_bootstrap_is_zero`; existing numeric expected-loss test |
| Eq. 5 current-state expectation | M/I `actor_loss`, `num_multi_z_for_actor_loss=16` | Fresh same-state latent/prior pairs; arithmetic mean over `k_train=16` | K_actor, K_bootstrap, collection and evaluation are separate configuration roles | profile test; real smoke K_train/bootstrap=16 |
| §4 adaptive RLPD | A `get_current_expert_ratio`; B `_sample_rlpd`, `int(B*r)` | `expert_fraction`, `sample_rows(fixed_split=true)` | Schedule units explicitly controller commands; new Can-derived duration 160000. Audit profile's online-only mode forces ratio 0 | ratio endpoint/clamp and fixed split tests |
| Eq. 6 value improvement gate | M/I `actor_loss`: strict `Q_cur > Q_pre` | `regularization_mask(mode="candidate")` uses paper's inclusive comparison | Paper and code differ at equality; explicit `mode="replay"` retains strict code semantics | candidate/equality/replay gate test |
| Eq. 6 Monte Carlo gate | A `update_networks`: **pre-update** min-Q of replay action minus MC return; M/I broadcast it over K | Paper profile calibrates each candidate `Q_cur[b,k]-G[b]`; code-compatible replay mode also implemented | Material paper/code conflict, resolved in favor of Eq. 6 for paper profile; neither signal is silently substituted | gate test distinguishes outcomes; real smoke exercises both retained and relaxed penalties |
| Eq. 7–8 regularization | M/I `actor_loss`: mean squared current-minus-same-z-prior action, no expert-action target | `DICE.update`: squared residual, filtered per candidate, averaged over valid time/motor coordinates | Paper writes squared L2 norm; pinned implementation uses coordinate/time MSE. We retain MSE and beta=50, explicitly a norm convention adaptation | `test_regularizer_is_not_regression_to_demonstration_actions` changes replay actions and preserves actor update while critic changes |
| Eq. 6–8 filtering enabled | C `use_soft_q_filtering=false`; all inspected main recipes also false | New profiles set `filter_enabled=true` | Released configs disable a central paper component. Disabled filtering is an ablation, not our paper profile | config rejects unlabeled disabled filter; gate warmup and conditions tests |
| Demonstrations in §4/Algorithm 1 | B `_sample_expert_data`; D supplies action, reward, state, next state, MC; C masks expert actor-Q and retains expert regularization | `load_demonstrations` validates task/protocol/checkpoint/normalizer/return convention; both sources train TD; actor receives demo states with fresh latent/prior pairs | Paper profile allows actor Q and filtering at demo states, as its stated expectation does. Optional code switches `disable_expert_actor_q`, `retain_expert_regularization` are exposed | provenance rejection tests; regularizer-not-regression test; no real demo-assisted smoke claimed |
| §4 best-of-N; Algorithm 1 interaction | M/I `get_action_with_exploration_strategy`; C `num_exploration_samples=16`, `max_q_min` | `DICE.select`: `argmax_k min_i Q_i`, action retains selected latent/prior association | Collection K=16 only after warmup; primary benchmark evaluation K=1 has no candidate advantage; K=16 evaluation separately costed | index tests; real best-of-16 oracle |
| Appendix B/C ensemble and TD reduction | M `DistilledCritic.forward` **forces min even when return_mean=True**; `critic_loss` sums ensemble MSEs | Ten independent critics; sum of per-network mean MSE; min for actor, calibration, target and selection | Previous pilot had two critics. Min-of-mean over samples would be wrong | pinned-source critic oracle copies weights and matches all ten outputs and forced-min result |
| Algorithm 1 target updates | M `update_target_networks`; A `update_networks` delayed gate `(training_step+1)%freq` | `DICE.update`, tau=.01, configurable collection-clock actor/target delay=2; target initialized from critics | Code's burst cadence retained; actual counters corrected rather than copying its off-by-one metric fields | burst/Polyak test; saved actual optimizer step counts verified |
| Appendix C optimization | C: GELU, three 1024 layers, LayerNorm, Adam 1e-4, weight decay 1e-5, gradient clip 1, scheduler | `mlp`; optimizers; licensed `dice_scheduler.py`; all states persisted | Zero final residual layer is a declared initialization adaptation. Paper profile omits code's Q normalization/expert Q mask to follow Eq. 8 | architecture/source oracle, finite update, scheduler resume tests |
| Appendix B/C n-step returns | B `_sample_standard` walks sliding windows; D aggregates non-overlapping chunks, but next-state index remains one horizon ahead | `finish_episode` aggregates up to n non-overlapping executed chunks and uses actual last next observation/discount | Fixes duplicated online rewards and demo n-step next-state mismatch; does not imitate these defects | hand-calculated n-step/reward/next-state test; no cross-episode labels |
| Eq. 4 terminal bootstrap | A combines termination and truncation; B finalizes finite episodes | Terminal and time-limit chunks have zero bootstrap; shortened final prefixes masked | Time limits define finite-task horizon here; budget interruption is **not** a task terminal, so its unfinished episode is excluded from MC replay | terminal/bootstrap test; command-cap test; real shortened-horizon truncation |
| Algorithm 1 update step; Appendix C vs README | A `run` calls G `update_networks(training_step=total_steps)` after one vector chunk step | Per-chunk online updates on completed replay, rational G/E credit with a serial collector | Preserves updates per collected chunk, not parallel timing. No catch-up for empty replay; see accounting below | full runner/save-resume/cap tests; logged actual ratios |
| Checkpoint/evaluation protocol: domain adaptation | No paper requirement for our Object 0 split/cap/checkpoint eligibility | Exact controller snapshots, episode-boundary restart, per-episode JAX key; validation-only selection and separate test/retention | Historical artifacts have incompatible K, milestone opportunities and budgets; new configs do not retroactively repair them | matched ordered-state tests; exact mid-chunk snapshots; uninterrupted vs resumed runner bitwise equality |

## Paper/code conflicts and chosen settings

The [config snapshot](pinned-configs.json) is authoritative for values from the pin. Selected **Can pixel** settings: batch 256; replay maximum 2000000 rows; 4 parallel environments; 150000 vector collection iterations; chunk/execution horizon 4; gamma .99; G=10; update frequency 1; actor and target frequency 2; ten critics; three 1024-wide GELU layers with LayerNorm; both Adam learning rates 1e-4, weight decay 1e-5; tau .01; clip norm 1; cosine restart cycle 1000 optimizer steps, warmup 10, minimum LR 1e-6. Code uses actor K=16, bootstrap K=4, collection/evaluation candidates=16; beta=50; epsilon=-.5; filter warmup 2000 vector steps; candidate-selection warmup 3000; expert fraction .7→.1 over 10000 vector steps. The source config seed is 42, and its environment limit is 400 commands. Twenty Can pretraining demonstrations are a source-task setting, not a requirement to use our initial ten-demo pack.

Common paper K=16 and Eq. 6 take precedence in the **paper algorithm profile**: actor/bootstrap K=16, candidate calibration, filtering enabled, no automatic demo-state actor-Q exclusion, no forced demo regularization, raw Q objective. This differs from pinned code's bootstrap K=4, replay calibration, disabled filter, expert-state masking and Q-normalized actor objective. The optional code behaviors remain named settings. Choosing candidate rather than replay calibration is a documented resolution, not proof that either published artifact is a typo.

Other recipe discrepancies must not be hidden behind a single “paper default”:

| Pixel task | pinned E/G/h | paper Table 2 E/G | pinned beta / epsilon | paper beta / epsilon | pinned actor/bootstrap K | pinned expert schedule |
|---|---|---|---|---|---|---|
| Can | 4/10/4 | 4/10 | 50 / -.5 | 50 / -.5 | 16/4 | .7→.1, 10000 vector steps |
| Square | 8/20/4 | 8/20 | 50 / -.25 | 50 / -.25 | 16/4 | .5→.1, 10000 vector steps |
| Transport | 4/10/8 | 4/10 | 100 / -.55 | 100 / -.5 | 16/4 | .9→.1, 10000 vector steps |
| Tool Hang | 8/10/8 | 8/20 | 100 / -.85 | 100 / -.75 | 16/4 | .5→.1, 10000 vector steps |

All four pixel YAMLs disable the filter and enable Q normalization and expert actor-Q masking. Their n is 3/3/3/5. State YAMLs also differ: Square uses E=4/G=10/beta=100; Can uses initial expert ratio .5 and beta=100; Tool Hang uses E=4/G=10/epsilon=-.55. Full settings, including pretraining checkpoint paths, image metadata, flow steps, logging, evaluation and dataset choices, are retained in JSON. We do not silently blend state and pixel recipes.

Appendix A's pi0 experiment is architectural context, not proof of pi0.5 fidelity or a transferable LIBERO-PRO score. Appendix B motivates explicit filter, sample-count and replay ablations. Appendix D describes a different real-robot batching recipe and has its own horizon-description inconsistencies; its episodic update schedule and five-critic choice are not imported into this simulation profile.

## Update-ratio discrepancy resolved through the loop

Let **E** be parallel environments, **h** executed commands per complete chunk, **G** gradients per update, **u** collection iterations between updates, **f** actor delay. One vector collection iteration supplies E chunks and at most E*h controller commands. A critic optimizer step updates every critic; it is **one ensemble update**, not ten separate gradient steps.

The loop at A `run` increments `total_steps` **once per vector chunk step**, and passes the same zero-based counter to every one of its G gradient calls. Thus, after replay warmup:

- Critic updates/collected chunk = G/(u*E); per command = G/(u*E*h).
- Actor optimizer updates/collected chunk = G/(u*E*f) only when the update/delay gates cover all phases (u=1 in these recipes). Here f=2. Target updates have the same average frequency.
- All G actor and target updates occur on eligible collection iterations, followed by none on the next iteration. They are not interleaved every other gradient call. The actor backward pass on ineligible iterations is not an optimizer step.
- Actual shortened terminal chunks reduce commands below E*h; exact observed ratios therefore require command counters. Replay warmup also reduces observed ratios. Offline rows do not add environment interactions.

| Pinned recipe | critic/chunk | actor/chunk | critic/command, full chunks | actor/command, full chunks |
|---|---:|---:|---:|---:|
| Can state/pixel; Square state | 2.5 | 1.25 | .625 | .3125 |
| Square pixel | 2.5 | 1.25 | .625 | .3125 |
| Transport state/pixel; Tool Hang state | 2.5 | 1.25 | .3125 | .15625 |
| Tool Hang pixel, **actual G=10** | 1.25 | .625 | .15625 | .078125 |
| Tool Hang using paper table G=20 | 2.5 | 1.25 | .3125 | .15625 |

The README's UTD “around 1” and Appendix C's “10–20” do not describe the same denominator. The actual recipes execute 10 or 20 critic updates per **vector chunk collection call**, not per individual controller command. Neither prose statement can replace G/E/h accounting. Batch size is samples per optimizer step and is not another multiplier on optimizer-step counts; sample reuse can separately be reported as B*updates/new rows.

For our serial Can-derived profiles, `gradient_steps_per_chunk=10` and `reference_parallel_envs=4` mean rational credit `floor((d+1)*10/4)-floor(d*10/4)` per serial chunk d. This is alternately 2 and 3 critic updates once replay is ready. Delay/filter clocks use `floor(d/4)`, preserving groups corresponding to the reference vector clock. With execution prefix 5, steady full-chunk ratios are **2.5 critic and 1.25 actor updates/chunk; .5 critic and .25 actor updates/command**. This preserves per-transition learning budget while explicitly changing wall-clock parallel collection; it does not claim equivalent trajectories. `updates_per_episode` is used only by legacy configs and ignored when the per-chunk field is present.

The optional `sample_multi_z_for_actor_loss` flag does not control the actual pinned `actor_loss`: that function always samples its configured K, even though the constructor default of the flag is false. The audit follows the function, not the flag name.

RLPD units also differ across artifacts. In C, 10000 vector steps correspond to 160000 nominal commands (4 envs × 4 actions). We preserve **160000 actual commands** for the Can-derived mixing schedule. The new five-command execution prefix means this is intentionally not 10000 groups of four serial chunks. Filter and selection warmups retain the collection-group clock; selection warmup is 3000 reference groups (roughly 12000 serial decisions). Labels/units are explicit.

Online MC labels become available on completing an episode, as in B. Between episode ends, updates use previously completed replay; updates are not all postponed to the end of the current episode. Empty-replay warmup earns no deferred update debt. Once an episode finishes, its records enter replay before that chunk's update. Only executed decision prefixes are replay rows here; official B constructs sliding windows at every primitive offset. We keep non-overlapping decision rows as an explicit VLA adaptation, with consistent demonstration and online returns.

## pi0.5/LIBERO adaptations

| Boundary | Implemented convention and evidence |
|---|---|
| Frozen checkpoint | Existing native `checkpoints/openpi/pi05_libero`, OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`; no weights replaced or downloaded. The shared adapter names expert arrays `params`, but DICE never optimizes them. LoRA is rejected by DICE config validation. |
| Cameras/instruction | Object 0 PRO task limit 280 commands (source Can: 400); ten reset-settling commands excluded from budget. PRO adapter: 256 RGB frames, rotate 180 degrees, resize-with-pad to 224; agentview + wrist; checkpoint's third camera padding/mask unchanged; BDDL instruction verified against task provenance. |
| Features | Masked mean of final VLM tokens (2048), concatenate checkpoint-normalized padded state (32), normalize each combined 2080-vector to zero mean/unit variance with epsilon 1e-5. No running data statistics, action tokens or reward/validation features. Shared frozen prefix implementation reused. |
| Proprioception | EEF xyz, xyzw quaternion converted to axis-angle, two gripper positions (8 values) before checkpoint padding and quantile normalization. |
| Latent coordinates | Full Gaussian `[10,32]` sampled by native JAX sampler, flattened to 320 for residual input. Padded and unexecuted noise channels remain part of the prior's deterministic generation map. |
| Flow time | **Noise at t=1, data at t=0**; times `1-i/T`, update `x_next=x-v/T`, T=10. The paper's forward-time flow notation is not substituted. `Pi05Flow.sample` and real metadata verify this. |
| Actions/horizons | Native prediction `[10,32]`; residual and critic act on first 5×7 normalized motor coordinates, including gripper. Remaining predicted times/channels preserve the original prior sample when decoding. No residual bounds or arbitrary unit scaling added. |
| Normalization | Native input transforms and `assets/physical-intelligence/libero/norm_stats.json`; native inverse transforms decode actions. Replay importer runs controller actions through the actual checkpoint transform. Asset hashes and checkpoint metadata/manifest hashes are recorded. |
| Clipping/replay | Online replay stores the **normalized command requested before environment clipping**; environment clips decoded controller command to bounds. This follows the existing pilot/requested-action convention and must not be described as measured executed motion or clipped-normalized actions. `clipped_commands` exposes the mismatch; the real smoke observed clipping. No unverified inverse clipping transform is introduced. |
| Partial final prefix | Actual duration and action mask stored. Critic inputs and residual penalty ignore unexecuted tail coordinates for this row; penalties divide by valid coordinates. Full requested chunk still exists for decode/selection. |
| Replay transitions | One actual decision prefix plus next observation, sum of rewards, one-chunk discount and MC label; TD data additionally aggregates up to n=3 decision chunks. Next observation advances to the last aggregated chunk, never another episode. Demos use the same segmentation, normalization, masks, n-step and reward scale; archive metadata must match. |
| Rewards | Simulator's binary success predicate; scale=1; one success ends episode. New profiles sum primitive rewards without within-chunk discount, gamma=.99 per chunk; legacy primitive-time behavior remains available. No reward shaping or demo reward multiplier. |
| Endings/budgets | Success termination and actual time limit both end finite-horizon task return and zero bootstrap. Reset settling is outside interaction budget. A budget interruption counts all executed commands and discards its unfinished MC episode. Its commands still count in ratios. |
| Initialization/resources | Residual output layer starts exactly zero; CPU Torch learner; frozen JAX prior on allocated GPU (bf16 VLM, float32 action expert under the shared audit adapter); serial simulation. Raw images are retained for fresh prior proposals; a 4 GiB replay byte cap can reduce the effective capacity below the source's two-million feature rows. Source simulation starts an ordinary random MLP and parallel environments. These are recorded adaptations, not claimed paper requirements. |

Requested-action replay is a known modeling limitation when clipping is frequent: Q is over requested commands followed by the fixed clipping environment boundary. It does not establish equivalence to a critic trained on clipped controls. A future change must transform actor candidates and replay consistently, rather than silently changing replay alone.

## Separate launch profiles and the benchmark audit

1. [Paper algorithm profile](../../configs/dice_paper_pi05_object0.toml): demo-assisted, ten critics, K_train=K_bootstrap=16, K_collection=16 after the reference warmup, paper candidate filter, Can pixel numerical settings with the listed adaptations. Seed 42 follows the source configuration. Its 3000000-command ceiling corresponds nominally to 150000×4×5 serial-domain commands, and is a **domain training ceiling**, not a new published requirement. Primary evaluation defaults to K=1; K=16 can be separately requested.
2. [Audit benchmark profile](../../configs/dice_audit_pi05_object0.toml): same algorithm mechanics, **`no_demonstration_replay` ablation**, canonical Object 0 protocol, exact 16000 executed commands, snapshots at 0/4000/8000/12000/16000, K=1 primary evaluation. Its local filter/selection warmups are 11/16 reference groups: the source 2000/3000 fractions of a 150000-iteration run are scaled to the budget's nominal 800 reference groups and rounded upward. Keeping the unscaled warmups would effectively disable both components for the whole 16K benchmark. Inclusive warmup gates and actor delays still apply. Collection also requires at least one completed critic update before using K>1, so a short-budget run does not rank candidates with an entirely untrained critic. This eligibility guard leaves the long reference warmup behavior unchanged. This is an explicit budget adaptation, not a paper setting. Seed 0 is the supplied local instance; seeds 1 and 2 are optional local replications via `--train-seed`. They are not paper mandates.
3. [Best-of-16 evaluation profile](../../configs/dice_bestof16_pi05_object0.toml): identical trained policy settings, only `[evaluation].k=16`. No new checkpoint selection on test. Candidate counts, expert-forward equivalents, prior/scoring seconds, total rollout seconds and controller commands are reported. Sixteen candidates at ten ODE steps mean 160 candidate action-expert forward equivalents per decision, versus ten at K=1; batched wall time is measured, not assumed 16×. VLM encoder cost and model loading are not included in that arithmetic.

K_train is the actor's latent expectation size. `k_bootstrap` is separate and explicitly 16 here. K_collection governs online value-guided interaction after warmup. K_evaluation is `[evaluation].k`; changing it never changes the learner configuration or invalidates checkpoint identity. `particles` and `selection_samples` remain compatibility aliases and must agree with their explicit counterparts in new profiles.

The existing audits were read at `/workspace/smrithi/rl-vla-config-audit/report.md` and `/workspace/smrithi/rl-vla-comparison-audit/report.md`, plus repository `docs/flashsac_shared_contract.md`. The second audit correctly says historical artifacts do not support a controlled three-method comparison. It reports historical DICE 16000 commands/3212 decisions/268 actor and critic updates, no demos, train K=2/eval K=4. Its observed ratios are approximately **.08344 updates/chunk and .01675 updates/command**, not the simulation recipe. Historical checkpoints near 4361 and 8063 commands cannot become exact 4K/8K/12K snapshots by relabeling.

New benchmark config uses the verified Oct31 protocol SHA256 `a1e84a70c766c7ad74898b3bec9a12e272b1c326cb94a08cfaf0bc432e1bafa4`, states `9f4d57d3368a345730395ff5955b4c6c377ed01a3f17c607cbfbaa3523219af2`, and BDDL recorded there. The local 100 train/50 validation/50 test split, 50 retention states, 16K cap, three proposed seeds, and initial ten-demo pack are **benchmark/data choices**, not paper requirements. Original Para/Kai bundle identity and fair cross-method base-weight equivalence remain limited by the audit's evidence. Reused test states are not fresh blind confirmation.

`launch_dice_benchmark.py` uses the shared reset/noise protocol: ordered split IDs; reset seed 100000+i; JAX PRNGKey(100000+i), split once per prior query; native ten-step sampling; K1. It validates all five snapshots, includes initialization as eligible (a declared adoption of the audit draft), chooses most validation successes, ties earliest command, writes selection before testing, and evaluates the selected K1 model and prior on test and source retention. It does not run a new production experiment automatically.

Milestone files are inference snapshots immediately after the exact controller command, **before pending chunk/episode updates**. Collection continues with its pending episode intact; no artificial terminal or shortened replan is introduced at a milestone. This is a declared method-specific update phase. `final.pt` can have later updates than the exact-cap snapshot; benchmark selection uses `step_016000.pt`, not `final.pt`. `resume.pt` is episode-boundary restart state, not a simulator snapshot; an unfinished budget-stopped episode is explicitly discarded. Exact mid-episode simulator continuation is not claimed.

## Validation completed and remaining limits

The dedicated suites passed **58 tests** (`runs/dice_fidelity_20261105/final-tests.log`; portable copy [test-results.txt](test-results.txt)), including existing pi0.5 tests, Q normalization/expert masking, unexecuted-tail invariance and non-selection-eligible diagnostic cost reporting. Numerical/source evidence covers the matrix, deterministic optimizer/scheduler restoration, bitwise uninterrupted-vs-resumed **fixture** runner equivalence, partial-prefix masks, exact mid-chunk caps and pending MC isolation. The upstream critic oracle actually executes the extracted pinned class and copied upstream MLP; it is not a restatement of our implementation.

A real GPU-0 pi0.5/LIBERO smoke used new output paths under `runs/dice_fidelity_20261105/`. Only the smoke task horizon was shortened to ten actual commands; task geometry/states/reward/backend remained real. Smoke ablations: online-only replay, batch 2, two gradient calls per chunk, reference parallelism 1, warmups zero. It retained ten 1024-wide critics, actual frozen native policy, and K_train/K_bootstrap=16. Stage 1 `final_first_v2/resume.pt`: one completed episode, ten commands, two decisions, two critic/two actor/two target updates. Fresh process `final_resumed_v2/resume.pt`: two completed episodes, twenty cumulative commands, four decisions, six critic/four actor/four target updates; replay grew 2→4 rows; reset IDs were 85 then 47. Counters, every restored optimizer moment tensor/group, replay counts, and NumPy/Torch/JAX RNG states were checked against both saved files before continuing optimization. The cumulative smoke ratios were 1.5 critic and 1 actor optimizer updates per collected chunk, or .3 critic and .2 actor updates per executed command; these reflect its smaller diagnostic update budget and empty-replay warmup. No task success in this deliberately shortened diagnostic is evidence about production performance.

[smoke-verification.json](smoke-verification.json) preserves `verification_v3.json` and confirms real prior features/noise/actions bitwise unchanged by a seventh critic/fifth actor update; native zero-residual decode equality; and best-of-16 index/action equality with direct min-ensemble scoring. It does **not** claim bitwise simulator equivalence across a separate uninterrupted real run; that equivalence test used fixtures. Earlier `first/`, `final_first/` and `final_resumed/` artifacts were implementation-stage attempts before final collection eligibility; only `final_first_v2/`, `final_resumed_v2/`, and `verification_v3.json` support the final smoke claims.

A separate real K1/K16 **diagnostic** evaluated the earlier collector attempt's checkpoint `final_resumed/final.pt` (its identity/hash is retained in the cost record) on canonical validation IDs 100 and 101, with the same shortened ten-command task horizon and per-episode RNG contract. Both are explicitly ineligible for checkpoint selection. Recorded costs (model loading/reset excluded; first-query compilation/cache effects retained) were:

| Evaluation | Commands | Decisions / prior candidates | Action-expert forward equivalents | Prior seconds | Scoring seconds | Rollout seconds |
|---|---:|---:|---:|---:|---:|---:|
| K1 | 20 | 4 / 4 | 40 | 11.193 | .159 | 22.839 |
| K16 | 20 | 4 / 64 | 640 | 12.432 | .223 | 22.071 |

These tiny, cache-sensitive timings do not establish production throughput or a K16 speed advantage. Neither shortened diagnostic succeeded; this is not a task-performance evaluation. Machine-readable records are [diagnostic-costs.json](diagnostic-costs.json). The first attempt to shrink the validation protocol was rejected by split-coverage validation; the corrected run used the unchanged complete smoke protocol with an explicit non-selection-eligible subset flag.

No full-budget retraining, demo-assisted real run, held-out success improvement, benchmark seed sweep or measured production best-of-N evaluation was performed. The initial ten-demonstration pack was not found in searched local dataset/DICE directories, and its task/protocol remains unverified. The paper config therefore names an explicit not-yet-exported replay path and an **UNVERIFIED** source ID. It refuses to train without the archive; this is a data prerequisite, not a permission gate. Do not substitute online policy traces and call them demonstrations. The benchmark online-only ablation is launchable without that pack. Any demo-assisted comparison belongs in its own result stratum.

## Commands

From the repository root, use the existing OpenPI and simulator environments. Set `CUDA_VISIBLE_DEVICES` to an allocated device. All outputs below must be new directories.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_dice.sh check --config configs/dice_audit_pi05_object0.toml
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  third_party/openpi/.venv/bin/python -m pytest \
  tests/test_dice.py tests/test_dice_fidelity.py tests/test_rldt_pi05.py -q

# Full local benchmark pipeline; --seed is a local replication choice.
DICE_PYTHON="$PWD/third_party/openpi/.venv/bin/python" \
  third_party/openpi/.venv/bin/python scripts/launch_dice_benchmark.py \
  --config configs/dice_audit_pi05_object0.toml --seed 0 \
  --output runs/dice_audit_seed0_new --best-of-n 16

# Manual training and episode-boundary continuation.
bash scripts/run_dice.sh train --config configs/dice_audit_pi05_object0.toml \
  --train-seed 0 --output runs/dice_audit_manual_new
# Resume only after interruption before the cap, from the last episode checkpoint.
# Increase episodes if needed; an already completed controller budget is rejected.
bash scripts/run_dice.sh train --config configs/dice_audit_pi05_object0.toml \
  --resume runs/dice_audit_manual_new/resume.pt --output runs/dice_audit_resumed_new

# Separate inference cost profile on an already validation-selected checkpoint.
bash scripts/run_dice.sh evaluate --config configs/dice_bestof16_pi05_object0.toml \
  --split validation --checkpoint <selected-checkpoint.pt> \
  --output runs/dice_selected_validation_k16_new
```

For paper-profile data, first verify the archive's exact task, instruction, environment protocol and source split; replace its UNVERIFIED source ID with the actual identity. Export the documented `dice-libero-episodes-v1` format (T+1 verified observations, T controller actions/rewards). Then run:

```bash
# Requires a verified episode pack and matching configured provenance.
third_party/openpi/.venv/bin/python -m rl_vla.dice_data \
  --config configs/dice_paper_pi05_object0.toml --input <verified-episodes.mpk> \
  --output datasets/dice/object_0_chunk_n3.mpk --task-key object_0 \
  --source-id <verified-source-id> --source-split train
bash scripts/run_dice.sh train --config configs/dice_paper_pi05_object0.toml \
  --output runs/dice_paper_adaptation_new
```

A primitive-discounted legacy replay archive cannot be reused for the new chunk-time/n-step profile by changing its metadata. Re-export it from verified episodes. The importer validates the conventions and the archive's task/checkpoint/normalization identity before replay is used. Profile exports also require binary rewards, one success only at the end, and matching termination/truncation labels. Post-success repeated rewards are rejected; prepare a verified first-success endpoint and its final observation rather than guessing or relabeling it.
