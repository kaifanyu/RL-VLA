# FlashSAC residual on LIBERO-PRO Object 0 (Position): results and failure analysis

Run: `runs/flashsac_object_00/`, trained 2026-11-03 to 11-04, seed 0. Test evaluation
`ev20261104T004954`. Retention evaluation `ev20261104T035723`.

## Summary

On the 50 held-out test states, a small residual on frozen π0.5 raises success from 24/50
(48%) to 33/50 (66%). The paired test does not reach significance (16 gained, 7 lost, exact
McNemar p = 0.093), and this is one seed. The headline number is less interesting than how
the policy gets there:

1. **The benchmark perturbation is an object swap, and the base follows position, not
   language.** PRO Position exchanges the soup and the butter. In 22 of its 26 failures the
   base puts the butter in the basket, which sits exactly where the soup sat in the original
   task.
2. **The residual changes which object the base picks, and it does so with a nudge.** The
   target is picked in 50/50 states against 28/50 for the base (p = 5e-7). At step 15 the
   gripper is only 1.35 cm from where the base alone would have it. After that the frozen
   base's own command changes and it descends on the soup.
3. **Validation swings are switches in object choice, not noise.** Validation went 36, 66,
   36, 2, 74%. Re-running the checkpoints shows 0 butter picks at 4k and 16k, and 7 and 9
   of 12 at 8k and 12k. Smooth critic curves show none of this.
4. **The residual moves the failure from picking to placing.** All 17 residual failures hold
   the right object. Success given a pick falls from 24/28 (86%) to 33/50 (66%). The
   failures split into four mechanisms, all at the grasp or the release. The residual cannot
   command the gripper.
5. **The most useful next experiment needs no training.** Turning the residual off after the
   pick could keep the 100% pick rate and the base's placing skill. If the base's 86% held,
   that would be about 86% success, against 66% now.
6. **The gain has a small cost on the original task.** On unperturbed LIBERO, success goes from
   48/50 to 44/50 (6 lost, 2 gained, p = 0.29). The residual never picks the wrong object
   there, but its grasps come later and episodes get longer.

## Setup

| Item | Value |
|---|---|
| Task | "Pick up the alphabet soup and place it in the basket", PRO Position, case `pro_libero_object_6b0ff024b794ff53` |
| States | 100 train, 50 validation (generated), 50 test (published PRO states), RLDT `object_0` seeds |
| Policy | Frozen `pi05_libero` plus a FlashSAC residual at every env step |
| Correction limit | 0.1 on xyz (5 mm per step), 0.05 on rotation, **0 on the gripper** |
| Reward | Sparse: 1 on success, 280-step limit |
| Budget | 16,105 env steps, 77 episodes, 14,106 updates |
| Renderer | OSMesa throughout, so episodes are repeatable |

Training took 9.2 h, of which rollouts were 2.1 h. Almost all the rest was five validation
passes on the CPU renderer. RLDT reports about 2.0 h, but the two numbers are not like for
like because of this validation cost. Only 55 of the 100 train states were ever sampled.

## Result

| Held-out test | Frozen base | Base + residual |
|---|---|---|
| Success | 24/50 (48%) | 33/50 (66%) |
| 95% Wilson interval | 35–61% | 52–78% |
| Timeouts | 26 | 17 |
| Mean length of successful episodes | 168 steps | 126 steps |

| Both succeed | Residual only | Base only | Both fail |
|---|---|---|---|
| 17 | 16 | 7 | 10 |

The base numbers are trustworthy: all 45 states with a robustness-queue record reproduce it
exactly. Validation (74%) overstates the selected checkpoint by 8 points, as expected for the
best of four. Mean absolute correction is about 0.035 on xyz, or 1.8 mm per step.

![Run summary](../runs/flashsac_object_00/summary_plot.png)

## Why the base fails: position beats language

The PRO BDDL differs from the original in exactly two lines: the soup and the butter trade
regions. The butter now starts at (-0.12, -0.24), inside the original `target_object_region`.
The soup starts at (-0.199, -0.080), the same point in every recorded episode.

| Base outcome (50 states) | Count |
|---|---|
| Success | 24 |
| Butter placed in the basket | 22 |
| Soup lifted, not placed | 4 |

From the camera these 22 episodes look like clean successes, so video alone misleads here.
The base has not lost the skill. It runs the trained motion on the object at the trained
location, while the instruction names something else.

The choice is a sharp fork. The gripper first travels in -y, and the two groups differ by
only 2.5 mm at step 15. By step 30 they are about 11 cm apart, and only 3 of the 50 states
lie between the two clusters. The base is therefore bistable on
this task rather than uniformly confused. It picks the soup in 56% of states.

What tips the fork is not semantic. Base butter states have the basket 8 mm farther in +y
(0.2639 against 0.2558, Mann-Whitney p = 0.0004; four position features were checked, so
this survives Bonferroni). This is a correlation, not a causal result. If it holds, a
distractor shifted by under 1 cm is deciding which object a language-conditioned policy
picks.

## What the residual learned: a nudge across the fork

![Mechanism analysis](../runs/flashsac_object_00/mechanism_analysis.png)

The obvious reading is that the residual drives the arm to the soup. The trajectories rule
that out:

- **The push is small.** Over the first 45 steps the correction adds 5.8 cm in y, against
  37.8 cm of -y motion commanded by the base. At step 15 the two paths differ by 1.35 cm
  (range 0.75–2.14).
- **The frozen base does the rest.** In the 22 butter states, the base's own y command at
  steps 20–30 is -0.62 in the base episode and 0.00 in the residual episode, with the same
  weights. A slightly different image and proprio state puts it in the other mode.
- **The effect is complete.** All 50 gripper paths at step 30 sit at y between -0.096 and
  -0.072, over the soup.

The residual acts as a **mode selector for a bistable policy**, not as a controller. This
explains how a correction capped at 5 mm per step can fix a failure that looks semantic.

The correction is also largely a function of time. Time alone explains 76%, 47% and 53% of
its variance on x, y and z over the first 100 steps, and the same push appears in every
state. Because the soup never moves in this benchmark, nothing here shows the residual reads
the instruction or locates the soup. It could be a learned open-loop opening move.

Retention rules out a purely forcing push. In the original task the soup is back at y = -0.24,
yet the same +y correction (6.3 cm over 45 steps, against 5.8 cm on PRO) never sends the arm to
y = -0.08. All 50 gripper paths reach the soup side by step 30 (y from -0.25 to -0.18), and no
episode lifts the butter. The same nudge has a different effect in a scene where the base is not
near a fork, which fits the mode-selector reading.

## Why validation swung: discrete mode switches

Validation records no object poses, so we re-ran four checkpoints on the first 12 validation
states, capped at 100 steps (`runs/flashsac_object_00_diag/`).

| Env steps | Validation success (50 states) | Soup lifted | Butter lifted | Nothing lifted by step 100 |
|---|---|---|---|---|
| 0 (base) | 18 | not run | not run | not run |
| 4,068 | 33 | 10 | 0 | 2 |
| 8,157 | 18 | 3 | 7 | 2 |
| 12,151 | 1 | 0 | 9 | 3 |
| 16,105 | 37 | 12 | 0 | 0 |

The 2% at 12k is the policy steering nearly everything to the butter. It is not damaged
motor control: mean correction there (0.015) is smaller than at 16k (0.023). Success is a
cliff in policy space, because a change of a few millimeters at step 15 moves all states to
the other side of the fork at once.

Three consequences follow:

- **Standard diagnostics are blind to this.** Critic loss falls smoothly from 5.0 to 3.1 and
  Q tracks its target throughout. Nothing marks the collapse at 12k.
- **Rollouts and the mean policy can disagree badly.** Stochastic rollouts between 10k and
  14k steps still succeed in 5 of 17 episodes (29%), while the mean policy at 12k scores 2%. Our reading is that
  exploration noise crosses the fork at random, so a rollout success rate says little about
  the deterministic policy. This is a hypothesis.
- **We do not know that the final policy is stable.** Rollouts went from six straight failures
  to twelve straight successes at episodes 64–75, about 1,800 steps before the budget ended.
  The selected checkpoint is the first sign of this regime, and the run stops there. With 16k
  steps, "best checkpoint" here means "which side of the fork training ended on".

Spatial 1 showed a similar non-monotone curve (12, 34, 18%), in that run under the noisier
EGL renderer. Whether the same mechanism applies there is untested.

## Why the residual fails: from picking to placing

Success factors into picking the right object and then placing it:

| | P(pick soup) | P(place given pick) | Success |
|---|---|---|---|
| Frozen base | 28/50 = 56% | 24/28 = 86% | 48% |
| Base + residual | 50/50 = 100% | 33/50 = 66% | 66% |

The pick gain is overwhelming (22 states fixed, 0 broken, McNemar p = 5e-7). The place loss
is suggestive but not established: Fisher p = 0.069, and on the 28 states where both pick the
soup, 24 against 19 (p = 0.18). Residual placing is about the same on the 22 states the base
got wrong (14/22) and the 28 it got right (19/28), so the newly gained states are not simply
harder.

The automatic label "held at timeout" hides real differences, so we reclassified the 17
failures by when the soup left the gripper and what the gripper was commanded to do.

| Failure class | n | Evidence | Gripper role |
|---|---|---|---|
| Slip in transport | 6 | Soup leaves while closed is commanded, 12–34 cm from the basket; fingers close to a gap of 0.7–2 cm | Grip was too marginal |
| Never released | 5 | Soup still held at timeout; open commands flicker (5.4 re-closes per episode, against 1.6 in residual successes); arm drifts past the basket | Base fails to commit to opening |
| Early release | 3 | Opens 9–11 cm from basket center, soup lands on the floor | Base opens before the soup is over the basket |
| Lodged on rim | 3 | Opens 7.2–7.6 cm from center, soup stays at z = 0.177 | Same, by a smaller margin |

**Release geometry is sharp.** All 30 residual successes with a measurable release let go
within 6.6 cm of the basket center. Lodging starts at 7.2 cm and floor drops at 8.7 cm. The
margin is about a centimeter, and the mean correction during transport, 3 mm per step in +x
and 2.5 mm in +y for steps 45–80, is large enough to use it up. The base opens on its own
schedule from what it sees, and it cannot know about a correction it did not plan. Residual
successes release earlier than the base's (median step 109 against 139), which accounts for
most of their 42-step shorter episodes.

**Grasp quality predicts slips.** All 6 slips have the soup at least 2.5 cm to the -x side of
the gripper at lift (residual success against failure, Mann-Whitney p = 0.0003). With a -2.4 cm
cut, the residual succeeds on 29/34 centered grasps and only 4/16 off-center ones. The
residual does not grasp worse: median offset is -2.1 cm against the base's -2.3 cm, and the
paired change is -0.04 cm. It tolerates off-center grasps worse, since the base succeeds on
9/13 of them (Fisher p = 0.027, with a cut chosen after seeing the data, so treat this as an
exploratory result). The likely explanation is that a steady sideways push loads a marginal
grip. We have not tested this.

The 7 states lost relative to the base are 4 slips and 3 never-released. Correction size does
not separate outcomes: transport totals are about 12 cm in x for success, drop and held
episodes alike. The residual applies much the same action in every state, whether or not this
state's grip or alignment can afford it.

## Retention: the cost on the original task

Retention runs the same checkpoint on the unperturbed LIBERO task, with original object positions
and 50 original initial states.

| Retention | Frozen base | Base + residual |
|---|---|---|
| Success | 48/50 (96%) | 44/50 (88%) |
| 95% Wilson interval | 87–99% | 76–94% |
| Timeouts | 2 | 6 |
| Mean length of successful episodes | 149 steps | 174 steps |

| Both succeed | Residual only | Base only | Both fail |
|---|---|---|---|
| 42 | 2 | 6 | 0 |

The 8-point drop is not significant (exact McNemar p = 0.29), but its direction and its mechanism
are both informative. Each of the 6 lost states picks the correct object or attempts to:

| Lost state class | n | States |
|---|---|---|
| Grasp missed or too late (no lift, or lift at step 276) | 2 | 20, 34 |
| Slip in transport | 2 | 17, 35 |
| Never released | 2 | 19, 26 |

Slips and failures to release are the same classes as on PRO. Missed grasps are new. The base lifts
the soup by step 69 in 90% of states, while the residual's 90th percentile is step 119 and 14 states
lift after step 80. Successful episodes are 26 steps longer, the opposite of the 42 steps saved on
PRO. A likely cause is that the opening push, tuned for the soup at y = -0.08, works against the
approach to a soup at y = -0.24, so the base must correct or retry. We have not tested this.

Taken together: +18 points on the perturbed task and -8 on the original. The residual is a
specialization to this perturbation, not a strict improvement. A gate that turns it off after the
pick (experiment 1) would address the slips and failed releases, though not the slower grasps.

## Interpretation

**Sparse-reward residual RL found the cheapest lever, which was the fork.** Crossing it earns
the full reward, and placing precision only matters afterward. In 77 episodes the critic gets
one bit per episode, with no separate signal for a good grasp or a good release. Getting the
first decision right and paying a small tax on the rest is what this objective would be
expected to produce.

**The residual's action space fits the failure pattern.** Every failure happens at the grasp
or the release, and the gripper scale is 0. That keeps the base's gripper timing, but the residual
moves the arm under that timing, so position and release can drift apart. The design
decoupled two things that have to agree.

**This does not show improved language grounding.** The soup is at a fixed location, so
"go toward y = -0.08 early" solves the benchmark. A policy that really grounded the
instruction would be distinguishable only under variation this split does not contain. The
66% should be read as a task-specific correction of a position prior.

**VLA failures here look like selection among competent behaviors.** The base can already
pick the soup and place it well. That points to a cheaper class of method: rather than
retraining motor skill, steer the policy at its few decision points. It also suggests a
concrete robustness probe for VLAs: find forks and measure how little perturbation crosses
them. A policy whose object choice flips under about 1 cm is fragile in a way that average
success rate does not show.

## Next experiments, in priority order

| # | Experiment | Cost | What it decides |
|---|---|---|---|
| 1 | **Gate the residual off after the pick** (for example at lift), using `best.pt` | No training, about 3 h | Whether the place loss is caused by the residual. If so, expect well above 66%, near 86% if the base's rate carries over |
| 2 | **Open-loop replay:** apply the mean correction by time step, without the actor | No training, about 3 h | Whether the actor's observation matters at all. Parity would make this a fixed nudge |
| 3 | **Minimal crossing:** scale the first 15–20 steps by 0, 0.25, 0.5, 1 | No training | How close the base sits to the fork, and how large the nudge must be |
| 4 | **Vary the target location,** or swap the instruction to name the butter | Evaluation only | Whether the residual reads the task. A nudge toward the soup under a "butter" instruction would settle this |
| 5 | **Seeds 1 and 2, with extended budget** | About 10 h each | Whether 66% is typical, and whether the regime after 14k persists |
| 6 | **Log object poses in validation, and report pick rate** | Small code change | Makes the mode switch visible during training, and enables selection on it |
| 7 | **Dense or staged reward (grasp, over-basket, release)** | New training | Gives RL a signal for the phase it currently damages |
| 8 | **Paired comparison with RLDT `object_0`** | Needs the PARCC results | Compare each method's paired change against its own base |

Experiments 1–3 are the best value: they need no training and directly test the two central
claims.

## Limitations

- **One seed, 50 states, p = 0.093.** The overall gain is not statistically established. The
  pick effect is, within this seed and task.
- **The analysis is post hoc.** The failure classes, the -2.4 cm cut and the feature checks
  were all defined after seeing the test data. Those test results were used for analysis only,
  never for selection, but the p-values above are exploratory, and the test split should now
  be treated as seen for any method development based on this report.
- **The diagnostic is small.** It covers 12 validation states and 100 steps. "Nothing lifted"
  is not failure: some of those episodes lift later and succeed in the full validation.
- **Causal claims are limited.** The nudge and the base's response are observed in paired
  rollouts. That the correction causes the switch is strongly suggested but has not been
  isolated, which is what experiments 2 and 3 are for. The explanation for the sideways-load
  slips is a hypothesis.
- **Failure classes come from logged poses.** We checked frames for one episode in each of
  four cases (never released, rim, early release, slip), not all 17. The rim class rests on
  the z = 0.177 resting height plus one frame.
- **Retention base numbers have no external check.** Stored original-task episodes exist only for
  the Spatial suite, so the 48/50 base result is not cross-checked against a record.
- **RLDT state identity is unverified.** The `object_0` state hash has not been compared with
  RLDT's file on PARCC. Eval seeds also differ from RLDT's `100000 + i`.

## Reproduce

```bash
R=runs/flashsac_object_00; E=ev20261104T004954
.venv/bin/python scripts/plot_flashsac_run.py $R --eval-id $E
.venv/bin/python scripts/analyze_flashsac_run.py $R --eval-id $E
.venv/bin/python scripts/analyze_flashsac_mechanism.py $R --eval-id $E --diag ${R}_diag
```

Per-state classes are in `runs/flashsac_object_00/mechanism_episodes.csv`, and failure modes in
`failure_episodes.csv`. The diagnostic run used, for each checkpoint:

```bash
bash scripts/run_flashsac_pro.sh evaluate --run ${R}_diag --split validation --variants residual \
  --episodes 12 --max-episode-steps 100 --checkpoint $PWD/$R/checkpoints/step_00012151.pt --eval-id diag00012151
```

![Failure analysis](../runs/flashsac_object_00/failure_analysis.png)

![Training diagnostics](../runs/flashsac_object_00/training_diagnostics.png)
