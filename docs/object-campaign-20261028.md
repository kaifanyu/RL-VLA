# Active experiment direction — October 28, 2026

The user cancelled all prior active experiments and pending queues. Retain
Slurm GPU allocation 1470624. Run only frozen baseline, RLDT and Flow-SDE;
spatial runs and Flow-Noise are paused. Full task launchers enforce
`configs/experiment_policy.json`. Historical results remain available.

The active campaign is `runs/object_adaptation_20261028`. Its queue runs
Object2 RLDT, Object2 Flow-SDE, Object1-hard RLDT, Object1-hard Flow-SDE.

| Task | Prompt | Condition |
| --- | --- | --- |
| Object2 | Pick the salad dressing and place it in the basket | Published LIBERO-PRO Position/swap; the earlier native baseline screen was 0/5, not an established population success rate |
| Object1-hard | Pick the cream cheese and place it in the basket | Local six-item position cycle; all food objects move relative to the original layout; basket, prompt, goal and region geometry stay fixed |

For Object1, cream cheese goes to the original orange-juice region, alphabet
soup to cream cheese's region, milk to alphabet soup's region, tomato sauce to
milk's region, butter to tomato sauce's region, and orange juice to butter's
region. This aims to make the task harder; baseline evaluation determines
whether it actually does. It is a custom LIBERO task, not an official PRO score.

Each method starts independently from the original pi05_libero checkpoint
(converted original weights for Torch). No previous adapted weights, added
demonstrations or demonstration SFT warm start are used. Measure separate
frozen native-JAX and converted-Torch baselines because their RNG implementations
differ. Baseline is iteration zero and has no training iterations.

Each task has 100 train, 50 validation and 50 final test states, plus 50 original
states for retention. Object2 retains the 50 published PRO states for final
testing. Object1 uses 200 newly generated custom-layout states, partitioned
100/50/50. Generation excludes the published, original and development-screen
states; it does not filter by policy success. Check all 250 resets and cameras
before starting a task.

Train one continuous 100-iteration trajectory per method, 32 decisions per
iteration, up to five controller steps per decision. At 25, 50 and 100,
save the checkpoint, pause training, and evaluate all 50 validation states in
a separate process. The learner retains optimizer, replay, environment and
RNG state. Only validation states enter these checks. Select the highest
validation success count, breaking ties at the earliest iteration, then run
paired final test and original-task retention evaluations. Keep validation and
test results distinct. This is one training seed, not a full-suite benchmark.

The user removed iteration-75 validation on October 28. Recovery checkpoints
may still be saved at iteration 75; they are neither evaluated nor eligible
for selection. The amendment preserves the in-progress Object2 baseline and
records its original input contract and source snapshot separately from the
updated controller schedule. Training has not been restarted.

`checkpoint_validations.jsonl` records online results and validation costs.
`training/summary.json` separates training time from checkpoint-validation time;
`stage_timings.jsonl` also includes process/model startup costs. Actual training
controller steps, policy decisions, optimizer counts/diagnostics, all evaluation
outcomes and replay paths are retained. Campaign-level `validation_results.csv`
and `validation_results.json` update while the queue runs. Status files and
logs are authoritative; queued work is not a completed result.

The GPU reservation expires October 30 at 19:32:42 America/New_York. Work runs
inside that reservation; this change does not extend or release it.
