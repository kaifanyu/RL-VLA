# Results

Snapshot: 2026-10-29T14:44:05-04:00. These files are dated evidence, not live status.

| File | Contents |
|---|---|
| [Object validation](object-validation.csv) | Completed baseline and checkpoint validation passes |
| [Object test](object-test.csv) | Final test and original-task retention for completed runs |

Each completed evaluation contains 50 episodes. Experiments use one training seed.
The selected iteration is chosen on validation, never on test outcomes.

Object2 RLDT and Flow-SDE both completed 100 iterations with zero validation
or final test successes on the shifted task. Both retained high original-task
success. Object1-hard is still in progress; its CSV rows are validation only.

RoboTwin, RoboCasa, and ManiSkill have no completed production results in this
snapshot. Setup smoke checks are not task-success measurements.

Raw outcomes, videos, checkpoints, and historical comparisons remain in the
original local checkout. See the [run index](../../runs/README.md).
