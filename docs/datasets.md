# Dataset space

Online SAC/PPO can run without a dataset. Place untouched data in `datasets/raw/`, conversions in `datasets/processed/`, and copy `datasets/manifest.example.json` to describe the actual dataset when chosen. All bulk data is ignored by Git.

No loader assumes that an unknown "standard dataset" is LeRobot, RLDS or HDF5. Keep its native metadata and write one adapter once its format is known. The baseline does not silently train from files in these folders.

Record at least:

| Field | Why it matters |
|---|---|
| Format/version, source, split IDs | Reproducible conversion and held-out evaluation |
| Robot, task/instruction, controller | Compatible checkpoint and action semantics |
| Camera names, orientation, RGB layout, calibration | Match model input transforms |
| Action names, units, frame, rate, normalization | Correct residual composition |
| Episode IDs, timestamps, terminal/truncated flags | Sequence boundaries and discounting |
| State, images, executed actions, verified success | Supervised extraction and evaluation |
| Base checkpoint/config, nominal chunk, sampled residual | Valid residual replay labels |

For SAC residual replay, ordinary demonstration actions are insufficient on their own. Their conditioning base chunk and sampling protocol must be known, and actions must be representable within the residual bounds. Controller-clipped actions cannot generally be inverted uniquely into the original residual. Preserve the actual sampled residual during online collection.

PPO uses newly collected on-policy rollouts, not demonstration replay. Future demonstrations can support pretraining or a separate supervised anchor. For OpenPI distillation, construct native action chunks with padding masks and compatible normalization; use the upstream flow loss. These data training stages are not implemented in this baseline.

