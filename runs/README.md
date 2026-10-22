Use a new directory per experiment. Bulk outputs are ignored by Git; only this
README is published. The filename `validation` alone does not establish whether
a run used a real VLA: inspect its manifest and the [verification record](../docs/verification.md).

| Output | Contents |
|---|---|
| SAC / PPO / FlashSAC training | Manifest, metrics, periodic `.pt` checkpoints, and final checkpoint/summary when completed |
| RLDT training | Config, manifest, iteration metrics, `expert.npz` and `learner.npz`, final counters |
| DICE training | Metadata, episode metrics, `latest.pt` and `final.pt`; no exact replay/RNG resume |
| Generic residual evaluation | JSON report with per-episode outcomes and confidence interval |
| Baseline / robustness evaluation | `summary.json`, `protocol.json`, `episodes/`, telemetry, logs and selected videos |

Existing local reference results:

- `evaluation/pi05_libero_spatial_20260928/`: completed 495/500 Spatial baseline.
- `evaluation/pi05_libero_robustness_20260929/`: ongoing 20,030-case Plus/PRO queue.
- `rldt_pi05_k8_validation/`: real expert optimizer/save smoke; not a learning result.
- `dice_pi05_smoke_01/`: two-episode collection/update smoke; not a benchmark score.

For before/after policy comparisons, preserve the base checkpoint fingerprint,
task definitions, state/seed schedule and control settings. Keep train and
evaluation states separate. Full RLDT training snapshots are about 8.1 GiB each;
set a retention budget before launching multiple runs.

