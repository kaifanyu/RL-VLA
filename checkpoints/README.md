# Checkpoint locations

- `openpi/pi05_libero/`: put the complete frozen openpi checkpoint here, including `params/` and `assets/`.
- RL residual actor/critic checkpoints belong to each experiment's output directory, not inside the frozen base checkpoint.

No weights are bundled in Git. The prepared server already has the complete
`openpi/pi05_libero/` checkpoint and has evaluated it in LIBERO. A fresh clone
must download its own checkpoint. The official source is
`gs://openpi-assets/checkpoints/pi05_libero`, served with config `pi05_libero`.
Follow [download and server instructions](../docs/openpi.md). Use another
task-compatible checkpoint and matching transforms for a different robot.

Large checkpoint contents are ignored by version control. Keep this README.
