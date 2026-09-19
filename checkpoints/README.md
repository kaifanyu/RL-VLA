# Checkpoint locations

- `openpi/pi05_libero/`: put the complete frozen openpi checkpoint here, including `params/` and `assets/`.
- RL residual actor/critic checkpoints belong to each experiment's output directory, not inside the frozen base checkpoint.

No weights are bundled or downloaded. The official example checkpoint is `gs://openpi-assets/checkpoints/pi05_libero`, served with config `pi05_libero`. Follow [download and server instructions](../docs/openpi.md). Use another task-compatible checkpoint and its matching observation/action transforms for a different robot.

Large checkpoint contents are ignored by version control. Keep this README.
