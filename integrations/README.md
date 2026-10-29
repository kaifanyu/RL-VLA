# Integrations

| Directory | Purpose |
|---|---|
| `pi_rl/` | Checkpoint conversion, policy parity, and readiness checks |
| `rlinf_bridge/` | RLinf Flow-SDE training and paired LIBERO evaluation |
| `benchmark_bridge/` | Frozen-policy development screens for PRO, Plus, and LIBERO-90 |
| `multibench/` | RoboTwin, RoboCasa, and ManiSkill training and evaluation |

`multibench/core/rl_vla/` is an intentional source snapshot. It isolates the
multi-benchmark campaign from the active LIBERO runtime; `source_origin.json`
records its provenance. Keep it separate until those campaigns are complete.

Benchmark setup guides:
[RoboTwin](multibench/robotwin/README.md),
[RoboCasa](multibench/robocasa/README.md),
[ManiSkill](multibench/maniskill/README.md).

Commit source, configuration templates, dependency locks, and normalization
provenance. Generated logs, screenshots, downloads, and machine-specific setup
reports stay local. The [campaign guide](../docs/multibench-campaign-20261028.md)
defines the evaluation protocol.
