# Runs

Raw runs are local and excluded from Git. Active GRASP artifacts remain in
`/mnt/projects/jg/kaifany/RL-VLA/runs/`; the clean publishing checkout contains
source and documentation only. [Compact results](../docs/results/README.md)
are retained in Git.

## Run index

| Label | Local directory | Purpose |
|---|---|---|
| LIBERO objects | `object_adaptation_20261028/` | Current Object2 and Object1-hard campaign |
| Multi-benchmark | `multibench_20261028/` | Current RoboTwin, RoboCasa, and ManiSkill campaign |
| GPU allocation | `gpu_allocation_1470624/` | Active allocation metadata and shared lock |
| Setup | `setup/` | Environment checks and readiness evidence |
| PRO comparison | `pro_multimethod_20261027/` | Earlier experiments and required Flow-SDE readiness evidence |
| Spatial0 | `pro_spatial0_rldt_20261026/` | Completed spatial adaptation experiment |
| Initial RLDT | `rldt_pi05_libero_1470466/` | Completed original-task experiment |
| Baseline screens | `benchmark_screen*_20261027/` | Task-selection evidence |
| Multi-object pilots | `multiobject*_20261027/` | Completed layout pilots |
| Analysis | `analysis_20261027/`, `analysis_20261028/` | Plots, comparisons, and failure reviews |
| Cleanup archive | `archive/cleanup_20261029/` | Compact diagnostics from removed failed attempts |

## Reading a run

- `status.json`: current stage and completion/failure state.
- `protocol.json` and `input_contract.json`: fixed task, split, and source identity.
- `training/metrics*.jsonl`: training progress; not held-out performance.
- `baseline_validation.json` and `validation_*.json`: checkpoint selection evidence.
- `selection.json`: selected checkpoint and rule.
- `baseline_test.json` and `selected_test.json`: final paired comparison.
- `baseline_source.json` and `selected_source.json`: original-task retention.

## Retention

Keep active runs, unique results (including failures), source snapshots, protocols,
seeds, selected policies, and referenced readiness evidence. Remove abandoned
setup retries and redundant smoke-test weights after preserving their compact
diagnostics. Do not rename active run paths: launchers and contracts reference them.

Use `<task>_<method>_<YYYYMMDD>` for new standalone runs. Campaign directories
may group tasks and methods under one dated root.
