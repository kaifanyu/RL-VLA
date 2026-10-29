# Documentation

## Current experiments

| Guide | Purpose |
|---|---|
| [GRASP setup](grasp-setup.md) | Environments, checkpoints, and GPU launch commands |
| [LIBERO campaign](object-campaign-20261028.md) | Object2 and Object1-hard; baseline, RLDT, Flow-SDE |
| [Multi-benchmark campaign](multibench-campaign-20261028.md) | RoboTwin, RoboCasa, ManiSkill; raw `pi05_base` |
| [Results](results/README.md) | Compact evidence retained in Git |
| [Cleanup](cleanup.md) | Repository preparation and artifact retention |

## Methods and interfaces

| Guide | Purpose |
|---|---|
| [RLDT](rldt.md) | Native JAX action-expert adaptation |
| [OpenPI](openpi.md) | Checkpoint download and policy service |
| [LIBERO](libero.md) | Residual SAC/PPO reference task |
| [Environment interface](environment.md) | Add a task adapter |
| [Data](datasets.md) | Demonstrations and replay formats |
| [FlashSAC](flashsac.md) | Optional residual learner |
| [DICE](dice.md) | Optional demonstration-based adaptation |

## Other installations and historical work

| Guide | Scope |
|---|---|
| [Docker](docker.md), [GPU demos](gpu-demos.md) | Containers and small-network demos |
| [PARCC](parcc-setup.md), [PARCC PRO pilot](pro-rldt-parcc.md) | Separate installation and older protocol |
| [Local server](local-baseline.md) | Earlier gpu9 installation |
| [Controlled PRO comparison](controlled-pro-experiments.md) | Superseded spatial/multimethod campaign |
| [Robustness](robustness-evaluation.md) | Earlier LIBERO-Plus/PRO queue |
| [Verification](verification.md) | Dated checks; not live status |
| [Design](design.md), [Research](research.md) | Original design and source notes |

The supplied papers and roadmap remain at the repository root. Live status
belongs in each local run's `status.json`; a historical guide is not a queue monitor.
