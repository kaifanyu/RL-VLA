# Scripts

Run commands from the repository root, using the environment in the linked guide.

| Group | Entry points | Guide |
|---|---|---|
| Setup | `env.sh`, `download_pi05.py` | [GRASP](../docs/grasp-setup.md) |
| LIBERO campaign | `prepare_object_campaign_grasp.py`, `run_object_campaign_grasp.sh` | [Protocol](../docs/object-campaign-20261028.md) |
| Multi-benchmark campaign | `integrations/multibench/configure.py`, `integrations/multibench/submit.py` | [Protocol](../docs/multibench-campaign-20261028.md) |
| RLDT | `run_rldt.sh` | [Method](../docs/rldt.md) |
| Flow-SDE | `run_pi_rl.sh`, `run_pi_rl_task_grasp.py` | [Integration](../integrations/README.md) |
| Analysis | `analyze_rldt.py`, `compare_pro_methods.py`, `plot_training.py` | [Run index](../runs/README.md) |
| Readiness | `run_pi_rl_readiness_grasp.sh`, `run_pi_rl_smokes_grasp.slurm` | [Controlled comparison](../docs/controlled-pro-experiments.md) |
| Earlier PRO campaigns | `prepare_pro_*`, `run_pro_*`, `launch_pro_spatial_grasp.py` | [Controlled comparison](../docs/controlled-pro-experiments.md) |
| Development screens | `prepare_benchmark_screen.py`, `prepare_libero90_screen.py` | [Controlled comparison](../docs/controlled-pro-experiments.md) |
| Optional baselines | `run_baseline.sh`, `run_robustness.sh`, `run_dice.sh`, `fetch_flashsac.sh` | [Documentation](../docs/README.md) |

Keep launcher filenames stable: saved experiment contracts refer to these paths.
An installed launcher does not mean its campaign is active.
