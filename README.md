# RL-VLA

Fine-tune and evaluate OpenPI policies on robotics tasks.

The current experiments compare a frozen baseline, **RLDT**, and **Flow-SDE**.
LIBERO experiments start from `pi05_libero`; RoboTwin, RoboCasa, and ManiSkill
experiments start from `pi05_base`. These use separate baseline measurements.

## Start here

| Need | Guide |
|---|---|
| Install on GRASP | [Setup](docs/grasp-setup.md) |
| Run Object2 and Object1-hard | [LIBERO campaign](docs/object-campaign-20261028.md) |
| Run RoboTwin, RoboCasa, and ManiSkill | [Multi-benchmark campaign](docs/multibench-campaign-20261028.md) |
| Find a launcher | [Scripts](scripts/README.md) |
| Find results and retained runs | [Run index](runs/README.md) |
| Recreate upstream dependencies | [Dependencies](third_party/README.md) |
| Other methods and earlier experiments | [Documentation](docs/README.md) |

## Layout

| Directory | Contents |
|---|---|
| `src/rl_vla/` | Learners, policy adapters, and evaluation |
| `integrations/` | RLinf and benchmark bridges; [map](integrations/README.md) |
| `configs/` | Example settings and experiment policy |
| `scripts/` | Setup, launch, and analysis commands |
| `tests/` | CPU tests and optional integration checks |
| `docs/` | Setup, methods, protocols, and compact results |
| `third_party/` | Pinned upstream checkouts, downloaded separately |
| `checkpoints/`, `datasets/`, `runs/` | Local weights, data, and raw outputs; excluded from Git |

## CPU quick start

Use Python 3.11. Install GPU policy and simulator environments separately using
the setup guides.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python -m rl_vla train --config configs/sac_toy.toml --steps 128 --output runs/toy_sac
```

For Windows, use `.venv/Scripts/python.exe`. For containers, follow the
[Docker guide](docs/docker.md). Toy runs check the software; they are not VLA
benchmark results.

## Experiment rules

- Use a new output directory for each experiment.
- Measure the frozen baseline on the same evaluation states as the adapted policy.
- Keep training, validation, and test states separate. Select checkpoints using validation only.
- Report validation and final test results separately, with successes and episode counts.
- Preserve source snapshots, configurations, seeds, and compact results. Bulk artifacts stay local.

The current campaigns train for 100 iterations and validate at 25, 50, and 100.
Spatial campaigns and Flow-Noise are paused. Historical protocols retain their
original schedules.

## Review and push

```bash
git status --short
git log -1 --oneline
git show --stat HEAD
git push -u origin cleanup
```

The `cleanup` branch is prepared separately from the active GRASP checkout.
See the [cleanup record](docs/cleanup.md) for retained artifacts and validation.
