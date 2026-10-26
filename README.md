# RL-VLA: OpenPI reinforcement learning and LIBERO evaluation

Train and evaluate policies built around the LIBERO-adapted OpenPI `pi05_libero`
checkpoint.

For the installation in `/vast/projects/jgu32/lab/kai/RL-VLA`, use the
[PARCC setup and verification guide](docs/parcc-setup.md). Its environments,
checkpoint, simulator assets and caches are inside this repository. The
`/workspace/upenn` results below are historical results from another installation.

The repository includes several learning methods with distinct entrypoints
and checkpoint formats:

| Method | What learns | Entrypoint and guide |
|---|---|---|
| Residual SAC / PPO | Small PyTorch actor and critics; OpenPI stays frozen | `python -m rl_vla train`; [LIBERO setup](docs/libero.md) |
| Residual FlashSAC | FlashSAC actor and critics around frozen OpenPI | `python -m rl_vla train`; [FlashSAC guide](docs/flashsac.md) |
| RLDT | Native JAX π0.5 action expert; vision/language backbone stays frozen | `bash scripts/run_rldt.sh`; [RLDT guide](docs/rldt.md) |
| DICE adaptation | PyTorch residual and critics using frozen π0.5 features and demonstration replay | `bash scripts/run_dice.sh`; [DICE guide](docs/dice.md) |

Real-checkpoint smoke tests verify RLDT updates and DICE collection/updates;
they do not establish a training improvement or reproduce those papers' results.
The example training configs target ordinary LIBERO Spatial. Training on
LIBERO-PRO still needs its task adapter and an explicit train/evaluation split.

The original Spatial baseline completed **495/500 episodes successfully (99%)**.
The separate [LIBERO-Plus / PRO queue](docs/robustness-evaluation.md) evaluates
20,030 perturbation episodes, including 2,000 explicitly local PRO Environment
cases. Its ongoing results are partial. See the [verification record](docs/verification.md)
for measured results and limitations.

For the prepared `/workspace/upenn/RL-VLA` installation, start with the
[local server and baseline guide](docs/local-baseline.md). As checked on
**2026-10-21 (America/New_York)**, its container has **one 64 GiB GPU** exposed as
CUDA device 0; ten physical GPUs exist on the host. More devices require a new
allocation from the administrator. The three local Python environments have
different roles and should remain separate.

Source, configurations, tests, launch scripts and documentation belong in Git.
Model weights, demonstration datasets, virtual environments, upstream checkouts,
and raw results stay local. A fresh clone needs the dependencies and assets
described below. Docker covers the residual baseline and simulator; the new
RLDT/DICE launchers use the documented local environments.

## Run with Docker

From this repository in PowerShell or Bash, with Docker running in Linux-container mode:

```text
docker compose build learner test libero
docker compose run --rm test
docker compose run --rm learner train --config configs/sac_toy.toml --steps 128 --output runs/docker_sac
docker compose run --rm --entrypoint python libero /opt/rl-vla/smoke.py --libero
```

Results persist locally in `runs/`, which is ignored by Git and excluded from Docker builds. No host Python environment or `third_party/` checkout is required for these builds. See the [Docker deployment guide](docs/docker.md) for PPO, evaluation, remote OpenPI, GPU service deployment, checkpoint mounts and dependency updates. The full OpenPI GPU Docker build remains unverified; local checkpoint inference and LIBERO rollouts have now passed on gpu9. See [verification results](docs/verification.md).

For an NVIDIA GPU exposed to Docker, build the CUDA demo image and run one task:

```text
docker compose -f compose.gpu-tests.yaml build gpu-tests
docker compose -f compose.gpu-tests.yaml run --rm gpu-tests --output runs/my_gpu_demos --tasks pendulum
docker compose -f compose.gpu-tests.yaml run --rm --entrypoint python gpu-tests /opt/rl-vla/benchmark_report.py runs/my_gpu_demos
```

The image includes CUDA PyTorch, benchmark environments, tests and report tooling; dependency installation is part of the build. Omit `--tasks pendulum` to run the full sequence. See [GPU demos](docs/gpu-demos.md) for GPU checks, task budgets and local reports. These demos train small SAC/PPO networks from numerical observations and do not run the OpenPI model.

## Run it now

First follow [Install on another machine](#install-on-another-machine) to create the local Python environment. Then, from this folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m rl_vla train --config configs/sac_toy.toml --output runs/sac_trial
.\.venv\Scripts\python.exe -m rl_vla train --config configs/ppo_toy.toml --output runs/ppo_trial
.\.venv\Scripts\python.exe -m rl_vla evaluate --config configs/sac_toy.toml --output runs/base_eval.json
.\.venv\Scripts\python.exe -m rl_vla evaluate --config configs/sac_toy.toml --checkpoint runs/sac_trial/final.pt --output runs/sac_eval.json
.\.venv\Scripts\python.exe -m rl_vla evaluate --config configs/ppo_toy.toml --checkpoint runs/ppo_trial/final.pt --output runs/ppo_eval.json
```

Use a new output path for each run. `--steps 128` gives a short training check; `--seed 1` changes the training seed. Evaluation defaults to separate seeds starting at 100000 and stochastic residuals. Add `--deterministic` for a residual-mean ablation. Run `python -m rl_vla --help` for commands.

Each run contains `manifest.json` (configuration, versions, base metadata), `metrics.jsonl` (losses, success events, timing and counts), `step_*.pt`, `final.pt`, and `summary.json`. Evaluation writes per-episode outcomes and a success confidence interval.

## Use a real OpenPI checkpoint

1. Follow [OpenPI setup](docs/openpi.md) to obtain the source for a local installation, or use the [Docker service](docs/docker.md#optional-local-gpu-policy-service). Place the complete checkpoint in `checkpoints/openpi/pi05_libero/` and start its separate GPU server.
2. For the reference task, follow [LIBERO setup](docs/libero.md) and use `configs/libero_sac.toml` or `configs/libero_ppo.toml`. For your own task, implement the [environment contract](docs/environment.md) and register `env.factory = "your_task.adapter:make_env"`.
3. For a custom task, copy `configs/openpi_sac.example.toml` or `configs/openpi_ppo.example.toml`. Set the factory, server host and physical residual bounds. The examples expect the seven LIBERO controller coordinates, with gripper corrections disabled.
4. Evaluate the frozen base first by omitting `--checkpoint`. Then train and evaluate using the same commands as above and your new config.

The example download source is `gs://openpi-assets/checkpoints/pi05_libero`. It is task-specific; use a different checkpoint/config pair if your robot or task differs. [Exact download and serve commands](docs/openpi.md#2-download-the-checkpoint-when-ready).

## Where things go

| Path | Purpose |
|---|---|
| `src/rl_vla/` | Learners, collection, evaluation, OpenPI and task adapters |
| `configs/` | Runnable toy settings and real-policy templates |
| `third_party/openpi/` | Optional local upstream checkout at the roadmap's commit; fetched separately |
| `checkpoints/openpi/` | Local VLA weights plus normalization assets; ignored by Git |
| `datasets/raw/`, `datasets/processed/` | Local demonstrations and converted episodes; see [data locations](datasets/README.md) |
| `datasets/manifest.example.json` | Dataset metadata to fill when the format is known |
| `runs/` | Policy/optimizer checkpoints, logs and evaluation results; ignored by Git |
| `scripts/` | Local launchers, resource monitors, analysis and video tools |
| `tests/` | Numerical correctness and integration checks |
| `docs/` | [Design](docs/design.md), [task integration](docs/environment.md), [data](docs/datasets.md), [sources](docs/research.md) |

The original PDF and Markdown roadmap remain unchanged at the folder root.

See [verification results](docs/verification.md) for exactly what was tested and what still needs GPU/simulator validation.

## Install on another machine

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). NumPy stays below 2 to match the upstream OpenPI client. For a CPU baseline:

```powershell
uv venv --python 3.11
uv pip install --python .venv/Scripts/python.exe torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

On Linux, replace `.venv/Scripts/python.exe` with `.venv/bin/python`. Install the OpenPI client using [its setup instructions](docs/openpi.md). The heavyweight OpenPI server has its **own** Linux/GPU environment. `requirements-tested.txt` records the exact baseline packages tested here; the OpenPI source commit is recorded in `third_party/README.md`.

RLDT and DICE use the separate OpenPI environment described in their guides;
do not install the root project's extras into that locked environment.
The historical [design notes](docs/design.md) describe the original roadmap;
the method guides and verification record describe the current implementation.

## Review and push manually

Run these commands from the repository root. Review the staged paths and diff
before committing; `git diff` alone does not show new, untracked source files.

```bash
git status --short
git diff --check
git add README.md .gitignore .dockerignore Dockerfile .github compose.yaml pyproject.toml src tests configs scripts docs \
  third_party/README.md checkpoints/README.md datasets/README.md runs/README.md rldt.pdf
git diff --cached --stat
git diff --cached --check
git diff --cached
git commit -m "Add RLDT, DICE, FlashSAC and LIBERO robustness evaluation"
git push origin main
```

The local FlashSAC handoff patch is already applied and ignored. The DICE source
checkout is ignored as well; its pinned fetch instructions and license notice
are retained. `rldt.pdf` is the supplied paper referenced by the RLDT guide.
GitHub will receive source and documentation, not the local `runs/` results.
No commit or push is performed by the preparation workflow itself.
