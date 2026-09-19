# RL-VLA baseline

Train **residual SAC or PPO around a frozen OpenPI policy**. OpenPI proposes an action chunk; a small PyTorch actor learns bounded corrections. Both algorithms use the same observations, action interface and reward accounting.

**Included:** CPU toy runs, both learners, tests, evaluation, checkpoints and the real OpenPI websocket client adapter. Docker builds fetch pinned upstream sources and include the learner and a headless LIBERO simulator with pinned dependencies/assets. The `gpu` target adds CUDA learning and sequential simulation demos. A separate OpenPI GPU server image is configured. **Needed for VLA runs:** a compatible checkpoint and sufficient GPU memory. No model weights or demonstration datasets are included. The toy is a software check, not evidence of robot performance.

## Run with Docker

From this repository in PowerShell or Bash, with Docker running in Linux-container mode:

```text
docker compose build learner test libero
docker compose run --rm test
docker compose run --rm learner train --config configs/sac_toy.toml --steps 128 --output runs/docker_sac
docker compose run --rm --entrypoint python libero /opt/rl-vla/smoke.py --libero
```

Results persist locally in `runs/`, which is ignored by Git and excluded from Docker builds. No host Python environment or `third_party/` checkout is required for these builds. See the [Docker deployment guide](docs/docker.md) for PPO, evaluation, remote OpenPI, GPU service deployment, checkpoint mounts and dependency updates. The full OpenPI server build and checkpoint inference still require validation on a suitable GPU machine; see [verification results](docs/verification.md).

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
| `checkpoints/openpi/` | Future VLA weights plus normalization assets |
| `datasets/raw/`, `datasets/processed/` | Future demonstrations and converted episodes |
| `datasets/manifest.example.json` | Dataset metadata to fill when the format is known |
| `runs/` | Residual weights, logs and evaluation results |
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

This baseline updates the residual actor and critics. Direct flow-policy fine-tuning, distillation, dataset replay, batched Warp tasks, and visual robustness training remain later research stages described in [the design notes](docs/design.md).
