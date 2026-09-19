# Sequential GPU demos

These experiments use the existing residual SAC/PPO implementations with CUDA
for the learner and CPU physics/rendering. The image includes Pendulum, MuJoCo
Reacher, and Panda reaching, pushing, and pick-and-place environments. It does
not load or fine-tune OpenPI. The conservative near-zero actor initialization
and core learners are unchanged.

## Build and verify

Use Docker Engine with Compose v2 and NVIDIA Container Toolkit on Linux, or
Docker Desktop with its WSL2 Linux-container backend and an NVIDIA driver
supporting GPU containers on Windows. The image targets Linux x86_64 and uses
PyTorch 2.7.0 with CUDA 12.6. No host Python environment, CUDA toolkit,
`third_party/` checkout, prebuilt image or prepared container is required.
The first build downloads several gigabytes of CUDA dependencies. Budget at
least 30 GB of free space on Docker's storage drive for the first CUDA build;
this is a conservative planning estimate that includes cache and image-export
copies, and actual usage varies.

The packaging check installed all locked packages, passed `uv pip check` for
95 installed packages, and passed the Torch 2.7.0+cu126, CUDA 12.6 and benchmark
import assertions. Image export/unpacking then filled the Windows `C:` drive
and failed. The incomplete image and its 9.01 GB build cache were removed,
Docker recovered, and saved runs remain intact. A completed GPU image and its
runtime checks are **not yet verified**. Earlier demos used a manually prepared
container and do not validate this new image. See [verification](verification.md#github-packaging-check-2026-09-19).

From the repository root:

```text
docker compose -f compose.gpu-tests.yaml config --quiet
docker compose -f compose.gpu-tests.yaml build gpu-tests
docker compose -f compose.gpu-tests.yaml run --rm --entrypoint nvidia-smi gpu-tests
docker compose -f compose.gpu-tests.yaml run --rm --entrypoint python gpu-tests -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
docker compose -f compose.gpu-tests.yaml run --rm --entrypoint python gpu-tests -m pytest -q -p no:cacheprovider
```

The Compose service builds Dockerfile target `gpu` as `rl-vla:gpu`. Its CUDA
packages and benchmark tools are installed from the hash-pinned
`docker/requirements-gpu.lock` during the build. Source, tests and scripts are
included in the image; rebuild after changing them. Only `runs/`, `configs/`,
`datasets/`, and `checkpoints/` are mounted, with `runs/` writable. On Linux,
follow the [directory ownership setup](docker.md#volumes-ownership-and-cleanup)
before running jobs.

## Run and inspect

Start with one task, which trains SAC and then PPO:

```text
docker compose -f compose.gpu-tests.yaml run --rm gpu-tests --output runs/my_pendulum --tasks pendulum
docker compose -f compose.gpu-tests.yaml run --rm --entrypoint python gpu-tests /opt/rl-vla/benchmark_report.py runs/my_pendulum
```

Run all tasks sequentially with:

```text
docker compose -f compose.gpu-tests.yaml run --rm gpu-tests --output runs/my_gpu_demos
docker compose -f compose.gpu-tests.yaml run --rm --entrypoint python gpu-tests /opt/rl-vla/benchmark_report.py runs/my_gpu_demos
```

The runner executes one task/algorithm at a time: toy, weaker-base toy,
Pendulum, MuJoCo Reacher, Panda reaching, pushing, and pick-and-place. Each
task evaluates the base, trains SAC then PPO, reloads checkpoints for stochastic
and mean-action evaluations on the same 100 held-out reset seeds, and saves one
preselected video episode per controller (seed 100000, mean actions).

| Task argument | Default training decisions per algorithm |
|---|---:|
| `toy` | 2,048 |
| `weak_toy` | 8,192 |
| `pendulum` | 16,384 |
| `reacher` | 8,192 |
| `panda_reach` | 8,192 |
| `panda_push` | 8,192 |
| `panda_pick_place` | 8,192 |

Optional arguments: `--tasks pendulum`, `--algorithms sac`, `--steps 32768`,
`--seed 1`, `--episodes 100`, or `--no-video`. Run the service without arguments
to print help. Use a fresh output directory when changing settings. Completed
runs are skipped if rerunning the same plan; incomplete training directories
are preserved and require a fresh output path. The runner requires CUDA and
fails if it is unavailable.

Open the generated `runs/my_gpu_demos/report.html` locally in a browser.
`status.json` identifies the current stage; `metrics.jsonl` in each training
directory gives live progress. Videos are `base.mp4`, `sac.mp4`, and `ppo.mp4`
in each task directory. GPU telemetry includes other host activity; CUDA peak
allocation/reservation in each run summary measures the learner process more
directly. All generated outputs stay in the Git-ignored `runs/` directory and
are excluded from Docker's build context. GitHub Actions does not upload runs,
checkpoints, reports or videos.

Pendulum and Reacher have no native binary success metric. Compare return and
video; generic evaluator success fields are marked unavailable by the runner.
Panda success means reaching the environment's native goal criterion at least
once during the episode. Single training seeds and limited budgets are initial
diagnostics, not convergence evidence, tuned benchmark results, or an algorithm
ranking. The weaker toy uses `base_gain=0.3`. External benchmarks use a zero base,
one action per decision, and the full physical action range.

The real LIBERO reset/render/step check is separate:

```text
docker compose -f compose.gpu-tests.yaml run --rm --entrypoint python gpu-tests /opt/rl-vla/smoke.py --libero
```

OpenPI-driven LIBERO learning still needs model weights and sufficient inference
memory. The separate OpenPI server image and checkpoint inference have not been
fully validated. This suite does not establish VLA inference or robot
performance from the simulator smoke check.

Each job exits when it finishes, and `--rm` removes its container while keeping
host outputs. No idle container needs to remain running between experiments.
