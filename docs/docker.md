# Docker deployment

Run all commands from the repository root. The learner is a command-line training/evaluation job: use `docker compose run --rm` for each run. `docker compose up learner` only prints CLI help and exits. The optional OpenPI policy server is the long-running service.

Use Docker Engine with Compose v2 on Linux, or Docker Desktop with its WSL2 Linux-container backend on Windows. Hardware virtualization must be enabled and `docker info` must reach the engine. Images target `linux/amd64`. The CPU learner and headless simulator need no GPU. The CUDA demo image and the separate OpenPI service need an NVIDIA GPU accessible to Docker; OpenPI also needs sufficient memory for the chosen model. See [OpenPI hardware requirements](openpi.md#1-install-the-pinned-server-source).

## CPU learner: build, train, evaluate

```text
docker info
docker compose config --quiet
docker compose build learner
docker compose run --rm learner --help
docker compose run --rm learner train --config configs/sac_toy.toml --output runs/docker_sac
docker compose run --rm learner evaluate --config configs/sac_toy.toml --checkpoint runs/docker_sac/final.pt --episodes 20 --output runs/docker_sac_eval.json
docker compose run --rm learner train --config configs/ppo_toy.toml --output runs/docker_ppo
docker compose run --rm learner evaluate --config configs/ppo_toy.toml --checkpoint runs/docker_ppo/final.pt --episodes 20 --output runs/docker_ppo_eval.json
```

Outputs persist under the host's `runs/`. Choose a new output path for each experiment. Omit `--checkpoint` to evaluate the frozen base. Add `--steps 128` to a training command for a short diagnostic run. These toy tasks exercise the pipeline; they do not measure VLA or robot performance.

The default `docker build -t rl-vla:cpu .` also builds the CPU learner. The OpenPI client is included, so external task adapters can use a separately hosted policy server without installing the full OpenPI training environment.

## Tests and smoke checks

```text
docker compose --profile test build test
docker compose --profile test run --rm test
docker compose --profile test run --rm --entrypoint python test -m ruff check --no-cache src tests
docker compose run --rm --entrypoint python learner /opt/rl-vla/smoke.py
```

The test image includes the real OpenPI websocket client and serializer, allowing the websocket integration test to run against its local test server even when the host has no `third_party/openpi` checkout. The smoke script trains both SAC and PPO for 128 decisions, saves and reloads checkpoints, and evaluates two episodes per algorithm. Its temporary outputs are removed afterward.

## CUDA learner and sequential demos

The `gpu` target in the root Dockerfile builds a CUDA learner on top of the LIBERO simulator. It includes PyTorch 2.7.0+cu126, benchmark environments, tests and report scripts. Dependencies are installed from a hash-pinned GPU lock during the build, so recreating a container needs no manual package installation. Source and tests are packaged in the image, with no host source mounts or `third_party/` prerequisites.

Budget at least 30 GB of free space on Docker's storage drive for a first CUDA build. This is a conservative estimate including build cache and export copies, not a fixed requirement. The local packaging check installed all locked packages, passed `uv pip check` for 95 packages and passed the Torch 2.7.0+cu126/CUDA 12.6/import assertions. Image export/unpacking then filled `C:` and failed. The incomplete image and its 9.01 GB build cache were removed, Docker recovered, and saved runs remain intact. The completed CUDA image and runtime checks remain **unverified**; previous manually prepared GPU demos do not validate this image. See the [packaging verification record](verification.md#github-packaging-check-2026-09-19).

```text
docker compose -f compose.gpu-tests.yaml config --quiet
docker compose -f compose.gpu-tests.yaml build gpu-tests
docker compose -f compose.gpu-tests.yaml run --rm --entrypoint python gpu-tests -m pytest -q -p no:cacheprovider
docker compose -f compose.gpu-tests.yaml run --rm gpu-tests --output runs/my_gpu_demos
docker compose -f compose.gpu-tests.yaml run --rm --entrypoint python gpu-tests /opt/rl-vla/benchmark_report.py runs/my_gpu_demos
```

This builds `rl-vla:gpu` and executes one task/algorithm at a time. Add `--tasks pendulum --algorithms sac` to select a single demo. The runner requires CUDA; physics and headless rendering use the CPU. GPU access is provided through Compose's GPU reservation, using NVIDIA Container Toolkit on Linux or Docker Desktop's WSL2 GPU support on Windows. See [GPU demos](gpu-demos.md) for prerequisites, device checks, budgets and reporting.

Results stay in the host's `runs/` directory, which is ignored by Git and excluded from image builds. The CI workflow does not upload experiment artifacts. These numerical-state demos use small residual networks; they do not load an OpenPI checkpoint or validate the separate OpenPI server.

## Headless LIBERO and a remote policy server

First validate the simulator independently of model weights and a GPU:

```text
docker compose --profile libero build libero
docker compose --profile libero run --rm --entrypoint python libero /opt/rl-vla/smoke.py --libero
```

This check loads real LIBERO task assets, restores an official initial state, renders a nonblank RGB image, and steps the simulator. LIBERO paths are preconfigured in `/opt/libero-config/config.yaml`; no interactive path setup or demonstration dataset download is needed for online rollouts.

The `libero` image keeps the learner on CPU and uses Mesa's software renderer for MuJoCo (`MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`). This leaves GPU memory available for OpenPI and supports headless Docker Desktop without an X server. The separate `gpu` target above supplies CUDA PyTorch when GPU learning is desired.

Copy `.env.example` to `.env` if it does not already exist (`Copy-Item .env.example .env` in PowerShell, or `cp .env.example .env` in Bash), then set the reachable policy endpoint:

```dotenv
RL_VLA_OPENPI_HOST=your-gpu-server
RL_VLA_OPENPI_PORT=8000
```

Use `host.docker.internal` for a server running on the Docker host. `localhost` inside the learner refers to the learner container. A host server must listen on an interface reachable from the container; a host-only loopback listener may not be reachable through this gateway. A remote server likewise needs a reachable listening address and port.

Once the matching `pi05_libero` policy server is ready:

```text
docker compose --profile libero run --rm libero evaluate --config configs/libero_sac.toml --episodes 5 --output runs/docker_libero_base.json
docker compose --profile libero run --rm libero train --config configs/libero_sac.toml --output runs/docker_libero_sac
docker compose --profile libero run --rm libero evaluate --config configs/libero_sac.toml --checkpoint runs/docker_libero_sac/final.pt --episodes 5 --output runs/docker_libero_sac_eval.json
docker compose --profile libero run --rm libero train --config configs/libero_ppo.toml --output runs/docker_libero_ppo
```

Both endpoint variables override the TOML only when `base.kind = "openpi"`. Host whitespace is stripped and empty values are rejected; the port must be an integer from 1 through 65535. Unset variables preserve TOML values in direct CLI use. Compose supplies `openpi:8000` as both learner services' default endpoint. Toy/mock policies ignore the overrides. Read [LIBERO's task and observation contract](libero.md) before changing tasks or checkpoints.

## Optional local GPU policy service

The OpenPI server image uses its own environment and builds from pinned upstream source in `docker/openpi.Dockerfile`. It is separate from the CUDA learner image. Neither a local source checkout nor a host Python installation is required. First download the **complete checkpoint** as described in [OpenPI checkpoint setup](openpi.md#2-download-the-checkpoint-when-ready), placing it at:

```text
checkpoints/openpi/pi05_libero/
  params/                    # JAX weights, or model.safetensors for PyTorch
  assets/.../norm_stats.json
```

The service mounts `checkpoints/openpi` read-only at `/checkpoints`. `OPENPI_CHECKPOINT_DIR` must name a directory inside that container mount; it defaults to `/checkpoints/pi05_libero`. The entrypoint requires local mounted weights and normalization assets, rejects download URIs, and checks GPU access before starting inference. Checkpoints are excluded from the build context.

For a server in the same Compose project, keep these `.env` values:

```dotenv
RL_VLA_OPENPI_HOST=openpi
RL_VLA_OPENPI_PORT=8000
OPENPI_MODEL_CONFIG=pi05_libero
OPENPI_CHECKPOINT_DIR=/checkpoints/pi05_libero
OPENPI_BIND_ADDRESS=127.0.0.1
OPENPI_PORT=8000
```

```text
docker compose --profile gpu build openpi
docker compose --profile gpu up -d --wait --wait-timeout 900 openpi
docker compose --profile gpu ps
docker compose --profile gpu logs --tail 100 openpi
```

Readiness probes the upstream `/healthz` endpoint after the checkpoint loads; GPU initialization and model loading can take several minutes. Run the LIBERO training/evaluation commands above only after the service becomes healthy. A standalone GPU diagnostic after building is `docker compose --profile gpu run --rm openpi nvidia-smi`.

The policy listens on port **8000 inside Docker**. `OPENPI_PORT` in `.env` changes only the host-published port; it does not change the service-to-service endpoint `openpi:8000`. `OPENPI_BIND_ADDRESS=127.0.0.1` makes that published port available only through the host's loopback interface. To serve a different machine, set an appropriate reachable host bind address and restrict access to the intended clients; configure that remote client's `RL_VLA_OPENPI_PORT` to the published port.

Full OpenPI server image build and real checkpoint inference have **not yet been verified**. The setup machine has an 8 GB GPU, while the selected model's documented inference requirement exceeds 8 GB. Weights and the server build also need additional disk space. No checkpoint was downloaded. GPU passthrough (`nvidia-smi` inside Docker), Dockerfile checks and an upstream frozen-lock installation dry run passed during setup. These checks do not establish model inference; see the [verification record](verification.md).

## Dependencies and build inputs

| Component | Pinned deployment baseline |
|---|---|
| Learner runtime | Python 3.11, Debian Bookworm slim; base image pinned by digest in `Dockerfile` |
| Installer | uv 0.11.24; image pinned by digest in both Dockerfiles |
| CPU learner | Torch 2.7.0+cpu, NumPy 1.26.4, Gymnasium 1.3.0 |
| CUDA learner/demos | Torch 2.7.0+cu126, NumPy 1.26.4, Gymnasium 1.3.0, panda-gym 3.0.7, PyBullet 3.2.7 |
| OpenPI/client source | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LIBERO source/assets | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| Simulator | robosuite 1.4.1, MuJoCo 3.2.3, Numba 0.61.2, SciPy 1.15.3 |
| OpenPI GPU server | CUDA 12.6.3 runtime on Ubuntu 22.04, Python 3.11.13, frozen upstream `uv.lock` |
| Server import checks | JAX 0.5.3, Torch 2.7.1, Transformers 4.53.2 with upstream replacement files |

CPU learner/test/LIBERO and CUDA learner dependency locks include hashes, and builds run `uv pip check`. The simulator installs only its online-rollout dependencies; it does not install LIBERO's incompatible legacy training requirements. Source checkouts are fetched and commit-verified during builds, with LIBERO assets retained inside the image. Host virtual environments, credentials, runs, datasets, and checkpoints are excluded by the build-context allow-list.

LIBERO uses setuptools' compatibility editable mode to expose its namespace package while retaining the complete, root-owned asset tree. The simulator lock explicitly includes `future`, an undeclared BDDL runtime dependency. The build imports the real environment as its non-root runtime user to catch packaging and missing-import failures.

The CPU image installs `ca-certificates` and `libgomp1`. The LIBERO image adds OSMesa, GL/EGL/GLES, GLFW, GLEW, GLib and X11 support libraries; temporary compiler packages are removed after dependency installation. The CUDA learner inherits those libraries and gets CUDA user-space dependencies from its PyTorch lock. The separate OpenPI server runtime includes FFmpeg, GL/EGL, GLib, OpenMP and X11 support; compilers, Git and Git LFS stay in its builder stage. See the Dockerfiles for exact OS package names. Base images are digest-pinned, but apt packages still come from the distribution's current repositories.

To refresh dependencies, edit the corresponding `.in` file and use uv 0.11.24. Compile the CPU lock first, then preserve it as a constraint for the test and LIBERO environments. The GPU lock is independent so it can select CUDA PyTorch:

```text
uv pip compile docker/requirements-cpu.in --python-version 3.11 --python-platform x86_64-unknown-linux-gnu --torch-backend cpu --generate-hashes --output-file docker/requirements-cpu.lock
uv pip compile docker/requirements-test.in --constraint docker/requirements-cpu.lock --python-version 3.11 --python-platform x86_64-unknown-linux-gnu --torch-backend cpu --generate-hashes --output-file docker/requirements-test.lock
uv pip compile docker/requirements-libero.in --python-version 3.11 --python-platform x86_64-unknown-linux-gnu --torch-backend cpu --generate-hashes --emit-index-url --output-file docker/requirements-libero.lock
uv pip compile docker/requirements-gpu.in --python-version 3.11 --python-platform x86_64-unknown-linux-gnu --torch-backend cu126 --default-index https://pypi.org/simple --generate-hashes --emit-index-url --output-file docker/requirements-gpu.lock
```

The test and LIBERO `.in` files constrain the CPU lock; keep shared simulator versions aligned with the GPU `.in` file when updating them. Review lock changes, rebuild affected images, and rerun tests plus the toy and real simulator smoke checks. Validate CUDA availability and a short demo on a GPU host after GPU dependency changes. Updating the OpenPI source pin also requires reviewing its frozen lock, client compatibility, replacement files and version assertions; it is a separate server dependency update. The `Container checks` GitHub Actions workflow validates both Compose files and Dockerfile targets, then repeats CPU tests and both smoke checks on pushes and pull requests. It does not build the large CUDA image or run GPU training/inference on the standard runner, and it uploads no experiment artifacts.

## Volumes, ownership and cleanup

| Host path / volume | Container path | Access |
|---|---|---|
| `runs/` | `/app/runs` | Learner read/write |
| `configs/` | `/app/configs` | Learner read-only |
| `checkpoints/` | `/app/checkpoints` | Learner read-only |
| `datasets/` | `/app/datasets` | Learner read-only |
| `checkpoints/openpi/` | `/checkpoints` | GPU server read-only |
| `openpi-cache` named volume | `/home/app/.cache` | GPU server read/write |

Containers run as a non-root user. Docker Desktop users can keep UID/GID 1000. On Linux, create the host directories as your own user before running Compose (`mkdir -p runs configs checkpoints/openpi datasets`), then set `RL_VLA_UID` and `RL_VLA_GID` in `.env` to the output of `id -u` and `id -g`. Ensure existing `runs/` is writable by that user. The GPU server retains UID/GID 1000; checkpoint files must be readable by it. LIBERO's Numba cache uses writable `/tmp/numba` when learner IDs are overridden.

Profiles in `compose.yaml` enable optional services: `test`, `libero`, and `gpu` (the OpenPI server). The default profile contains only the CPU learner, so it does not download a model or start a GPU service. The separate `compose.gpu-tests.yaml` runs the CUDA learner as one-off jobs. Stop the main project with:

```text
docker compose --profile gpu --profile libero --profile test down
```

This retains host outputs, model weights and the named policy cache. Omit `-v` when retaining caches. One-off jobs use `--rm` so their stopped containers do not accumulate.
