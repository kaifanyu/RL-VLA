# Verification

Validated locally on Windows with Python 3.11.15, PyTorch 2.7.0 CPU, NumPy 1.26.4 and Gymnasium 1.3.0. Full dependency versions are in `requirements-tested.txt`; `uv pip check` reports no conflicts.

The 56-test suite covers residual probability calculations and saturation, SAC target masking, variable chunk durations, PPO likelihood recomputation, GAE boundaries, real parameter updates, checkpoint/optimizer restoration, warm-start setting checks, CLI use, external adapters, camera/state mapping, and OpenPI action validation. It also tests the actual upstream websocket client and NumPy serialization against a local test server.

End-to-end runs saved under `runs/`:

| Run | RL decisions | Controller commands | Update calls |
|---|---:|---:|---:|
| `validation_sac/` | 2000 | 9504 | 1937 SAC steps |
| `validation_ppo/` | 2048 | 9740 | 8 PPO rollout updates |

Their `final.pt` checkpoints were reloaded for 20-episode stochastic evaluation on seeds 100000-100019. The mock frozen base succeeded in 20/20 episodes, SAC in 19/20 and PPO in 20/20. These exercise training and evaluation plumbing; they do **not** establish improvement or rank algorithms. The toy base is already competent. Per-episode results and confidence intervals are in `runs/validation_*_eval.json`.

## Docker verification (2026-09-19)

Docker Desktop's Linux engine and GPU passthrough were verified after hardware virtualization was enabled on the Windows host. Images target Linux x86_64; the digest-pinned learner base supplied Python 3.11.16. Torch 2.7.0+cpu, NumPy 1.26.4 and Gymnasium 1.3.0 match the project's runtime contract. Container dependency locks live in `docker/`; the Windows baseline above remains recorded separately in `requirements-tested.txt`.

| Check | Result |
|---|---|
| Build `learner`, `test`, `libero` from pinned upstream sources | Passed; images available as `rl-vla:cpu`, `rl-vla:test`, `rl-vla:libero` |
| Container `uv pip check` | Passed for learner, tests and simulator |
| Full container test suite | 72 passed, including the actual OpenPI websocket client and 16 endpoint-override cases |
| SAC and PPO smoke training | Each completed 128 decisions, saved a checkpoint and reloaded it for two evaluation episodes |
| Windows bind-mounted output | SAC trained to `runs/docker_validation_sac_20260919/`; a separate container reloaded its checkpoint and wrote `runs/docker_validation_sac_20260919_eval.json` |
| Real LIBERO/MuJoCo smoke check | Spatial task 0, initial-state ID 0, seed 7: reset, ten settling actions, nonblank 224x224 RGB rendering and one controller step passed |
| Docker GPU visibility | `nvidia-smi` inside the container detected the RTX 4060, 8188 MiB |
| Both Dockerfile checks and all Compose profiles | Passed |
| Application/test lint inside Linux container | Passed |
| OpenPI frozen-lock installation dry run | Passed under Linux/Python 3.11; 202 packages planned |
| GPU server helper checks | Syntax/lint, checkpoint preflight, diagnostic exec and local HTTP health probe passed |

The real simulator check uses OSMesa software rendering as the non-root runtime user. It validates one task/reset/step, not all LIBERO suites or long-run training. Installation fixes include setuptools compatibility editable mode for LIBERO's namespace package, BDDL's undeclared `future` dependency, and compiler headers for `evdev`.

No OpenPI weights or demonstration datasets were downloaded. The GPU server Dockerfile and health/startup configuration are supplied, but its **full image build, heavyweight import checks during that build, and real checkpoint inference remain unverified**. The local 8 GB GPU falls below upstream's stated inference memory requirement; available disk space is also insufficient for the heavyweight build plus checkpoint. Build and validate that service on a suitable GPU host before relying on VLA training. Robot learning, MuJoCo Warp throughput and visual robustness remain unmeasured.

Recheck container behavior using the [Docker guide](docker.md); the `Container checks` CI workflow runs the CPU tests, both learner smoke checks and real simulator check. Manual root-environment checks remain `python -m pytest -q` and `ruff check src tests`. GPU/server setup instructions are in [openpi.md](openpi.md).

## GitHub packaging check (2026-09-19)

The Docker configuration now builds CUDA dependencies into a dedicated `gpu`
target instead of relying on packages installed into a persistent container.
The GitHub workflow validates both Compose files and the Dockerfile targets,
retains CPU and LIBERO checks, and uploads no experiment artifacts.

| Check | Result |
|---|---|
| Test-image build using only publishable repository files | Passed; no host environment, ignored source checkout or run artifacts required |
| Git candidate-file audit | Passed; runs, checkpoints, datasets and local environment artifacts excluded |
| Dockerfile and Compose checks | Passed, including the new GPU target and GPU demo Compose file |
| Refreshed CPU test suite | 89 passed in 20.64 seconds; container exited with status 0 after Docker recovery |
| CPU learner smoke checks and lint | SAC and PPO each trained 128 decisions, reloaded checkpoints and evaluated two episodes; container lint passed |
| GPU dependency installation | All locked packages installed; `uv pip check` passed for 95 installed packages |
| GPU build import checks | Torch 2.7.0+cu126, CUDA 12.6 and benchmark imports passed |
| GPU image export/unpacking | Failed when Docker exhausted free space on `C:` |
| Completed GPU image and runtime checks | Unverified; requires rebuilding with sufficient disk space |

Docker recovered, and the incomplete GPU image and its 9.01 GB build cache
were removed. Saved runs remain intact and are excluded from Git and the Docker
build context. For a first CUDA build, budget at least 30 GB free on Docker's
storage drive as a conservative estimate including cache and export copies;
actual usage varies. Earlier GPU demos ran in a manually prepared container and
do not validate this new image. The separate OpenPI server and checkpoint
inference limitations above still apply.
