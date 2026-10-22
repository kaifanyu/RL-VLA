# Verification

## Repository review and GPU allocation (2026-10-21, America/New_York)

Live inspection confirmed one CUDA-visible GPU with 65,536 MiB VRAM, mapped from
host `/dev/nvidia9` to container device 0. Ten physical GPU records exist on the
host; the remaining cards' utilization and available VRAM are not visible here.
The robustness server used about 8.4 GiB and was the only GPU compute process at
inspection. Container RAM is limited to 96 GiB. See [local setup](local-baseline.md)
for the dated inventory, CPU/storage limits and device-selection details.

The publication review preserved the existing algorithm changes and corrected
the overview, environment descriptions, upstream pins and stale setup status.
DICE's reference checkout, the applied FlashSAC handoff patch and local caches
are ignored. Docker now copies `scripts/`, required by the robustness tests, and
retains the DICE license notice. Compose keeps its explicit one-GPU reservation
and no longer declares `NVIDIA_VISIBLE_DEVICES: all`.

| Check | Result |
|---|---|
| Root CPU test suite, including local FlashSAC source | 125 passed, 10 optional JAX/OpenPI checks skipped |
| RLDT and DICE tests in the separate OpenPI environment, with JAX forced to CPU | 54 passed |
| Source publication snapshot without local upstream checkouts or artifacts, using the existing CPU interpreter | 117 passed, 18 optional checks skipped |
| Ruff over `src`, `tests`, `scripts` | Passed |
| Launcher Bash syntax; YAML and TOML parsing | Passed |
| Relative Markdown file links in the publication snapshot | All resolve |
| Candidate files | About 3.8 MB total, including the supplied 2.5 MB RLDT paper; no bulk checkpoint/dataset/run files |
| Existing robustness evaluation's recorded source hashes | Unchanged |
| Remote `main` versus local HEAD at review | Same commit, `c0f925bd1800bb3d30f4e01ec23b917d0b86f900` |

Tests were limited to CPUs 4–7 with one-thread BLAS and CUDA hidden; no new model
load or GPU training was started. The source snapshot used already-installed
dependencies and does not validate a fresh dependency installation. Docker is
unavailable in this container: YAML parsing and copying required sources do not
establish a successful Docker build or Compose deployment. The GitHub container
workflow remains the build/runtime check. No commit or push was made by this review.

To repeat the local CPU checks from the repository root:

```bash
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  nice -n 10 taskset -c 4-7 .venv/bin/python -m pytest -q
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  nice -n 10 taskset -c 4-7 third_party/openpi/.venv/bin/python -m pytest \
  tests/test_rldt_math.py tests/test_rldt_pi05.py tests/test_rldt_pipeline.py tests/test_dice.py -q
.venv/bin/ruff check src tests scripts
```

Additional local DICE evidence predating this review:
`runs/dice_pi05_validation_01/validation.json` passed real-policy sampling with
zero simulator steps/updates; `runs/dice_pi05_smoke_01/metrics.jsonl` records two
successful episodes, 30 decisions and two optimizer updates, with saved final
weights. This is a small diagnostic using a single-demonstration replay export,
not a held-out improvement measurement. The FlashSAC run
`runs/libero_flashsac_s0/` has partial training metrics and no final checkpoint
or summary. The method guides distinguish these statuses from completed runs.

## RLDT and native π0.5 integration (2026-10-06)

Read all 16 pages of `rldt.pdf` and cross-checked the algorithm against the
authors' release at commit `3ab73e775efd0f041ad28170c616146be86bb6ea`.
The [RLDT guide](rldt.md) records paper/code differences, native OpenPI time
conversion, implementation choices, runnable commands, and pseudocode.

| Check | Observed result |
|---|---|
| RLDT CPU numerical/integration tests | 45 passed in the pinned OpenPI JAX environment |
| Existing root-environment suite | 116 passed, 3 skipped |
| Ruff and shell syntax | New RLDT modules/tests/parity script and launcher passed |
| CPU toy pipeline | 48 decisions, 190 primitive steps, 12 critic updates, 4 actor updates; save/reload and evaluation passed |
| Actual π0.5/LIBERO training | Full 430,098,464-parameter action expert; frozen VLM; K=2 and K=8 runs each completed 10 primitive steps, one critic update, one actor update, and checkpoint save |
| K=8 finite update | Actor gradient norm 0.0323868; optimizer-update norm 0.00901871; all logged metrics finite |
| Native checkpoint export | Compact trained expert exported with full backbone and normalization assets; upstream OpenPI policy factory loaded it successfully |
| GPU sampler comparison | Same noise and mixed-precision checkpoint: maximum absolute error 0.00173307 normalized and 0.00173256 decoded, within atol=rtol=0.025 |
| Saved-expert real evaluation | Reloaded K=8 expert completed one LIBERO Spatial task-0 episode successfully in 80 primitive steps, seed 100000; infrastructure validation only |

Evidence: `runs/rldt_toy_validation/`, `runs/rldt_pi05_smoke_v2/`,
`runs/rldt_pi05_k8_validation/`, `runs/rldt_pi05_native_load.json`, and
`runs/rldt_pi05_parity.json`, and `runs/rldt_pi05_k8_evaluation.json`.
These artifacts stay local under ignored `runs/`.
The K=8 run used `--smoke --particles 8`, with a reduced update budget and one
bootstrap sample. No episode completed during these short training checks, and
no improved success rate or reproduction of the paper's results is claimed.
The existing robustness server was preserved.

The first attempted GPU check was interrupted during excessive JAX compilation.
The adapter was fixed to pass frozen and trainable weights as explicit JIT
arguments, and a regression test now covers that boundary. The completed checks
above use the corrected implementation.

## Robustness setup (2026-09-29)

The original Spatial baseline completed **495/500 successes (99%)**, with no
infrastructure errors. A separate `.venv-robustness` environment isolates the
LIBERO-Plus and PRO forks. The 99-test repository suite passed, followed by eight
robustness tests after adding a capacity-wait regression test (100 tests total).
These cover restricted
initial-state decoding, complete queue scheduling, and separate partial-result
denominators. See the [robustness evaluation guide](robustness-evaluation.md)
for the 20,030-episode plan, locally generated PRO Environment limitation, and
saved real-simulator preflight evidence. All 15 simulator checks passed: one
case from each of the 12 Plus/PRO categories, plus Environment in PRO's other
three suites. The five Plus noise functions also passed at severities 1 and 10.
Initial camera images were inspected. Asset checksum verification completed;
workspace usage after installation was 115.6 GB against the 200 GB cap.
During the first model startup, an unrelated test process allocated about
49 GiB on the shared GPU. The monitor terminated only this queue's processes
when aggregate GPU memory crossed 56 GiB. After that test finished, the queue
waited for stable capacity and resumed. The event is preserved under
`runs/evaluation/pi05_libero_robustness_20260929/startup_events/`.
Preflight checks establish that tested
cases execute; they do not imply success on the full benchmark.

The persistent queue then completed its first full Plus Background Textures
episode (success, 82 primitive steps) and locally generated PRO Environment
episode (success, 145 steps), and advanced to Plus Camera Viewpoints. Checkpoint
fingerprint matches the completed Spatial baseline. The policy server used
about 8.3 GiB; warm inference was about 115 ms. This verifies rollout, video,
metrics, and queue transitions, not final benchmark accuracy. Live evidence is
saved in `runs/evaluation/pi05_libero_robustness_20260929/live_validation.json`.

## Local gpu9 setup and baseline (2026-09-28)

The local uv environment at `/workspace/upenn/RL-VLA/.venv` now passes all
92 tests, dependency checks and lint. The existing pinned OpenPI GPU environment
is separate. A real LIBERO reset, two-camera observation pipeline and controller
step pass with OSMesa and `NUMBA_DISABLE_JIT=1`; optional Numba compilation
segfaulted on this container, so its Python fallbacks are enabled. The evaluation
client also disables websocket keepalive pings during the first JAX compilation,
using an explicit inference timeout instead.

The real `pi05_libero` checkpoint completed five episodes on LIBERO Spatial task
0, official initial states 0–4, with zero residual actions:

| Measurement | Observed |
|---|---:|
| Successes | 5 / 5 |
| Wilson 95% interval | 56.55%–100% |
| Mean primitive controller steps | 83.6 |
| Median / p95 inference latency | 113.08 / 113.71 ms |
| First inference, including compilation | 72.76 s |
| Sampled peak GPU memory | 8.33 GiB |
| Sampled peak GPU temperature | 44 C |
| Sampled peak combined owned CPU use | 3.98 logical CPUs |

Both initial camera views and a successful rollout's final frame were inspected.
Success agreed with LIBERO termination and sparse reward. Results, videos,
checkpoint hashes and resource samples are saved in
`runs/evaluation/pi05_libero_smoke_20260928/`. This small test establishes that
the complete evaluation path works; it is not an estimate for all LIBERO tasks.

The larger Spatial run completed its plan of 10 tasks x 50 initial states with
495 successes. Final results are in
`runs/evaluation/pi05_libero_spatial_20260928/summary.json`; its model server exited
when evaluation finished. A separate emergency-stop
test used a disposable child process and a simulated threshold crossing; the
monitor correctly terminated the supplied child and wrote its stop reason.

See [local baseline instructions](local-baseline.md) for activation, launch,
resume, resource limits and comparison with future VLA checkpoints. Older
Windows/Docker verification below describes those environments at their original
dates; the local GPU validation does not establish a successful GPU Docker build.

## Earlier Windows verification

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
