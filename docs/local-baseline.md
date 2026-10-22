# Local server, GPU access, and baseline evaluation

## Server snapshot: 2026-10-21 (America/New_York)

The host exposes records for ten physical NVIDIA GPUs, but this container can
use **one GPU with 64 GiB VRAM**. Host `/dev/nvidia9` maps to CUDA device `0`
inside the container, at PCI address `0000:0f:00.0`. The driver identifies it as
`NVIDIA Graphics Device`; the other cards' available memory and occupancy cannot
be established from this container. Device visibility is not an exclusive
reservation of the GPU's capacity.

At inspection, the robustness server was the only GPU compute process and used
about 8.4 GiB. The host has two Xeon E5-2699 v4 CPUs (44 physical cores, 88 logical
threads); all threads are visible but shared. Container RAM is capped at 96 GiB.
The local workspace used about 141 GB against the requested 200 GB soft cap.
These are dated measurements, not permanent free-capacity guarantees.

Check the current assignment and load without loading a model:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/cpuset.cpus.effective
```

`compose.yaml` requests one GPU for a newly created OpenPI service. It does not
manage this pre-existing container. Changing `CUDA_VISIBLE_DEVICES` or editing
Compose cannot expose additional host devices; the administrator must assign
them. Docker is not installed in this container, so Compose runtime validation
must be performed on a Docker host. See [Docker GPU deployment](docker.md#optional-local-gpu-policy-service).

For four independent RL experiments, a planning allocation is six GPUs total:
four training slots, one evaluator and one development slot. Confirm per-card
VRAM, CPU/RAM allocation and storage before increasing concurrency. RLDT has
passed a single-GPU eight-particle smoke test, not a sustained full-budget memory
profile. Each saved expert plus learner state uses about 8.1 GiB of storage.
Its launcher supports `RLDT_CPUSET` and `CUDA_VISIBLE_DEVICES`; give parallel jobs
distinct assigned GPUs, CPU sets and output directories. The baseline and DICE
wrappers currently select device 0 and CPUs 0–3, so they need explicit allocation
support before being used as a multi-GPU scheduler.

## Local environments

The setup in `/workspace/upenn/RL-VLA` uses three isolated Python 3.11 environments:

| Environment | Role |
|---|---|
| `.venv` | CPU PyTorch learner, LIBERO / Robosuite / MuJoCo simulator, tests and evaluation |
| `third_party/openpi/.venv` | CUDA JAX OpenPI inference and RLDT; DICE combines the frozen JAX prior with its PyTorch learner |
| `.venv-robustness` | Separate simulator environment for the pinned LIBERO-Plus and PRO forks |

Activate the local settings from Bash:

```bash
cd /workspace/upenn/RL-VLA
source scripts/env.sh
python -m pytest -q
```

`scripts/env.sh` selects the local interpreter, uv cache, dataset directory,
headless CPU renderer, and thread limits. `NUMBA_DISABLE_JIT=1` avoids a native
Numba compiler crash during Robosuite initialization on this container. MuJoCo
physics and OSMesa rendering remain native. A real simulator reset, image check
and controller step passed with these settings. Do not remove the workaround
without repeating `python docker/smoke.py --libero`.

The learner installation used `docker/requirements-libero.lock` with CPU Torch
and `docker/requirements-test.lock`, plus editable installs of this repository,
`third_party/openpi/packages/openpi-client` and the pinned LIBERO source. Video
recording adds `imageio==2.37.0` and `imageio-ffmpeg==0.6.0`. Exact installed
package lists and verification logs are in `runs/setup_20260928/`.

## Run a baseline

The launcher starts an owned OpenPI server on localhost port 8001, waits for it,
runs one simulator sequentially, and stops its own server and monitor afterward.
It uses the local `checkpoints/openpi/pi05_libero` JAX checkpoint and its bundled
normalization statistics. This is the LIBERO-adapted pi0.5 checkpoint, not the
generic pretrained pi0.5 base model.

```bash
# Quick end-to-end check on the configured reference task.
bash scripts/run_baseline.sh runs/evaluation/my_smoke \
  --task-ids 0 --episodes-per-task 5

# Reference task: every one of its 50 official initial states.
bash scripts/run_baseline.sh runs/evaluation/my_task_baseline \
  --task-ids 0 --episodes-per-task 50

# LIBERO Spatial: 10 tasks x 50 initial states.
bash scripts/run_baseline.sh runs/evaluation/my_spatial_baseline

# Resume an interrupted run with exactly the same options and checkpoint.
bash scripts/run_baseline.sh runs/evaluation/my_spatial_baseline --resume

# Inspect progress and resource peaks while a run is active or after it ends.
python scripts/baseline_status.py runs/evaluation/my_spatial_baseline
```

Use a new output directory for a new experiment. To keep a run alive after the
terminal closes, run the launcher in `tmux`. The monitor remains a child of the
launcher. Stop the launcher with
SIGTERM to clean up its owned server, evaluator and monitor. Do not kill unrelated
Python processes. The launcher refuses an occupied port or less than 32 GiB of
free GPU memory.

The 2026-09-28 Spatial baseline **completed with 495/500 successes (99%)**.
Its output directory is `runs/evaluation/pi05_libero_spatial_20260928`; its former
`pi05-baseline` tmux session closed on completion. The separate robustness queue
uses `tmux -L rl-vla`, session `robustness`, and writes to
`runs/evaluation/pi05_libero_robustness_20260929`. Follow the
[robustness guide](robustness-evaluation.md) for status and resume commands.
These output directories are local and ignored by Git.

## Evaluation protocol and saved results

- Frozen VLA with zero residual; controller actions are projected into the
  simulator's supported bounds. Prediction horizon 10, execute 5 actions before
  replanning, matching the repository's residual-training interface.
- Each task uses official initial state IDs in order, with a fixed per-episode
  environment seed and fixed inference-noise seed per policy call. Resume and
  task order do not advance a shared random stream.
- Spatial tasks have a 220-controller-step horizon, preceded by 10 settling
  steps. Success comes from LIBERO's task predicate, with sparse reward 1 on
  success. The adapter distinguishes success from timeouts.
- Both camera streams are rotated and resized by the existing adapter. First
  observations are saved per task; default videos cover initial states 0 and 1.
- `protocol.json`: exact episode plan, seeds, action chunk, model fingerprint
  and protocol settings. `manifest.json` and `server_manifest.json`: installed
  versions, configuration, source hashes, checkpoint file hashes and server
  provenance.
- `episodes/*.json`: individual outcomes, returns, primitive steps, clipping,
  inference timings and reset metadata, written after each episode.
- `summary.json`: completed/planned counts, overall and per-task success rates,
  Wilson 95% intervals, return, successful-episode steps, and inference latency.
  A partial run counts only completed episodes; a simulator/server error is
  recorded separately in `error.json`, not scored as policy failure.
- `telemetry.jsonl` / `resources.jsonl`: resource samples. `server.log` and
  `evaluation.log`: progress and diagnostic logs. `observations/` and `videos/`:
  visual checks.

Inference latency includes local transport and the first compilation; p50/p95
are reported separately. Episode elapsed time includes reset, rendering and the
deliberate request pause. It is not an unthrottled hardware speed benchmark.
This is a repository-protocol baseline, not a claim to reproduce a paper's
aggregate score or confidence interval assumptions.

For a later fine-tuned **OpenPI JAX** checkpoint, use the same command and episode
options with a new output directory:

```bash
RL_VLA_MODEL_DIR=/absolute/path/to/finetuned/checkpoint \
RL_VLA_MODEL_CONFIG=pi05_libero \
  bash scripts/run_baseline.sh runs/evaluation/my_finetuned_spatial
```

Keep the model/config pair compatible, including normalization assets. Pair
before/after outcomes by `episode_id`; compare success rate, failures repaired
and successes lost, and steps among successful episodes. The checkpoint hashes
will differ, while the episode plan and control protocol should match. The new
runner evaluates VLA weights; residual SAC/PPO `.pt` weights still use the
existing `rl_vla evaluate --checkpoint ...` command and need the same explicit
episode schedule added before claiming a paired comparison.

## Resource limits

Server, evaluator and monitor share CPU affinity 0–3: at most four of the host's
88 logical CPUs can execute this work concurrently. Nice priority 10, low I/O
priority and one-thread BLAS / Torch / software-rendering settings keep it modest.
Only container CUDA device 0 is selected; the other host GPUs are not used.

JAX preallocation is disabled and its allocator fraction is 0.35. This is not a
hard device-wide memory quota. The evaluator pauses between requests if total
GPU memory reaches 48 GiB, GPU temperature reaches 78 C, or available RAM falls
to 8 GiB; persistent pressure aborts the run. An independent five-second monitor
stops the owned processes at 56 GiB GPU memory, 83 C, or less than 4 GiB available
RAM. These checks cannot impose
an instantaneous GPU-utilization cap; short utilization spikes are expected.
The default 0.1-second pause between calls and sequential simulation also leave
gaps between GPU requests. No driver, power, clock or other user's process is
changed.

The verified demonstration dataset is shared at
`/workspace/upenn/datasets/physical-intelligence/libero`; online simulator
evaluation does not read it or make another copy. Keep all of `/workspace`
below the requested 200 GB persistent-storage soft cap.
