# Queued π0.5 robustness evaluation

The run `runs/evaluation/pi05_libero_robustness_20260929` evaluates the existing
`checkpoints/openpi/pi05_libero` policy with zero residual actions. It alternates
between benchmark categories, executes one simulator episode at a time, and uses
one persistent OpenPI server on the container's assigned GPU. This is inference,
not training; the demonstration dataset is not required by the rollout loop.

| Benchmark | Coverage | Planned episodes |
| --- | --- | ---: |
| [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) | All four suites, seven perturbation categories, one rollout per published variant | 10,030 |
| [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) published cases | Object, Position, Language, Task; four suites × ten tasks × 50 states | 8,000 |
| PRO Environment, locally generated | Four suites × ten tasks × 50 saved, seeded states | 2,000 |

Plus covers background textures, camera viewpoints, robot initial states,
language instructions, lighting, object layouts, and sensor noise. PRO includes
changed task goals and success predicates. These are broad robustness checks,
but they do not exhaust all possible LIBERO perturbations, combinations, or task
generalization settings. The available checkpoint's complete pretraining data
is not known, so these tests do not establish universal training-set novelty.

The completed original Spatial baseline is 495/500 successes (99%). Original
Object, Goal, and Long baselines are **not included in this robustness queue**
and have not yet been measured here. Retain those separate original-suite
baselines when comparing a later fine-tuned policy.

## Protocol and limitations

`protocol.json` freezes every case, upstream source commit, dataset revision,
BDDL and state-file checksums, rollout horizon, and category. The Plus asset
archive is SHA-256 verified; extraction checks every member's CRC. The PRO
download verifies each file against the pinned Hugging Face metadata. Source
forks live separately under `third_party/libero_plus` and `third_party/libero_pro`.
Each simulator subprocess imports exactly one fork in `.venv-robustness`.
The original baseline environment remains in `.venv`.

The published PRO dataset does not contain the `_env` BDDL/state directories
referenced by its source task map. These cases are explicitly **locally generated**
using the pinned upstream `EnvironmentReplacePerturbator`. That implementation
hardcodes `living_room_table`; five Long tasks already use that table and therefore
serve as unchanged controls. Initial states use the sampling point in upstream
`notebooks/generate_init_states.py`: a freshly constructed environment before its
first explicit reset. They are seeded and saved under `generated_states/` for
the later policy comparison. These results must not be presented as a direct
reproduction using a published PRO Environment state release.

The worker reads the actual BDDL instruction rather than deriving it from a
filename; this preserves PRO's language changes and avoids giving Plus camera
parameters to the policy. It loads the fork's native success detector. Images
use the same baseline transforms (256-pixel rendering, 180-degree rotation,
224-pixel resize); the model predicts ten actions and executes five per request.
The suite horizons are Spatial 220, Object 280, Goal 300, and Long 520 primitive
steps, after ten settling actions. Episode seeds and diffusion-noise seeds are
deterministic and independent of queue order. An invalid reset or infrastructure
error stops the queue and is recorded as an error, never as a policy failure.

`summary.json` reports separate benchmark/category/suite success counts, rates,
and Wilson intervals. Incomplete results are labelled partial. Plus and PRO are
not combined into one headline score because their sampling weights differ.
Each `episodes/*.json` retains the task, prompt, seed, initial-state index,
success, return, steps, inference latencies, and wall time. Selected videos and
initial observations are retained, with one video per benchmark/category/suite.
For publication, report per-category results and the exact protocol; raw pooled
counts are not automatically identical to the papers' aggregation choices.

## Resource controls

- CPU affinity 0–3, nice level 10, low I/O priority; BLAS/OpenMP/Numba/Mesa each
  restricted to one thread. All children inherit the four-CPU affinity.
- One simulator and one policy server, with a 0.25-second delay per inference
  request and a two-second gap between episodes.
- OSMesa rendering and `NUMBA_DISABLE_JIT=1`, retaining the workaround validated
  for this server's Mesa/LLVM interaction.
- GPU memory preallocation disabled. Pause requests at 48 GiB GPU usage,
  78°C, or less than 8 GiB available RAM. Stop after five minutes of pressure.
- Independent telemetry every five seconds; stop owned processes at 56 GiB
  GPU usage, 83°C, or less than 4 GiB available RAM. Other users' processes are
  never managed by this runner.
- Check workspace usage at startup and hourly; stop at 190 GB apparent usage
  or less than 10 GB filesystem free space, leaving room below the 200 GB cap.
- Stop a worker after 30 minutes without a progress heartbeat. Stop if the
  policy server or resource monitor exits.
- Before model startup, wait until GPU use is below 32 GiB, temperature below
  70°C, and available RAM above 24 GiB for 30 seconds. After an emergency resource
  stop, the outer queue waits and retries the same saved plan. Past stop events
  are retained under `startup_events/`; simulator and code errors do not trigger
  automatic retries.

Brief utilization spikes during compilation/inference are expected. Affinity
limits sustained CPU consumption to four of the host's 88 logical CPUs. The
telemetry's GPU values cover the whole visible GPU, including other projects;
the CPU/RSS process list covers this queue's owned processes. The
single-GPU queue may take days to weeks; early timing estimates should not be
treated as final because slow sensor corruptions and Long failures cost more.

## Inspect, stop, and resume

From `/workspace/upenn/RL-VLA`:

```bash
source scripts/env.sh
python scripts/baseline_status.py runs/evaluation/pi05_libero_robustness_20260929
tail -f runs/evaluation/pi05_libero_robustness_20260929/launcher.log
tmux -L rl-vla attach -t robustness
```

Detach with Ctrl-b then d. To stop the queue cleanly, send SIGTERM to the PID in
`launcher.pid`; the supervisor shuts down only its own worker, server, and
monitor. Do not kill unrelated GPU jobs. To resume after addressing the error:

```bash
tmux -L rl-vla new-session -d -s robustness \
  'cd /workspace/upenn/RL-VLA && bash scripts/run_robustness.sh runs/evaluation/pi05_libero_robustness_20260929 >> runs/evaluation/pi05_libero_robustness_20260929/launcher.log 2>&1'
```

Only one queue can hold the lock. Existing completed episodes are skipped;
the protocol, evaluation code, and checkpoint fingerprint must match. If code
or protocol changes are necessary, start a new output directory and preserve
the old results. Inspect `queue.json`, `heartbeat.json`, `telemetry.jsonl`,
`storage.json`, `error.json`, and `resource_stop.json` for progress and failures.
`queue.json` is the current state.
The outer queue moves prior stop/error files into `startup_events/` when starting
another attempt. It records capacity waits in `resource_wait.json`.

After verified assets are installed, the reproducible plan builder is:

```bash
source scripts/env.sh
nice -n 10 taskset -c 0-3 .venv-robustness/bin/python scripts/prepare_robustness.py \
  --output runs/evaluation/NEW_RUN
```

Use a separate result directory for the fine-tuned policy and the same frozen
cases, initial states, prompts, simulator versions, transforms, horizons, and
seed protocol. The current launcher explicitly serves `pi05_libero`; configure
and verify a future checkpoint before making a before/after comparison.
