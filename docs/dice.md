# DICE–OpenPI pi0.5 integration

This is a project-specific adaptation of **From Prior to Pro: Efficient Skill
Mastery via Distribution Contractive RL Finetuning** (Sun and Song, 2026).
Paper/project: https://zhanyisun.github.io/dice.rl.2026/
Official simulation source: https://github.com/zhanyisun/dice-rl
Inspected revision: `98d936e52523878f0172e606ea32403176497100`.
The MIT notice is retained in `docs/dice-upstream-LICENSE.txt`.

For a fresh PrismaX GPU host, use the [portable pipeline guide](dice-prismax.md).

## What is implemented

The existing JAX `pi05_libero` checkpoint remains frozen. A separate CPU PyTorch
MLP receives frozen visual/language/proprioceptive features and the **same full
Gaussian noise** used to generate a prior action. It predicts a residual in
normalized action coordinates. The model scores and corrects the five actions
actually executed from the ten-action OpenPI chunk, including the gripper.
Padded coordinates 7–31 remain unchanged. The checkpoint's existing output
transforms convert the corrected actions back to LIBERO controller units.

An independent double-Q critic learns from executed chunk transitions. The
actor maximizes minimum Q and penalizes deviation from the corresponding prior
sample. After warmup, that penalty is removed for a sample only when:

1. Its Q exceeds the Q of the prior action generated from the same noise.
2. The replay action's predicted Q minus its empirical discounted return is
   below the configured underestimation threshold.

Expert transitions always retain the prior penalty. The critic's calibration
signal is detached and computed before its update. Candidate selection supports
maximum minimum Q or maximum ensemble standard deviation. No SAC entropy term,
policy likelihood, flow-policy gradient, or VLA action-head update is used.

The runner collects complete episodes, computes finite-episode Monte Carlo
returns, mixes demonstration and online replay, recomputes frozen prior samples
for actor/target updates, updates target critics, and saves the residual/critic
weights and optimizer states separately from OpenPI. It runs one simulator
through Kai's existing environment IPC. It starts no websocket server and does
not connect to the baseline/robustness server.

## Differences from the published recipe

This is **not a validated reproduction of the paper**. The pi0.5 integration,
frozen pooled VLM features, executed-prefix action representation, finite-horizon
return objective, zero-initialized residual, small batch/particle counts, and
serial collector are project adaptations. The published task thresholds and
training schedules have not been tuned for LIBERO. A ten-step OpenPI prior is
still queried; the method does not make it a one-step policy or guarantee higher
throughput. Environment limits clip decoded commands; replay retains the
normalized command requested before clipping. Terminal chunks may execute fewer
than five commands. Both success and time limits end the finite-episode objective
and disable bootstrapping, which must also hold for demonstration returns.

The default configuration **requires demonstration replay**. An explicitly named
`online_only_ablation` data mode is supported but must not be reported as the
demonstration-assisted paper recipe. The local smoke uses one exported Spatial
task-0 demonstration; the default full-training replay path is still a template.
No arbitrary HDF5 layout, camera
orientation, action normalization, success label, or missing final observation
is guessed by the importer.

## Install and check without training

The adaptation is now included in `src/rl_vla/dice*.py`, with configs and its
launcher in this repository. The original external handoff installer is not
needed after cloning. First prepare the separate OpenPI/simulator environments
and the [pinned DICE reference checkout](../third_party/README.md), then use the
local server settings:

```bash
cd /workspace/upenn/RL-VLA
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  third_party/openpi/.venv/bin/python -m pytest tests/test_dice.py -q
bash scripts/run_dice.sh check
```

The tests use synthetic arrays and in-memory fixtures only. They do not load
pi0.5, open LIBERO, download assets, or touch the GPU. A numerical optimizer
step on a tiny synthetic network tests the objective; no experiment is trained.
`check` reads local paths and source revisions, reports whether demonstration
data is ready, and does not initialize JAX or open an environment.

To validate the actual checkpoint **without training**, run this once when the
shared GPU has room:

```bash
bash scripts/run_dice.sh validate-policy --output runs/dice_pi05_validation_01
```

This loads a separate frozen checkpoint, resets one simulator, performs one
prior query, checks finite shapes and exact zero-residual decode equivalence,
then exits. It performs **zero environment steps and zero optimizer updates**.
The output directory must be new. Expect normal model load/JIT startup latency;
this is not an action rollout test. Preserve `validation.json` for review.

## Demonstration data preparation

`rl_vla.dice_data` accepts a msgpack archive written with
`openpi_client.msgpack_numpy.packb`. Its schema is:

```python
{
    "format": "dice-libero-episodes-v1",
  "provenance": {
    "task": {"task_key": "explicit-key", "task_suite": "explicit-suite",
         "task_id": 0, "instruction": "exact verified prompt"},
    "environment": {"adapter": "explicit-adapter", "protocol_id": "explicit-id",
            "protocol_sha256": "verified-hash", "protocol_version": 2,
            "split": "train"},
    "demonstration": {"source_id": "explicit-dataset-revision",
              "source_split": "train"},
  },
    "episodes": [
        {
      "provenance": "same complete provenance object as archive",
            "observations": [obs_0, ..., obs_T],  # T+1, including the final state
            "actions": primitive_actions,       # float32 [T,7], controller units
            "rewards": primitive_rewards,       # float32 [T], verified task rewards
            "ending": "terminated",             # or "truncated" at the time limit
        },
    ],
}
```

Every observation uses the existing LIBERO adapter structure:
`{"openpi": {"observation/image": uint8_RGB_224,
"observation/wrist_image": uint8_RGB_224,
"observation/state": float32_8, "prompt": instruction}}`.
Images must already have the same 180-degree rotation and resize as online
observations. State is eef XYZ, axis-angle rotation, and two gripper positions.
Use task 0 of the configured suite, the correct task instruction, and the same
reward/success predicate and episode horizon. Verify these against the actual
dataset before exporting. Reconstructing a missing terminal observation needs
simulator replay, not copying the last pre-action frame.

Do not infer missing task, instruction, protocol, source, or split values. Both
the archive and each episode must contain complete provenance; prompts must
exactly match the declared instruction. The loader compares all provenance to
the selected DICE config and rejects mismatched or unverifiable replay before
sampling it.

The converter uses the **actual checkpoint input transforms** to normalize the
demonstration actions, chunks them using the configured execution horizon,
computes primitive-discounted chunk rewards and MC returns, and records
checkpoint/normalization fingerprints. It makes no RL updates. It rejects empty
episodes, invalid shapes, missing boundaries, non-finite values, and overwrites.

```bash
source scripts/env.sh
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
third_party/openpi/.venv/bin/python -m rl_vla.dice_data \
  --config /absolute/path/to/explicit-dice-config.toml \
  --input /absolute/path/to/verified_episodes.mpk \
  --output /absolute/path/to/normalized_replay.mpk \
  --task-key explicit-key --source-id explicit-dataset-revision --source-split train
```

The training loader validates task/instruction/environment/protocol/source
provenance as well as horizon, gamma, checkpoint and normalization fingerprints
before using this replay. Raw RGB observations are retained; KV caches are
recomputed rather than filling replay with GPU caches. Replay count and array
bytes are bounded.

## Training

After container validation and demonstration verification, the explicit entry
point is:

```bash
bash scripts/run_dice.sh train --output runs/dice_pi05_FIRST_RUN
```

This command starts training. None of the installation or checking commands
above invoke it. Choose a new output directory and review the task, data, budget,
and action selection strategy with the team first. The current Spatial baseline
is already 99%; an initial task-0 run is a pipeline experiment, not evidence that
DICE improves robustness. Use separate training and held-out evaluation cases.

`final.pt` and `latest.pt` contain residual/critic weights and optimizer state,
configuration, and prior fingerprint. `DICE.load_state_dict` restores the
learner. The runner now writes `resume.pt` with replay, optimizer state and RNGs for
episode-boundary continuation; see the portable guide. It does not save a
partially collected episode or simulator state. It also does not export
a native finetuned OpenPI checkpoint. The `evaluate` command applies the residual and selection rule. The separate
robustness launcher cannot consume these weights automatically.

## GPU host resource controls

The portable launcher preserves scheduler GPU assignments (defaults to GPU 0
when unset), disables JAX preallocation and sets one BLAS/Torch thread. CPU
affinity is optional via `DICE_CPUSET`. GPU checks query assigned devices, with
thresholds configured through `DICE_MIN_GPU_FREE_MB`, `DICE_MAX_GPU_TEMP`,
`DICE_MIN_RAM_MB` and `DICE_MIN_DISK_GB`. These checks are periodic observations,
not allocator limits. No other processes are stopped. See the portable guide.

## Verification status

The synthetic CPU tests cover gating, prior detachment, selection, variable-duration
returns, data normalization/chunking, numerical updates, serialization and an
in-memory collector. They run in the existing Python 3.11 environments; current
counts are recorded in [verification](verification.md).

On this server, `runs/dice_pi05_validation_01/validation.json` records successful
real-checkpoint sampling and zero-residual equivalence, with zero environment
steps and optimizer updates. The subsequent `runs/dice_pi05_smoke_01/` contains
two successful diagnostic episodes, 30 decisions, two optimizer updates, and
saved `latest.pt` / `final.pt` files, using a small single-demonstration export.
These local artifacts are ignored by Git. They establish a short working path,
not a comparative success rate, a complete training dataset or an improvement.
The shipped `dice_pi05_libero.toml` still requires its own verified replay file.

The portable DICE launcher and resource guard no longer use the prepared
server's absolute paths or CPU assignment. Historical validation artifacts
mentioned above belong to that server, not a fresh checkout.
