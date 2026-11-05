# Object-0 online pilot

This directory records the seed-0 comparison of RLDT, FlashSAC, and DICE on
LIBERO Object task 0, PRO Position/swap: **pick the alphabet soup and place it
in the basket**. All methods start from the same `pi05_libero` checkpoint.

RLDT was launched on GPU 0 on November 5, 2026, at 5:21 PM America/New_York,
from source commit `fde81e0e5327b246a609329e883cefebd3616fae`. The archived
[launch confirmation](evidence/launch-confirmed.json) records its first training
window, not its final performance. Consult the live attempt for current status.
The publishable snapshot preserves that learner and all nine method/seed configs.

## Protocol and configuration

| Setting | Primary protocol |
|---|---|
| Training | Fresh, uninterrupted, online only; exactly 16,000 controller commands |
| Seeds | Seed 0 is executable; seed 1/2 configs are prepared but launchers reject them |
| Milestones | 0, 4,000, 8,000, 12,000, 16,000 actual controller commands |
| Validation | 50 ordered episodes per milestone; K=1 |
| Selection | Highest validation success count; ties choose earliest, including initialization |
| Test | Selected policy and initial policy on the same 50 test states, after selection |
| Optional work | Retention and DICE K=16 evaluations disabled by default |

Configs are in [`configs/online_object0/`](../../configs/online_object0/).
[Resolved configurations](evidence/resolved-seed-configs.json) preserve the
effective settings used in the original readiness review. They are historical
host-specific records; paths resolved on another checkout will differ.

RLDT collects 32 decisions per window, executes at most five commands per
decision, uses critic batch 64, actor batch 1, and eight particles. Each window
has 64 critic updates and, after two critic-only warmup windows, four actor
updates. The learning rates, losses, and replay settings match the prior
Object-0 run. Command milestones are saved after the command and before the
current chunk enters replay and subsequent updates. They are not interchangeable
with the earlier post-update iteration-25/50/100 checkpoints.

FlashSAC uses batch 256, a 2,000-command warmup, and a residual at every controller
command. DICE uses batch 256, K_train=K_bootstrap=16, and ten width-1024 critics.
The proposed DICE acceleration failed numerical equivalence; its presence in
the configs is not approval to run it.

## Readiness and evidence

[`readiness.json`](readiness.json) is the executable admission record. RLDT alone
has a positive method-specific review. Global `production_ready` remains false,
so the three-method queue refuses execution. The other methods are not admitted
by RLDT's review.

- The handoff recorded 115 passing regressions and real-policy RLDT
  collection, updates, and learner/optimizer/RNG/replay recovery checks.
- Launch preparation verified all 392 files in the saved source manifest and
  the canonical task hashes, then passed 54 RLDT/protocol tests.
- A saved RLDT milestone loaded and completed one 280-command validation
  timeout with exactly acknowledged command accounting. This was a runtime
  check, not evidence of improved success rate.
- DICE's serial/batched action difference was approximately 0.05563 against a
  0.0001 tolerance. Its subsequent update-equivalence check was skipped.

Compact evidence and hashes are in [`evidence/`](evidence/manifest.json).
The [November 5 handoff](handoff-20261105.md) is preserved as a **historical
pre-launch snapshot**. Its statement that no production was launched describes
that earlier point; the RLDT-specific review and launch followed it.
The exact host-side [bounded-check runner](host-scripts/finish_bounded_checks.sh),
[regression runner](host-scripts/run_final_tests.sh), and
[handoff-state exporter](host-scripts/save_handoff_state.py) are archived with
their original absolute paths. Review and adapt those paths before reusing them
on a different checkout; the underlying diagnostic commands are in `scripts/`.

## Prerequisites

Use the [standalone-host setup](../test-box-1-setup.md),
[`setup_local_core.sh`](../../scripts/setup_local_core.sh), and
[`setup_vla_trace_pro.sh`](../../scripts/setup_vla_trace_pro.sh). Policy learning
and LIBERO-PRO simulation use separate Python environments. Upstream revisions
are pinned in [`dependencies.lock.json`](../../third_party/dependencies.lock.json).
The checkout does not include local environment symlinks, upstream checkouts,
weights, datasets, replay, compilation caches, or raw run outputs.

The saved configs intentionally retain the actual `/workspace/upenn/RL-VLA`
asset paths. Before using another machine, recreate those assets or explicitly
rebind the policy checkpoint, simulator Python, protocol, and FlashSAC state-set
paths in new configs and review their resolved hashes. The canonical
[protocol record](evidence/object0-protocol.json) records state partitions,
dataset/source revisions, and BDDL/state hashes; it does not contain the state
array itself. [`prepare_vla_trace_objects.py`](../../scripts/prepare_vla_trace_objects.py)
and [`make_flashsac_canonical_state_set.py`](../../scripts/make_flashsac_canonical_state_set.py)
provide the preparation pipeline. Preserve existing assets and verify their
hashes; do not silently substitute regenerated states in an existing experiment.

One allocated GPU is sufficient for RLDT. The actual launch used a 64-GiB GPU,
CPU cores 0-7, a 48-GiB host RAM budget, and 35 GiB of output allowance; its wrapper
required 100 GiB filesystem free. The launcher also retains the 2,300-GiB
workspace guard. Those are resource checks, not a claim of minimum GPU VRAM.
Obtain a real allocation and check current usage before launching.

## Launch and monitor

From the repository root, inspect the exact plan first:

```bash
third_party/openpi/.venv/bin/python scripts/launch_online_object0.py \
  --method rldt --seed 0 --gpu 0 \
  --output-root /ABSOLUTE/NEW/OUTPUT --dry-run
```

For an allocated GPU 0 and CPU cores 0-7, start a persistent terminal with
`tmux new-session -s object0-rldt`, then run:

```bash
export RLDT_CPUSET=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
export JAX_COMPILATION_CACHE_DIR=/ABSOLUTE/NEW/CACHE
third_party/openpi/.venv/bin/python scripts/launch_online_object0.py \
  --method rldt --seed 0 --gpu 0 --output-root /ABSOLUTE/NEW/OUTPUT
```

**The single-method launcher executes unless `--dry-run` is supplied.** The
three-method queue, `launch_online_object0_seed0.py`, instead defaults to a dry
run and requires `--execute`; its global readiness gate remains closed.

The launcher trains, validates all five snapshots, saves selection durably,
and runs the paired tests. Each attempt receives a unique directory with
`attempt.json`, `phases.jsonl`, command accounting, `training.log`, and phase
logs. Recovery files do not support restarting the same benchmark attempt.

```bash
third_party/openpi/.venv/bin/python scripts/launch_online_object0.py \
  --monitor /ABSOLUTE/ATTEMPT/DIRECTORY
tail -f /ABSOLUTE/ATTEMPT/DIRECTORY/training.log
```

The existing run uses tmux session `rldt_object0_seed0` and the output directory
recorded in [launch-confirmed.json](evidence/launch-confirmed.json). The exact
[host-specific wrapper](launch-rldt-20261105.py) is archived for provenance; use
the commands above for a new run instead of executing that archived wrapper.

Allow approximately 12-27 hours for the full RLDT training/validation/test
pipeline, subject to actual episode lengths and contention. No final performance
claim follows from this launch or from a single training seed.
