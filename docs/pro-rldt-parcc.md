# LIBERO-PRO RLDT pilot on PARCC

This experiment starts from the verified official native JAX `pi05_libero`
checkpoint and runs independent RLDT adaptations for PRO Position task 0 in
Spatial and Object. No demonstration or SFT warm start is used. The full action
expert is trained; the VLM remains frozen. It uses the normal 100-iteration,
eight-particle configuration, up to 16,000 controller steps per task.

The experiment directory is `runs/pro_rldt_20261025/`. Its `protocol.json`
records source versions, checksums, and exact state IDs. The 50 published PRO
states per task are divided deterministically into 30 training and 20 held-out
states using split seed 20261025. These are different initial states of the
same published position perturbation, **not disjoint perturbation ranges**.
This pilot therefore does not complete the proposal's stronger generalization
experiment or establish a full LIBERO-PRO benchmark score.

Submitted on 2026-10-25 (America/New_York): baseline array **9552302** and
dependent RLDT array **9552303**. Task index 0 is Spatial; index 1 is Object.
The Spatial baseline started on `dgx028` using a 45 GB B200 MIG slice.
Consult Slurm and stage status files for current state.

Each task runs these stages:

1. Frozen-policy evaluation on all 20 held-out PRO states.
2. Frozen-policy retention evaluation on original-task states 0–19.
3. RLDT from the original checkpoint, training seed 0, 100 iterations.
4. Trained-policy evaluation on the same 20 PRO and 20 original states.

Evaluation enumerates each state once, uses reset/noise seeds 100000–100019,
and resets policy noise independently per episode. Both evaluations use the
native RLDT sampler with its float32 expert and the checkpoint normalization
assets. This keeps before/after numerical behavior consistent; it is not a
claim to reproduce upstream's serving precision or full benchmark protocol.

## Installed dependencies

- PRO source: `third_party/libero_pro`, commit
  `eafdb809426b13153aa1e4c42d6601844217dfec`.
- Official dataset: `zhouxueyang/LIBERO-Pro`, revision
  `c86fc3b8293185a6f373677018ff3e37f8391602`. Four required BDDL/state files
  are under `datasets/benchmark_assets/libero_pro/`, verified against the
  pinned Hugging Face Git blob IDs and recorded with SHA-256 checksums.
- Simulator: `.venv-pro`, using the repository's locked CPU LIBERO dependencies,
  the PRO editable package, root package, and OpenPI client.
- Learner: existing `third_party/openpi/.venv`.
- Simulator paths: `.cache/libero-pro/config.yaml`; rendering uses OSMesa.

The original `.venv` and ordinary LIBERO installation remain separate. The
adapter rejects accidentally importing another LIBERO fork, modified task
assets, overlapping splits, or extra evaluation episodes.

## Submission and resources

From the repository root in a login shell:

```bash
source scripts/env.sh
python scripts/submit_pro_rldt.py --experiment runs/pro_rldt_20261025
```

This is a **one-time submission command**: it refuses a second submission when
`jobs.json` exists. Inspect that file for the actual job IDs already submitted.
Each array has two task indices, 0 = Spatial and 1 = Object, with concurrency
limited to one. Baseline jobs request one 45 GB B200 MIG slice, six CPUs, 48 GB
host RAM, and a four-hour limit. Training jobs request one full B200, 28 CPUs,
224 GB host RAM, and a twelve-hour limit, including post-training evaluation.
Both use account `jgu32-lab` and QOS `dgx`.

The training array uses `aftercorr` dependencies: each task starts only after
its corresponding baseline finishes successfully. Thus at most one baseline
slice and one full training GPU run concurrently. A failed baseline cancels
its dependent training task. Jobs do not automatically requeue or overwrite
partial runs. Model/optimizer checkpoints do not implement exact replay resume.

`launch_contract.json` freezes hashes of project/upstream Python code,
launchers, protocol, configs, and normalization assets. A mismatch stops a job
before model startup. Preserve this experiment and prepare a new one when
changing code or protocol.

## Monitoring

```bash
squeue -u "$USER"
cat runs/pro_rldt_20261025/jobs.json
cat runs/pro_rldt_20261025/spatial_0/baseline_status.json
cat runs/pro_rldt_20261025/spatial_0/train_status.json
tail -F runs/pro_rldt_20261025/spatial_0/baseline_heldout.log
tail -F runs/pro_rldt_20261025/spatial_0/train.log
tail -F runs/pro_rldt_20261025/spatial_0/training/metrics.jsonl
```

Replace `spatial_0` with `object_0` for the second task. Stage files appear only
after the corresponding job starts. Slurm logs are `runs/pro_rldt_JOBID_INDEX.log`.
Evaluation writes each finished episode immediately to `*.episodes.jsonl`,
including its initial-state ID; the full JSON report appears at completion.
After training and both evaluations, `comparison.json` reports paired baseline
and trained success rates. GPU CSV logs on MIG include physical-device totals
and must not be interpreted as this job's exclusive memory usage.

The simulator checks are in `setup_check/report.json` and initial camera PNGs.
Package snapshots and the numerical RLDT test report are in the experiment
directory. Scheduling limits are allowances, not measured runtimes.
