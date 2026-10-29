# Repository cleanup

Prepared October 29, 2026, on branch `cleanup` in
`/mnt/projects/jg/kaifany/RL-VLA-clean`.

## Changes

- Short README and clear indexes for documentation, launchers, integrations, and runs.
- Source, configuration, tests, and compact results included; generated evidence and bulk artifacts excluded.
- Twelve upstream revisions recorded in a dependency lock; local RLinf changes saved as a patch.
- Existing lint errors fixed in the publishing checkout.
- CPU configuration tests use small, licensed RLinf fixtures instead of an untracked checkout.
- Docker test packaging includes integrations and a pinned PyYAML dependency.

## Local artifact cleanup

Removed 65 obsolete paths containing **17,249,389,150 bytes (16.06 GiB)**:
abandoned setup/smoke attempts, four expired allocation-lock directories,
two redundant smoke checkpoint directories, and an empty root log.

Preserved 166 diagnostic files in a **1.66 MiB** archive at
`runs/archive/cleanup_20261029/diagnostics.tar.gz` in the original checkout.
Its `manifest.json` records every removed path and size.

Active campaigns, completed experiments, negative results, selected policies,
baseline screens, source snapshots, and referenced readiness evidence remain local.
The original runtime source is unchanged because active jobs verify frozen hashes.
The cleaned source is isolated in the publishing checkout.

## Validation

- Root CPU suite: **215 passed, 21 skipped** (optional runtime dependencies).
- Multi-benchmark evaluation, lifecycle, and environment tests: **11 passed**.
- Ruff checks passed for `src`, `tests`, and `scripts`.
- All **3,169 protected files** survived unchanged: size and modification time
  checked for every file; files smaller than 8 MiB also rehashed.
- The saved RLinf patch passed a reverse-application check against the running checkout.
- No files larger than 10 MiB or detected credential patterns are included in the publishing tree.

Docker image builds require another machine because Docker is not installed on
this GRASP login host. GPU training was not launched as part of cleanup.

## Push

Review the prepared commit, then push manually:

```bash
cd /mnt/projects/jg/kaifany/RL-VLA-clean
git status --short
git show --stat HEAD
git push -u origin cleanup
```

This publishes the cleanup branch. It does not replace `main` or modify a running
experiment. No remote push was performed during preparation.
