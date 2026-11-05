# VLM → action-policy bottleneck diagnostics

These diagnostics test whether object information survives perception and is used
by the action expert. They do **not** assume that the VLM is innocent. High
attention, high rank, and a successful linear probe are individually insufficient
to establish that the action policy caused a failure.

## What is implemented

- `scripts/analyze_vla_trace_rank.py`: singular spectra, numerical/effective/stable/
  participation rank, 90%/99% energy rank, Frobenius/cosine drift, attention-mass
  drift, and conditional Jensen–Shannon divergence. Each layer, head, state, flow
  time and checkpoint is retained. Masked cameras and language padding are excluded.
- `scripts/analyze_vla_trace_value_transport.py`: rank, drift and centered linear
  CKA of the actual cached value contribution `A_block @ V_block`, with the native
  grouped-query head mapping. It verifies the original KV fingerprints against
  the saved lossless context. This examines the information transported by the
  attention operator, before the expert output projection, residual and MLP.
- `scripts/run_vla_trace_probe_controls.py`: train-only ridge object probes on
  image embeddings and final VLM prefix features, plus a strong categorical
  camera/patch-position baseline and three independently shuffled-label controls.
  It evaluates swap-layout validation and original-layout transfer separately.
- `scripts/analyze_vla_trace_compiler_bottleneck.py`: measures the same-weight
  attention drift caused by the two archived compiler profiles, using the
  existing first-five-state controls and verifying their source hashes.
- `scripts/report_vla_trace_bottleneck.py`: generates the comparison report and
  PNG/PDF figures from completed analysis outputs.
- Pure NumPy metric implementations live in
  `src/rl_vla/diagnostics/{attention_spectrum,value_transport,probe_controls}.py`.
  The existing native capture, rollout interventions, noun edits and spatial
  localization probes remain the input/evidence sources.

No policy parameters, old runs or training pipelines are modified. Outputs use
new directories, refuse overwrite, include input hashes, and preserve per-head
records. Probe split checks compare both simulator-state and observation hashes.
Uncertainty resamples complete observations with all their cameras/patches/heads,
not individual patches treated as independent data.

## Meaning of “cross attention” in this architecture

The pi0.5 expert uses joint attention over cached VLM prefix keys and expert suffix
keys. We analyze its **action-query × prefix-key block**. It is not a separate
cross-attention module, and its rows do not necessarily sum to one because action
suffix keys also receive probability mass. We preserve that original denominator.
Camera, combined visual, valid text, and combined prefix blocks are reported.

For singular values `s_i`, effective rank is `exp(-sum(p_i log p_i))`, where
`p_i=s_i/sum(s)`. Stable rank is `sum(s_i²)/max(s_i²)`. Participation rank is
`sum(s_i²)²/sum(s_i⁴)`. Zero matrices have rank zero; undefined cosine, relative
drift and conditional JS values are JSON `null`, with missing-query counts.

There are ten action queries in these captures, so an attention matrix's rank is
at most **10**, regardless of VLM width or number of image tokens. Low rank may
be appropriate for a coordinated action chunk. Native BF16 rounding can produce
small singular values; the numerical-rank tolerance is explicit (`1e-6` relative
to the largest singular value), and the complete spectra support sensitivity checks.
No universal “good rank” threshold is assumed.

The value analysis computes `A V` in float64 using exported native BF16 values;
it is a contribution analysis, **not bit-exact BF16 execution or a causal
intervention**. Its CKA uses the ten action queries as rows and is descriptive,
not a sample-level generalization estimate. Attention can change with no value
change, and visually similar attention can carry different value information.

Drift pairs must have identical observation/state hashes, noise, `x_t`, time and
token layout. This avoids confusing checkpoint changes with different flow
trajectories. The original campaign used different compiler profiles for baseline
and trained experts; small differences must be checked against its existing
`traces/<task>/compiler_controls/` before attribution to learned weights.

## Run on the saved campaign

From the repository root, use its CPU NumPy environment:

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=2
OLD=runs/vla_trace_objects_20261031
NEW=runs/vlm_policy_bottleneck_20261103

.venv/bin/python scripts/analyze_vla_trace_rank.py \
  --traces "$OLD/traces/object_2" --output "$NEW/rank/object_2"

.venv/bin/python scripts/analyze_vla_trace_value_transport.py \
  --traces "$OLD/traces/object_2" --output "$NEW/value_transport/object_2"

.venv/bin/python scripts/run_vla_trace_probe_controls.py \
  --root "$OLD" --task object_2 --model pi05_libero \
  --output "$NEW/probes/object_2/pi05_libero"

.venv/bin/python scripts/analyze_vla_trace_compiler_bottleneck.py \
  --root "$OLD" --output "$NEW/compiler_controls"

.venv/bin/python scripts/report_vla_trace_bottleneck.py --root "$NEW"
```

Repeat for Object 0/1/2 and both `pi05_base` and `pi05_libero` probe feature banks.
Choose a fresh `NEW` when rerunning. Rank defaults to all 50 archived states,
checkpoints 25/50/100, and ODE indices 0/5/9. Value transport uses the first five
states, the same predetermined subset as the prior projection ablations. The
probe uses 100 training states and 50 each for validation and original-layout
transfer. This is exploratory reanalysis of already-opened campaign data, not
new final-test evidence or a way to choose checkpoints.
The trace bank contains initial-decision observations. ODE indices are internal
flow-denoising times, not successive simulator timesteps; these measurements do
not by themselves localize failures later in a rollout.

Rank outputs contain `heads.jsonl` (spectra and drift) and `summary.json` (state
bootstrap summaries). Probe outputs contain fitted weights, results, intervals,
and source hashes. The original fixed 2048→256 projection is reused; feature
normalization is fitted only on train. The classifier predicts physical identity
(background, target, distractor), not the requested role inferred from an edited
instruction. Background predictions on object patches count as errors in the
reported object-only balanced accuracy.

## What would isolate the failure

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Image-feature identity decoding survives held-out positions | Visual identity is recoverable by that readout | The VLM grounds the requested object |
| Final-prefix probe transfers and beats position/shuffled controls | Contextual features carry accessible object information | The action expert reads that information |
| Requested-object localization follows noun edits with fixed pixels | Instruction-conditioned grounding | Successful control or multi-step reasoning |
| Per-layer KV probes and `A V` preserve the signal | Information reaches the interface/readout contribution | The remaining expert computation uses it correctly |
| Selective value/representation patching restores target-directed action | Causal evidence about an interface component | Generality beyond the tested states and controls |
| Closed-loop rescue on held-out layouts with VLM frozen | Evidence that an expert/interface change is sufficient in those cases | Every VLM failure mode has been ruled out |

The existing train/validation scenes alone are insufficient for the first two
rows: a position-only baseline can solve their patch-label task. Cross-layout
transfer is therefore essential. Failure of a linear probe is also ambiguous:
the representation may contain nonlinear information or the probe may be poorly
trained for the transfer distribution.

For a stronger follow-up, collect balanced target/distractor placements in
multiple positions with held-out layouts, vary the requested noun on the same
pixels, and include spatial-relation questions. Fit layerwise readouts on visual
embeddings, VLM residual states, and **each layer's K/V interface**, using separate
train/evaluation scenes. Current cached feature banks expose image and final
prefix features; they do not support training those layerwise readouts. The final
VLM hidden state is not the entire pi0.5 interface: the expert consumes per-layer
KV caches. Separate genuinely unseen categories from familiar objects in new
positions.

For causal testing, compare target-region edits against equal-area background,
distractor, and original-target-position controls. Use the existing
`scripts/run_vla_trace_interventions.py` and
`scripts/run_vla_trace_grid.py` for knockout, noun-edit, mask, and projection-swap
evidence. A targeted per-layer **KV activation-patching rescue** with a matched
donor remains a separate experiment; these new scripts do not implement or claim
that rescue. Check simulator reachability/contact confounds before labeling a
failed grasp a semantic error. Raw `pi05_base` is a representation reference here,
not an interchangeable LIBERO action policy.
Also, `pi05_base` already includes VLA pretraining: its VLM component is not an
untouched original PaliGemma checkpoint. A claim about loss from the original
foundation VLM needs that additional reference and matched preprocessing.

## Related studies

- [VLA-Trace](https://arxiv.org/abs/2605.30117) combines representation CKA,
  checkpoint drift, attention knockouts, spatial grounding, masks and instruction
  edits. It finds architecture-specific routing and limited fine-grained semantic
  control; pi0.5 shows concentrated visual-to-action dependencies. The spectral
  rank and shortcut-controlled linear probes here are extensions, not a claim
  that its published protocol already proves our model's failure mechanism.
- [Act2Answer / Does VLA Even Know the Basics?](https://arxiv.org/abs/2606.19297)
  separates knowledge-sensitive action selection from broader control evaluation
  and uses layerwise intent probing. It reports answer-related signals strongest
  in middle layers and weaker in upper layers, directly relevant to an information
  availability versus action-use hypothesis.
- [Depth-Wise Probing and Pruning of the Planning Token](https://arxiv.org/abs/2608.07361)
  distinguishes early linear decodability of intent from compatibility with a
  frozen native planner. This is a close conceptual match, but its experiments
  concern a driving VLA, not our pi0.5 manipulation checkpoint.
- [What Frozen VLAs Already Know About Success](https://arxiv.org/abs/2605.28527)
  probes outcome information in frozen features, includes shortcut/shuffled
  controls, and tests action selection. It provides a useful precedent for
  separating recoverable information from its behavioral use.

None of these studies establishes the cause of a particular local failure without
matched tests on that checkpoint and environment.
