# Native JAX RL-VLA adapter for VLA-Trace

VLA-Trace checkout: `/workspace/upenn/VLA-Trace`, commit
`027e14443ea4532337e0376ed18e8e247e32d927`. Plotting environment:
`third_party/vla_trace_tools/.venv`. Native inference remains in the pinned
OpenPI environment; no Torch extras are installed there.

`src/rl_vla/diagnostics/pi05_attention.py` calls the original OpenPI prefix,
attention, and sampler. A scoped Linen interceptor taps the original
`jax.nn.softmax` result while the native `gemma.Attention` method is tracing.
The original softmax value returns unchanged to attention. Ordered JAX debug
callbacks export the native probabilities in scan layer order. The temporary
Python tap is restored on exit, including exceptions. Run diagnostics in a
separate process; concurrent tracing in the same process is unsupported.
Training never imports this module. No upstream source files are modified.

The campaign uses separate immutable diagnostic compiler profiles:
`runs/vla_trace_objects_20261031/compiler_profiles/native_baseline_v1/manifest.json`
for the untouched LIBERO expert and
`runs/vla_trace_objects_20261031/compiler_profiles/native_trained_v2/manifest.json`
for the requested trained experts. Each starts in its own Python process with an
isolated cache. The original native baseline and trained executables use
different tuning choices for an identical projection graph/device key, so one
startup profile cannot represent both simultaneously. Each profile reproduces
its independently retrieved original native executable. This deployment changes neither the
native training/evaluation runtime nor the shared compilation cache. The
pre-diagnostic amendment at
`compiler_profile_integration/fidelity_protocol_amendment_v1.json` distinguishes
instrumentation fidelity from historical compiler reproducibility: all requested
checkpoints require exact agreement with independent uninstrumented native
outputs obtained through verified original-cache hits. Historical recorded-action
differences remain separately reported, without an invented acceptance tolerance.
Actual held-out rollout traces still require exact replay of their saved actions.
The original unsuccessful technical attempts and prior profile release remain
available. A single passing technical case does not release the research queues.

Use the guarded queue launchers below so the compiler environment is set before
the native Python process starts. They record profile, helper and effective
environment hashes, and wait for the required per-task fidelity and selection
gates. The parent-safe `diagnostics.compiler_dispatch` helper routes the two immutable
profile helpers and never initializes a
JAX backend or changes the parent environment. Child package versions and profile
bytes are rechecked before model loading and during execution. Existing campaign
queues own their destinations; these commands are reproduction examples, not an
instruction to start duplicate workers.

The official-PRO Object0 technical gate is
`runs/vla_trace_objects_20261031/gate_b_actual/gate_summary.json`. It uses a
generated training candidate, not a held-out test scene. The complete native
10-step sampler, all intermediate flow latents, velocities at steps 0/5/9,
prefix features, and KV cache were bit-identical with capture enabled.
The largest softmax row-sum error was 0.002821, reflecting native BF16
probabilities; masked dummy-camera mass was zero. This is a software-fidelity
result, not evidence of policy success or semantic grounding. Initial failed
synthetic attempts remain in their own directories and are not research data.

Run one matched scene:

```bash
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src
taskset -c 48-55 third_party/openpi/.venv/bin/python scripts/trace_pi05_attention.py \
  --observation runs/vla_trace_objects_20261031/gate_b_pro_train_input/observation.npz \
  --steps 0,5,9 --output runs/NEW_TRACE
```

Add `--expert methods/rldt/<task>/training/iteration_000025` to load a saved
expert through `Pi05Flow.load_params`, which checks original checkpoint and
normalization provenance. All fixed-time comparisons use the original
baseline denoising path, regardless of the selected expert. Each expert's own
trajectory is stored separately. When an observation has an adjacent
`initial_noise.npy`, the trace consumes that exact persisted bank latent and
records its SHA256; `--noise` can select an explicit file. Both the common
baseline path and each policy's own path start from that same array. Without
one, the seed defaults to 100000 and actual JAX noise is saved in `actions.npz`.
The original technical gate used native JAX noise independently of the bank's
NumPy diagnostic-noise generator.

A rollout observation can instead contain `sampling_key` (uint32[2]) and
`recorded_normalized_actions` ([10,32]). The trace reconstructs the exact JAX
noise from that recorded key and requires bit-exact replay of the original
policy actions. Stored rollout keys take precedence over adjacent bank noise;
an explicit contradictory `--noise` override is rejected. Such frames are
labeled rollout-conditioned, rather than fixed initial-state bank comparisons.

Input NPZ fields are `image`, `wrist_image`, `state`, and `prompt`, without
pickle. Images are uint8 224×224 RGB already rotated at the LIBERO policy
boundary; the adapter adds no rotation. State is the original eight-dimensional
LIBERO observation. `prefix.npz` contains original image embeddings,
contextualized prefix features, masks, prompt IDs, and prefill attention for
every valid instruction token when requested. `expert_stepXX_*.npz` contains
attention `[layer,batch,head,action_query,key]`, velocity, common `x_t`, and `t`.
`own_trajectory_attention.npz` adds a leading ODE-step axis.

The current native model has 18 layers, 8 query heads, and 10 action queries.
Actual spans are derived from the image encoder output and recorded, rather
than taken from toolkit examples: external image 0:256, wrist 256:512, masked
dummy camera 512:768, text 768:968, suffix actions 968:978. Each camera is a
16×16 patch grid in row-major image order. Padded text and dummy camera remain
in the key axis and retain their native zero attention. Token IDs, pieces,
valid-token mask, noun-subword indices, and query/key offsets are recorded.
Attention values are the original full-denominator probabilities, not
min–max heatmaps or per-camera renormalizations.

`none`, `no_image`, `no_text`, and explicit prefix-key knockouts alter the
actual native boolean mask before softmax. They are applied to expert queries
in all layers. The final action projection has no attention map. The separately
validated layer adapter also masks early, middle, late, or individual expert layers.
Output-projection hidden-state swaps are an explicitly separate secondary analysis;
neither is inferred from the image masks.

Batch feature extraction reuses one model and compiled functions. Feature-bank
NPZs are uncompressed, lossless float32 exports of native BF16 features; this
reduced measured steady-state time from 6.52 seconds to approximately 0.21
seconds per frame, at a cost of roughly 17 GB for both models' 1,200
train/validation/test frames:

```bash
taskset -c 48-55 third_party/openpi/.venv/bin/python scripts/trace_pi05_attention.py \
  --bank PATH/TO/manifest.jsonl --features-only --output runs/NEW_FEATURE_BANK
```

`--bank-root PATH/TO/observation_bank` waits for completed manifests in each
`<task>/<split>` directory and processes all ready banks with one model.
`--splits source` extracts original-layout retention features. `--prompt`
supports an explicitly offline counterfactual instruction; it never changes
a simulator goal or authorizes scoring success for the changed instruction.

Output is `state_<six-digit-ID>/{prefix.npz,metadata.json}` plus progress and
completion records. The first frame checks native prefix fidelity; later
feature-only frames reuse the same capture graph and explicitly record that
native fidelity was not recomputed per frame. Their diagnostic contexts must
not be used for actions. Action traces always recompute the original canonical
native context. RLDT prefix representations can be stored once after proving
frozen-backbone identity; do not count duplicate features as independent data.

Raw base uses a separate loader:

```bash
taskset -c 48-55 third_party/openpi/.venv/bin/python scripts/trace_pi05_attention.py \
  --raw-base --bank PATH/TO/manifest.jsonl --features-only \
  --output runs/NEW_RAW_BASE_FEATURE_BANK
```

This is an **image/text-only representation reference**, with native OpenPI
image resizing/range and PaliGemma vocabulary. It deliberately omits discrete
robot-state tokens because a valid LIBERO-to-base embodiment/normalization
mapping is unavailable. This omission is a declared representation intervention,
not the raw-base native full-policy contract. No LIBERO normalization or
LIBERO-trained weights are borrowed. There are no raw-base action/sample/decode
methods: closed-loop success, expert attention, and causal action effects are
`N/A — incompatible action contract`. The raw-base state placeholder is never
tokenized and must not be treated as a physical robot state.

The pinned `pi05_libero` configuration also sets `discrete_state_input=False`.
Its captured VLM prefix therefore contains instruction tokens without discrete
state tokens; normalized numeric state is carried separately in the native
observation/context. Do not describe the observed raw-base/LIBERO prefix
comparison as a contrast between absent and present proprioceptive state tokens.
The campaign's CPU preprocessing audit also verified exact image arrays, camera
masks, prompt IDs and prompt masks for all 150 frozen original-prompt test
observations. These matched VLM inputs do not establish a raw-base action adapter;
see `compiler_profile_integration/vlm_input_contract_correction_v2.json` in the run.

VLA-Trace's packed all-to-all `attention-export` assumes shared query/key
coordinates and therefore cannot directly consume action-only generation
queries. `views_stepXX_*.npz` supplies one direct camera slice per key, with
`base_0_rgb` and `left_wrist_0_rgb` mean vectors and separate per-layer/head
arrays. Use those with the plotting environment:

```bash
third_party/vla_trace_tools/.venv/bin/vla-trace attention-overlay \
  --image PATH/TO/external_rgb.npy --attention PATH/TO/views_step00_none.npz \
  --attention-key base_0_rgb --masks PATH/TO/masks.npz \
  --mask-key image__alphabet_soup_1 --grid-size 16 --output overlay.png
```

Check the installed CLI help for its exact grid syntax. Metrics must use
simulator instance masks in the same policy-oriented image coordinates,
preserve per-camera denominators, and report mask area alongside mass/density.
An overlay is descriptive; causal claims require paired interventions and
closed-loop behavior on predeclared cases.

Verification:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  third_party/openpi/.venv/bin/python -m pytest -q \
  tests/test_pi05_token_spans.py tests/test_pi05_attention_capture.py
```

The tests cover native cached attention, exact no-op hidden outputs, zero mass
on masked keys, changed outputs under actual knockout, reconstruction of the
native value aggregation from exported probabilities, and distinct generation
query/key coordinates. The full-checkpoint gate additionally verifies the
native ten-step sampler and real PRO observation.

The full matched checkpoint grid waits for all 25/50/100 snapshots and the
validation-only selection lock before using test cases. It verifies frozen
bank IDs and observation/mask/noise hashes, preserves selection immutability,
and exports raw-base VLM-only maps, all four compatible expert conditions,
counterfactual noun maps, exact prefix equality, frozen-backbone SHA,
per-layer/projection weight drift, and fifteen open-loop mask/pathway
conditions. For 50 cases, retained native contexts occupy roughly 0.9 GB;
allow approximately 70–100 GB for trace files across all tasks/policies. The
grid queue runs separate `baseline`, `trained`, and `baseline_projection` native
processes, then a CPU-only `aggregate` stage per task. The baseline stage saves
exact contexts, noise, latent paths, and hidden/projection references; the trained
stage consumes hash-verified immutable copies. The baseline postpass completes
baseline-weight projection swaps and records same-hidden compiler differences
separately. Only the parent publishes global completion after all three tasks.

```bash
source scripts/env.sh
PYTHONPATH=src .venv/bin/python scripts/queue_vla_trace_diagnostics.py grid \
  --run runs/vla_trace_objects_20261031 --gpu 3 --cpuset 48-55 \
  --compiler-profile runs/vla_trace_objects_20261031/compiler_profiles/native_trained_v2/manifest.json \
  --baseline-compiler-profile runs/vla_trace_objects_20261031/compiler_profiles/native_baseline_v1/manifest.json
```

Closed-loop interventions follow the native evaluation RNG sequence:
`PRNGKey(100000 + episode)`, then one split per decision. Pixel masks come
from current simulator geometry at chunk boundaries in the separate PRO
process, and change only policy input pixels. Each condition verifies an
unchanged native no-op and exact explicit-noise recurrence. Ten predetermined
states form an exploratory paired analysis; they are not additional
independent cases in the primary 50-state test denominator.

```bash
source scripts/env.sh
PYTHONPATH=src RL_VLA_OBJECT_TELEMETRY=1 LIBERO_CONFIG_PATH="$PWD/.cache/libero-pro" \
  .venv/bin/python scripts/queue_vla_trace_diagnostics.py interventions \
  --run runs/vla_trace_objects_20261031 --task object_0 --gpu 4 --cpuset 0-7 --workers 4 \
  --compiler-profile runs/vla_trace_objects_20261031/compiler_profiles/native_trained_v2/manifest.json \
  --baseline-compiler-profile runs/vla_trace_objects_20261031/compiler_profiles/native_baseline_v1/manifest.json
```

The intervention queue covers the baseline and all three requested checkpoints.
`no_image` and `no_text` remove
expert-to-prefix routes; contextualized image keys can already carry language
information and text keys can carry visual information. These are pathway
interventions, not complete modality-information erasure. Decoded action
differences are LIBERO OSC/gripper command differences, not Cartesian metres.
Old-target controls use separate immutable auxiliary masks: the original
BDDL target region is projected with native camera calibration as a floor
polygon and an extruded prism using the requested object's native height.
The prism is the declared pixel-intervention mask; its artifact, source BDDL,
and matching observation hashes are checked. This spatial-region mask is
separate from the distractor instance currently occupying that region.
The layer adapter passed bit-exact native velocity and full ten-step sampler
checks with both recorded JAX and persisted Gaussian noise. Its 42-condition
grid software check covers image/text masks in early/middle/late windows and
each of 18 individual layers; every blocked probability is zero. These secondary
causal interventions run on the first five predetermined frames per policy.
`layer_knockout_summary.json` stores actual action deltas and common-latent
velocity deltas at steps 0/5/9; `layer_knockout_arrays.npz` retains outputs.
Descriptive attention changes remain distinct from these intervention effects.

For the first five predetermined test frames, the grid additionally captures
final normalized expert suffix hidden states at common flow steps 0/5/9 and
calls the actual native NNX output projection with all four combinations of
baseline/checkpoint hidden states and baseline/checkpoint projection weights
(including bias). Own-checkpoint combinations must exactly reconstruct native
velocity. These secondary open-loop artifacts are
`projection_ablation_stepXX.{npz,json}`; values are normalized flow velocities,
not metre displacements or closed-loop performance. The actual-model software
gate using one real optimizer-smoke expert passed exact reconstruction at all
three times: `projection_swap_technical_smoke/gate_summary.json`.

Recorded rollout phases use `scripts/trace_pi05_rollout_bank.py`. Each phase
replays the exact stored native JAX sampling key and checks normalized and
decoded action chunks against the original recording before exporting attention.
Expert captures follow that policy's own denoising path. The trace metadata marks
these as rollout-conditioned observations, since pixels and robot states can
differ between policies. The direct trace CLI supports
`--bank <episode-phase-directory>` for a technical fidelity gate. Research traces
use the startup-profile queue, which waits for all three tasks and four action
policies and retains the predefined first five test episodes:

```bash
source scripts/env.sh
PYTHONPATH=src .venv/bin/python scripts/queue_vla_trace_diagnostics.py rollout \
  --run runs/vla_trace_objects_20261031 --gpu 5 --cpuset 68-71,82-85 \
  --compiler-profile runs/vla_trace_objects_20261031/compiler_profiles/native_trained_v2/manifest.json \
  --baseline-compiler-profile runs/vla_trace_objects_20261031/compiler_profiles/native_baseline_v1/manifest.json
```

`plot_vla_trace_qualitative.py` consumes completed trace artifacts without loading
models. Its phase panels use one linear raw-attention scale across all phases
and ODE columns within each camera, with target, distractor and prior-region
contours. Supplemental pinned VLA-Trace CLI overlays preserve their commands
and disclose their individually scaled color bars. Both PNG/PDF figures and
the exact panel attention arrays are saved.

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/plot_vla_trace_qualitative.py \
  --run runs/vla_trace_objects_20261031 \
  --output runs/vla_trace_objects_20261031/rollout_trace_analysis
```

The campaign already has a waiting renderer; do not launch a duplicate into
that destination. A reproduction needs a fresh output directory. The output
contains `<task>/<policy>/state_ID` rollout panels and
`fixed_input_qualitative/<task>/<candidate>_<category>_state_ID` matched-input
panels. Fixed examples select the minimum-ID recovered and jointly-unsolved
case per RLDT candidate, prioritizing independently reviewed baseline
wrong-object recoveries where available. This outcome selection is explicit
and does not replace full-set statistics. Each policy must share the exact
bank observation, persisted diagnostic noise, baseline latent and flow time;
recorded rollout initial observations must also match the bank.

Fixed-example selection waits for
`video_reviews/<task>/baseline_test_review_complete.json` with
`complete:true`, task, baseline report SHA, and `review_records:[{path,sha256}]`.
Every failed baseline test episode must be covered by immutable reviewed
records; an empty review set is valid only when there are no baseline failures.
The renderer emits review requests as baseline reports arrive. Actual rollout
phase rendering does not wait for outcome reviews. Final report completion
requires the global panel summary, and missing outcome categories are recorded
explicitly rather than replaced with attention-selected examples.

Required figure readiness is checked separately by
`scripts/audit_vla_trace_figure_artifacts.py`. The CPU-only analysis environment
uses `pypdf==5.9.0` to decode PDFs, alongside PNG/table/map validation. The
27-cell manifest covers original and counterfactual probes, causal sensitivity
and supplemental controls/vectors; E/F cannot pass from early metrics markers
alone. Explicit undefined metrics need matching Boolean validity masks, while
raw actions and controller requests must be finite. The campaign already runs
a validator; its launch and CPU-affinity records are in `figure_artifacts/`.

The report's paired failure-transition tables use the existing verified review
labels without new fitting or statistical settings. A wrong-object failure
becoming a target-grasp placement failure is distinct from recovery to native
success. OOD visibility remains camera-specific and layout variants remain
separate because their new-center neighborhoods and base states overlap.
Supplemental token controls, modality curves, layer-response plots and action
vectors are explicitly post-outcome exploratory additions. Nominal directional
alignment is not achieved motion or predicted first-choice accuracy.
