# RLDT with VLM-backbone LoRA

Set `policy.lora_target = "vlm"` with a positive `policy.lora_rank` to adapt
PaliGemma's image/text transformer (expert 0). The existing default remains
`"action_expert"`. The VLM variant uses `VlmLoraFlow` automatically, including
through direct `Pi05Flow.from_config` calls.

```toml
[policy]
kind = "pi05"
checkpoint = "/workspace/upenn/RL-VLA/checkpoints/openpi/pi05_libero"
model_config = "pi05_libero"
observation_action_adapter = "libero"
normalization_assets = "/workspace/upenn/RL-VLA/checkpoints/openpi/pi05_libero/assets"
projection_only = false
lora_target = "vlm"
lora_rank = 128
lora_alpha = 128.0
lora_seed = 0
```

LoRA covers the VLM transformer's query, key/value, output-attention, and MLP
weights. Each scanned layer/head has factors on the last two weight axes,
matching the existing action-expert adapter parameterization. All original
checkpoint weights remain frozen. SigLIP's image encoder, token embeddings,
normalization weights, action expert, and action/time projections receive no
adapters. The RL critic is trained independently.

The original action-expert-only pipeline detaches a frozen VLM KV cache before
each actor update. That would sever the gradient to VLM adapters. This variant
instead stores frozen image/text embeddings and recomputes the transformer
prefix **inside the actor loss**, using current adapters. Gradients pass from
the RLDT velocity objective through the frozen motor expert and its KV inputs
into VLM adapters. Transport and constraint velocities share one differentiable
prefix. Reference velocities use reference adapters, not current-policy caches.

Each observation's prefix is computed once and gathered for its action
particles. Sampling computes one current-policy prefix per rollout, reusing it
for all ODE steps. Rollout trajectories remain detached as required by RLDT.
Adapters are materialized inside a rematerialized layer scan using the native
OpenPI transformer block. This bounds temporary dense-weight gradients to one
layer instead of allocating them for the full VLM simultaneously. A regression
test compares both KV-cache values and adapter gradients against native full
materialization on a reduced model.
Critic features use the original frozen checkpoint so replay features stay in
the same coordinates as training progresses.

`scripts/smoke_rldt_vlm_lora.py` checks zero-adapter sampling against the original
policy, performs a real RLDT actor update, reloads its saved adapters, checks
that actions respond to the update, and hashes every frozen tensor before and
after. It is a technical gate, not a success-rate evaluation.

VLM backpropagation requires more compute and memory than action-expert LoRA.
The current learner uses one GPU. Independent evaluations can use additional
GPUs. Checkpoint metadata records VLM versus action-expert targets and rejects
loading adapters with an incompatible target.
