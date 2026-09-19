# Baseline design

The supplied roadmap's first comparison is implemented: frozen base + residual SAC versus frozen base + residual PPO. The actor uses deployable state, optional frozen observation features, and the sampled base prefix. The critic can additionally use `critic_state`. The base consumes checkpoint-compatible images, state and language through its own adapter.

```text
task observation -> frozen OpenPI -> cached nominal chunk B
       |                                   |
       +-> state/features + B[:K] -> residual actor -> v = tanh(u)
                                                   |
                       clip(B[:K] + scale * v) -> task controller
                                                   |
            reward, final observation, flags -> SAC replay / PPO rollout
```

## Probability and time conventions

- Policy density and SAC entropy use normalized residual coordinates `v`, before controller projection. The stable tanh Jacobian is included. Physical scaling belongs to the environment; `log p(delta) = log p(v) - sum(log(scale))` on active coordinates if physical density is ever needed. Never mix that density with the normalized entropy target.
- A zero residual scale freezes that motor coordinate and removes it from the modeled action dimension. Actor mean starts at zero and standard deviation at `exp(-3)`.
- `H` is the VLA prediction horizon; `K <= H` is the maximum executed prefix. One decision earns `R = sum(gamma**j * r_j)` and discount `gamma**k`, where `k` is the actual number of commands executed before an early boundary.
- True terminal states never bootstrap. Time-limit truncations bootstrap from the final pre-reset observation and its sampled base chunk. GAE stops across either episode boundary. GAE lambda is per decision.
- PPO stores the exact conditioning vector, sampled pre-tanh residual and old log probability. It never re-queries OpenPI during optimization. SAC replay stores both current and next conditioning vectors, so the sampled base prefix is preserved.
- Corrections apply only to actual robot coordinates, not OpenPI's internal padded dimensions. Entire sampled decision densities remain scored if an episode ends before its full prefix executes.

## Implemented learners

SAC has twin Q networks, target networks, reparameterized actor updates, automatic temperature tuning, and uniform online replay. Replay warm-up uses the initially small residual around the base, rather than uniform motor commands. One collected decision produces the configured number of updates after warm-up. Logs report gradient steps per decision.

PPO has clipped policy updates, a separate value network, macro-step GAE, batch advantage normalization, gradient clipping and KL stopping. The entropy bonus, if enabled, uses samples from the current transformed policy. Each rollout is fresh; replay is not used for PPO.

Both use float32 MLPs without dropout. The compact replay stores state/features and base action context, not full images. Images still go to OpenPI. To give the residual direct visual input, supply a **fixed** image embedding as `features`; include task/language embeddings there for a multi-task residual. A state-plus-base-only residual can have limited perceptual recovery ability. If training an encoder, redesign replay to retain/recompute images; cached features would become stale.

## Checkpoints and reproducibility

Residual checkpoint files contain network weights, optimizers, SAC temperature, configuration and counters. `--init-from runs/old/final.pt` warm-starts a **new** run. It resets the environment, replay/rollout, counters and RNG; it is not an exact paused-run continuation. Effective learner settings must match the saved optimizer configuration. OpenPI weights are separate and remain fixed.

The ordinary websocket server owns its base-policy RNG. Local seeds reproduce the toy and residual sampling, but cannot guarantee identical remote OpenPI chunks across runs. Use the same serving configuration, report multiple training seeds, and add server-side noise control before claiming tightly paired base-noise comparisons.

The standard server may return empty metadata. `base.checkpoint_id`, `base.model_config` and `base.source_commit` in example configurations are declared provenance, not remote verification. Keep the actual server checkpoint fixed throughout a run and verify that it matches these fields before warm-starting or evaluating. A host/port alone does not establish checkpoint identity.

## Scope of this first version

The implementation starts with one synchronous environment. An optional LIBERO adapter follows the upstream reset, camera and controller conventions; its contract tests use a simulated backend, not an installed renderer. Other tasks use the generic factory. There is no MuJoCo Warp benchmark implementation. GPU batching requires a vector collector, separate final observations for each world, and device-resident base inference. Preserve the tested chunk semantics when adding it.

Later stages from the roadmap: demonstrations/offline replay with valid residual labels; held-out lighting/camera evaluation; paired-view training; direct stochastic-flow PPO using an established implementation; and native OpenPI flow-loss distillation. None is silently represented as implemented by a placeholder trainer. Current checkpoints improve the composite controller, not the standalone VLA weights.
