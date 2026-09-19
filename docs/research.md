# Sources and implementation decisions

All supplied material was read: the pasted setup summary, the complete 22-page `openpi_RL_research_roadmap.pdf`, and `openpi_RL_research_roadmap_2026-09-19.md`. The original documents remain at the root. Their central recommendation determines this version: matched bounded residual SAC/PPO first, with the VLA frozen.

Primary sources checked for this implementation on 2026-09-19:

- [Pinned OpenPI README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md): installation, supported platform, checkpoint choices. The roadmap commit was also current upstream HEAD at inspection.
- [Policy server CLI](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/serve_policy.py), [websocket client](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/websocket_client_policy.py): separate frozen serving process and exact client API.
- [LIBERO example](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py), [adapter](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/libero_policy.py), [config](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py): checkpoint-specific image conventions, 8-value state, 7-coordinate action and 10-action prediction horizon.
- [SAC algorithms and applications](https://arxiv.org/abs/1812.05905): entropy-regularized twin-critic updates and temperature learning.
- [PPO](https://arxiv.org/abs/1707.06347): clipped likelihood-ratio updates. Chunk-aware GAE and stopping masks follow the supplied roadmap's explicitly defined decision process.
- [Gymnasium time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/): separate true termination and bootstrap-eligible truncation.

The implementation chooses normalized residual entropy, explicit physical correction scales and a serial PyTorch collector for a small testable baseline. Those are engineering choices, not reproductions of EXPO-FT or direct flow pi-RL. No literature benchmark claims have been measured here. The broader roadmap bibliography was read as context; it is not represented as a fresh replication of every cited paper.

