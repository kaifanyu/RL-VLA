# Verification

Validated locally on Windows with Python 3.11.15, PyTorch 2.7.0 CPU, NumPy 1.26.4 and Gymnasium 1.3.0. Full dependency versions are in `requirements-tested.txt`; `uv pip check` reports no conflicts.

The 56-test suite covers residual probability calculations and saturation, SAC target masking, variable chunk durations, PPO likelihood recomputation, GAE boundaries, real parameter updates, checkpoint/optimizer restoration, warm-start setting checks, CLI use, external adapters, camera/state mapping, and OpenPI action validation. It also tests the actual upstream websocket client and NumPy serialization against a local test server.

End-to-end runs saved under `runs/`:

| Run | RL decisions | Controller commands | Update calls |
|---|---:|---:|---:|
| `validation_sac/` | 2000 | 9504 | 1937 SAC steps |
| `validation_ppo/` | 2048 | 9740 | 8 PPO rollout updates |

Their `final.pt` checkpoints were reloaded for 20-episode stochastic evaluation on seeds 100000-100019. The mock frozen base succeeded in 20/20 episodes, SAC in 19/20 and PPO in 20/20. These exercise training and evaluation plumbing; they do **not** establish improvement or rank algorithms. The toy base is already competent. Per-episode results and confidence intervals are in `runs/validation_*_eval.json`.

No OpenPI weights, task datasets, LIBERO simulator, or GPU inference environment were downloaded. Actual VLA inference, real LIBERO rendering, robot learning, MuJoCo Warp throughput and visual robustness remain unmeasured. LIBERO's legacy dependency stack needs separate validation in a compatible simulator environment before using the concrete configs.

Recheck after edits with `python -m pytest -q` and `ruff check src tests` in the root environment. GPU/server setup instructions are in [openpi.md](openpi.md).

