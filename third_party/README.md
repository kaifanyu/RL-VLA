# Upstream sources

These directories are separate, ignored Git checkouts, not submodules. A fresh
RL-VLA clone does not contain them. Keep each upstream license with its checkout.
The prepared server already has OpenPI weights and a separate GPU environment;
neither is published with this repository.

| Directory | Source | Pinned commit |
|---|---|---|
| `openpi/` | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| `FlashSAC/` | [Holiday-Robot/FlashSAC](https://github.com/Holiday-Robot/FlashSAC) | `87edc9061150ae9e962dd84e6544e27a1554b3ab` |
| `dice_rl_upstream/` | [zhanyisun/dice-rl](https://github.com/zhanyisun/dice-rl) | `98d936e52523878f0172e606ea32403176497100` |
| `libero_plus/` | [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) | `4976dc30028e805ff8094b55501d532c48fec182` |
| `libero_pro/` | [Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) | `eafdb809426b13153aa1e4c42d6601844217dfec` |

Recreate OpenPI with its [setup guide](../docs/openpi.md). For FlashSAC, run
`bash scripts/fetch_flashsac.sh` from the repository root; the loader uses its
PyTorch source files without installing the upstream training stack.

DICE uses a local adaptation and checks the upstream revision for provenance.
To create its reference checkout when absent, run from the repository root:

```bash
git clone https://github.com/zhanyisun/dice-rl.git third_party/dice_rl_upstream
git -C third_party/dice_rl_upstream checkout --detach 98d936e52523878f0172e606ea32403176497100
```

Do not install that upstream repository's dependency stack into OpenPI. The
adaptation's license notice is in `docs/dice-upstream-LICENSE.txt` and is also
copied into Docker images. Follow the [DICE guide](../docs/dice.md) for data setup.

The Plus/PRO forks additionally require their task assets, state files, and
separate simulator path configurations. Their commits alone are insufficient to
run the queue; see the [benchmark protocol](../docs/robustness-evaluation.md).
The RLDT implementation is local Python source audited against the commit linked
in its [guide](../docs/rldt.md); it does not import a separate RLDT checkout.

