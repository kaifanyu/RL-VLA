# Dependencies

Upstream repositories are downloaded separately and excluded from Git. Full revisions
and URLs are recorded in [dependencies.lock.json](dependencies.lock.json).

| Checkout | Purpose | Revision |
|---|---|---|
| `openpi` | Native policy | `215abfb217db` |
| `libero_plus` | LIBERO-Plus | `4976dc30028e` |
| `libero_pro` | LIBERO-PRO | `eafdb809426b` |
| `rlinf` | Flow-SDE learner | `050543189957` |
| `rlinf_openpi` | RLinf policy | `c5dc4b9296a1` |
| `robotwin` | RoboTwin simulator | `dca9ec682688` |
| `robocasa365` | RoboCasa simulator | `dc73a8ce18e4` |
| `robocasa365/dependencies/robosuite` | RoboCasa robot simulator | `5ce6643f3092` |
| `robocasa365/dependencies/openpi_reference` | RoboCasa preprocessing reference | `5a6beda9ff99` |
| `maniskill3` | ManiSkill simulator | `33967b9e3ead` |
| `FlashSAC` | Optional residual baseline | `87edc9061150` |
| `dice_rl_upstream` | Optional DICE reference | `98d936e52523` |

## Restore a checkout

Use the URL, path, and full commit from the lock file:

```bash
git clone <url> <path>
git -C <path> checkout --detach <commit>
```

RLinf also needs the saved local compatibility patch:

```bash
git -C third_party/rlinf apply --check ../patches/rlinf.patch
git -C third_party/rlinf apply ../patches/rlinf.patch
```

The patch records lazy environment imports, benchmark action conversion, and the
action-expert FSDP integration. It preserves the working runtime without publishing
an entire upstream repository or discarding local changes.
The [RLinf license](patches/RLinf-LICENSE) accompanies the patch.

Simulator assets, Python environments, model weights, and normalization setup are
separate from source checkout. Follow the [GRASP guide](../docs/grasp-setup.md) and
[benchmark setup guides](../integrations/README.md). Keep upstream licenses intact.

The OpenPI and RLinf OpenPI revisions are intentionally different. Do not merge
their environments. The vendored multi-benchmark core is documented in the
[integration map](../integrations/README.md).
