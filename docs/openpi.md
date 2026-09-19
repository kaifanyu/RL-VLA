# Connect a frozen openpi policy

The RL actor learns bounded corrections around a frozen action chunk. Openpi runs in a separate GPU process; the lightweight RL client uses NumPy over a websocket. The included mock policy needs no model weights. No weights have been downloaded.

For container deployment, use the [Docker GPU service](docker.md#optional-local-gpu-policy-service). It builds the pinned server separately from the learner, applies the required Transformers replacement, and includes startup checks and HTTP readiness. Its checkpoint must be mounted locally. The instructions below remain the manual source-environment alternative.

## 1. Install the pinned server source

Use a Linux GPU machine or Ubuntu under WSL2 for the server. Upstream tests Ubuntu 22.04 and does not support native Windows. Its stated inference memory requirement is **more than 8 GB**, before adding the simulator and RL learner. WSL2 is a deployment suggestion, not an upstream-tested guarantee. See the [upstream requirements and installation](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md#requirements).

The baseline includes the source checkout in `third_party/openpi`; no weights are included. Use **Python 3.11** for both environments. The following are **Bash** commands for preparing the server. Start at the RL-VLA repository root, with Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```bash
RL_VLA_ROOT="$PWD"
mkdir -p third_party
# Only when the source checkout is absent (for example, on a new machine):
if [ ! -d third_party/openpi ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/Physical-Intelligence/openpi.git third_party/openpi
fi
cd third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 git checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python 3.11
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

This creates `third_party/openpi/.venv`, separate from the RL learner's root `.venv`. The commit was verified against upstream on 2026-09-19; it matches the supplied roadmap and was upstream HEAD at inspection. The source pin is an integration reference, not a claim that GPU deployment has been tested here.

## 2. Download the checkpoint when ready

The concrete example is `pi05_libero`, which is trained for LIBERO's controller, observations and tasks. It is not a universal robot checkpoint. The [official checkpoint table](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md#model-checkpoints) lists alternatives.

| Setting | Value |
|---|---|
| Training/model config | `pi05_libero` |
| Official checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` |
| Local checkpoint directory | `RL-VLA/checkpoints/openpi/pi05_libero/` |
| External chunk | `10 x 7` |
| Example executed prefix | `5` actions |

When you choose to download, install Google Cloud CLI's `gsutil`, then run in the same Bash session. This follows the download command used by [upstream's downloader](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/shared/download.py):

```bash
mkdir -p "$RL_VLA_ROOT/checkpoints/openpi/pi05_libero"
gsutil -m cp -r 'gs://openpi-assets/checkpoints/pi05_libero/*' "$RL_VLA_ROOT/checkpoints/openpi/pi05_libero/"
```

Preserve the full checkpoint, including `params/` and `assets/`; its normalization statistics are required. Download completion is required before serving this local path. Alternatively, serving the `gs://` URI directly makes openpi download automatically into `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero`. `OPENPI_DATA_HOME` changes the cache root, not the complete checkpoint path.

## 3. Start the server and install the client

In the server environment:

```bash
cd "$RL_VLA_ROOT/third_party/openpi"
uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir="$RL_VLA_ROOT/checkpoints/openpi/pi05_libero"
```

In a separate terminal, activate the root RL environment and install only the lightweight client:

```bash
uv pip install -e third_party/openpi/packages/openpi-client
uv pip install 'websockets>=14,<16'
```

On Windows, activate the root environment with `.\.venv\Scripts\Activate.ps1`; on Linux use `source .venv/bin/activate`. Run the client installation from the root, using Python 3.11 and NumPy below version 2 (as required by the root project and upstream client). Its connection constructor waits and retries if the server is unavailable, so start the server first. The source [server](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/scripts/serve_policy.py) listens on all interfaces; use localhost or your trusted private machine connection.

The Python API used by this baseline is:

```python
from rl_vla.base_policy import make_base_policy

base = make_base_policy(
    {"kind": "openpi", "host": "localhost", "port": 8000},
    action_dim=7,
    horizon=10,
)
chunk = base.infer({"openpi": checkpoint_observation})  # float32 [10, 7]
```

Replace the `base` section in your RL TOML configuration and use the matching environment adapter:

```toml
[base]
kind = "openpi"
host = "localhost"
port = 8000
```

The mock environment is a software diagnostic; its observations do not substitute for a LIBERO task.

## 4. Match the checkpoint's observation and action contract

For `pi05_libero`, the environment supplies this dictionary under observation key `openpi`:

```python
checkpoint_observation = {
    "observation/image": external_rgb,       # uint8 [224, 224, 3]
    "observation/wrist_image": wrist_rgb,    # uint8 [224, 224, 3]
    "observation/state": state,             # float32 [8], unnormalized
    "prompt": "pick up the red block",      # actual task instruction
}
```

The eight state values are end-effector XYZ, three axis-angle rotation values, and two gripper joint positions. For raw LIBERO images, rotate both camera images 180 degrees, then use `openpi_client.image_tools.resize_with_pad` and `convert_to_uint8`. This is a LIBERO-specific training convention, not a transform to apply indiscriminately to another robot. Follow the [official LIBERO example](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py).

The server handles checkpoint normalization and unnormalization. Apply the residual to the **returned controller-space actions**, with bounds defined in those same units. In LIBERO these are six delta pose controller coordinates plus one gripper command. Do not apply an additional delta transform or treat the seven coordinates as seven joint positions. The checkpoint internally pads actions to 32 dimensions, but its [LIBERO output adapter](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/libero_policy.py) already returns seven.

The wrapper rejects non-finite actions, a chunk shorter than the configured horizon, and the wrong action dimension. It never silently slices padded dimensions. Other checkpoint adapters may explicitly configure `action_indices` (one distinct index per environment action coordinate). A `[base.observation_map]` table with entries such as `"observation/state" = "robot_state"` selects environment dictionary keys instead of using a nested `openpi` payload; supply all inputs required by that checkpoint. Without either mapping or a nested payload, the entire observation dictionary is sent.

At each episode reset, discard any pending action chunk and call `base.reset()`. The upstream websocket client's `reset()` is a no-op; it does not reset server RNG. The ordinary server API also does not let the client seed each inference. Record this when comparing runs: environment seeding alone does not make base action sampling identical. PPO must reuse the stored sampled nominal chunk when recomputing likelihoods.

## 5. Scale after the contract is correct

This is a serial correctness baseline: one observation request returns one CPU/NumPy chunk. Upstream `Policy.infer` inserts its own single-example batch. Do not pass batched images through this API and assume batched GPU execution. A future MuJoCo Warp collector should preserve these transforms while adding batched, device-resident inference and per-environment reset/final-observation handling. Keep that optimization separate from the initial SAC/PPO comparison.

Direct VLA weight updates, flow-policy likelihoods, GPU-batched Warp training and visual robustness claims are outside this frozen-policy connector. The supplied roadmap describes those later stages.
