"""Compute calibration statistics from 50 official training demonstrations.

This never loads or updates model weights and never reads evaluation rollouts.
The same pinned policy functions are used to map states and actions. Loading
their AST avoids importing the training GPU stack in the simulator environment.
"""
import ast
import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "third_party/rlinf/rlinf/models/embodiment/openpi/policies/aloha_policy.py"
ARCHIVE = HERE / "handover_block_demo_clean.zip"
EXPECTED = "9959f4daab727215a06bad6b44821bfdd38b9c3ba9d6a4eca074fe58f4d920e3"
HORIZON = 50
DELTA_MASK = np.array([True] * 6 + [False] + [True] * 6 + [False])


def load_transform_functions():
    selected = {"_joint_flip_mask", "_normalize", "_unnormalize", "_gripper_to_angular",
                "_gripper_from_angular", "_gripper_from_angular_inv", "_decode_state",
                "_encode_actions", "_encode_actions_inv"}
    tree = ast.parse(SOURCE.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    assert {node.name for node in tree.body} == selected
    namespace = {"np": np}
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def native_array(h5, kind):
    return np.concatenate([h5[f"{kind}/{arm}_{name}"][:] for arm in ("left", "right")
                           for name in ("arm_joint_states", "ee_joint_states")], axis=-1)


def stats(values):
    assert values.ndim == 2 and values.shape[1] == 14 and np.isfinite(values).all()
    return {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist()}


def main():
    assert sha256(ARCHIVE) == EXPECTED
    functions = load_transform_functions()
    states, actions, episodes = [], [], []
    max_roundtrip_error = 0.
    with zipfile.ZipFile(ARCHIVE) as archive:
        filenames = sorted(name for name in archive.namelist() if name.endswith(".hdf5"))
        assert len(filenames) == 50
        for name in filenames:
            with h5py.File(io.BytesIO(archive.read(name)), "r") as h5:
                native_state, native_action = native_array(h5, "state"), native_array(h5, "action")
                assert native_state.shape == native_action.shape
                state = np.stack([functions["_decode_state"](x.copy(), adapt_to_pi=True) for x in native_state])
                encoded = functions["_encode_actions_inv"](native_action.copy(), adapt_to_pi=True)
                recovered = functions["_encode_actions"](encoded.copy(), adapt_to_pi=True)
                max_roundtrip_error = max(max_roundtrip_error, float(np.max(np.abs(recovered - native_action))))
                indices = np.minimum(np.arange(len(state))[:, None] + np.arange(HORIZON)[None, :], len(state) - 1)
                chunks = encoded[indices].copy()
                chunks[:, :, DELTA_MASK] -= state[:, None, DELTA_MASK]
                states.append(state)
                actions.append(chunks.reshape(-1, 14))
                episodes.append({"archive_member": name, "frames": len(state)})
        seeds = archive.read("handover_block/aloha_agilex/seed.txt").decode()
    assert max_roundtrip_error < 1e-6
    (HERE / "norm_stats.json").write_text(json.dumps({"norm_stats": {
        "state": stats(np.concatenate(states)), "actions": stats(np.concatenate(actions)),
    }}, indent=2) + "\n")
    (HERE / "normalization_manifest.json").write_text(json.dumps({
        "dataset": "TianxingChen/RoboTwin2.0",
        "revision": "3dc3b798668feb99ac61cc9086d84cbcc3d79186",
        "file": "dataset/handover_block/demo_clean.zip", "archive_sha256": EXPECTED,
        "embodiment": "aloha-agilex", "demonstrations": 50,
        "use": "normalization calibration only; no SFT or demonstration training",
        "policy_transform_source": str(SOURCE), "policy_transform_sha256": sha256(SOURCE),
        "adapt_to_pi": True, "action_horizon": HORIZON,
        "action_chunk_padding": "repeat final action at episode boundary",
        "joint_delta_mask": DELTA_MASK.tolist(), "grippers": "absolute; official Aloha encode/decode",
        "joint_order": "left six, left gripper, right six, right gripper",
        "quantiles": "exact numpy q01/q99 over all training frames and padded horizon actions",
        "action_roundtrip_max_error": max_roundtrip_error,
        "norm_stats_sha256": sha256(HERE / "norm_stats.json"),
        "demonstration_seeds": seeds, "episodes": episodes,
    }, indent=2) + "\n")
    print(json.dumps({"episodes": len(episodes), "frames": sum(x["frames"] for x in episodes),
                      "roundtrip_max_error": max_roundtrip_error, "path": str(HERE / "norm_stats.json")}))


if __name__ == "__main__":
    main()
