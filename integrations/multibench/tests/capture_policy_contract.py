"""Capture preprocessing in each real policy interpreter for parity comparison.

No model weights, GPU, simulator, or synthetic reward is used. The fixture
exercises the native state widths, all camera slots, tokenizer, normalization,
and output actions in both installed OpenPI runtimes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from multibench.policy_spec import BENCHMARKS, build_transforms, model_config


def observation(benchmark):
    width = BENCHMARKS[benchmark]["state_dim"]
    state = np.linspace(-0.2, 0.3, width, dtype=np.float32)
    if benchmark == "robotwin":
        state[[6, 13]] = [0.25, 0.75]
    elif benchmark == "robocasa":
        state[[3, 4, 5, 6]] = [0, 0, 0, 1]
        state[[10, 11, 12, 13]] = [0, 0, 0, 1]
        state[[14, 15]] = [0.03, -0.03]
    else:
        state[-2:] = [0.02, -0.02]
    height, width = (480, 640) if benchmark == "maniskill" else (224, 224)
    y, x = np.mgrid[:height, :width]
    image = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)
    return {
        "observation/state": state,
        "observation/image": image,
        "observation/wrist_image": np.roll(image, 17, axis=1),
        "observation/right_wrist_image": np.roll(image, 23, axis=0),
        "prompt": "Pick up the object and place it in the target.",
    }


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        path = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, path + "/"))
        else:
            result[path] = np.asarray(item)
    return result


def capture(output):
    output.mkdir(parents=True, exist_ok=True)
    arrays, metadata = {}, {}
    cfg = model_config()
    assert cfg.pi05 and cfg.action_horizon == 50 and cfg.action_dim == 32
    assert cfg.discrete_state_input is True
    for benchmark, spec in BENCHMARKS.items():
        inputs, outputs, calibration = build_transforms(benchmark, cfg)
        transformed = inputs(copy.deepcopy(observation(benchmark)))
        assert transformed["state"].shape == (32,)
        assert sum(bool(mask) for mask in transformed["image_mask"].values()) == spec["images"]
        normalized = np.linspace(-0.7, 0.7, 50*32, dtype=np.float32).reshape(50, 32)
        decoded = outputs({"state": transformed["state"].copy(), "actions": normalized.copy()})
        assert decoded["actions"].shape == (50, spec["action_dim"])
        assert np.isfinite(decoded["actions"]).all()
        arrays.update(flatten(transformed, benchmark + "/input/"))
        arrays[benchmark + "/output/actions"] = decoded["actions"]
        metadata[benchmark] = calibration
    forbidden = [name for name in sys.modules if name.split(".")[0] in {"robocasa", "robosuite", "mani_skill", "sapien"}]
    assert not forbidden, f"Simulator imports leaked into policy interpreter: {forbidden}"
    np.savez(output / "fixture.npz", **arrays)
    record = {
        "python": sys.executable, "benchmarks": metadata,
        "model": {"pi05": cfg.pi05, "action_horizon": cfg.action_horizon,
                  "action_dim": cfg.action_dim, "discrete_state_input": cfg.discrete_state_input,
                  "max_token_len": cfg.max_token_len},
        "fixture_sha256": hashlib.sha256((output / "fixture.npz").read_bytes()).hexdigest(),
        "arrays": {key: {"shape": list(a.shape), "dtype": str(a.dtype)} for key, a in arrays.items()},
    }
    (output / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output": str(output), "arrays": len(arrays), "passed": True}), flush=True)


def compare(first, second, report):
    left = np.load(first / "fixture.npz", allow_pickle=False)
    right = np.load(second / "fixture.npz", allow_pickle=False)
    assert set(left.files) == set(right.files)
    result = {"passed": True, "first": str(first), "second": str(second), "arrays": {}}
    for key in left.files:
        a, b = left[key], right[key]
        assert a.shape == b.shape, key
        if np.issubdtype(a.dtype, np.floating):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6, err_msg=key)
            error = float(np.max(np.abs(a-b)))
        else:
            np.testing.assert_array_equal(a, b, err_msg=key)
            error = 0.0
        result["arrays"][key] = {"shape": list(a.shape), "maximum_absolute_error": error}
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "arrays": len(left.files), "report": str(report)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare, args.report)
    else:
        capture(args.output)
