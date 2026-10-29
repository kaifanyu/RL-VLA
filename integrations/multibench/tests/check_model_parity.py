"""Shared-noise numeric inference parity of raw pi05_base across all embodiments.

Fixtures are synthetic and explicitly test inference/conversion, not task
success. Each backend loads the same policy weights once and uses each task's
verified transform. Training weights are never updated.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from multibench.policy_spec import BENCHMARKS, ROOT, build_transforms, model_config
from multibench.tests.capture_policy_contract import flatten, observation

ORDER = ("robocasa", "robotwin", "maniskill")
TOLERANCES = {"executed_normalized_mean_abs": .01,
              "executed_normalized_max_abs": .05,
              "first_velocity_rms": .05}


def noise_for(benchmark):
    return np.random.default_rng(20261028 + ORDER.index(benchmark)).normal(size=(1, 50, 32)).astype(np.float32)


def write(output, benchmark, transformed, velocity, actions, decoded):
    arrays = flatten(transformed, "input/")
    arrays.update(first_velocity=np.asarray(velocity, np.float32),
                  normalized_actions=np.asarray(actions, np.float32),
                  decoded_actions=np.asarray(decoded, np.float32), noise=noise_for(benchmark))
    path = output / f"{benchmark}.npz"
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def native(output):
    import jax
    import jax.numpy as jnp
    from rl_vla.rldt_pi05 import Pi05Flow

    start = time.perf_counter()
    flow = Pi05Flow.from_config({"checkpoint": str(ROOT / "checkpoints/openpi/pi05_base"), "benchmark": ORDER[0]})

    @jax.jit
    def forward(params, frozen, context, x):
        first = flow.velocity(params, context, x, jnp.ones((1,)), frozen=frozen)
        def step(value, index):
            t = jnp.full((1,), 1.0-index/10, dtype=jnp.float32)
            v = flow.velocity(params, context, value, t, frozen=frozen)
            return value-v/10, None
        actions, _ = jax.lax.scan(step, x, jnp.arange(10, dtype=jnp.float32))
        return first, actions

    records = {}
    for benchmark in ORDER:
        flow._input_transform, flow._output_transform, _ = build_transforms(benchmark, model_config())
        flow.motor_dim = BENCHMARKS[benchmark]["action_dim"]
        raw = observation(benchmark)
        transformed = flow._transform_observation(raw)
        context = flow.prepare([raw])
        velocity, actions = forward(flow.params, flow.frozen, context, jnp.asarray(noise_for(benchmark)))
        decoded = flow.decode(actions, [raw])
        records[benchmark] = write(output, benchmark, transformed, velocity, actions, decoded)
        print(json.dumps({"backend": "native", "benchmark": benchmark, "elapsed": time.perf_counter()-start}), flush=True)
    return {"devices": [str(d) for d in jax.devices()], "records": records, "elapsed_seconds": time.perf_counter()-start}


def torch_backend(output):
    import torch
    from omegaconf import OmegaConf
    from openpi.models.model import Observation
    from torch.utils._pytree import tree_map
    from multibench.torch_model import get_model

    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = OmegaConf.create({"benchmark": ORDER[0],
        "model_path": str(ROOT / "checkpoints/openpi/pi05_base_rlinf"),
        "openpi": {"action_chunk": 5, "num_steps": 10, "add_value_head": True,
                   "value_after_vlm": True, "detach_critic_input": True}})
    model = get_model(cfg).eval().to("cuda")
    records = {}
    for benchmark in ORDER:
        spec = BENCHMARKS[benchmark]
        model.config = dataclasses.replace(model.config, action_env_dim=spec["action_dim"], num_images_in_input=spec["images"])
        model._input_transform, model._output_transform, _ = build_transforms(benchmark, model.config)
        transformed = model._input_transform(copy.deepcopy(observation(benchmark)))
        batch = tree_map(lambda v: torch.as_tensor(np.asarray(v)[None].copy(), device="cuda"), transformed)
        obs = Observation.from_dict(batch)
        with torch.no_grad():
            images, masks, tokens, token_masks, state = model._preprocess_observation(obs, train=False)
            _, prefix_masks, cache = model._build_prefix_cache(images, masks, tokens, token_masks)
            noise = torch.from_numpy(noise_for(benchmark)).to("cuda")
            velocity, _ = model.get_velocity(state, noise, torch.ones((1,), device="cuda"), prefix_masks, cache)
            actions = model.sample_actions(obs, noise=noise, mode="eval", compute_values=False)["actions"]
        result = actions.float().cpu().numpy()
        decoded = model._output_transform({"state": transformed["state"].copy(), "actions": result[0].copy()})["actions"][None]
        records[benchmark] = write(output, benchmark, transformed, velocity.float().cpu().numpy(), result, decoded)
        print(json.dumps({"backend": "torch", "benchmark": benchmark, "elapsed": time.perf_counter()-start}), flush=True)
    return {"device": torch.cuda.get_device_name(), "records": records,
            "elapsed_seconds": time.perf_counter()-start, "provenance": model.multibench_metadata}


def compare(first, second, output):
    report = {"fixture_type": "synthetic inference fixtures; not task-success evaluation", "tolerances": TOLERANCES,
              "benchmarks": {}, "passed": True}
    for benchmark in ORDER:
        a = np.load(first / f"{benchmark}.npz", allow_pickle=False)
        b = np.load(second / f"{benchmark}.npz", allow_pickle=False)
        assert set(a.files) == set(b.files)
        identical = all(np.array_equal(a[k], b[k]) for k in a.files if k.startswith("input/") or k == "noise")
        finite = all(np.isfinite(a[k]).all() and np.isfinite(b[k]).all() for k in ("first_velocity", "normalized_actions", "decoded_actions"))
        motor_dim = BENCHMARKS[benchmark]["action_dim"]
        delta = a["normalized_actions"][:, :5, :motor_dim] - b["normalized_actions"][:, :5, :motor_dim]
        vd = a["first_velocity"].astype(np.float64) - b["first_velocity"].astype(np.float64)
        dd = a["decoded_actions"][:, :5] - b["decoded_actions"][:, :5]
        axes = {"robocasa": [6, 11], "robotwin": [], "maniskill": [6]}[benchmark]
        discrete_match = all(np.array_equal(a["decoded_actions"][:, :5, axis] > .5,
                                            b["decoded_actions"][:, :5, axis] > .5) for axis in axes)
        entry = {"inputs_identical": identical, "finite": finite, "controller_discrete_commands_identical": discrete_match,
                 "executed_normalized_mean_abs": float(np.abs(delta).mean()),
                 "executed_normalized_max_abs": float(np.abs(delta).max()),
                 "first_velocity_rms": float(np.sqrt(np.mean(vd**2))),
                 "executed_decoded_mean_abs": float(np.abs(dd).mean()),
                 "executed_decoded_max_abs": float(np.abs(dd).max())}
        entry["passed"] = bool(identical and finite and discrete_match and all(entry[key] <= limit for key, limit in TOLERANCES.items()))
        report["benchmarks"][benchmark] = entry
        report["passed"] &= entry["passed"]
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("Cross-backend inference parity exceeded predeclared tolerances")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["native", "torch", "compare"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    args = parser.parse_args()
    if args.backend == "compare":
        compare(args.first, args.second, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    result = (native if args.backend == "native" else torch_backend)(args.output)
    result.update(backend=args.backend, fixture_type="synthetic inference fixtures; not task-success evaluation", tolerances=TOLERANCES)
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
