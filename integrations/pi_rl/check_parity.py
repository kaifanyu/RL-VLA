"""Compare native JAX RLDT inference and upstream Torch pi-RL, with shared noise.

Run ``native`` and ``torch`` separately in their respective environments, then
``compare``. An NPZ fixture stores original OpenPI camera/state observations.
The model's inference sampler remains upstream ODE with 10 denoising steps.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from model_configs import model_config
from model_contract import enforce_trainable_scope


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_provenance(checkpoint: Path, fixture: Path, output: Path) -> dict:
    checkpoint, fixture, output = checkpoint.resolve(), fixture.resolve(), output.resolve()
    manifest_path = checkpoint / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model_hash = file_sha256(checkpoint / "model.safetensors")
    if model_hash != manifest["weights_sha256"]:
        raise ValueError("Converted checkpoint weights no longer match conversion manifest")
    return {
        "fixture": {"path": str(fixture), "sha256": file_sha256(fixture)},
        "checkpoint": {
            "path": str(checkpoint), "source_checkpoint": manifest["source_checkpoint"],
            "conversion_manifest_path": str(manifest_path),
            "conversion_manifest_sha256": file_sha256(manifest_path),
            "model_sha256": model_hash,
        },
        "output_artifact": {"path": str(output), "sha256": file_sha256(output)},
    }


def payload_from_fixture(path: Path) -> tuple[dict, np.ndarray]:
    with np.load(path, allow_pickle=False) as fixture:
        return {
            "observation/image": fixture["image"],
            "observation/wrist_image": fixture["wrist_image"],
            "observation/state": fixture["state"],
            "prompt": str(fixture["prompt"].item()),
        }, fixture["noise"]


def write_result(output: Path, transformed: dict, velocity, actions, decoded) -> None:
    arrays = {"first_velocity": np.asarray(velocity, np.float32),
              "normalized_actions": np.asarray(actions, np.float32),
              "decoded_actions": np.asarray(decoded, np.float32),
              "state": np.asarray(transformed["state"]),
              "tokenized_prompt": np.asarray(transformed["tokenized_prompt"]),
              "tokenized_prompt_mask": np.asarray(transformed["tokenized_prompt_mask"])}
    for name, value in transformed["image"].items():
        arrays[f"image_{name}"] = np.asarray(value)
    np.savez_compressed(output, **arrays)


def native(args) -> None:
    import jax
    import jax.numpy as jnp

    from rl_vla.rldt_pi05 import Pi05Flow

    flow = Pi05Flow.from_config({"checkpoint": args.checkpoint, "model_config": "pi05_libero"})
    payload, noise = payload_from_fixture(Path(args.fixture))
    transformed = flow._transform_observation(payload)
    context = flow.prepare([payload])

    @jax.jit
    def forward(params, frozen, context, x):
        velocity = flow.velocity(params, context, x, jnp.ones((1,)), frozen=frozen)

        def step(value, index):
            t = jnp.full((1,), 1.0 - index / 10, dtype=jnp.float32)
            v = flow.velocity(params, context, value, t, frozen=frozen)
            return value - v / 10, None

        actions, _ = jax.lax.scan(step, x, jnp.arange(10, dtype=jnp.float32))
        return velocity, actions

    velocity, actions = forward(flow.params, flow.frozen, context, jnp.asarray(noise))
    decoded = flow.decode(actions, [payload])
    write_result(Path(args.output), transformed, velocity, actions, decoded)
    print(json.dumps({"backend": "jax", "devices": [str(d) for d in jax.devices()],
                      "output": args.output}), flush=True)


def torch_inference(args) -> None:
    import torch
    from omegaconf import OmegaConf
    from openpi.models.model import Observation
    from rlinf.models.embodiment.openpi import get_model
    from torch.utils._pytree import tree_map

    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = OmegaConf.create(model_config(args.checkpoint, "flow_sde"))
    model = get_model(cfg)
    counts = enforce_trainable_scope(model)
    model.eval().to(args.device)
    payload, noise = payload_from_fixture(Path(args.fixture))
    transformed = model._input_transform(copy.deepcopy(payload))
    batch = tree_map(lambda value: torch.as_tensor(np.asarray(value)[None].copy(),
                                                  device=args.device), transformed)
    observation = Observation.from_dict(batch)
    with torch.no_grad():
        images, masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
        _, prefix_masks, cache = model._build_prefix_cache(images, masks, tokens, token_masks)
        noise_t = torch.from_numpy(noise).to(args.device)
        velocity, _ = model.get_velocity(state, noise_t, torch.ones((1,), device=args.device),
                                        prefix_masks, cache)
        actions = model.sample_actions(observation, noise=noise_t, mode="eval", compute_values=False)["actions"]
    actions_np = actions.float().cpu().numpy()
    decoded = model._output_transform({"state": transformed["state"],
                                       "actions": actions_np[0]})["actions"][None]
    write_result(Path(args.output), transformed, velocity.float().cpu().numpy(), actions_np, decoded)
    actual_device = actions.device
    report = {"backend": "torch", "device": args.device,
              "actual_device": str(actual_device),
              "device_name": torch.cuda.get_device_name(actual_device) if actual_device.type == "cuda" else "cpu",
              "torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
              "parameter_counts": counts, "output": args.output,
              **inference_provenance(Path(args.checkpoint), Path(args.fixture), Path(args.output))}
    metadata_path = Path(args.output).with_suffix(".json")
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(metadata_path)
    print(json.dumps(report), flush=True)


def compare(args) -> None:
    result = {"inputs_identical": True, "finite": True, "outputs": {}}
    with np.load(args.native, allow_pickle=False) as native_data, np.load(args.torch, allow_pickle=False) as torch_data:
        for key in native_data.files:
            left, right = native_data[key], torch_data[key]
            if left.shape != right.shape:
                raise ValueError(f"Shape mismatch for {key}: {left.shape} vs {right.shape}")
            if key in {"first_velocity", "normalized_actions", "decoded_actions"}:
                difference = np.asarray(left, np.float64) - np.asarray(right, np.float64)
                result["outputs"][key] = {
                    "max_abs_difference": float(np.max(np.abs(difference))),
                    "mean_abs_difference": float(np.mean(np.abs(difference))),
                    "rms_difference": float(np.sqrt(np.mean(difference**2))),
                }
                result["finite"] &= bool(np.isfinite(left).all() and np.isfinite(right).all())
            else:
                result["inputs_identical"] &= bool(np.array_equal(left, right))
        delta = native_data["decoded_actions"][:, :5] - torch_data["decoded_actions"][:, :5]
        result["executed_prefix_max_abs_difference"] = float(np.max(np.abs(delta)))
        result["executed_prefix_mean_abs_difference"] = float(np.mean(np.abs(delta)))
        result["executed_gripper_signs_identical"] = bool(np.array_equal(
            np.sign(native_data["decoded_actions"][:, :5, 6]),
            np.sign(torch_data["decoded_actions"][:, :5, 6])))
    # Declared before observing Torch output. Controls use [-1,1] simulator
    # units: mean difference <= .01 and every coordinate <= .05. Also require
    # the same gripper direction. This rejects gross conversion/transform
    # errors; a separate Torch baseline still measures backend behavior.
    result["tolerances"] = {"executed_prefix_mean_abs": .01,
                            "executed_prefix_max_abs": .05,
                            "first_velocity_rms": .05}
    result["parity_passed"] = bool(
        result["inputs_identical"] and result["finite"]
        and result["executed_gripper_signs_identical"]
        and result["executed_prefix_mean_abs_difference"] <= .01
        and result["executed_prefix_max_abs_difference"] <= .05
        and result["outputs"]["first_velocity"]["rms_difference"] <= .05)
    result["note"] = "Shared initial noise; bfloat16 backend differences are expected. This is an inference check, not task-success equivalence."
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["parity_passed"]:
        raise RuntimeError("Cross-framework preprocessing/action parity exceeded its declared tolerance")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("native", "torch"):
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--fixture", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--device", default="cuda")
    sub = commands.add_parser("compare")
    sub.add_argument("--native", required=True)
    sub.add_argument("--torch", required=True)
    sub.add_argument("--output", required=True)
    args = parser.parse_args()
    {"native": native, "torch": torch_inference, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
