#!/usr/bin/env python3
"""Real-observation technical proof for isolated native layer-window knockouts."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.layer_knockout import (
    capture_layer_velocity,
    layer_keep_mask,
    sample_with_layer_knockout,
    velocity_with_layer_knockout,
)
from rl_vla.diagnostics.pi05_attention import (
    capture_prefix,
    key_keep_mask,
    load_observation,
    native_sample,
    native_velocity,
    sample_with_intervention,
)
from rl_vla.rldt_pi05 import Pi05Flow


def comparison(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {"exact": bool(np.array_equal(a, b)), "max_abs_error": float(np.max(np.abs(a - b))),
            "l2_error": float(np.linalg.norm((a - b).ravel())), "shape": list(a.shape)}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert", type=Path)
    parser.add_argument("--seed", type=int, default=100000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import jax
    import jax.numpy as jnp

    paths = [Path(__file__), Path(__file__).resolve().parents[1] /
             "src/rl_vla/diagnostics/layer_knockout.py", Path(__file__).resolve().parents[1] /
             "src/rl_vla/diagnostics/pi05_attention.py", args.observation, args.noise]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(args.output / "source_hashes.json", hashes)
    print("Loading native pi05_libero", flush=True)
    flow = Pi05Flow.from_config({"model_config": "pi05_libero"})
    params = flow.load_params(args.expert) if args.expert else flow.params
    raw = load_observation(args.observation)
    context, _, metadata = capture_prefix(flow, raw, capture_attention=False)
    save(args.output / "metadata.json", metadata)
    noise = np.load(args.noise, allow_pickle=False).astype(np.float32)
    if noise.ndim == 2:
        noise = noise[None]
    if noise.shape != (1, flow.horizon, flow.action_dim):
        raise ValueError(f"Invalid frozen initial noise shape {noise.shape}")
    key = jax.random.key(args.seed)
    x, t = jnp.asarray(noise), jnp.ones((1,), dtype=jnp.float32)
    fidelity, arrays = {}, {"noise": noise}
    print("Checking exact native velocity and both ten-step sampler paths", flush=True)
    reference_velocity = native_velocity(flow, params, context, x, t)
    noop_velocity = velocity_with_layer_knockout(flow, params, context, x, t, metadata)
    fidelity["velocity"] = comparison(reference_velocity, noop_velocity)
    arrays["native_velocity"], arrays["noop_velocity"] = reference_velocity, np.asarray(noop_velocity)
    reference = {}
    for label, explicit_noise in (("jax_noise", None), ("persisted_noise", noise)):
        original = native_sample(flow, params, context, args.seed, initial_noise=explicit_noise)
        adapted = sample_with_layer_knockout(flow, params, context, key, metadata,
                                             initial_noise=explicit_noise)
        for name, expected, actual in zip(("action", "prestep_path"), original, adapted, strict=True):
            fidelity[f"{label}_{name}"] = comparison(expected, actual)
            arrays[f"native_{label}_{name}"] = np.asarray(expected)
            arrays[f"noop_{label}_{name}"] = np.asarray(actual)
        reference[label] = original
    captured, attention = capture_layer_velocity(flow, params, context, x, t, metadata)
    fidelity["captured_velocity"] = comparison(reference_velocity, captured)
    arrays["noop_attention"] = attention
    save(args.output / "fidelity.json", fidelity)
    np.savez_compressed(args.output / "fidelity_arrays.npz", **arrays)
    if not all(row["exact"] for row in fidelity.values()):
        save(args.output / "summary.json", {"passed": False, "technical_only": True,
             "reason": "Native no-op was not bit exact", "fidelity": fidelity})
        raise RuntimeError("Native no-op fidelity failed; no intervention effects interpreted")

    print("Exact no-op passed; comparing all-layer interventions to existing native adapter", flush=True)
    for mode in ("no_image", "no_text"):
        expected = sample_with_intervention(flow, params, context, key, metadata,
                                            knockout=mode, initial_noise=noise)
        actual = sample_with_layer_knockout(flow, params, context, key, metadata,
                                            knockout=mode, initial_noise=noise)
        for name, a, b in zip(("action", "prestep_path"), expected, actual, strict=True):
            fidelity[f"all_layers_{mode}_{name}"] = comparison(a, b)
    save(args.output / "fidelity.json", fidelity)
    if not all(row["exact"] for row in fidelity.values()):
        save(args.output / "summary.json", {"passed": False, "technical_only": True,
             "reason": "All-layer adapter equivalence failed", "fidelity": fidelity})
        raise RuntimeError("All-layer adapter equivalence failed")

    baseline = np.asarray(reference["persisted_noise"][0])
    decoded_baseline = flow.decode(baseline, [raw])
    effects, intervention_arrays = [], {"baseline_action": baseline, "baseline_decoded": decoded_baseline}
    for name, layers in (("early", list(range(6))), ("middle", list(range(6, 12))),
                         ("late", list(range(12, 18)))):
        for mode in ("no_image", "no_text"):
            print(f"Measuring {name} {mode}", flush=True)
            action, path = sample_with_layer_knockout(flow, params, context, key, metadata,
                                                      knockout=mode, layers=layers, initial_noise=noise)
            action, path = np.asarray(action), np.asarray(path)
            velocity, probs = capture_layer_velocity(flow, params, context, x, t, metadata,
                                                      knockout=mode, layers=layers)
            captured_reference = velocity_with_layer_knockout(flow, params, context, x, t, metadata,
                                                               knockout=mode, layers=layers)
            if not np.array_equal(velocity, np.asarray(captured_reference)):
                raise RuntimeError("Attention capture changed intervention velocity")
            blocked_keys = ~key_keep_mask(metadata, mode)
            blocked_mass = float(probs[np.asarray(layers)][:, :, :, :, blocked_keys].sum())
            if blocked_mass != 0.0:
                raise RuntimeError("Blocked layer/key attention has nonzero mass")
            decoded = flow.decode(action, [raw])
            effects.append({"window": name, "layers": layers, "knockout": mode,
                            "normalized_action_l2": float(np.linalg.norm(action - baseline)),
                            "decoded_command_l2": float(np.linalg.norm(decoded - decoded_baseline)),
                            "velocity_l2": float(np.linalg.norm(velocity - reference_velocity)),
                            "selected_layer_blocked_attention_mass": blocked_mass,
                            "capture_velocity_exact": True,
                            "unselected_layer_blocked_key_mass": float(
                                probs[np.asarray([i for i in range(18) if i not in layers])]
                                [:, :, :, :, blocked_keys].sum())})
            prefix = f"{name}_{mode}"
            intervention_arrays.update({f"{prefix}_action": action, f"{prefix}_prestep_path": path,
                                       f"{prefix}_decoded": decoded, f"{prefix}_velocity": velocity,
                                       f"{prefix}_attention": probs,
                                       f"{prefix}_mask": layer_keep_mask(metadata, mode, layers)})
    np.savez_compressed(args.output / "interventions.npz", **intervention_arrays)
    save(args.output / "summary.json", {"passed": True, "complete": True, "technical_only": True,
         "fidelity": fidelity, "effects": effects, "observation_count": 1, "noise_count": 1,
         "observation": str(args.observation), "noise": str(args.noise),
         "expert": str(args.expert) if args.expert else "pi05_libero", "seed": args.seed,
         "steps": 10, "source_sha256": hashes, "elapsed_seconds": time.monotonic() - started,
         "interpretation": "One real observation technical validation and secondary open-loop action effects; no closed-loop outcome or generalization inference."})
    print("PASS: native no-op and all-layer controls exact; six layer-window interventions audited", flush=True)


if __name__ == "__main__":
    main()
