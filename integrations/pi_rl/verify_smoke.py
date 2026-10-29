"""Verify actual action-expert updates and frozen VLM against initial weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def group_for_key(key):
    if key.startswith("paligemma_with_expert.paligemma."):
        return "frozen_vlm"
    if key.startswith("paligemma_with_expert.gemma_expert.lm_head."):
        return "unused_head"
    return "action_expert"


def metric_values(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from metric_values(item, f"{prefix}/{key}".strip("/"))
    elif isinstance(value, list):
        for item in value:
            yield from metric_values(item, prefix)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        yield prefix, float(value)


def verify(initial, trained, metrics_path, output):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(1)
    state = torch.load(trained, map_location="cpu", weights_only=True, mmap=True)
    if "model" in state:
        state = state["model"]
    results = {group: {"parameters": 0, "changed_parameters": 0,
                       "max_abs_change": 0., "squared_l2_change": 0., "finite": True}
               for group in ("frozen_vlm", "action_expert", "unused_head")}
    examples = []
    with safe_open(initial, framework="pt", device="cpu") as source:
        keys = source.keys()
        for key in keys:
            if key not in state:
                raise KeyError(f"Converted initial weight missing from saved checkpoint: {key}")
            original, updated = source.get_tensor(key), state[key]
            if original.shape != updated.shape:
                raise ValueError(f"Weight shape changed: {key}")
            group = group_for_key(key)
            record = results[group]
            record["parameters"] += original.numel()
            before, after = original.reshape(-1), updated.reshape(-1)
            for start in range(0, original.numel(), 1024**2):
                left, right = before[start:start + 1024**2], after[start:start + 1024**2]
                record["finite"] &= bool(torch.isfinite(right).all())
                changed = int((left.to(right.dtype) != right).sum())
                record["changed_parameters"] += changed
                if changed:
                    delta = right.double() - left.double()
                    record["max_abs_change"] = max(record["max_abs_change"], float(delta.abs().max()))
                    record["squared_l2_change"] += float(delta.square().sum())
                    if group == "action_expert" and len(examples) < 12:
                        examples.append({"name": key, "block_start": start, "changed_values": changed})
    auxiliary = {name: bool(torch.isfinite(value).all()) for name, value in state.items()
                 if name.startswith(("value_head.", "noise_head."))}
    rows = [json.loads(line) for line in Path(metrics_path).read_text().splitlines()]
    diagnostics = list(metric_values([row.get("actor_training_metrics", {}) for row in rows]))
    loss_grad = [(key, value) for key, value in diagnostics if "loss" in key or "grad_norm" in key]
    undefined_loss_grad = [name for row in rows for name in row.get("nonfinite_diagnostics", [])
                           if "loss" in name or "grad_norm" in name]
    finite_loss_grad = bool(loss_grad) and all(math.isfinite(value) for _, value in loss_grad)
    has_gradient = any("grad_norm" in key and value > 0 for key, value in loss_grad)
    passed = bool(
        results["action_expert"]["parameters"] == 430_098_464
        and results["frozen_vlm"]["parameters"] == 2_923_335_408
        and results["action_expert"]["changed_parameters"] > 0
        and results["frozen_vlm"]["changed_parameters"] == 0
        and all(record["finite"] for record in results.values())
        and all(auxiliary.values()) and finite_loss_grad and has_gradient
        and not undefined_loss_grad)
    result = {"smoke_passed": passed, "initial": str(initial), "checkpoint": str(trained),
              "groups": results, "changed_expert_examples": examples,
              "auxiliary_parameters_finite": all(auxiliary.values()),
              "finite_loss_and_gradient_metrics": finite_loss_grad,
              "positive_gradient_norm_observed": has_gradient,
              "undefined_loss_gradient_fields": undefined_loss_grad,
              "iterations_logged": len(rows)}
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("PPO smoke failed: require changed expert, unchanged VLM, finite losses/gradients")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True)
    parser.add_argument("--checkpoint", required=True, help="Actual saved actor full_weights.pt")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.initial, args.checkpoint, args.metrics, args.output)


if __name__ == "__main__":
    main()
