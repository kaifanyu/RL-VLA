"""Check trainable scope without changing upstream pi-RL's policy or loss."""

from __future__ import annotations

NATIVE_EXPERT_PARAMETERS = 430_098_464
NATIVE_VLM_PARAMETERS = 2_923_335_408


def enforce_trainable_scope(model) -> dict:
    """Freeze VLM and the unused language head; count the actual action expert.

    OpenPI's Torch GemmaForCausalLM carries an unused vocabulary projection.
    The native JAX action expert has no such parameter. It is not part of
    action generation and must not enter the fine-tuning optimizer.
    """
    if not model.config.train_expert_only or not model.config.pi05:
        raise ValueError("This comparison requires pi05 with a frozen VLM")
    model.freeze_vlm()
    unused_head = model.paligemma_with_expert.gemma_expert.lm_head
    for parameter in unused_head.parameters():
        parameter.requires_grad_(False)
    counts = {name: 0 for name in ("action_expert", "frozen_vlm", "unused_head",
                                  "value_head", "noise_head")}
    for name, parameter in model.named_parameters():
        if name.startswith("paligemma_with_expert.paligemma."):
            group, expected_trainable = "frozen_vlm", False
        elif name.startswith("paligemma_with_expert.gemma_expert.lm_head."):
            group, expected_trainable = "unused_head", False
        elif name.startswith("value_head."):
            group, expected_trainable = "value_head", True
        elif name.startswith("noise_head."):
            group, expected_trainable = "noise_head", True
        else:
            group, expected_trainable = "action_expert", True
        if parameter.requires_grad != expected_trainable:
            raise ValueError(f"Wrong trainable scope: {name}, requires_grad={parameter.requires_grad}")
        counts[group] += parameter.numel()
    if counts["action_expert"] != NATIVE_EXPERT_PARAMETERS:
        raise ValueError(f"Action-expert parameter count changed: {counts}")
    if counts["frozen_vlm"] != NATIVE_VLM_PARAMETERS:
        raise ValueError(f"VLM parameter count changed: {counts}")
    counts["total_trainable"] = sum(counts[name] for name in
                                     ("action_expert", "value_head", "noise_head"))
    return counts
