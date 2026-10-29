"""Keep the native policy's frozen modules outside FSDP's flat parameters."""


def ignored_modules_for_native_openpi(model, model_config, *, device):
    """Return only the already-frozen VLM and unused expert vocabulary head.

    Torch 2.7 NO_SHARD with use_orig_params can leave Tensor views registered
    after backward when the root flat parameter is frozen and trainable
    children supply the loss. The next forward then rejects the original
    parameter shape. Excluding the frozen modules uses FSDP's public API and
    leaves their weights in the model and its full state dictionary.

    RLinf normally moves the entire model to its compute device before FSDP
    construction. Explicitly place these modules too: FSDP does not move
    ignored modules, and a future CPU-loading path must not leave them behind.
    This happens before wrapping and preserves every parameter's dtype.
    """
    if (model_config.get("model_type") != "openpi"
            or not model_config.get("enforce_native_scope", False)):
        return None
    if not model.config.pi05 or not model.config.train_expert_only:
        raise ValueError("Native OpenPI FSDP scope requires pi05 with a frozen VLM")

    backbone = model.paligemma_with_expert
    modules = [backbone.paligemma, backbone.gemma_expert.lm_head]
    names = ["VLM", "unused expert language head"]
    for name, module in zip(names, modules, strict=True):
        parameters = tuple(module.parameters())
        if not parameters or any(parameter.requires_grad for parameter in parameters):
            raise ValueError(f"Cannot exclude {name} from FSDP: it must already be frozen")
    for module in modules:
        module.to(device=device)
    return modules
