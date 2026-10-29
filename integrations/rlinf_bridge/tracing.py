"""Record already-computed official rollout outputs without changing PPO inputs."""

import numpy as np
import torch


def _numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        # NumPy has no bfloat16 dtype; widening preserves every represented value.
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy().copy()
    return np.asarray(value).copy()


def record_policy_trace(environment, rollout_result, model_config):
    inputs = rollout_result.forward_inputs
    if "model_action" not in inputs:
        raise ValueError("Controlled pi-RL recording requires full normalized model_action")
    raw = _numpy(inputs["model_action"])
    horizon = int(model_config["openpi"]["action_horizon"])
    action_dim = int(model_config["openpi"]["action_dim"])
    if raw.shape[0] != 1 or raw.size != horizon * action_dim:
        raise ValueError("Controlled recording expects one complete normalized action chunk")
    metadata = {"policy_versions": _numpy(rollout_result.versions).tolist()
                if rollout_result.versions is not None else None}
    if "denoise_inds" in inputs:
        metadata["denoise_indices"] = _numpy(inputs["denoise_inds"])[0].tolist()
    if "chains" in inputs:
        metadata["initial_noise"] = _numpy(inputs["chains"])[0, 0].tolist()
    environment.set_policy_trace(raw.reshape(horizon, action_dim), metadata)
