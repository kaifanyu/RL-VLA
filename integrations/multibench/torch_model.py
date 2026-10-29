"""Unchanged pi05_base loading with the official RLinf Flow-SDE implementation.

Only observation naming, checkpoint loading and benchmark transforms differ from
the upstream model. Sampling, likelihoods, PPO loss and value estimation are
inherited without changes.
"""
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from multibench.policy_spec import BENCHMARKS, build_transforms, model_config
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config, OpenPi0ForRLActionPrediction,
)


class MultiBenchmarkOpenPi(OpenPi0ForRLActionPrediction):
    def obs_processor(self, env_obs):
        result = {"observation/image": env_obs["main_images"],
                  "observation/state": env_obs["states"],
                  "prompt": env_obs["task_descriptions"]}
        if env_obs.get("wrist_images") is not None:
            result["observation/wrist_image"] = env_obs["wrist_images"]
        if env_obs.get("extra_view_images") is not None:
            result["observation/right_wrist_image"] = env_obs["extra_view_images"]
        return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def get_model(cfg, torch_dtype=None):
    """RLinf model factory; always starts with the verified base conversion."""
    from pi_rl.model_contract import enforce_trainable_scope
    from safetensors import safe_open
    from safetensors.torch import load_file

    benchmark = str(cfg["benchmark"])
    spec = BENCHMARKS[benchmark]
    settings = dataclasses.asdict(model_config())
    overrides = dict(cfg.get("openpi", {}))
    immutable = {"pi05": True, "action_horizon": 50, "action_dim": 32,
                 "discrete_state_input": True, "paligemma_variant": "gemma_2b",
                 "action_expert_variant": "gemma_300m"}
    for key, value in immutable.items():
        if key in overrides and overrides[key] != value:
            raise ValueError(f"Base checkpoint setting {key} must remain {value!r}")
    settings.update(overrides)
    settings.update(immutable)
    settings.update(config_name="pi05_base_multibench", action_env_dim=spec["action_dim"],
                    num_images_in_input=spec["images"], train_expert_only=True,
                    noise_method="flow_sde", joint_logprob=False)
    config = OpenPi0Config(**settings)
    if not 1 <= config.action_chunk <= config.action_horizon:
        raise ValueError("Invalid execution prefix")
    if cfg.get("action_dim", spec["action_dim"]) != spec["action_dim"]:
        raise ValueError("Actor motor action_dim differs from its benchmark")
    model_path = Path(cfg["model_path"]).resolve()
    provenance_path = model_path / "conversion_manifest.json"
    provenance = json.loads(provenance_path.read_text())
    if "pi05_base" not in str(provenance.get("source_checkpoint", "")):
        raise ValueError("Model provenance is not the original pi05_base")
    model = MultiBenchmarkOpenPi(config)
    model.freeze_vlm()
    weights_path = str(model_path / "model.safetensors")
    state = load_file(weights_path, device="cpu")
    # The official converter uses safetensors.save_model, which stores tied
    # embeddings once and records the omitted name as metadata. load_file
    # alone does not restore those aliases. Recreate them without a copy so
    # every inference parameter is explicitly included in the checked load.
    with safe_open(weights_path, framework="pt", device="cpu") as weights:
        aliases = {key: target for key, target in (weights.metadata() or {}).items()
                   if key not in state and target in state}
    model_state = model.state_dict()
    for key, target in aliases.items():
        if key not in model_state or target not in model_state:
            raise ValueError(f"Unknown checkpoint parameter alias: {key} -> {target}")
        if model_state[key].data_ptr() != model_state[target].data_ptr():
            raise ValueError(f"Checkpoint alias is not tied in the model: {key} -> {target}")
        state[key] = state[target]
    del model_state
    result = model.load_state_dict(state, strict=False)
    allowed_unused = {"paligemma_with_expert.paligemma.lm_head.weight",
                      "paligemma_with_expert.gemma_expert.lm_head.weight"}
    missing = [key for key in result.missing_keys if key not in allowed_unused and not key.startswith("value_head.")]
    if missing or result.unexpected_keys:
        raise ValueError(f"Incomplete base load: missing={missing}, unexpected={result.unexpected_keys}")
    del state
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model._input_transform, model._output_transform, transform_metadata = build_transforms(benchmark, config)
    counts = enforce_trainable_scope(model)
    model.multibench_metadata = {
        "benchmark": benchmark, "parameter_counts": counts,
        "conversion_manifest": provenance,
        "conversion_manifest_sha256": sha256(provenance_path),
        "restored_safetensors_aliases": aliases,
        "transforms": transform_metadata,
        "base_model": dataclasses.asdict(model_config()),
        "sampling_implementation": "Unmodified RLinf OpenPi0ForRLActionPrediction Flow-SDE",
        "initially_random_parameters": [key for key in result.missing_keys if key.startswith("value_head.")],
    }
    directory = cfg.get("scope_report_directory")
    if directory:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"model_{os.getpid()}.json").write_text(json.dumps(counts, indent=2) + "\n")
    return model


def load_model(config, checkpoint=None, device="cuda"):
    from omegaconf import OmegaConf
    from rlinf_bridge.evaluate import load_checkpoint, seed_episode

    settings = copy.deepcopy(config["actor"]["model"])
    seed_episode(0)
    model = get_model(OmegaConf.create(settings))
    provenance = load_checkpoint(model, checkpoint) if checkpoint else None
    model.eval().to(device)
    return model, model.multibench_metadata | {"loaded_checkpoint": provenance,
                                              "torch_version": torch.__version__, "device": str(device)}


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def predict_chunk(model, raw):
    from multibench.rl_env import ControlledEnv

    observation = ControlledEnv._wrap_obs(raw)
    with torch.inference_mode():
        motor, outputs = model.predict_action_batch(observation, mode="eval", compute_values=False)
    motor = _numpy(motor)[0]
    normalized = _numpy(outputs["forward_inputs"]["model_action"])[0].reshape(50, 32)
    processed = model._input_transform(copy.deepcopy(raw["openpi"]))
    decoded = np.asarray(model._output_transform({"state": processed["state"],
                                                  "actions": normalized.copy()})["actions"], np.float32)
    expected = (int(model.config.action_chunk), int(model.config.action_env_dim))
    if motor.shape != expected or decoded.shape != (50, expected[1]):
        raise ValueError(f"Wrong action shape: execution={motor.shape}, full={decoded.shape}")
    if not np.allclose(decoded[:len(motor)], motor, rtol=1e-6, atol=1e-6):
        raise ValueError("Full action decoding disagrees with official rollout decoding")
    if not np.isfinite(normalized).all() or not np.isfinite(decoded).all():
        raise FloatingPointError("Nonfinite policy actions")
    initial_noise = _numpy(outputs["forward_inputs"]["chains"])[0, 0]
    return normalized, decoded, motor, initial_noise


def install():
    """Replace the lazy OpenPI factory inside only this campaign's processes."""
    import rlinf.models.embodiment.openpi
    rlinf.models.embodiment.openpi.get_model = get_model
