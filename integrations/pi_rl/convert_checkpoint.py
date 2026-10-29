"""Convert the user's original JAX checkpoint with the official RLinf converter.

Run with the existing OpenPI Python, JAX_PLATFORMS=cpu and no visible GPUs.
This wrapper verifies every mapped parameter, preserves normalization assets,
and records provenance. It never downloads a different SFT checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from model_configs import RLINF_COMMIT, assert_native_model_settings


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rlinf", required=True)
    args = parser.parse_args()
    source, output, rlinf = (Path(value).resolve() for value in
                             (args.source, args.output, args.rlinf))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {output}")
    if "pi05" not in str(source):
        raise ValueError("The upstream v0.3 converter determines adaptive norms from 'pi05' in source path")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("Conversion must run CPU-only: CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu")
    commit = subprocess.check_output(["git", "-C", str(rlinf), "rev-parse", "HEAD"], text=True).strip()
    if commit != RLINF_COMMIT:
        raise ValueError(f"Unexpected RLinf source: {commit}")
    converter_path = rlinf / "rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py"
    started = time.time()
    import openpi.models_pytorch.pi0_pytorch as native_torch
    import openpi.training.config as native_config
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(0)
    model_settings = native_config.get_config("pi05_libero").model
    assert_native_model_settings(model_settings)
    spec = importlib.util.spec_from_file_location("official_rlinf_converter", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    load_reports = []
    original_load = native_torch.PI0Pytorch.load_state_dict

    def checked_load(model, state_dict, strict=True, **kwargs):
        result = original_load(model, state_dict, strict=False, **kwargs)
        # OpenPI inference does not use these language-generation output heads.
        allowed_missing = {
            "paligemma_with_expert.paligemma.lm_head.weight",
            "paligemma_with_expert.gemma_expert.lm_head.weight",
        }
        missing = set(result.missing_keys)
        if missing - allowed_missing or result.unexpected_keys:
            raise RuntimeError(f"Incomplete conversion: missing={sorted(missing)}, unexpected={result.unexpected_keys}")
        # Compare every mapped tensor before dtype conversion; this catches a
        # loader silently skipping a weight even with strict=False upstream.
        current = model.state_dict()
        for key, value in state_dict.items():
            if not torch.equal(current[key], value.to(current[key].dtype)):
                raise RuntimeError(f"Converted tensor was not loaded correctly: {key}")
        load_reports.append({
            "mapped_tensors": len(state_dict),
            "mapped_parameters": sum(value.numel() for value in state_dict.values()),
            "allowed_missing_unused_heads": sorted(missing),
            "unexpected_keys": list(result.unexpected_keys),
            "all_mapped_tensors_loaded_exactly": True,
        })
        return result

    native_torch.PI0Pytorch.load_state_dict = checked_load
    try:
        converter.convert_pi0_checkpoint(str(source), "bfloat16", str(output), model_settings)
    finally:
        native_torch.PI0Pytorch.load_state_dict = original_load
    if len(load_reports) != 1:
        raise RuntimeError("Expected exactly one fully verified model load")
    gc.collect()
    # Upstream's converter assumes a different assets-root convention.
    shutil.copytree(source / "assets", output / "assets", dirs_exist_ok=True)
    source_norm = source / "assets/physical-intelligence/libero/norm_stats.json"
    rlinf_norm = output / "physical-intelligence/libero/norm_stats.json"
    rlinf_norm.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_norm, rlinf_norm)
    assets = {}
    for path in sorted((source / "assets").rglob("*")):
        if path.is_file():
            relative = path.relative_to(source)
            original_hash = digest(path)
            if digest(output / relative) != original_hash:
                raise RuntimeError(f"Normalization asset changed: {relative}")
            assets[str(relative)] = original_hash
    source_files = {str(path.relative_to(source)): {
        "bytes": path.stat().st_size, "sha256": digest(path)
    } for path in sorted((source / "params").rglob("*")) if path.is_file()}
    weights = output / "model.safetensors"
    manifest = {
        "source_checkpoint": str(source), "source_parameters": source_files,
        "source_assets_sha256": assets,
        "rlinf_commit": commit, "converter_path": str(converter_path),
        "converter_sha256": digest(converter_path),
        "conversion_dtype": "bfloat16",
        "dtype_note": "Matches the bfloat16 restore used by native RLDT before action-expert weights are cast to float32.",
        "model_config": dict(model_settings.__dict__),
        "mapping_verification": load_reports[0],
        "weights_bytes": weights.stat().st_size, "weights_sha256": digest(weights),
        "rlinf_normalization_sha256": digest(rlinf_norm),
        "elapsed_seconds": time.time() - started,
        "numerical_inference_parity": "pending separate shared-noise check",
    }
    (output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({"output": str(output), "weights_bytes": weights.stat().st_size,
                      "mapping": load_reports[0], "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
