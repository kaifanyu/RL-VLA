"""Audit native prefix mask topology with synthetic tokens, without loading weights."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.models.model import Observation


def audit(output):
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("This topology-only audit must run with JAX_PLATFORMS=cpu")
    # Stub embeddings isolate the exact native packing and mask construction;
    # they are neither model features nor scored experimental observations.
    model = SimpleNamespace(PaliGemma=SimpleNamespace(
        img=lambda image, train: (jnp.zeros((image.shape[0], 2, 4)), None),
        llm=lambda ids, method: jnp.zeros((*ids.shape, 4)),
    ))
    observation = Observation(
        images={"valid_camera": jnp.zeros((1, 2, 2, 3)), "masked_camera": jnp.zeros((1, 2, 2, 3))},
        image_masks={"valid_camera": jnp.array([True]), "masked_camera": jnp.array([False])},
        state=jnp.zeros((1, 32)),
        tokenized_prompt=jnp.array([[1, 2, 0]]),
        tokenized_prompt_mask=jnp.array([[True, True, False]]),
    )
    _, valid, autoregressive = Pi0.embed_prefix(model, observation)
    mask = np.asarray(make_attn_mask(valid, autoregressive))[0]
    checks = {
        "all_native_prefix_autoregressive_flags_false": bool(np.all(~np.asarray(autoregressive))),
        "image_queries_can_attend_valid_text_keys": bool(mask[0:2, 4:6].all()),
        "text_queries_can_attend_valid_image_keys": bool(mask[4:6, 0:2].all()),
        "masked_camera_and_text_padding_excluded_as_queries_and_keys": bool(
            not mask[[2, 3, 6], :].any() and not mask[:, [2, 3, 6]].any()),
        "all_valid_prefix_pairs_allowed": bool(np.array_equal(
            mask, np.asarray(valid)[0, :, None] & np.asarray(valid)[0, None, :])),
    }
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for relative in (
        "third_party/openpi/src/openpi/models/pi0.py",
        "src/rl_vla/rldt_pi05.py",
        "src/rl_vla/diagnostics/pi05_attention.py",
        "scripts/audit_vla_trace_prefix_mask.py",
    ):
        path = root / relative
        sources[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {
        "complete": True, "passed": all(checks.values()), "checks": checks,
        "scope": "Technical native mask-topology check using synthetic embedding stubs; no policy weights, actions, research scores or protocol changes",
        "native_methods": {
            name: {"start_line": inspect.getsourcelines(function)[1], "source": inspect.getsource(function)}
            for name, function in (("Pi0.embed_prefix", Pi0.embed_prefix), ("make_attn_mask", make_attn_mask))
        },
        "prefix_validity": np.asarray(valid).tolist(),
        "prefix_autoregressive_flags": np.asarray(autoregressive).tolist(),
        "prefix_attention_mask": mask.astype(int).tolist(),
        "sources": sources,
        "interpretation": "Valid image and text tokens share a bidirectional prefix block. Later-layer cached image keys can carry instruction information and cached text keys can carry visual information. Direct expert key masks preserve this precomputed context; the audit establishes a permitted information path, not its magnitude or behavioral use.",
        "reproduction": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' PYTHONPATH=src third_party/openpi/.venv/bin/python scripts/audit_vla_trace_prefix_mask.py --output " + str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    if not result["passed"]:
        raise AssertionError(checks)
    print(json.dumps({"passed": result["passed"], "checks": checks, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
