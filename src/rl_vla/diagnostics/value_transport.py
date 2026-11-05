"""Inspect the actual cached values read by pi0.5's action-query attention.

Reconstructs A_block @ V_block in float64 before the output projection. This is
a diagnostic contribution, not a bit-exact replay of the native BF16 block or
an intervention: suffix values, residuals, output projection and MLP are absent.
"""

import hashlib

import numpy as np

from rl_vla.diagnostics.attention_spectrum import key_groups, spectrum


def stored_cache_hash(values, dtype):
    """Recover the capture fingerprint without a JAX/ml_dtypes dependency.

    The context serializer stores native BF16 leaves losslessly as float32.
    Their low 16 bits must therefore be zero; the upper bits are BF16 bytes.
    """
    if dtype == "bfloat16":
        a = np.asarray(values, dtype=np.float32)
        bits = a.view(np.uint32)
        if not np.isfinite(a).all() or np.any(bits & 0xFFFF):
            raise ValueError("Context float32 values are not lossless finite BF16")
        payload = (bits >> 16).astype(np.uint16).tobytes()
    else:
        a = np.asarray(values, dtype=dtype)
        payload = a.tobytes()
    return hashlib.sha256(str((a.shape, dtype)).encode() + payload).hexdigest()


def linear_cka(left, right):
    """Centered linear CKA across queries; None for a constant representation."""
    x, y = np.asarray(left, float), np.asarray(right, float)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("CKA needs finite matrices with matched query rows")
    x, y = x - x.mean(0), y - y.mean(0)
    gx, gy = x @ x.T, y @ y.T
    denominator = np.linalg.norm(gx) * np.linalg.norm(gy)
    return float(np.clip(np.sum(gx * gy) / denominator, 0, 1)) if denominator else None


def compare_value_transport(reference, candidate, values, metadata):
    """Native GQA head mapping: contiguous query-head groups share a KV head.

    Attention: [L,B,H,Q,K_total]. Cached values: [B,L,K_prefix,H_kv,D].
    Both policies must use the same frozen prefix cache on the same observation.
    """
    a, b, v = map(np.asarray, (reference, candidate, values))
    if a.ndim != 5 or a.shape != b.shape or v.ndim != 5:
        raise ValueError("Expected matched 5D attention and 5D cached values")
    layers, batches, heads, _, keys = a.shape
    if (
        v.shape[:2] != (batches, layers)
        or v.shape[2] != metadata["prefix_tokens"]
        or v.shape[2] > keys
        or v.shape[3] < 1
        or heads % v.shape[3]
    ):
        raise ValueError("Cached value axes or grouped-query head count differ")
    if any(not np.isfinite(x).all() for x in (a, b, v)) or (a < 0).any() or (b < 0).any():
        raise ValueError("Nonfinite values or negative attention")
    sharing = heads // v.shape[3]
    groups = key_groups(metadata, keys)
    for layer, batch, head in np.ndindex(a.shape[:3]):
        for group in ("visual", "text", "prefix"):
            if group not in groups:
                continue
            indices = groups[group]
            value = v[batch, layer, :, head // sharing][indices].astype(np.float64)
            left = a[layer, batch, head][:, indices].astype(np.float64) @ value
            right = b[layer, batch, head][:, indices].astype(np.float64) @ value
            norm = np.linalg.norm(left)
            yield {
                "layer": layer,
                "batch": batch,
                "head": head,
                "kv_head": head // sharing,
                "key_group": group,
                "reference_spectrum": spectrum(left),
                "candidate_spectrum": spectrum(right),
                "query_linear_cka": linear_cka(left, right),
                "frobenius_delta": float(np.linalg.norm(right - left)),
                "relative_frobenius_delta": float(np.linalg.norm(right - left) / norm)
                if norm
                else None,
            }
