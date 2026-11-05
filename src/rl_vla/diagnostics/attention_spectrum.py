"""CPU diagnostics on unaveraged native attention, never on plotted heatmaps.

For pi0.5, cross attention means the action-query/prefix-key block of the
joint expert attention. Its row sums need not be one: suffix keys compete.
"""

from __future__ import annotations

import numpy as np


def _matrix(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError("Expected a nonempty finite [query,key] matrix")
    return a


def spectrum(matrix, *, rtol=1e-6):
    """Entropy rank uses singular values; participation rank uses squared values.

    No centering: this measures the attention operator, not covariance rank.
    Float64 analysis cannot undo the native BF16 quantization of saved attention.
    """
    a = _matrix(matrix)
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("rtol must be finite in (0,1)")
    singular = np.linalg.svd(a, compute_uv=False)
    energy = singular**2
    total = energy.sum()
    if total == 0:
        effective = stable = participation = 0.0
        numerical = rank90 = rank99 = 0
    else:
        p = singular[singular > 0] / singular.sum()
        effective = float(np.exp(-np.sum(p * np.log(p))))
        stable = float(total / energy[0])
        participation = float(total**2 / np.sum(energy**2))
        numerical = int(np.count_nonzero(singular > rtol * singular[0]))
        cumulative = np.cumsum(energy) / total
        rank90 = int(np.searchsorted(cumulative, 0.9) + 1)
        rank99 = int(np.searchsorted(cumulative, 0.99) + 1)
    return {
        "queries": a.shape[0],
        "keys": a.shape[1],
        "rank_ceiling": min(a.shape),
        "numerical_rank": numerical,
        "numerical_rank_rtol": rtol,
        "effective_rank": effective,
        "stable_rank": stable,
        "participation_rank": participation,
        "energy90_rank": rank90,
        "energy99_rank": rank99,
        "frobenius_norm": float(np.sqrt(total)),
        "singular_values": singular.tolist(),
    }


def attention_drift(reference, candidate):
    """Separate absolute routing/mass drift from conditional spatial drift.

    JS is computed per query, then averaged, in nats. Zero-mass rows have no
    conditional distribution and are counted explicitly, never assigned JS=0.
    """
    a, b = _matrix(reference), _matrix(candidate)
    if a.shape != b.shape or (a < 0).any() or (b < 0).any():
        raise ValueError("Attention must have matched shapes and nonnegative entries")
    ma, mb = a.sum(-1), b.sum(-1)
    valid = (ma > 0) & (mb > 0)
    js = None
    if valid.any():
        p, q = a[valid] / ma[valid, None], b[valid] / mb[valid, None]
        midpoint = (p + q) / 2

        def kl(x):
            terms = np.zeros_like(x)
            positive = x > 0
            terms[positive] = x[positive] * np.log(x[positive] / midpoint[positive])
            return terms.sum(-1)

        js = float(np.mean((kl(p) + kl(q)) / 2))
    na, nb, delta = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(b - a)
    cosine = float(np.clip(np.sum(a * b) / (na * nb), -1, 1)) if na and nb else None
    return {
        "frobenius_delta": float(delta),
        "relative_frobenius_delta": float(delta / na) if na else None,
        "cosine_distance": 1 - cosine if cosine is not None else None,
        "reference_mass_all_keys": float(ma.mean()),
        "candidate_mass_all_keys": float(mb.mean()),
        "mass_delta_all_keys": float((mb - ma).mean()),
        "mean_absolute_query_mass_delta": float(np.abs(mb - ma).mean()),
        "conditional_js_nats": js,
        "conditional_valid_queries": int(valid.sum()),
        "conditional_undefined_queries": int((~valid).sum()),
    }


def key_groups(metadata, key_count):
    """Exclude padded language and masked cameras without renormalizing rows."""
    prefix = int(metadata["prefix_tokens"])
    if prefix > key_count:
        raise ValueError("Prefix exceeds captured key count")
    groups, visual = {}, []
    occupied = set()
    for span in metadata["camera_spans"]:
        lo, hi = int(span["start"]), int(span["end"])
        if not 0 <= lo < hi <= prefix or occupied.intersection(range(lo, hi)):
            raise ValueError("Invalid or overlapping camera spans")
        occupied.update(range(lo, hi))
        if span["valid"]:
            groups[span["name"]] = np.arange(lo, hi)
            visual.extend(range(lo, hi))
    lo, hi = map(int, metadata["text_span"])
    mask = np.asarray(metadata["prompt_mask"], dtype=bool)
    if (
        not 0 <= lo <= hi <= prefix
        or mask.shape != (hi - lo,)
        or occupied.intersection(range(lo, hi))
    ):
        raise ValueError("Invalid text span/mask")
    text = lo + np.flatnonzero(mask)
    groups.update(
        visual=np.asarray(visual, dtype=int),
        text=text,
        prefix=np.asarray(visual + text.tolist(), dtype=int),
    )
    return {name: indices for name, indices in groups.items() if len(indices)}


def compare_attention(reference, candidate, metadata, *, rtol=1e-6):
    """Yield one record per layer, batch, head and key group; no head averaging."""
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.ndim != 5 or min(a.shape) < 1:
        raise ValueError("Expected matched [layer,batch,head,query,key] attention")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError("Attention must be finite and nonnegative")
    if not np.allclose(a.sum(-1), 1, atol=0.01, rtol=0) or not np.allclose(
        b.sum(-1), 1, atol=0.01, rtol=0
    ):
        raise ValueError("Expected native full-denominator attention with row sums near one")
    groups = key_groups(metadata, a.shape[-1])
    for layer, batch, head in np.ndindex(a.shape[:3]):
        for group, indices in groups.items():
            left, right = a[layer, batch, head][:, indices], b[layer, batch, head][:, indices]
            baseline, current = spectrum(left, rtol=rtol), spectrum(right, rtol=rtol)
            yield {
                "layer": layer,
                "batch": batch,
                "head": head,
                "key_group": group,
                "reference_spectrum": baseline,
                "candidate_spectrum": current,
                "effective_rank_delta": current["effective_rank"] - baseline["effective_rank"],
                **attention_drift(left, right),
            }


def validate_pair(left, right, left_meta, right_meta):
    """Reject drift comparisons that confound checkpoints with inputs/noise/time."""
    for key in ("x_t", "t"):
        if key not in left or key not in right or not np.array_equal(left[key], right[key]):
            raise ValueError(f"Paired traces require identical {key}")
    for key in (
        "camera_spans",
        "text_span",
        "prefix_tokens",
        "action_tokens",
        "prompt_ids",
        "prompt_mask",
    ):
        if key not in left_meta or key not in right_meta or left_meta[key] != right_meta[key]:
            raise ValueError(f"Paired token metadata differ: {key}")
    for key in ("observation_sha256", "state_sha256"):
        a = left_meta.get("bank_record", {}).get(key)
        b = right_meta.get("bank_record", {}).get(key)
        if not a or a != b:
            raise ValueError(f"Paired observation identity missing or different: {key}")
    if not left_meta.get("noise_sha256") or left_meta["noise_sha256"] != right_meta.get(
        "noise_sha256"
    ):
        raise ValueError("Paired noise identity missing or different")
