"""Auditable paired statistics and spatial metrics, with no simulator imports."""

import math
from statistics import NormalDist

import numpy as np


def wilson(successes, count, confidence=.95):
    if count < 1 or not 0 <= successes <= count:
        raise ValueError("Expected 0 <= successes <= count and count > 0")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p, z2 = successes / count, z * z
    center = (p + z2 / (2 * count)) / (1 + z2 / count)
    half = z * math.sqrt(p * (1 - p) / count + z2 / (4 * count**2)) / (1 + z2 / count)
    return [max(0., center - half), min(1., center + half)]


def episode_identity(row):
    reset = row["reset"]
    return (row["seed"], reset["initial_state_id"], reset.get("backend_seed"),
            reset.get("task_suite"), reset.get("task_id"), reset.get("experiment_split"))


def paired_success(baseline, candidate, *, seed=20261031, draws=20000):
    """Match by reset AND noise identity; never silently drop unmatched episodes."""
    def index(rows):
        result = {episode_identity(r): r for r in rows}
        if not rows or len(result) != len(rows):
            raise ValueError("Episodes must be nonempty with unique reset/noise identities")
        if any(type(r["success"]) is not bool for r in rows):
            raise ValueError("Success must be an explicit boolean")
        return result
    left, right = index(baseline), index(candidate)
    if left.keys() != right.keys():
        raise ValueError("Paired evaluations have different reset/noise identities")
    a = np.array([left[k]["success"] for k in left], dtype=bool)
    b = np.array([right[k]["success"] for k in left], dtype=bool)
    changes = b.astype(int) - a.astype(int)
    recovered, regressed = int((~a & b).sum()), int((a & ~b).sum())
    discordant = recovered + regressed
    pvalue = min(1., 2 * sum(math.comb(discordant, k) for k in
                           range(min(recovered, regressed) + 1)) / 2**discordant)
    # Resample complete pairs, not independent before/after outcomes.
    boot = np.random.default_rng(seed).choice(changes, size=(draws, len(a))).mean(axis=1)
    return {"episodes": len(a), "baseline_successes": int(a.sum()),
            "candidate_successes": int(b.sum()), "baseline_rate": float(a.mean()),
            "candidate_rate": float(b.mean()), "difference": float(changes.mean()),
            "baseline_wilson95": wilson(int(a.sum()), len(a)),
            "candidate_wilson95": wilson(int(b.sum()), len(b)),
            "paired_bootstrap95": np.quantile(boot, [.025, .975]).tolist(),
            "bootstrap_seed": seed, "bootstrap_draws": draws,
            "recovered": recovered, "regressed": regressed,
            "both_success": int((a & b).sum()), "both_failure": int((~a & ~b).sum()),
            "mcnemar_exact_two_sided_p": pvalue,
            "caveat": "One rollout per state; one RL training seed; exploratory comparisons are unadjusted."}


def patch_occupancy(mask, grid_shape):
    """Fraction of each patch covered by a simulator mask; preserves small objects."""
    mask = np.asarray(mask, dtype=float)
    gh, gw = grid_shape
    if mask.ndim != 2 or mask.shape[0] % gh or mask.shape[1] % gw:
        raise ValueError("Mask dimensions must be divisible by the patch grid")
    if not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError("Mask occupancy must be finite in [0,1]")
    return mask.reshape(gh, mask.shape[0] // gh, gw, mask.shape[1] // gw).mean((1, 3))


def grounding_scores(attention, mask, *, top_fraction=.1):
    """Camera slice scores retain the original all-key softmax denominator."""
    a = np.asarray(attention, dtype=float)
    occupancy = np.asarray(mask, dtype=float)
    if a.shape != occupancy.shape or a.ndim != 2 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Expected finite nonnegative attention and matching 2D occupancy")
    if not np.isfinite(occupancy).all() or np.any((occupancy < 0) | (occupancy > 1)):
        raise ValueError("Invalid occupancy")
    image_mass = float(a.sum())
    mass = float((a * occupancy).sum())
    area = float(occupancy.sum())
    binary = occupancy > 0
    selected = np.zeros(a.size, dtype=bool)
    # Deterministic cardinality even for tied attention values.
    selected[np.argsort(-a.ravel(), kind="stable")[:max(1, math.ceil(a.size * top_fraction))]] = True
    selected = selected.reshape(a.shape)
    probability = a / image_mass if image_mass > 0 else a
    yy, xx = np.indices(a.shape)
    center_error = None
    if area > 0 and image_mass > 0:
        center_error = math.hypot(float((probability * yy).sum() - (occupancy * yy).sum() / area),
                                 float((probability * xx).sum() - (occupancy * xx).sum() / area))
    positive = probability[probability > 0]
    return {"region_mass_all_keys": mass, "camera_mass_all_keys": image_mass,
            "region_fraction_camera": mass / image_mass if image_mass else None,
            "region_patch_area": area, "region_density": mass / area if area else None,
            "peak_hit": bool(binary.ravel()[a.argmax()]) if area and image_mass else None,
            "top10_iou_any_overlap": float((selected & binary).sum() / (selected | binary).sum()) if area else None,
            "centroid_error_patches": center_error,
            "camera_conditional_entropy_nats": float(-(positive * np.log(positive)).sum()) if image_mass else None}


def jensen_shannon(left, right):
    a, b = np.asarray(left, float).ravel(), np.asarray(right, float).ravel()
    if a.shape != b.shape or min(a.sum(), b.sum()) <= 0 or not np.isfinite([a, b]).all():
        raise ValueError("Expected matched finite distributions with positive mass")
    if (a < 0).any() or (b < 0).any():
        raise ValueError("Negative probabilities")
    a, b = a / a.sum(), b / b.sum()
    midpoint = (a + b) / 2
    def kl(x):
        positive = x > 0
        return float((x[positive] * np.log(x[positive] / midpoint[positive])).sum())
    return (kl(a) + kl(b)) / 2
