"""Controls for spatial shortcut leakage in frozen-feature object probes."""

import numpy as np


def validate_disjoint_records(train, evaluation):
    """State IDs alone are not globally unique; compare saved state/image hashes."""
    for name, records in (("train", train), ("evaluation", evaluation)):
        if not records:
            raise ValueError(f"Empty {name} records")
        for key in ("state_sha256", "observation_sha256"):
            hashes = [r.get(key) for r in records]
            if not all(hashes) or len(set(hashes)) != len(hashes):
                raise ValueError(f"Missing or duplicate {key} in {name}")
    for key in ("state_sha256", "observation_sha256"):
        if {r[key] for r in train} & {r[key] for r in evaluation}:
            raise ValueError(f"Probe split leakage: overlapping {key}")


def position_features(rows, cameras=("image", "wrist_image")):
    """Categorical camera/patch baseline: permits a separate prior at every patch.

    Stronger than just x/y coordinates; it tests memorized spatial regions.
    It has no image pixels, state values, instruction, or learned VLM features.
    """
    if not rows:
        raise ValueError("Empty probe rows")
    count = len(rows[0]["labels"])
    basis = np.eye(count * len(cameras), dtype=np.float32)
    result = []
    for row in rows:
        if len(row["labels"]) != count or row["camera"] not in cameras:
            raise ValueError("Position control requires matched grids and known cameras")
        start = cameras.index(row["camera"]) * count
        result.append(basis[start : start + count][row["valid"]])
    return np.concatenate(result)


def shuffled_patch_labels(rows, *, seed):
    """Shuffle labels within each scene/camera, retaining its class frequencies."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(r["labels"][r["valid"]]) for r in rows])


def state_metrics(probe, rows, scores, *, draws=2000, seed=20261103):
    """Aggregate confusion by state; all cameras/patches stay together in bootstrap."""
    classes = probe["classes"]
    predictions = classes[np.asarray(scores).argmax(-1)]
    groups = {}
    offset = 0
    for row in rows:
        truth = row["labels"][row["valid"]]
        pred = predictions[offset : offset + len(truth)]
        offset += len(truth)
        if len(pred) != len(truth):
            raise ValueError("Scores do not align with patch rows")
        confusion = np.array(
            [[(int(((truth == a) & (pred == b)).sum())) for b in classes] for a in classes]
        )
        groups.setdefault(row["state_id"], np.zeros_like(confusion))
        groups[row["state_id"]] += confusion
    if offset != len(predictions):
        raise ValueError("Extra scores after last patch row")
    matrices = np.stack(list(groups.values()))

    def balanced(matrix, object_only=False):
        count = matrix.sum(-1)
        recall = np.divide(
            np.diagonal(matrix, axis1=-2, axis2=-1),
            count,
            out=np.full(count.shape, np.nan),
            where=count > 0,
        )
        if object_only:
            recall = recall[..., np.isin(classes, [1, 2])]
        return np.nanmean(recall, axis=-1)

    sampled = np.random.default_rng(seed).integers(0, len(matrices), (draws, len(matrices)))
    bootstrap = matrices[sampled].sum(1)
    total = matrices.sum(0)
    result = {
        "states": len(matrices),
        "bootstrap_unit": "whole state, all cameras and patches",
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "confusion": total.tolist(),
    }
    for name, object_only in (("balanced_accuracy", False), ("object_balanced_accuracy", True)):
        value = balanced(total, object_only)
        samples = balanced(bootstrap, object_only)
        defined = samples[np.isfinite(samples)]
        result[name] = float(value) if np.isfinite(value) else None
        result[name + "_state_bootstrap95"] = (
            np.quantile(defined, [0.025, 0.975]).tolist()
            if len(defined) and len(matrices) > 1
            else None
        )
    return result
