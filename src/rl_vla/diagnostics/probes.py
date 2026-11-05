"""Diagnostic linear probes. Fitted weights never enter the policy or learner."""

import numpy as np


def fit_linear_probe(features, labels, *, ridge=10., fit_split="train"):
    """Class-balanced ridge classifier with training-only standardization."""
    if fit_split != "train":
        raise ValueError("Diagnostic probe fitting is restricted to the training split")
    x, y = np.asarray(features, float), np.asarray(labels)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("Expected finite [samples,features] and aligned labels")
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    if len(classes) < 2 or ridge <= 0:
        raise ValueError("At least two classes and positive fixed regularization are required")
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    z = np.column_stack(((x - center) / scale, np.ones(len(x))))
    weight = 1 / counts[inverse]
    weight *= len(x) / weight.sum()
    labels_onehot = np.eye(len(classes))[inverse]
    penalty = np.eye(z.shape[1]) * ridge
    penalty[-1, -1] = 0
    coefficients = np.linalg.solve(z.T @ (z * weight[:, None]) + penalty,
                                   z.T @ (labels_onehot * weight[:, None]))
    return {"center": center, "scale": scale, "coefficients": coefficients,
            "classes": classes, "ridge": np.asarray(ridge), "fit_samples": np.asarray(len(x))}


def probe_scores(probe, features):
    x = np.asarray(features, float)
    if x.ndim != 2 or x.shape[1] != len(probe["center"]) or not np.isfinite(x).all():
        raise ValueError("Feature dimensions or finiteness differ from fit data")
    z = np.column_stack(((x - probe["center"]) / probe["scale"], np.ones(len(x))))
    return z @ probe["coefficients"]


def binary_auc(labels, scores):
    """Pairwise AUC with exact half credit for ties; no sklearn dependency."""
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    positives, negatives = scores[labels], scores[~labels]
    if not len(positives) or not len(negatives):
        return None
    negative_sorted = np.sort(negatives)
    less = np.searchsorted(negative_sorted, positives, side="left")
    equal = np.searchsorted(negative_sorted, positives, side="right") - less
    return float((less + .5 * equal).sum() / (len(positives) * len(negatives)))


def score_probe(probe, features, labels):
    scores = probe_scores(probe, features)
    truth = np.asarray(labels)
    classes = probe["classes"]
    prediction = classes[scores.argmax(axis=1)]
    if not np.isin(truth, classes).all():
        raise ValueError("Evaluation labels include an unfitted class")
    confusion = np.array([[(int(((truth == a) & (prediction == b)).sum()))
                           for b in classes] for a in classes])
    denominator = confusion.sum(axis=1)
    recalls = np.divide(confusion.diagonal(), denominator,
                        out=np.full(len(classes), np.nan), where=denominator > 0)
    return {"samples": len(truth), "accuracy": float((prediction == truth).mean()),
            "balanced_accuracy": float(np.nanmean(recalls)), "classes": classes.tolist(),
            "confusion": confusion.tolist(),
            "one_vs_rest_auc": {str(c): binary_auc(truth == c, scores[:, i])
                                 for i, c in enumerate(classes)},
            "meaning": "Feature decodability only; does not establish causal use by the action expert."}
