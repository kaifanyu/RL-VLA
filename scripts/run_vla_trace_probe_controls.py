"""Re-fit train-only object probes with position and shuffled-label controls.

Uses saved native feature banks. Validation and original-layout source are
evaluated separately. This is post-hoc diagnostic analysis, not policy selection.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from run_vla_trace_probes import OBJECTS, examples

from rl_vla.diagnostics.probe_controls import (
    position_features,
    shuffled_patch_labels,
    state_metrics,
    validate_disjoint_records,
)
from rl_vla.diagnostics.probes import fit_linear_probe, probe_scores


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(root, task, model, output, *, shuffle_seeds=(17, 29, 43), ridge=10.0):
    root, output = Path(root).resolve(), Path(output).resolve()
    features = root / "features" / model / task
    source = root / "features_retention" / model / task / "source"
    bank = root / "observation_bank" / task
    manifests = {
        split: bank / split / "manifest.jsonl" for split in ("train", "validation", "source")
    }
    records = {s: read_records(p) for s, p in manifests.items()}
    for split in ("validation", "source"):
        validate_disjoint_records(records["train"], records[split])
    # Reuse the predeclared projection, not a PCA fit on evaluation examples.
    projection_path = root / "probes" / task / model / "fixed_projection.npy"
    projection = np.load(projection_path, allow_pickle=False)
    output.mkdir(parents=True, exist_ok=False)
    provenance = {str(p): sha(p) for p in [projection_path, *manifests.values()]}
    for split, entries in records.items():
        for record in entries:
            folder = (
                source if split == "source" else features / split
            ) / f"state_{record['initial_state_id']:06d}"
            for path in (
                folder / "prefix.npz",
                folder / "metadata.json",
                Path(record["masks_path"]),
            ):
                provenance[str(path)] = sha(path)
    results = []
    for kind in ("image_features", "prefix_features"):
        train = examples(features, bank, task, "train", kind, projection, source)
        x = np.concatenate([r["features"][r["valid"]] for r in train])
        y = np.concatenate([r["labels"][r["valid"]] for r in train])
        probes = {"features": fit_linear_probe(x, y, ridge=ridge)}
        for seed in shuffle_seeds:
            probes[f"shuffled_{seed}"] = fit_linear_probe(
                x, shuffled_patch_labels(train, seed=seed), ridge=ridge
            )
        if kind == "image_features":
            probes["position_only"] = fit_linear_probe(position_features(train), y, ridge=ridge)
        for control, probe in probes.items():
            np.savez_compressed(output / f"{kind}_{control}.npz", **probe)
        del x, y, train
        for split in ("validation", "source"):
            rows = examples(features, bank, task, split, kind, projection, source)
            x = np.concatenate([r["features"][r["valid"]] for r in rows])
            for control, probe in probes.items():
                inputs = position_features(rows) if control == "position_only" else x
                scores = probe_scores(probe, inputs)
                metrics = state_metrics(probe, rows, scores)
                results.append({"features": kind, "control": control, "split": split, **metrics})
                print(
                    json.dumps(
                        {
                            "task": task,
                            "model": model,
                            "features": kind,
                            "control": control,
                            "split": split,
                            "balanced_accuracy": metrics["balanced_accuracy"],
                        }
                    ),
                    flush=True,
                )
            del x, rows
    result = {
        "complete": True,
        "task": task,
        "model": model,
        "results": results,
        "fit_states": len(records["train"]),
        "fit_split": "train",
        "ridge": ridge,
        "shuffle_seeds": list(shuffle_seeds),
        "projection_shape": list(projection.shape),
        "sources_sha256": provenance,
        "interpretation": [
            "Physical object identity labels: background=0, target=1, distractor=2; ambiguous overlap excluded.",
            "Image features precede VLM contextualization; prefix features are its final normalized output, not the per-layer KV interface.",
            "Position-only uses a categorical camera/patch prior; shuffled controls preserve per-scene class counts.",
            "State bootstrap intervals account for patch/camera dependence, not training-run uncertainty.",
            "Source is an original-layout transfer test, not unseen-category generalization.",
            "Decodability is neither reasoning nor causal use; negative transfer may reflect a probe failure.",
            "Use counterfactual noun prompts, object moves, target/control masks and closed-loop interventions to localize causality.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=list(OBJECTS))
    parser.add_argument("--model", required=True, choices=("pi05_base", "pi05_libero"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.task, args.model, args.output)
