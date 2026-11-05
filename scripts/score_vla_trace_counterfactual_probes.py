"""Score frozen train-only object probes under paired valid distractor prompts."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.metrics import patch_occupancy
from rl_vla.diagnostics.probes import binary_auc, probe_scores, score_probe

OBJECTS = {"object_0": ("alphabet_soup_1", "butter_1"),
           "object_1": ("cream_cheese_1", "alphabet_soup_1"),
           "object_2": ("salad_dressing_1", "tomato_sauce_1")}
CAMERAS = {"base_0_rgb": "image", "left_wrist_0_rgb": "wrist_image"}
KINDS = ("image_features", "prefix_features")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def ready(root, task, model):
    paths = {"selection": root/"methods/rldt"/task/"selection.json",
             "probe": root/"probes"/task/model/"metrics.json",
             "grid": root/"traces"/task/"summary.json"}
    if not all(path.is_file() for path in paths.values()):
        return False
    selection, probe, grid = (read(paths[k]) for k in ("selection", "probe", "grid"))
    counts = selection.get("validation_counts", {})
    if set(counts) != {"25", "50", "100"} or any(type(v) is not int or not 0 <= v <= 50 for v in counts.values()):
        raise ValueError("Invalid predeclared validation counts")
    chosen = min((25, 50, 100), key=lambda i: (-counts[str(i)], i))
    if (selection.get("iteration") != chosen or not selection.get("locked") or
            selection.get("test_results_seen") is not False or
            Path(selection["checkpoint"]).resolve() != (paths["selection"].parent/"training"/f"iteration_{chosen:06d}").resolve()):
        raise ValueError("Selection is not locked validation-only with earliest tie handling")
    if not probe.get("complete") or not grid.get("complete", grid.get("completed")):
        return False
    if probe.get("task") != task or probe.get("model") != model or grid.get("task") != task or grid.get("states") != 50:
        raise ValueError("Probe/grid task, model, or complete50-state identity differs")
    spec = probe["specification"]
    if (spec.get("fit_split") != "train" or spec.get("fit_states") != 100 or
            spec.get("feature_kinds") != list(KINDS) or
            spec.get("selection_sha256") != sha(paths["selection"]) or
            grid.get("selection_sha256") != sha(paths["selection"])):
        raise ValueError("Probe was not fit on frozen training data or selection identity differs")
    return True


def localization(scores, occupancy, label):
    predicted = scores.argmax(axis=-1).reshape(occupancy.shape) == label
    actual = occupancy >= .1
    union = actual | predicted
    a, p = np.argwhere(actual), np.argwhere(predicted)
    return {"visible_patches": int(actual.sum()), "predicted_patches": int(predicted.sum()),
            "iou": float((actual & predicted).sum() / union.sum()) if union.any() else None,
            "centroid_error_patches": float(np.linalg.norm(a.mean(0)-p.mean(0))) if len(a) and len(p) else None,
            "peak_hit": bool(actual.ravel()[np.argmax(scores[:, label])]) if actual.any() else None}


def binary_roc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    if not labels.any() or labels.all():
        return {"false_positive_rate": [], "true_positive_rate": [], "thresholds": []}
    order = np.argsort(-scores, kind="stable")
    truth, sorted_scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores)-1]
    tp = np.cumsum(truth)[ends]
    fp = (ends+1)-tp
    return {"false_positive_rate": np.r_[0., fp/(~labels).sum()].tolist(),
            "true_positive_rate": np.r_[0., tp/labels.sum()].tolist(),
            "thresholds": [None, *sorted_scores[ends].tolist()]}


def summarize_localization(rows, role):
    selected = [r for r in rows if r["region"] == role and r["visible_patches"] > 0]
    distances = [r["centroid_error_patches"] for r in selected if r["centroid_error_patches"] is not None]
    return {"visible_camera_states": len(selected),
            "mean_iou_visible": float(np.mean([r["iou"] for r in selected])) if selected else None,
            "mean_centroid_error_when_predicted": float(np.mean(distances)) if distances else None,
            "visible_but_no_prediction": sum(r["predicted_patches"] == 0 for r in selected),
            "peak_hit_accuracy_visible": float(np.mean([r["peak_hit"] for r in selected])) if selected else None}


def score_kind(probe, rows, output, kind):
    """Fixed physical identity classes; requested role changes with the prompt."""
    if not np.array_equal(probe["classes"], [0, 1, 2]):
        raise ValueError("Expected fitted background/actual-target/actual-distractor classes0/1/2")
    by_condition, score_arrays, local_rows = {}, {}, []
    labels = np.concatenate([row["labels"][row["valid"]] for row in rows])
    rocs = {}
    for condition, requested_label, requested_role in (("original", 1, "target"), ("counterfactual", 2, "distractor")):
        features = np.concatenate([row[condition][row["valid"]] for row in rows])
        metrics = score_probe(probe, features, labels)
        scores = [probe_scores(probe, row[condition]) for row in rows]
        score_arrays[condition] = np.stack(scores)
        valid_scores = np.concatenate([s[row["valid"]] for s, row in zip(scores, rows, strict=True)])
        rocs[condition] = {str(label): binary_roc(labels == label, valid_scores[:, label]) for label in (0, 1, 2)}
        for row, score in zip(rows, scores, strict=True):
            for role, label in (("target", 1), ("distractor", 2)):
                local_rows.append({"state_id": row["state_id"], "camera": row["camera"],
                                   "condition": condition, "region": role, "class": label,
                                   "requested": label == requested_label,
                                   **localization(score, row[f"{role}_occupancy"], label)})
        condition_rows = [r for r in local_rows if r["condition"] == condition]
        metrics["identity_localization"] = {role: summarize_localization(condition_rows, role) for role in ("target", "distractor")}
        prediction = valid_scores.argmax(axis=1) == requested_label
        truth = labels == requested_label
        tp, fp, fn = int((prediction & truth).sum()), int((prediction & ~truth).sum()), int((~prediction & truth).sum())
        metrics["requested_object"] = {"physical_identity_class": requested_label, "physical_identity_role": requested_role,
            "one_vs_rest_auc": binary_auc(truth, valid_scores[:, requested_label]),
            "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None,
            "classification_iou": tp/(tp+fp+fn) if tp+fp+fn else None,
            **metrics["identity_localization"][requested_role]}
        by_condition[condition] = metrics
    changes = []
    for index, row in enumerate(rows):
        a, b = score_arrays["original"][index], score_arrays["counterfactual"][index]
        valid = row["valid"]
        changes.append({"state_id": row["state_id"], "camera": row["camera"],
                        "identity_prediction_flip_fraction": float(np.mean(a[valid].argmax(1) != b[valid].argmax(1))),
                        "mean_patch_score_l2": float(np.linalg.norm(b-a, axis=1).mean()),
                        "max_abs_score_change": float(np.max(np.abs(b-a))),
                        "original_requested_target_mean_score_on_target": float(a[row["target_occupancy"].ravel() >= .1, 1].mean()) if (row["target_occupancy"] >= .1).any() else None,
                        "counterfactual_requested_distractor_mean_score_on_distractor": float(b[row["distractor_occupancy"].ravel() >= .1, 2].mean()) if (row["distractor_occupancy"] >= .1).any() else None})
    if kind == "image_features" and not np.array_equal(score_arrays["original"], score_arrays["counterfactual"]):
        raise ValueError("Language-free image control changed under prompt-only intervention")
    np.savez_compressed(output/f"{kind}_prediction_maps.npz", **score_arrays,
        labels=np.stack([r["labels"] for r in rows]), valid=np.stack([r["valid"] for r in rows]),
        state_ids=np.asarray([r["state_id"] for r in rows]), cameras=np.asarray([r["camera"] for r in rows]),
        target_occupancy=np.stack([r["target_occupancy"] for r in rows]),
        distractor_occupancy=np.stack([r["distractor_occupancy"] for r in rows]), classes=probe["classes"])
    for name, records in (("localization", local_rows), ("paired_prompt_changes", changes)):
        with (output/f"{kind}_{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write(output/f"{kind}_roc.json", rocs)
    by_condition["paired_prompt_change"] = {
        "camera_states": len(rows), "identity_prediction_flip_fraction_mean": float(np.mean([r["identity_prediction_flip_fraction"] for r in changes])),
        "mean_patch_score_l2": float(np.mean([r["mean_patch_score_l2"] for r in changes])),
        "max_abs_score_change": max(r["max_abs_score_change"] for r in changes),
        "identity_accuracy_difference": by_condition["counterfactual"]["accuracy"]-by_condition["original"]["accuracy"],
        "image_control_bit_exact": True if kind == "image_features" else None,
        "meaning": "Physical identity labels stay fixed; requested-object metrics switch from actual target/class1 to actual distractor/class2."}
    return by_condition


def figures(output, measurements, task, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for kind, row_axes in zip(KINDS, axes, strict=True):
        for condition, ax in zip(("original", "counterfactual"), row_axes, strict=True):
            matrix = np.asarray(measurements[kind][condition]["confusion"], float)
            matrix = np.divide(matrix, matrix.sum(1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(1, keepdims=True)>0)
            ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white" if matrix[i,j]>.6 else "black")
            ax.set(title=f"{kind} · {condition}", xticks=range(3), yticks=range(3),
                   xticklabels=["background", "actual target", "actual distractor"],
                   yticklabels=["background", "actual target", "actual distractor"], xlabel="Predicted identity", ylabel="Simulator identity")
    fig.suptitle(f"{task} · {model} · frozen train-only probe transfer\nRequested identity: original=class1, counterfactual=class2; physical identity labels unchanged")
    for extension in ("png", "pdf"):
        fig.savefig(output/f"confusion.{extension}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for kind, row_axes in zip(KINDS, axes, strict=True):
        rocs = read(output/f"{kind}_roc.json")
        for condition, ax in zip(("original", "counterfactual"), row_axes, strict=True):
            for label, name in enumerate(("background", "actual target", "actual distractor")):
                curve = rocs[condition][str(label)]
                auc = measurements[kind][condition]["one_vs_rest_auc"][str(label)]
                if curve["thresholds"]:
                    ax.plot(curve["false_positive_rate"], curve["true_positive_rate"], label=f"{name}: AUC={auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", linewidth=.7)
            ax.set(title=f"{kind} · {condition}", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
            ax.legend(fontsize=8)
    fig.suptitle(f"{task} · {model} · identity ROC under matched prompts")
    for extension in ("png", "pdf"):
        fig.savefig(output/f"roc.{extension}", dpi=160)
    plt.close(fig)
    archives = {}
    for kind in KINDS:
        with np.load(output/f"{kind}_prediction_maps.npz", allow_pickle=False) as data:
            archives[kind] = {name: data[name] for name in data.files}
    first_state = int(archives["image_features"]["state_ids"][0])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    colors = ListedColormap(["#eeeeee", "#e67e22", "#2471a3"])
    for camera, row_axes in zip(("image", "wrist_image"), axes, strict=True):
        index = np.flatnonzero((archives["image_features"]["state_ids"] == first_state) & (archives["image_features"]["cameras"] == camera))[0]
        shape = archives["image_features"]["target_occupancy"][index].shape
        maps = [archives["image_features"]["labels"][index],
                archives["image_features"]["original"][index].argmax(1),
                archives["prefix_features"]["original"][index].argmax(1),
                archives["prefix_features"]["counterfactual"][index].argmax(1)]
        for label, values, ax in zip(("Simulator patch identity", "Image-only probe (both prompts)", "Original-prefix probe", "Counterfactual-prefix probe"), maps, row_axes, strict=True):
            ax.imshow(values.reshape(shape), vmin=0, vmax=2, cmap=colors, interpolation="nearest")
            ax.set(title=label, xlabel="Patch column", ylabel=f"{camera}\nPatch row")
    fig.suptitle(f"{task} · {model} · first predetermined state {first_state}\nOrange=actual target/class1; blue=actual distractor/class2; gray=background. Requested identity switches1→2.")
    for extension in ("png", "pdf"):
        fig.savefig(output/f"prediction_maps.{extension}", dpi=160)
    plt.close(fig)


def run(root, task, model):
    root = Path(root).resolve()
    while not ready(root, task, model):
        print(f"Waiting for locked selection, complete original probes and full50-case grid: {task}/{model}", flush=True)
        time.sleep(15)
    output = root/"probes_counterfactual"/task/model
    output.mkdir(parents=True, exist_ok=False)
    probe_root = root/"probes"/task/model
    bank_root = root/"observation_bank"/task/"test"
    trace_root = root/"traces"/task/model
    protocol_path = root/"tasks"/task/"protocol.json"
    manifest = bank_root/"manifest.jsonl"
    bank_summary = read(bank_root/"summary.json")
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    expected_ids = read(protocol_path)["tasks"][task]["test_state_ids"]
    if len(records) != 50 or [r["initial_state_id"] for r in records] != expected_ids or not bank_summary["complete"] or bank_summary["manifest_sha256"] != sha(manifest):
        raise ValueError("Need exactly the complete frozen50-state test manifest")
    paths = [Path(__file__), root/"methods/rldt"/task/"selection.json", protocol_path, manifest,
             bank_root/"summary.json", probe_root/"specification.json", probe_root/"metrics.json",
             probe_root/"fixed_projection.npy", root/"traces"/task/"summary.json"]
    hashes = {str(p): sha(p) for p in paths}
    projection = np.load(probe_root/"fixed_projection.npy", allow_pickle=False)
    target, distractor = OBJECTS[task]
    all_rows = {kind: [] for kind in KINDS}
    for record in records:
        state = record["initial_state_id"]
        directory = trace_root/f"state_{state:06d}"
        orig_metadata_path = directory/"metadata.json"
        cf_metadata_path = directory/("counterfactual_prefix_metadata.json" if model == "pi05_base" else "counterfactual_metadata.json")
        orig_meta, cf_meta = read(orig_metadata_path), read(cf_metadata_path)
        fitted_feature_metadata = root/"features"/model/task/"test"/f"state_{state:06d}"/"metadata.json"
        if orig_meta.get("prefix_features_sha256") != read(fitted_feature_metadata)["prefix_features_sha256"]:
            raise ValueError("Original grid prefix features differ from original probe scoring features")
        hashes[str(fitted_feature_metadata)] = sha(fitted_feature_metadata)
        expected_prompt = record["prompt"].replace(target.rsplit("_", 1)[0].replace("_", " "), distractor.rsplit("_", 1)[0].replace("_", " "))
        if (orig_meta["bank_record"] != record or cf_meta["instruction"] != expected_prompt or expected_prompt == record["prompt"] or
                orig_meta["camera_spans"] != cf_meta["camera_spans"] or record["protocol_sha256"] != hashes[str(protocol_path)]):
            raise ValueError("Counterfactual prompt, camera geometry, or frozen observation identity differs")
        masks_path = Path(record["masks_path"])
        observation_path = Path(record["observation_path"])
        if sha(masks_path) != record["masks_sha256"] or sha(observation_path) != record["observation_sha256"]:
            raise ValueError("Frozen observation/mask changed")
        original_path, changed_path = directory/"prefix.npz", directory/"counterfactual_prefix.npz"
        for path in (orig_metadata_path, cf_metadata_path, masks_path, observation_path, original_path, changed_path):
            hashes[str(path)] = sha(path)
        with np.load(original_path, allow_pickle=False) as a, np.load(changed_path, allow_pickle=False) as b:
            features = {kind: (a[kind][0].copy(), b[kind][0].copy()) for kind in KINDS}
        if not np.array_equal(*features["image_features"]):
            raise ValueError("Prompt-only intervention changed language-free image features")
        with np.load(masks_path, allow_pickle=False) as masks:
            for span in orig_meta["camera_spans"]:
                if not span["valid"]:
                    continue
                camera, grid = CAMERAS[span["name"]], tuple(span["grid"])
                target_area = patch_occupancy(masks[f"{camera}__{target}"], grid)
                distractor_area = patch_occupancy(masks[f"{camera}__{distractor}"], grid)
                labels = np.zeros(target_area.size, np.int32)
                labels[target_area.ravel() >= .1] = 1
                labels[distractor_area.ravel() >= .1] = 2
                valid = ~((target_area.ravel() >= .1) & (distractor_area.ravel() >= .1))
                for kind, (original, changed) in features.items():
                    lo, hi = span["start"], span["end"]
                    if hi-lo != target_area.size or original.shape != changed.shape or original.shape[-1] != projection.shape[0]:
                        raise ValueError("Feature dimensions or camera patch alignment differs")
                    all_rows[kind].append({"state_id": state, "camera": camera, "labels": labels, "valid": valid,
                        "target_occupancy": target_area, "distractor_occupancy": distractor_area,
                        "original": original[lo:hi].astype(np.float32)@projection,
                        "counterfactual": changed[lo:hi].astype(np.float32)@projection})
    measurements = {}
    for kind in KINDS:
        path = probe_root/f"{kind}_probe.npz"
        hashes[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as archive:
            probe = {key: archive[key].copy() for key in archive.files}
        measurements[kind] = score_kind(probe, all_rows[kind], output, kind)
        original_predictions = probe_root/f"{kind}_test_predictions.npz"
        hashes[str(original_predictions)] = sha(original_predictions)
        with np.load(original_predictions, allow_pickle=False) as original, np.load(output/f"{kind}_prediction_maps.npz", allow_pickle=False) as replay:
            patch_count = replay["labels"].shape[1]
            expected = {"scores": replay["original"].reshape(-1, 3), "labels": replay["labels"].ravel(),
                        "state_ids": np.repeat(replay["state_ids"], patch_count),
                        "cameras": np.repeat(replay["cameras"], patch_count)}
            if any(not np.array_equal(original[name], value) for name, value in expected.items()):
                raise ValueError("Original-prompt probe score replay differs in values or state/camera order")
        measurements[kind]["original_probe_replay_bit_exact"] = True
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise ValueError("Frozen probe, input, or selection changed during scoring")
    figures(output, measurements, task, model)
    write(output/"input_manifest.json", hashes)
    write(output/"metrics.json", {"complete": True, "task": task, "model": model,
        "states": 50, "split": "test", "fit_split": "train", "refit": False,
        "classes": {"0": "background", "1": "actual target identity", "2": "actual distractor identity"},
        "requested_identity": {"original": 1, "counterfactual": 2}, "measurements": measurements,
        "original_probe_metrics": str(probe_root/"metrics.json"), "input_manifest": str(output/"input_manifest.json"),
        "interpretation": "Transfer of unchanged original-prompt train-only linear probes to matched valid distractor prompts. Identity localization and requested-object metrics are distinct. Positive scores show decodability, not causal policy use. Image features are a bit-exact language-free control.",
        "limitations": ["Counterfactual transfer uses50 frozen test states only; no source-layout counterfactual transfer is scored.",
                        "The environment goal is unchanged; counterfactual prompts receive no policy success score.",
                        "Probes were fitted under original target prompts only; no counterfactual refit or hyperparameter selection."]})
    print(json.dumps({"complete": True, "task": task, "model": model, "states": 50}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=list(OBJECTS), required=True)
    parser.add_argument("--model", choices=["pi05_base", "pi05_libero"], required=True)
    args = parser.parse_args()
    run(args.run, args.task, args.model)
