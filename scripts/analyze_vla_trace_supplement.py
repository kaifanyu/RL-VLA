"""CPU-only supplementary controls, modality curves, knockouts and action vectors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from analyze_vla_trace_grid import resolve_noun_spans

from rl_vla.diagnostics.metrics import grounding_scores, patch_occupancy

REPO = Path(__file__).resolve().parents[1]
POLICIES = ("pi05_libero", "rldt_000025", "rldt_000050", "rldt_000100")
MODELS = ("pi05_base", "pi05_libero")
CONDITIONS = ("prefix", "counterfactual_prefix")
CONTROLS = ("pick", "and", "in", "basket", "random_valid_non_noun")
CAMERAS = {"image": "base_0_rgb", "wrist_image": "left_wrist_0_rgb"}
OBJECTS = {"object_0": ("alphabet_soup_1", "butter_1"),
           "object_1": ("cream_cheese_1", "alphabet_soup_1"),
           "object_2": ("salad_dressing_1", "tomato_sauce_1")}
STEPS = (0, 5, 9)
OPEN = ("none", "no_image", "no_text", *[f"{region}:{mode}" for region in
        ("target", "distractor", "random", "robot", "background", "old_target")
        for mode in ("background_fill", "mosaic")])
METRICS = ("region_mass_all_keys", "camera_mass_all_keys", "region_fraction_camera",
           "region_patch_area", "region_density", "peak_hit", "top10_iou_any_overlap",
           "centroid_error_patches")
AMENDMENT_SHA = "e701c3c1d8db598917deb70cf77dfc6c4be383b510a55ee5743d743989d5b359"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty required table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class Evidence:
    def __init__(self):
        self.inputs = {}

    def bind(self, path):
        path = Path(path).resolve()
        value = sha(path)
        previous = self.inputs.setdefault(str(path), value)
        if previous != value:
            raise ValueError(f"Input changed during analysis: {path}")
        return path

    def verify(self):
        for path, expected in self.inputs.items():
            if sha(path) != expected:
                raise ValueError(f"Input changed during analysis: {path}")


def select_controls(metadata, task, state_id, condition):
    """Selection depends on task/state/prompt condition, never policy or outcomes."""
    nouns = [value.removesuffix("_1").replace("_", " ") for value in OBJECTS[task]] + ["basket"]
    noun_spans = resolve_noun_spans(metadata, nouns)
    named = resolve_noun_spans(metadata, CONTROLS[:4])
    if set(named) != set(CONTROLS[:4]):
        raise ValueError("A required named control is missing from the actual prompt")
    excluded = {index for record in noun_spans.values() for index in record["prompt_indices"]}
    start = metadata["text_span"][0]
    eligible = [index for index, (piece, valid) in enumerate(zip(
        metadata["prompt_token_pieces"], metadata["prompt_mask"], strict=True))
        if valid and index not in excluded and piece.replace("▁", " ").strip()
        and not re.fullmatch(r"<[^>]+>", piece)
        and start + index in metadata["prefill_query_indices"]]
    if not eligible:
        raise ValueError("No valid non-noun random-token control")
    material = f"20261031|{task}|{state_id}|{condition}"
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little")
    selected = int(np.random.default_rng(seed).choice(eligible))
    named[CONTROLS[-1]] = {"prompt_indices": [selected], "prefix_positions": [start + selected]}
    return {"controls": named, "eligible_random_prompt_indices": eligible,
            "random_seed_material": material, "random_seed": seed,
            "random_overlaps_named": [name for name in CONTROLS[:4]
                                      if selected in named[name]["prompt_indices"]]}


def nominal_requests(decoded, contract):
    actions = np.asarray(decoded, np.float64)
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise ValueError("Expected ten finite native decoded seven-dimensional commands")
    if (contract.get("control_dim") != 6 or contract.get("use_delta") is not True
            or contract.get("impedance_mode") != "fixed"):
        raise ValueError("Unsupported controller contract")
    low, high = np.asarray(contract["input_min"]), np.asarray(contract["input_max"])
    out_low, out_high = np.asarray(contract["output_min"]), np.asarray(contract["output_max"])
    if any(value.shape != (6,) for value in (low, high, out_low, out_high)) or np.any(high <= low):
        raise ValueError("Invalid native affine controller bounds")
    result = np.empty_like(actions)
    result[:, :6] = ((np.clip(actions[:, :6], low, high) - (low + high)/2)
                     * np.abs(out_high-out_low)/np.abs(high-low) + (out_low+out_high)/2)
    result[:, 6] = np.clip(actions[:, 6], -1, 1)
    return result


def cosine(request, origin, destination):
    request, direction = np.asarray(request, float), np.asarray(destination, float)-np.asarray(origin, float)
    denominator = np.linalg.norm(request)*np.linalg.norm(direction)
    return float(np.dot(request, direction)/denominator) if denominator > 0 else None


def geometry_contract(root, evidence):
    """Verify the actual world-coordinate source chain without importing a simulator."""
    contract_path = evidence.bind(root/"native_controller_contract.json")
    contract = read(contract_path)
    controller = evidence.bind(contract["controller_source"])
    if sha(controller) != contract["controller_source_sha256"]:
        raise ValueError("Audited native controller source changed")
    site = REPO/".venv-pro/lib/python3.11/site-packages/robosuite"
    requirements = {
        REPO/"src/rl_vla/envs/libero.py": [
            'position = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)',
            '(position, quaternion_to_axis_angle(raw["robot0_eef_quat"]), gripper)'],
        REPO/"src/rl_vla/diagnostics/observation_bank.py": [
            'np.asarray(sim.data.geom_xpos[geoms]).mean(axis=0).tolist()'],
        site/"robots/single_arm.py": [
            'return np.array(self.sim.data.site_xpos[self.eef_site_id])',
            'arm_action = action[: self.controller.control_dim]'],
        site/"controllers/base_controller.py": [
            'self.ee_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_name)])'],
        site/"utils/control_utils.py": ['goal_position = current_position + delta'],
        controller: ['scaled_delta[:3], self.ee_pos, position_limit=self.position_limits, set_pos=set_pos'],
        REPO/"third_party/openpi/src/openpi/policies/libero_policy.py": [
            'return {"actions": np.asarray(data["actions"][..., :7])}'],
    }
    checks = []
    for path, fragments in requirements.items():
        text = evidence.bind(path).read_text()
        checks.append({"path": str(path), "sha256": sha(path),
                       "verified_fragments": fragments, "passed": all(fragment in text for fragment in fragments)})
    passed = all(row["passed"] for row in checks)
    return contract, {"passed": passed, "status": "supported" if passed else "N/A",
        "reason": "Native XYZ requests and EEF/geom positions share MuJoCo world coordinates" if passed else
                  "The retained sources do not establish the required coordinate chain",
        "checks": checks, "object_position": "Mean native instance geom origins, not an inferred visual centroid or grasp point",
        "eef_position": "First three unnormalized bank state slots from native robot0_eef_pos",
        "scope": "First-command directional alignment only; no integrated endpoint, achieved displacement, or predicted success"}


def figure(path, fig):
    import matplotlib.pyplot as plt
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=150)
    plt.close(fig)


def token_controls(root, task, ids, output, evidence):
    import matplotlib.pyplot as plt
    # model,state,prompt,control,layer,head,camera,region,metric
    scores = np.full((2, 50, 2, 5, 18, 8, 2, 3, len(METRICS)), np.nan)
    choices, comparisons, representative = [], {}, {}
    target, distractor = OBJECTS[task]
    for si, state in enumerate(ids):
        bank = root/"observation_bank"/task/"test"/f"state_{state:06d}"
        bank_metadata = read(evidence.bind(bank/"metadata.json"))
        if sha(bank/"masks.npz") != bank_metadata["masks_sha256"]:
            raise ValueError("Frozen bank mask hash changed")
        with np.load(evidence.bind(bank/"masks.npz"), allow_pickle=False) as archive:
            masks = {key: archive[key] for key in archive.files}
        auxiliary = root/"observation_bank_aux"/task/"test"/bank.name/"old_target_region.npz"
        with np.load(evidence.bind(auxiliary), allow_pickle=False) as archive:
            masks.update({key: archive[key] for key in archive.files})
        for mi, model in enumerate(MODELS):
            directory = root/"traces"/task/model/bank.name
            for ci, condition in enumerate(CONDITIONS):
                filename = ("prefix_metadata.json" if ci == 0 else "counterfactual_prefix_metadata.json") if mi == 0 else (
                    "metadata.json" if ci == 0 else "counterfactual_metadata.json")
                metadata_path = evidence.bind(directory/filename)
                metadata = read(metadata_path)
                selected = select_controls(metadata, task, state, condition)
                identity = (state, condition)
                selection_identity = (metadata["prompt_ids"], metadata["prompt_mask"], selected)
                if mi == 0:
                    comparisons[identity] = selection_identity
                elif comparisons[identity] != selection_identity:
                    raise ValueError("Policy-independent token controls encountered different prompt inputs")
                choices.append({"task": task, "model": model, "state_id": state, "condition": condition,
                    "metadata_path": str(metadata_path), "metadata_sha256": sha(metadata_path),
                    "prompt_token_pieces": metadata["prompt_token_pieces"], **selected})
                with np.load(evidence.bind(directory/f"{condition}.npz"), allow_pickle=False) as archive:
                    attention = archive["prefill_attention"]
                    if not np.array_equal(archive["prompt_ids"][0], metadata["prompt_ids"]):
                        raise ValueError("Saved prompt IDs differ from attention metadata")
                if attention.shape[:3] != (18, 1, 8) or attention.shape[3] != len(metadata["prefill_query_indices"]):
                    raise ValueError("Unexpected captured attention layer/head/query coverage")
                for ki, name in enumerate(CONTROLS):
                    record = selected["controls"][name]
                    # All five declared controls are single-token in this frozen tokenizer.
                    if len(record["prompt_indices"]) != 1:
                        raise ValueError("Multi-token control needs explicit individual-subword export before scoring")
                    query = metadata["prefill_query_indices"].index(record["prefix_positions"][0])
                    for ai, (camera, camera_name) in enumerate(CAMERAS.items()):
                        span = next(value for value in metadata["camera_spans"] if value["name"] == camera_name)
                        if not span["valid"]:
                            raise ValueError("Required control camera is masked")
                        grid = tuple(span["grid"])
                        maps = attention[:, 0, :, query, span["start"]:span["end"]].reshape(18, 8, *grid)
                        if si == 0:
                            representative[mi, ci, ki, ai] = maps.mean(axis=1)
                        for ri, obj in enumerate((target, distractor, "old_target_prism")):
                            occupancy = patch_occupancy(masks[f"{camera}__{obj}"], grid)
                            for layer in range(18):
                                for head in range(8):
                                    metrics = grounding_scores(maps[layer, head], occupancy)
                                    scores[mi, si, ci, ki, layer, head, ai, ri] = [
                                        np.nan if metrics[key] is None else metrics[key] for key in METRICS]
        print(f"{task}: token controls {si+1}/50", flush=True)
    np.savez(output/"token_control_metrics.npz", values=scores, valid=np.isfinite(scores), states=ids, models=MODELS,
             conditions=CONDITIONS, controls=CONTROLS, cameras=list(CAMERAS),
             regions=["target", "distractor", "old_target"], metrics=METRICS)
    write_json(output/"token_control_choices.json", {"complete": True, "records": choices,
        "random_overlaps_named_count": sum(bool(value["random_overlaps_named"]) for value in choices),
        "selection_policy_independent": True, "source_subwords_per_control": 1})
    rows = []
    for mi, model in enumerate(MODELS):
        for ci, condition in enumerate(CONDITIONS):
            for ki, control in enumerate(CONTROLS):
                for ai, camera in enumerate(CAMERAS):
                    for ri, region in enumerate(("target", "distractor", "old_target")):
                        for layer in range(18):
                            data = scores[mi, :, ci, ki, layer, :, ai, ri]
                            row = {"task": task, "model": model, "condition": condition, "control": control,
                                   "camera": camera, "region": region, "layer": layer, "states": 50, "heads_per_state": 8}
                            for index, metric in enumerate(METRICS):
                                valid = np.isfinite(data[..., index])
                                row[metric] = float(data[..., index][valid].mean()) if valid.any() else None
                            rows.append(row)
    write_csv(output/"token_control_layer_means.csv", rows)
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), layout="constrained")
    for ai, camera in enumerate(CAMERAS):
        for ki, control in enumerate(CONTROLS):
            ax = axes[ai, ki]
            for mi, model in enumerate(MODELS):
                for ci, condition in enumerate(CONDITIONS):
                    # Mean over complete observations and heads, with denominator retained.
                    values = scores[mi, :, ci, ki, :, :, ai, 0, 0].mean(axis=(0, 2))
                    ax.plot(range(18), values, label=f"{model} {condition}", linestyle="-" if ci == 0 else "--")
            ax.set(title=f"{camera} · {control}", xlabel="VLM layer", ylabel="True task-target mass / all keys")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{task} · supplementary controls · all50 states/all8 heads\nExploratory amendment after some test outcomes; RLDT prefixes share verified LIBERO values.")
    figure(output/"token_control_layer_mass", fig)
    bank = root/"observation_bank"/task/"test"/f"state_{ids[0]:06d}"
    with np.load(evidence.bind(bank/"observation.npz"), allow_pickle=False) as archive:
        images = {camera: archive[camera] for camera in CAMERAS}
    for mi, model in enumerate(MODELS):
        for ci, condition in enumerate(CONDITIONS):
            fig, axes = plt.subplots(2, 5, figsize=(15, 6), layout="constrained")
            for ai, camera in enumerate(CAMERAS):
                vmax = max(float(representative[mi, ci, ki, ai][8].max()) for ki in range(5))
                for ki, control in enumerate(CONTROLS):
                    axes[ai, ki].imshow(images[camera])
                    axes[ai, ki].imshow(representative[mi, ci, ki, ai][8], extent=(-.5,223.5,223.5,-.5),
                        vmin=0, vmax=vmax, cmap="inferno", alpha=.55, interpolation="nearest")
                    axes[ai, ki].set(title=f"{camera} · {control}"); axes[ai, ki].set_axis_off()
            fig.suptitle(f"{task} state{ids[0]} · {model} {condition} · layer8/head mean\nFirst predetermined frame; same absolute scale within each camera; post-outcome exploratory controls.")
            figure(output/f"control_overlay_{model}_{condition}", fig)
    return {"states": 50, "models": list(MODELS), "prompt_conditions": list(CONDITIONS),
            "controls": list(CONTROLS), "layers": 18, "heads": 8, "cameras": list(CAMERAS),
            "metadata_records": len(choices), "all_head_metrics_shape": list(scores.shape),
            "rldt_reuse": list(POLICIES[1:]), "aggregation": "Single-token rows; arithmetic mean over states/heads for figures only"}


def modality(root, task, ids, output, evidence):
    import matplotlib.pyplot as plt
    rows = csv_rows(evidence.bind(root/"trace_analysis"/task/"routing.csv"))
    indexed = {}
    for row in rows:
        key = (row["policy"], int(row["state_id"]), int(row["step"]), int(row["layer"]), int(row["head"]))
        bucket = indexed.setdefault(key, {})
        if row["camera"] in bucket:
            raise ValueError("Duplicate modality camera row")
        bucket[row["camera"]] = row
    expected = {(p, s, t, l, h) for p in POLICIES for s in ids for t in STEPS for l in range(18) for h in range(8)}
    if set(indexed) != expected:
        raise ValueError("Incomplete modality policy/state/time/layer/head coverage")
    values = np.empty((4, 50, 3, 18, 8, 4)); times = {}
    for (policy, state, step, layer, head), pair in indexed.items():
        if set(pair) != set(CAMERAS):
            raise ValueError("Missing modality camera")
        a, b = pair["image"], pair["wrist_image"]
        for field in ("action_to_text_mass_all_keys", "action_to_action_mass_all_keys", "time"):
            if a[field] != b[field]:
                raise ValueError("Repeated camera-independent modality fields disagree")
        time_value = float(a["time"])
        if step in times and times[step] != time_value:
            raise ValueError("Flow-time metadata differ within a matched step")
        times[step] = time_value
        values[POLICIES.index(policy), ids.index(state), STEPS.index(step), layer, head] = [
            float(a["camera_mass_all_keys"]), float(b["camera_mass_all_keys"]),
            float(a["action_to_text_mass_all_keys"]), float(a["action_to_action_mass_all_keys"])]
    if not np.isfinite(values).all() or (values < 0).any() or np.max(np.abs(values.sum(-1)-1)) > .01:
        raise ValueError("Invalid full-denominator modality mass")
    np.savez(output/"modality_mass.npz", values=values, states=ids, policies=POLICIES,
             steps=STEPS, times=[times[s] for s in STEPS], groups=["external_image","wrist_image","instruction_text","action"])
    table = []; fig, axes = plt.subplots(3, 4, figsize=(15, 10), layout="constrained")
    for pi, policy in enumerate(POLICIES):
        for ti, step in enumerate(STEPS):
            means = values[pi, :, ti].mean(axis=(0, 2))
            for gi, group in enumerate(("external_image", "wrist_image", "instruction_text", "action")):
                axes[ti, pi].plot(range(18), means[:, gi], label=group)
                for layer in range(18):
                    table.append({"task": task, "policy": policy, "step": step, "time": times[step],
                        "layer": layer, "group": group, "mean_mass_all_keys": float(means[layer, gi]), "states": 50, "heads": 8})
            axes[ti, pi].set(title=f"{policy} · t={times[step]:.3g}", xlabel="Expert layer", ylabel="Mass / all keys", ylim=(0, 1))
            axes[ti, pi].grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{task} · same baseline latents/times · all50 states/all8 heads\nInstruction span has no proprioceptive tokens; contextual information can cross modality key groups.")
    write_csv(output/"modality_layer_means.csv", table); figure(output/"modality_layer_curves", fig)
    # Preserve original presentation scales and expose the values they can hide.
    density = read(evidence.bind(root/"trace_analysis"/task/"layer_summary.json"))
    density_values = []
    for camera in CAMERAS:
        for policy in POLICIES:
            for layer in range(18):
                pair = {row["region"]: row["region_density"] for row in density
                        if row["camera"] == camera and row["policy"] == policy
                        and row["step"] == 5 and row["layer"] == layer}
                a, b = pair["target"], pair["distractor"]
                if a is not None and b is not None:
                    density_values.append(float(np.log2((a+1e-12)/(b+1e-12))))
    js = np.empty((2, 3, 18, 8))
    for ai, camera in enumerate(CAMERAS):
        for pi, policy in enumerate(POLICIES[1:]):
            for layer in range(18):
                for head in range(8):
                    data = [float(indexed[policy,state,5,layer,head][camera]["camera_conditional_js_vs_libero_nats"])
                            for state in ids]
                    js[ai,pi,layer,head] = np.mean(data)
    scale_audit = {"density_ratio": {"original_display_limits": [-3,3],
        "actual_min": min(density_values), "actual_max": max(density_values),
        "below_display_limit_cells": sum(value < -3 for value in density_values),
        "above_display_limit_cells": sum(value > 3 for value in density_values), "cells": len(density_values)},
        "head_js_nats": {"original_display_limits": [0,float(np.log(2))],
            "actual_min": float(js.min()), "actual_max": float(js.max()), "cells": int(js.size)},
        "interpretation": "Original broad scales are preserved. Dark or saturated colors do not establish unchanged attention. Magnified JSD retains a shared absolute scale and can include compiler-profile effects; inspect the separate compiler controls."}
    write_json(output/"original_attention_plot_scale_audit.json",scale_audit)
    fig, axes = plt.subplots(2,3,figsize=(12,8),layout="constrained")
    for ai, camera in enumerate(CAMERAS):
        for pi, policy in enumerate(POLICIES[1:]):
            mesh = axes[ai,pi].imshow(js[ai,pi],aspect="auto",vmin=0,vmax=max(float(js.max()),1e-12),cmap="magma")
            axes[ai,pi].set(title=f"{camera} · {policy}",xlabel="Head",ylabel="Expert layer")
    fig.colorbar(mesh,ax=axes.ravel().tolist(),label="Camera-conditional JSD (nats), magnified shared scale",shrink=.7)
    fig.suptitle(f"{task} · supplementary magnification of original head means at t=.5\nOriginal0..ln2 figures remain unchanged; numerical/compiler effects are not separated by this plot.")
    figure(output/"head_attention_drift_magnified",fig)
    return {"states": 50, "policies": list(POLICIES), "steps": list(STEPS), "times": times,
            "layers": 18, "heads": 8, "groups": 4, "max_mass_sum_deviation": float(np.max(np.abs(values.sum(-1)-1)))}


def layer_responses(root, task, ids, output, evidence):
    import matplotlib.pyplot as plt
    source = root/"trace_analysis"/task
    actions = csv_rows(evidence.bind(source/"layer_knockout_actions.csv"))
    velocities = csv_rows(evidence.bind(source/"layer_knockout_velocities.csv"))
    windows = ("early", "middle", "late", *[f"layer_{i:02d}" for i in range(18)])
    expected = {(p, s, mode, w) for p in POLICIES for s in ids[:5] for mode in ("no_image", "no_text") for w in windows}
    def key(row): return row["policy"], int(row["state_id"]), row["mode"], row["window"]
    if len(actions) != len(expected) or {key(r) for r in actions} != expected:
        raise ValueError("Incomplete layer-knockout action coverage")
    if len(velocities) != 3*len(expected) or {(key(r), int(r["step"])) for r in velocities} != {(k,s) for k in expected for s in STEPS}:
        raise ValueError("Incomplete common-latent layer-knockout velocity coverage")
    write_csv(output/"layer_response_actions.csv", actions); write_csv(output/"layer_response_velocities.csv", velocities)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for mi, mode in enumerate(("no_image", "no_text")):
        for pi, policy in enumerate(POLICIES):
            for column, selected in enumerate((windows[3:], windows[:3])):
                means = [np.mean([float(r["normalized_action_l2"]) for r in actions
                                  if r["policy"] == policy and r["mode"] == mode and r["window"] == w]) for w in selected]
                axes[mi, column].plot(range(len(selected)), means, marker=".", label=policy)
                axes[mi, column].set(title=f"{mode} · {'single layer' if column == 0 else 'layer windows'}",
                    xlabel="Layer" if column == 0 else "Window", ylabel="Normalized full-chunk action L2 change")
                if column: axes[mi, column].set_xticks(range(3), ["early0–5", "middle6–11", "late12–17"])
                axes[mi, column].grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{task} · native key-pathway interventions · first5 predetermined states\nActions follow each intervened trajectory from identical bank noise; mean across five paired states.")
    figure(output/"layer_knockout_action_response", fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for mi, mode in enumerate(("no_image", "no_text")):
        for ti, step in enumerate(STEPS):
            for policy in POLICIES:
                means = [np.mean([float(r["normalized_velocity_l2"]) for r in velocities if r["policy"] == policy
                          and r["mode"] == mode and int(r["step"]) == step and r["window"] == w]) for w in windows[3:]]
                axes[mi, ti].plot(range(18), means, label=policy)
            axes[mi, ti].set(title=f"{mode} · flow step{step}", xlabel="Blocked layer", ylabel="Normalized velocity L2 change")
            axes[mi, ti].grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{task} · first5 states · velocity evaluated at the same baseline latent/time\nThis common-latent response differs from the intervened full-sampling action response.")
    figure(output/"layer_knockout_velocity_response", fig)
    return {"states": list(ids[:5]), "policies": list(POLICIES), "modes": ["no_image", "no_text"],
            "windows": list(windows), "action_rows": len(actions), "velocity_rows": len(velocities), "steps": list(STEPS)}


def action_vectors(root, task, ids, output, evidence):
    import matplotlib.pyplot as plt
    contract, geometry = geometry_contract(root, evidence)
    decoded = np.empty((4, 50, 15, 10, 7)); requests = np.empty_like(decoded)
    eef = np.empty((50, 3)); objects = np.empty((50, 2, 3)); geometries = []
    for si, state in enumerate(ids):
        bank = root/"observation_bank"/task/"test"/f"state_{state:06d}"
        meta = read(evidence.bind(bank/"metadata.json"))
        with np.load(evidence.bind(bank/"observation.npz"), allow_pickle=False) as archive:
            state_value = archive["state"]
        if state_value.shape != (8,) or not np.isfinite(state_value).all():
            raise ValueError("Unexpected retained native numeric state")
        eef[si] = state_value[:3]
        for ri, name in enumerate(OBJECTS[task]):
            objects[si, ri] = meta["instance_geom_centroids_xyz"][name]
        geometries.append({"state_id": state, "eef_world_xyz_m": eef[si].tolist(),
            "target_geom_centroid_world_xyz_m": objects[si, 0].tolist(),
            "distractor_geom_centroid_world_xyz_m": objects[si, 1].tolist(),
            "source_metadata_sha256": sha(bank/"metadata.json")})
        for pi, policy in enumerate(POLICIES):
            directory = root/"traces"/task/policy/bank.name
            record = read(evidence.bind(directory/"open_loop_summary.json"))
            if (not record.get("complete") or [r["condition"] for r in record["conditions"]] != list(OPEN)
                    or record["observation_sha256"] != sha(bank/"observation.npz")
                    or record["noise_sha256"] != meta["initial_noise_sha256"]):
                raise ValueError("Open-loop source identity, noise or condition order differs")
            with np.load(evidence.bind(directory/"open_loop_actions.npz"), allow_pickle=False) as archive:
                for ci, condition in enumerate(OPEN):
                    values = archive[condition.replace(":", "_")+"__decoded"]
                    if values.shape == (1, 10, 7): values = values[0]
                    decoded[pi, si, ci] = values
                    requests[pi, si, ci] = nominal_requests(values, contract)
    if not np.isfinite(objects).all(): raise ValueError("Nonfinite retained world geometry")
    own_delta = requests-requests[:, :, :1]
    policy_delta = requests[:, :, 0]-requests[0, :, 0]
    alignment = np.full((4, 50, 15, 2), np.nan)
    if geometry["passed"]:
        for pi in range(4):
            for si in range(50):
                for ci in range(15):
                    for ri in range(2):
                        value = cosine(requests[pi, si, ci, 0, :3], eef[si], objects[si, ri])
                        if value is not None: alignment[pi, si, ci, ri] = value
    np.savez(output/"action_vectors.npz", decoded_commands=decoded, nominal_requests=requests,
             intervention_request_deltas=own_delta, policy_none_request_deltas=policy_delta,
             first_command_cosine=alignment, first_command_cosine_valid=np.isfinite(alignment),
             eef_world_xyz_m=eef, object_geom_centroids_world_xyz_m=objects,
             states=ids, policies=POLICIES, conditions=OPEN, channel_units=["m","m","m","rad","rad","rad","unitless"])
    rows = []
    for pi, policy in enumerate(POLICIES):
        for si, state in enumerate(ids):
            for ci, condition in enumerate(OPEN):
                row = {"task": task, "policy": policy, "state_id": state, "condition": condition,
                    "first_command_target_cosine": float(alignment[pi,si,ci,0]) if np.isfinite(alignment[pi,si,ci,0]) else None,
                    "first_command_distractor_cosine": float(alignment[pi,si,ci,1]) if np.isfinite(alignment[pi,si,ci,1]) else None,
                    "alignment_is_predicted_choice": False}
                for name, delta in (("intervention_vs_own_none", own_delta[pi,si,ci]),
                                    ("policy_none_vs_libero_none", policy_delta[pi,si])):
                    for count in (5,10):
                        for group, sl in (("translation_m",slice(0,3)),("rotation_rad",slice(3,6)),("gripper_unitless",slice(6,7))):
                            row[f"{name}_first{count}_{group}_l2"] = float(np.linalg.norm(delta[:count,sl]))
                a, b = row["first_command_target_cosine"], row["first_command_distractor_cosine"]
                row["nominal_directional_preference_target_minus_distractor"] = a-b if a is not None and b is not None else None
                rows.append(row)
    write_csv(output/"action_vector_metrics.csv", rows)
    write_json(output/"action_geometry_contract.json", {**geometry, "controller_contract": str(root/"native_controller_contract.json"), "states": geometries})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, sl, name in zip(axes, (slice(0,3),slice(3,6),slice(6,7)),
                            ("Translation requests (m)","Rotation requests (rad)","Gripper commands (unitless)"), strict=True):
        for pi, policy in enumerate(POLICIES[1:], start=1):
            ax.plot(range(1,11), np.linalg.norm(policy_delta[pi,...,sl],axis=-1).mean(0), marker=".", label=policy)
        ax.axvline(5.5, color="gray", ls="--", lw=1);ax.set(title=name,xlabel="Predicted command index",ylabel="Mean per-command L2 delta vs LIBERO")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{task} · all50 matched initial observations/PCG bank noises\nFirst5 commands form native execution prefix; these are nominal targets, not achieved motion.")
    figure(output/"action_vector_changes", fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    if geometry["passed"]:
        for pi, policy in enumerate(POLICIES):
            axes[0].scatter(alignment[pi,:,0,1],alignment[pi,:,0,0],s=13,alpha=.6,label=policy)
            difference=alignment[pi,...,0]-alignment[pi,...,1]
            axes[1].plot(range(15),np.nanmean(difference-difference[:,:1],axis=0),marker=".",label=policy)
        axes[0].plot([-1,1],[-1,1],"k--",lw=.8)
        axes[0].set(xlim=(-1,1),ylim=(-1,1),xlabel="Cosine toward distractor geom centroid",ylabel="Cosine toward target geom centroid",title="Unmasked first-command world direction")
        axes[1].set(xticks=range(15),xticklabels=OPEN,ylabel="Change in target-minus-distractor cosine",title="Intervention minus same-policy none")
        axes[1].tick_params(axis="x",rotation=80)
        axes[0].legend(fontsize=7)
    else:
        for ax in axes:ax.text(.5,.5,f"Geometric approach N/A\n{geometry['reason']}",ha="center",va="center",wrap=True);ax.set_axis_off()
    fig.suptitle(f"{task} · nominal direction proxy only · initial retained world geometry\nNot a grasp/choice prediction, collision check, integrated endpoint or achieved displacement.")
    figure(output/"nominal_directional_alignment", fig)
    return {"states":50,"policies":list(POLICIES),"conditions":list(OPEN),"commands_per_chunk":10,
        "executed_prefix_length":5,"metric_rows":len(rows),"geometry_status":geometry["status"],
        "geometry_reason":geometry["reason"],"alignment_is_predicted_choice":False,
        "zero_request_or_direction_cosines_na":int((~np.isfinite(alignment)).sum()),
        "raw_base":"N/A — incompatible action contract"}


def ready(root, task):
    return all(path.exists() and read(path).get("complete") is True for path in
               (root/"traces"/task/"summary.json",root/"trace_analysis"/task/"summary.json"))


def run(root, task):
    import matplotlib
    matplotlib.use("Agg")
    root=Path(root).resolve(); output=root/"supplemental_analysis"/task
    if not ready(root,task):raise ValueError("Complete full50 trace and primary analysis are required")
    evidence=Evidence()
    amendment=evidence.bind(root/"compiler_profile_integration/supplemental_analysis_amendment_v1.json")
    if sha(amendment)!=AMENDMENT_SHA:raise ValueError("Supplement amendment changed")
    scale_addendum=evidence.bind(root/"compiler_profile_integration/supplemental_scale_disclosure_addendum_v1.json")
    if read(scale_addendum)["parent_amendment"]["sha256"]!=AMENDMENT_SHA:
        raise ValueError("Scale-disclosure addendum has a different amendment")
    for path in (Path(__file__),REPO/"scripts/analyze_vla_trace_grid.py",REPO/"src/rl_vla/diagnostics/metrics.py",
                 root/"traces"/task/"summary.json",root/"trace_analysis"/task/"summary.json"):
        evidence.bind(path)
    protocol=read(evidence.bind(root/"tasks"/task/"protocol.json"))
    ids=protocol["tasks"][task]["test_state_ids"]
    if len(ids)!=50 or len(set(ids))!=50:raise ValueError("Expected exactly50 unique predeclared states")
    for policy in POLICIES[1:]:
        audit=read(evidence.bind(root/"traces"/task/policy/"prefix_equality.json"))
        if audit.get("complete") is not True or audit.get("passed") is not True:
            raise ValueError("Frozen LIBERO prefix reuse is not verified")
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"plan.json",{"task":task,"states":ids,"amendment":{"path":str(amendment),"sha256":AMENDMENT_SHA},
                                 "source_sha256":sha(__file__),"policy_applicability":{"token_controls":[*MODELS,*POLICIES[1:]],"native_action_groups":list(POLICIES)}})
    coverage={};groups={}
    for group,call in (("token_controls",token_controls),("modality_routing",modality),
                       ("layer_response",layer_responses),("action_vectors",action_vectors)):
        before=set(output.iterdir())
        coverage[group]=call(root,task,ids,output,evidence)
        groups[group]=[str(path.resolve()) for path in sorted(set(output.iterdir())-before)]
    evidence.verify()
    write_json(output/"input_contract.json",evidence.inputs)
    artifacts=[]
    for path in sorted(output.iterdir()):
        if path.stat().st_size==0:raise ValueError(f"Empty artifact: {path}")
        artifacts.append({"path":str(path.resolve()),"sha256":sha(path),"kind":path.suffix.removeprefix(".")})
    summary={"complete":True,"task":task,"states":50,"state_ids":ids,"coverage":coverage,
        "required_artifact_groups":groups,"artifacts":artifacts,"amendment":{"path":str(amendment),"sha256":AMENDMENT_SHA},
        "scale_disclosure_addendum":{"path":str(scale_addendum),"sha256":sha(scale_addendum)},
        "source_input_sha256":evidence.inputs,"scope":read(amendment)["timing_disclosure"],
        "raw_base_action_applicability":"N/A — incompatible action contract",
        "geometry_approach_claim":"Nominal initial-direction proxy only where source-verified; no predicted choice/endpoint or achieved motion"}
    write_json(output/"summary.json",summary)
    print(json.dumps({"complete":True,"task":task,"summary":str(output/"summary.json"),"artifacts":len(artifacts)}),flush=True)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--tasks",nargs="+",choices=list(OBJECTS),default=list(OBJECTS))
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args(); pending=set(args.tasks)
    while pending:
        for task in sorted(pending):
            summary=args.run/"supplemental_analysis"/task/"summary.json"
            if summary.exists():
                value=read(summary)
                if value.get("complete") is not True or value["amendment"]["sha256"]!=AMENDMENT_SHA:
                    raise ValueError("Existing supplemental summary is not valid")
                for artifact in value["artifacts"]:
                    if sha(artifact["path"])!=artifact["sha256"]:raise ValueError("Supplemental artifact changed")
                pending.remove(task);continue
            if ready(args.run,task):run(args.run,task);pending.remove(task)
            elif not args.wait:raise ValueError(f"Task not ready: {task}")
        if pending:
            print(f"Waiting for primary CPU analysis: {sorted(pending)}",flush=True);time.sleep(15)
