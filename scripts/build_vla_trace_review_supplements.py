"""Extract paired-camera review evidence without assigning reviewed labels.

Consumes an existing build_vla_trace_review_sheets.py manifest. Native main and
OOD reports share this manifest schema. Maximum-lift frames are selected without
requiring contact, exposing strict-contact misses for explicit visual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_frames(peak_step, chunk_end_steps, frame_count):
    """Map a primitive step to its native post-chunk video frame, not a new view."""
    if frame_count != len(chunk_end_steps) + 1:
        raise ValueError("Native video must contain initial frame plus one frame per trajectory chunk")
    if not 1 <= peak_step <= chunk_end_steps[-1]:
        raise ValueError("Peak step is outside recorded trajectory")
    peak_frame = next(i + 1 for i, end in enumerate(chunk_end_steps) if end >= peak_step)
    indices = sorted({0, frame_count - 1, frame_count * 2 // 3,
                      *(max(0, min(frame_count - 1, peak_frame + delta)) for delta in (-2, -1, 0, 1, 2))})
    return peak_frame, indices


def build(manifest_path, output, *, state_ids=None, object_id=None):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text())
    if sha(manifest["report"]) != manifest["report_sha256"]:
        raise ValueError("Primary review manifest no longer matches its report")
    entries = manifest["entries"]
    by_id = {entry["state_id"]: entry for entry in entries}
    if len(by_id) != len(entries):
        raise ValueError("Primary manifest contains duplicate state IDs")
    if state_ids is None:
        selected = [entry for entry in entries if len(entry["evidence"]["sustained_grasp_candidates"]) != 1]
    else:
        if not set(state_ids) <= set(by_id):
            raise ValueError("Requested state IDs are absent from the primary manifest")
        selected = [by_id[state] for state in sorted(set(state_ids))]
    output.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    records = []
    for entry in selected:
        if sha(entry["video"]) != entry["video_sha256"]:
            raise ValueError("Native video changed after primary review evidence was extracted")
        observations, chunks, chunk_ends = [], [], []
        for chunk in sorted(Path(entry["trajectory"]).glob("chunk_*.npz")):
            before = chunk.stat()
            chunk_hash = sha(chunk)
            with np.load(chunk, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
            observations.extend(info["object_telemetry"]["objects"] for info in metadata["step_infos"])
            chunk_ends.append(len(observations))
            after = chunk.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("Trajectory chunk changed during evidence extraction")
            chunks.append({"path": str(chunk), "sha256": chunk_hash, "controlled_step_end": len(observations)})
        if not observations:
            raise ValueError("No native object telemetry found")
        objects = entry["evidence"]["objects"]
        if object_id is not None:
            if object_id not in objects or object_id == "basket_1":
                raise ValueError("Requested object is absent or is the basket")
            relevant = [object_id]
        else:
            relevant = sorted(name for name, value in objects.items() if name != "basket_1" and value["max_lift_m"] >= .03)
            if not relevant:
                relevant = [max((name for name in objects if name != "basket_1"), key=lambda name: objects[name]["max_lift_m"])]
        telemetry_path = output / f"state_{entry['state_id']}_object_measurements.json"
        write_json(telemetry_path, {"reviewed": False, "controlled_steps": len(observations),
                                   "chunk_end_steps": chunk_ends, "objects_by_step": observations})
        frames = list(iio.imiter(entry["video"]))
        height, width = frames[0].shape[:2]
        for name in relevant:
            peak_step = max(range(len(observations)), key=lambda index: observations[index][name]["lift_from_reset_m"]) + 1
            peak_frame, indices = select_frames(peak_step, chunk_ends, len(frames))
            canvas = Image.new("RGB", (2 * (width + 8) + 8, 80 + (height + 28) * ((len(indices) + 1) // 2)), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"{entry['task']} / {Path(entry['report']).stem} / state {entry['state_id']} / seed {entry['seed']}", font=font, fill="black")
            obj = objects[name]
            draw.text((8, 30), f"{name}: max lift {obj['max_lift_m']:.3f}m; both-pad steps {obj['contact_steps']}; sustained {obj['max_consecutive_lift_contact']}", font=font, fill="black")
            draw.text((8, 52), "Native paired cameras; peak lift selected independently of contact. Evidence only, unreviewed.", font=font, fill="black")
            for index, frame_index in enumerate(indices):
                x, y = 8 + (index % 2) * (width + 8), 80 + (index // 2) * (height + 28)
                canvas.paste(Image.fromarray(frames[frame_index]), (x, y))
                if frame_index:
                    step = chunk_ends[frame_index - 1]
                    measurement = observations[step - 1][name]
                    label = f"frame {frame_index} / step {step}: lift {measurement['lift_from_reset_m']:.3f}m; pads={measurement['both_fingerpads_contact']}"
                else:
                    label = "frame 0 / initial reset camera frame"
                draw.text((x, y + height + 2), label, font=font, fill="black")
            image_path = output / f"state_{entry['state_id']}_{name}.png"
            canvas.save(image_path)
            records.append({"task": entry["task"], "state_id": entry["state_id"], "seed": entry["seed"],
                "video": entry["video"], "video_sha256": entry["video_sha256"], "trajectory": entry["trajectory"],
                "trajectory_chunks": chunks, "object_id_for_peak_selection": name,
                "maximum_lift_step": peak_step, "maximum_lift_video_frame": peak_frame,
                "frame_indices": indices, "frame_controlled_step_ends": [0 if i == 0 else chunk_ends[i-1] for i in indices],
                "sheet": str(image_path), "sheet_sha256": sha(image_path),
                "telemetry": str(telemetry_path), "telemetry_sha256": sha(telemetry_path), "reviewed": False})
    result = {"schema_version": 1, "report": manifest["report"], "report_sha256": manifest["report_sha256"],
        "primary_manifest": str(manifest_path), "primary_manifest_sha256": sha(manifest_path),
        "generator": str(Path(__file__).resolve()), "generator_sha256": sha(__file__),
        "reviewed": False, "rule": "Supplemental evidence only. Explicit reviewer records are required; no inference of confirmed grasp from lift alone.",
        "default_selection": "Cases with zero or multiple sustained grasp candidates; each object lifted >=3cm, or the greatest-lift object if none meets that threshold.",
        "requested_state_ids": state_ids, "requested_object_id": object_id, "entries": records}
    write_json(output / "manifest.json", result)
    print(json.dumps({"states": len(selected), "sheets": len(records), "manifest": str(output / "manifest.json")}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", help="Comma-separated explicit state IDs; default is zero/multiple strict candidates")
    parser.add_argument("--object-id", help="Optional native object ID; otherwise render all objects with >=3cm lift")
    args = parser.parse_args()
    state_ids = [int(value) for value in args.states.split(",")] if args.states is not None else None
    build(args.manifest, args.output, state_ids=state_ids, object_id=args.object_id)
