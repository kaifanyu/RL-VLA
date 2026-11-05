"""Prepare auditable contact evidence and frame sheets for explicit review.

No labels are assigned by this utility. Native success is retained separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from review_vla_trace_episodes import TARGETS, episode_evidence


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact_frames(trajectory, frame_count):
    """Choose initial, first sustained lift, greatest lift, and final frames."""
    first, peak_step, peak_value = None, None, 0.0
    runs = {}
    step = 0
    for chunk in sorted(Path(trajectory).glob("chunk_*.npz")):
        with np.load(chunk, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        for info in metadata["step_infos"]:
            step += 1
            for name, state in info.get("object_telemetry", {}).get("objects", {}).items():
                if name == "basket_1":
                    continue
                contact = state["both_fingerpads_contact"]
                lift = state["lift_from_reset_m"]
                runs[name] = runs.get(name, 0) + 1 if contact and lift >= 0.03 else 0
                if runs[name] == 3 and first is None:
                    first = step
                if contact and lift > peak_value:
                    peak_step, peak_value = step, lift
    # Native evaluation records initial and each executed <=5-action chunk.
    # These are review images, not simulator states for matched-input inference.
    return [0, min(frame_count - 1, (first + 4) // 5) if first else frame_count // 3,
            min(frame_count - 1, (peak_step + 4) // 5) if peak_step else 2 * frame_count // 3,
            frame_count - 1]


def build(report_path, task, output, *, failed_only=True, rows_per_sheet=6):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads(report_path.read_text())
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 15)
    small = ImageFont.truetype(font_path, 13)
    entries, thumbnails = [], []
    for episode_index, episode in enumerate(report["episodes"]):
        if failed_only and episode["success"]:
            continue
        evidence = episode_evidence(episode["trajectory"], TARGETS[task])
        video = Path(episode["video"])
        frames = list(iio.imiter(video))
        indices = contact_frames(episode["trajectory"], len(frames))
        entry = {"task": task, "report": str(report_path), "report_sha256": sha(report_path),
                 "episode": episode_index + 1, "state_id": episode["reset"]["initial_state_id"],
                 "seed": episode["seed"], "success": episode["success"],
                 "video": str(video), "video_sha256": sha(video),
                 "trajectory": episode["trajectory"], "frame_indices": indices,
                 "sheet": f"sheet_{len(entries) // rows_per_sheet:03d}.jpg",
                 "row": len(entries) % rows_per_sheet, "evidence": evidence}
        entries.append(entry)
        # External camera retains full native224x224 content. Full paired-camera
        # video is linked in each entry for any ambiguous case.
        thumbnails.append([Image.fromarray(frames[i][:224, :224]) for i in indices])
    for batch in range(0, len(entries), rows_per_sheet):
        selected = entries[batch:batch + rows_per_sheet]
        canvas = Image.new("RGB", (1256, 52 + len(selected) * 254), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f"{task} / {report_path.stem} / explicit review required", fill="black", font=font)
        for column, label in enumerate(["initial", "first contact + lift", "peak contact + lift", "final"]):
            draw.text((350 + column * 226, 31), label, fill="black", font=small)
        for row_index, entry in enumerate(selected):
            y = 52 + row_index * 254
            draw.text((8, y + 4), f"state {entry['state_id']} / seed {entry['seed']}", fill="black", font=font)
            draw.text((8, y + 28), f"native success: {entry['success']}", fill="black", font=small)
            objects = entry["evidence"]["objects"]
            relevant = sorted(((name, x) for name, x in objects.items() if name != "basket_1"),
                              key=lambda x: x[1]["max_consecutive_lift_contact"], reverse=True)[:3]
            for j, (name, obj) in enumerate(relevant):
                draw.text((8, y + 58 + j * 47), name, fill="black", font=font)
                draw.text((8, y + 77 + j * 47),
                          f"lift {obj['max_lift_m']:.3f}m; contact {obj['contact_steps']}; sustained {obj['max_consecutive_lift_contact']}",
                          fill="black", font=small)
            for col, frame in enumerate(thumbnails[batch + row_index]):
                canvas.paste(frame, (350 + col * 226, y))
                draw.text((350 + col * 226, y + 226), f"frame {entry['frame_indices'][col]}", fill="black", font=small)
        canvas.save(output / selected[0]["sheet"], quality=95)
    record = {"schema_version": 1, "task": task, "report": str(report_path),
              "report_sha256": sha(report_path), "reviewed": False,
              "rule": "Sheets are evidence only. Explicit reviewer records are required for failure labels.",
              "camera": "Native external camera, full224x224; original paired-camera video retained",
              "entries": entries}
    (output / "manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "episodes": len(entries), "sheets": (len(entries) + rows_per_sheet - 1) // rows_per_sheet}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task", choices=list(TARGETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-successes", action="store_true")
    args = parser.parse_args()
    build(args.report, args.task, args.output, failed_only=not args.include_successes)
