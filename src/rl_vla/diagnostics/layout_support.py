"""Describe frozen OOD visibility/support without inspecting policy outcomes."""
import hashlib
import json
from pathlib import Path

VARIANTS = ("both_new_regions", "continuous_new_offsets", "target_moved_distractor_fixed", "distractor_moved_target_fixed")


def layout_support(root, *, expected_count=50):
    root = Path(root)
    rows = []
    for task in ("object_0", "object_1", "object_2"):
        declaration = root / "diagnostic_layouts" / task / "predeclaration.json"
        if not declaration.exists():
            continue
        specification = json.loads(declaration.read_text())
        if specification["task"] != task or specification["policy_used"] is not False:
            raise ValueError("OOD geometry inventory requires policy-free matching task declaration")
        for variant in VARIANTS:
            path = declaration.parent / variant / "generation.json"
            if not path.exists():
                continue
            accepted = json.loads(path.read_text())["accepted"]
            if len(accepted) != expected_count or sorted(row["state_id"] for row in accepted) != list(range(expected_count)):
                raise ValueError("OOD support lacks exactly the frozen state identities")
            if sorted(row["base_official_test_state_id"] for row in accepted) != list(range(150, 150 + expected_count)):
                raise ValueError("OOD support does not preserve one-to-one official base states")
            for camera, prefix in (("image", "external"), ("wrist_image", "wrist")):
                for role in ("target", "distractor"):
                    pixels = [row[f"{prefix}_{role}_pixels"] for row in accepted]
                    if any(not isinstance(value, int) or value < 0 for value in pixels):
                        raise ValueError("Invalid frozen visible pixel count")
                    either = sum(row[f"external_{role}_pixels"] >= 20 or row[f"wrist_{role}_pixels"] >= 20 for row in accepted)
                    rows.append({"task": task, "variant": variant, "camera": camera, "role": role,
                                 "states": expected_count, "visible_pixels_min": min(pixels), "visible_pixels_max": max(pixels),
                                 "states_below_generation_threshold_20_pixels": sum(value < 20 for value in pixels),
                                 "states_meeting_20_pixels_in_either_camera": either,
                                 "declared_new_target_center_xy": specification["new_center_xy"]["target"],
                                 "declared_new_distractor_center_xy": specification["new_center_xy"]["distractor"],
                                 "source": str(path.relative_to(root)), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "declaration": str(declaration.relative_to(root)), "declaration_sha256": hashlib.sha256(declaration.read_bytes()).hexdigest(),
                                 "scope": "Frozen generation audit, not policy success; variants reuse base states and overlapping new-center neighborhoods and are not independent novel-location replicates"})
    return rows
