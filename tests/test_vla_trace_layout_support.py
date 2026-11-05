import json
import pytest

from rl_vla.diagnostics.layout_support import layout_support


def fixture(tmp_path):
    folder = tmp_path / "diagnostic_layouts/object_0"
    folder.mkdir(parents=True)
    (folder / "predeclaration.json").write_text(json.dumps({"task": "object_0", "policy_used": False,
        "new_center_xy": {"target": [-.025, -.22], "distractor": [-.025, .035]}}))
    path = folder / "target_moved_distractor_fixed/generation.json"
    path.parent.mkdir()
    accepted = [{"state_id": index, "base_official_test_state_id": 150 + index,
                 "external_target_pixels": 30, "external_distractor_pixels": 3,
                 "wrist_target_pixels": 50, "wrist_distractor_pixels": 90} for index in range(2)]
    path.write_text(json.dumps({"accepted": accepted}))
    return path, accepted


def test_camera_specific_invisibility_does_not_become_both_camera_absence(tmp_path):
    fixture(tmp_path)
    rows = layout_support(tmp_path, expected_count=2)
    external = next(row for row in rows if row["camera"] == "image" and row["role"] == "distractor")
    wrist = next(row for row in rows if row["camera"] == "wrist_image" and row["role"] == "distractor")
    assert external["states_below_generation_threshold_20_pixels"] == 2
    assert external["states_meeting_20_pixels_in_either_camera"] == 2
    assert wrist["states_below_generation_threshold_20_pixels"] == 0


def test_duplicate_base_states_cannot_inflate_support_coverage(tmp_path):
    path, accepted = fixture(tmp_path)
    accepted[1]["base_official_test_state_id"] = 150
    path.write_text(json.dumps({"accepted": accepted}))
    with pytest.raises(ValueError, match="one-to-one"):
        layout_support(tmp_path, expected_count=2)
