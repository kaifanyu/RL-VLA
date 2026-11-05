import numpy as np
import pytest

from rl_vla.diagnostics.interventions import action_delta, matched_area_control, replace_region
from rl_vla.diagnostics.metrics import grounding_scores, paired_success, patch_occupancy, wilson
from rl_vla.diagnostics.probes import binary_auc, fit_linear_probe, score_probe


def episodes(successes):
    return [{"seed": 100000 + i, "reset": {"initial_state_id": i, "backend_seed": 100 + i}, "success": s}
            for i, s in enumerate(successes)]


def test_pairing_is_identity_based_and_rejects_silent_mismatch():
    a, b = episodes([True, False, False, False]), episodes([False, True, True, False])
    result = paired_success(a, b[::-1])
    assert result["recovered"] == 2 and result["regressed"] == 1
    assert result["difference"] == .25
    assert result["mcnemar_exact_two_sided_p"] == 1
    b[0]["seed"] += 42
    with pytest.raises(ValueError, match="different reset/noise"):
        paired_success(a, b)


def test_exact_discordance_and_uncertainty():
    result = paired_success(episodes([False]*10), episodes([True]*10))
    assert result["mcnemar_exact_two_sided_p"] == 2 / 2**10
    lo, hi = wilson(50, 50)
    assert .92 < lo < .94 and np.isclose(hi, 1)
    with pytest.raises(ValueError):
        wilson(0, 0)


def test_attention_mass_preserves_full_softmax_denominator():
    attention = np.array([[.1, .2], [.05, .05]])
    mask = np.array([[1., 0], [0, 0]])
    score = grounding_scores(attention, mask)
    assert score["region_mass_all_keys"] == .1
    assert np.isclose(score["camera_mass_all_keys"], .4)
    assert np.isclose(score["region_fraction_camera"], .25)
    assert score["region_density"] == .1
    assert not score["peak_hit"]


def test_fractional_patch_masks_preserve_tiny_objects():
    mask = np.zeros((224, 224), bool)
    mask[0, 0] = True
    occupancy = patch_occupancy(mask, (16, 16))
    assert occupancy[0, 0] == 1 / 196
    assert np.count_nonzero(occupancy) == 1


def test_mask_controls_modify_only_region_and_preserve_area():
    image = np.arange(16*16*3, dtype=np.uint8).reshape(16, 16, 3)
    mask = np.zeros((16, 16), bool)
    mask[3:7, 2:6] = True
    for mode in ("background_fill", "mosaic"):
        edited = replace_region(image, mask, mode)
        np.testing.assert_array_equal(edited[~mask], image[~mask])
    np.testing.assert_array_equal(replace_region(image, mask, "none"), image)
    control = matched_area_control(mask, np.ones_like(mask), seed=7)
    assert control.sum() == mask.sum() and not (control & mask).any()
    np.testing.assert_array_equal(control, matched_area_control(mask, np.ones_like(mask), seed=7))
    assert action_delta(np.zeros((10, 7)), np.zeros((10, 7)))["l2"] == 0


def test_probe_fit_excludes_test_and_auc_accounts_for_ties():
    x = np.array([[-2., 0], [-1, 0], [1, 0], [2, 0]])
    y = np.array([0, 0, 1, 1])
    probe = fit_linear_probe(x, y)
    assert score_probe(probe, x, y)["accuracy"] == 1
    assert binary_auc([True, False], [0., 0.]) == .5
    with pytest.raises(ValueError, match="training split"):
        fit_linear_probe(x, y, fit_split="test")
