"""Layer selection must preserve suffix keys and untouched layers exactly."""

import numpy as np
import pytest

from rl_vla.diagnostics.layer_knockout import layer_keep_mask


@pytest.fixture
def metadata():
    return {"layers": 18, "prefix_tokens": 12, "action_tokens": 3,
            "camera_spans": [{"start": 0, "end": 4}, {"start": 4, "end": 8}],
            "text_span": [8, 12]}


@pytest.mark.parametrize("start", [0, 6, 12])
@pytest.mark.parametrize("mode,span", [("no_image", (0, 8)), ("no_text", (8, 12))])
def test_mask_only_changes_requested_layers_and_keys(metadata, start, mode, span):
    keep = layer_keep_mask(metadata, mode, list(range(start, start + 6)))
    expected = np.ones((18, 15), bool)
    expected[start:start + 6, span[0]:span[1]] = False
    np.testing.assert_array_equal(keep, expected)
    assert keep[:, 12:].all()


def test_empty_layers_and_none_are_true_noops(metadata):
    assert layer_keep_mask(metadata, "no_image", []).all()
    assert layer_keep_mask(metadata, "none", [0, 6, 12]).all()


@pytest.mark.parametrize("layers", [[-1], [18], [0, 0], [1.0], [True]])
def test_invalid_layer_selection_rejected(metadata, layers):
    with pytest.raises(ValueError, match="Layer indices"):
        layer_keep_mask(metadata, "no_image", layers)


def test_explicit_keys_use_full_sequence_coordinates(metadata):
    keep = layer_keep_mask(metadata, "keys", [7], [2, 11])
    assert np.flatnonzero(~keep[7]).tolist() == [2, 11]
    assert keep[np.arange(18) != 7].all()
