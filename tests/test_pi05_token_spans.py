from types import SimpleNamespace

import numpy as np
import pytest

from rl_vla.diagnostics.pi05_attention import _token_metadata, key_keep_mask


def metadata():
    obs = SimpleNamespace(
        images={"base_0_rgb": None, "left_wrist_0_rgb": None, "right_wrist_0_rgb": None},
        image_masks={
            "base_0_rgb": np.array([True]),
            "left_wrist_0_rgb": np.array([True]),
            "right_wrist_0_rgb": np.array([False]),
        },
        tokenized_prompt=np.array([[2, 100, 200, 0]]),
        tokenized_prompt_mask=np.array([[True, True, True, False]]),
    )
    return _token_metadata(obs, 12, 10)


def test_images_then_text_and_dummy_camera_preserved():
    m = metadata()
    assert [s["start"] for s in m["camera_spans"]] == [0, 4, 8]
    assert m["camera_spans"][-1]["valid"] is False
    assert m["text_span"] == [12, 16]
    assert m["prefill_query_indices"] == [12, 13, 14]
    assert m["action_query_span"] == [0, 10]
    assert m["action_key_span"] == [16, 26]


def test_knockout_key_indices_are_not_query_indices():
    m = metadata()
    image = key_keep_mask(m, "no_image")
    assert not image[:12].any()
    assert image[12:].all()
    text = key_keep_mask(m, "no_text")
    assert text[:12].all() and not text[12:16].any() and text[16:].all()
    assert key_keep_mask(m).all()
    with pytest.raises(ValueError):
        key_keep_mask(m, "keys", [-1])
