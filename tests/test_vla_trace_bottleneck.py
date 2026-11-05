import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rl_vla.diagnostics.attention_spectrum import (
    attention_drift,
    compare_attention,
    key_groups,
    spectrum,
    validate_pair,
)
from rl_vla.diagnostics.probe_controls import (
    position_features,
    shuffled_patch_labels,
    state_metrics,
    validate_disjoint_records,
)
from rl_vla.diagnostics.probes import fit_linear_probe, probe_scores
from rl_vla.diagnostics.value_transport import (
    compare_value_transport,
    linear_cka,
    stored_cache_hash,
)


def metadata():
    return {
        "prefix_tokens": 5,
        "action_tokens": 2,
        "camera_spans": [
            {"name": "image", "start": 0, "end": 2, "valid": True},
            {"name": "dummy", "start": 2, "end": 3, "valid": False},
        ],
        "text_span": [3, 5],
        "prompt_mask": [True, False],
        "prompt_ids": [7, 0],
        "bank_record": {"observation_sha256": "image", "state_sha256": "state"},
        "noise_sha256": "noise",
    }


def test_spectrum_known_operators_and_scale_invariance():
    for key in ("effective_rank", "stable_rank", "participation_rank", "numerical_rank"):
        assert spectrum(np.eye(4))[key] == pytest.approx(4)
        assert spectrum(np.ones((4, 7)))[key] == pytest.approx(1)
        assert spectrum(np.zeros((4, 7)))[key] == 0
    a = np.diag([4.0, 2.0, 1.0])
    assert spectrum(a)["energy90_rank"] == 2
    assert spectrum(a)["stable_rank"] == pytest.approx(21 / 16)
    assert spectrum(a)["effective_rank"] == pytest.approx(spectrum(a * 0.001)["effective_rank"])
    with pytest.raises(ValueError):
        spectrum([[np.nan]])


def test_mass_drift_is_not_lost_to_conditional_normalization():
    a = np.array([[0.1, 0.3], [0.2, 0.2]])
    result = attention_drift(a, a * 0.1)
    assert result["conditional_js_nats"] == pytest.approx(0, abs=1e-15)
    assert result["relative_frobenius_delta"] == pytest.approx(0.9)
    assert result["mass_delta_all_keys"] == pytest.approx(-0.36)
    assert attention_drift([[1.0, 0]], [[0.0, 1]])["conditional_js_nats"] == pytest.approx(
        np.log(2)
    )
    zero = attention_drift(a, np.zeros_like(a))
    assert zero["conditional_js_nats"] is None
    assert zero["conditional_undefined_queries"] == 2


def test_masks_heads_and_full_denominator():
    meta = metadata()
    assert key_groups(meta, 7)["prefix"].tolist() == [0, 1, 3]
    attention = np.zeros((1, 1, 2, 2, 7))
    attention[..., [0, 1, 3, 5, 6]] = 0.2
    attention[0, 0, 1, 0] = [1, 0, 0, 0, 0, 0, 0]
    rows = list(compare_attention(attention, attention, meta))
    visual = [r for r in rows if r["key_group"] == "visual"]
    assert len(visual) == 2
    assert visual[0]["reference_mass_all_keys"] == pytest.approx(0.4)
    assert visual[0]["reference_spectrum"]["effective_rank"] == pytest.approx(1)
    assert visual[1]["reference_spectrum"]["effective_rank"] > 1
    assert all(r["frobenius_delta"] == 0 for r in rows)
    with pytest.raises(ValueError, match="denominator"):
        list(compare_attention(attention * 0.5, attention, meta))


def test_pair_identity_rejects_same_seed_different_inputs():
    a = {"x_t": np.zeros((1, 2, 3)), "t": np.ones(1)}
    validate_pair(a, a, metadata(), metadata())
    with pytest.raises(ValueError, match="x_t"):
        validate_pair(a, {**a, "x_t": a["x_t"] + 1}, metadata(), metadata())
    meta = metadata()
    meta["bank_record"]["observation_sha256"] = "changed"
    with pytest.raises(ValueError, match="observation"):
        validate_pair(a, a, metadata(), meta)


def rows():
    return [
        {
            "labels": np.array([0, 1, 2]),
            "valid": np.ones(3, bool),
            "state_id": state,
            "camera": camera,
        }
        for state in range(4)
        for camera in ("image", "wrist_image")
    ]


def test_position_prior_exposes_perfect_spatial_shortcut_and_shuffle_control():
    data = rows()
    x = position_features(data)
    y = np.concatenate([r["labels"] for r in data])
    probe = fit_linear_probe(x, y)
    result = state_metrics(probe, data, probe_scores(probe, x))
    assert result["states"] == 4
    assert result["balanced_accuracy"] == 1
    assert result["balanced_accuracy_state_bootstrap95"] == [1.0, 1.0]
    shifted = [{**r, "labels": np.roll(r["labels"], 1)} for r in data]
    assert state_metrics(probe, shifted, probe_scores(probe, x))["balanced_accuracy"] == 0
    shuffled = shuffled_patch_labels(data, seed=17)
    assert np.all(np.sort(shuffled.reshape(-1, 3), axis=1) == [0, 1, 2])
    assert not np.array_equal(y, shuffled)
    np.testing.assert_array_equal(shuffled, shuffled_patch_labels(data, seed=17))


def test_split_hash_leakage_rejected_even_if_ids_differ():
    train = [{"state_sha256": "a", "observation_sha256": "b"}]
    validate_disjoint_records(train, [{"state_sha256": "c", "observation_sha256": "d"}])
    with pytest.raises(ValueError, match="leakage"):
        validate_disjoint_records(train, [{"state_sha256": "c", "observation_sha256": "b"}])


def test_value_transport_respects_gqa_and_exposes_value_cancellation():
    meta = metadata()
    a = np.zeros((1, 1, 4, 2, 7))
    a[..., 0] = 0.5
    a[..., 1] = 0.5
    b = a.copy()
    b[..., 0] = 1
    b[..., 1] = 0
    v = np.zeros((1, 1, 5, 2, 2))
    v[:, :, 0, 0] = [1.0, 0.0]
    v[:, :, 1, 0] = [-1.0, 0.0]
    v[:, :, :2, 1] = [2.0, 0.0]
    rows = [r for r in compare_value_transport(a, b, v, meta) if r["key_group"] == "visual"]
    assert [r["kv_head"] for r in rows] == [0, 0, 1, 1]
    assert rows[0]["reference_spectrum"]["effective_rank"] == 0
    assert rows[0]["candidate_spectrum"]["effective_rank"] == pytest.approx(1)
    assert rows[0]["relative_frobenius_delta"] is None
    assert rows[2]["frobenius_delta"] == 0
    x = np.array([[1.0, 2.0], [4.0, 5.0], [1.0, 7.0]])
    assert linear_cka(x, -2 * x + 10) == pytest.approx(1)
    assert linear_cka(x, np.ones_like(x)) is None


def test_lossless_bf16_hash_without_accelerator_dependencies():
    a = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    expected = hashlib.sha256(
        str(((3,), "bfloat16")).encode() + np.array([0x3F80, 0xC000, 0], dtype=np.uint16).tobytes()
    ).hexdigest()
    assert stored_cache_hash(a, "bfloat16") == expected
    with pytest.raises(ValueError, match="lossless"):
        stored_cache_hash(np.array([1.00001], np.float32), "bfloat16")


def test_rank_cli_writes_auditable_results_and_rejects_missing_states(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_vla_trace_rank.py"
    spec = importlib.util.spec_from_file_location("rank_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for policy in ("pi05_libero", "rldt_000025"):
        for state in ("state_000000", "state_000001"):
            folder = tmp_path / "traces" / policy / state
            folder.mkdir(parents=True)
            (folder / "metadata.json").write_text(json.dumps(metadata()))
            np.savez(
                folder / "expert_step00_none.npz",
                attention=np.ones((1, 1, 1, 2, 7)) / 7,
                x_t=np.zeros((1, 2, 3)),
                t=np.ones(1),
            )
    result = module.run(
        tmp_path / "traces", tmp_path / "output", policies=["rldt_000025"], steps=[0]
    )
    assert result["complete"] and result["states"] == 2
    assert len(result["sources_sha256"]) == 8
    assert all(
        row["mean"] == 0 for row in result["summaries"] if row["metric"] == "effective_rank_delta"
    )
    (tmp_path / "traces/rldt_000025/state_000002").mkdir()
    with pytest.raises(ValueError, match="coverage"):
        module.run(tmp_path / "traces", tmp_path / "invalid", policies=["rldt_000025"], steps=[0])
