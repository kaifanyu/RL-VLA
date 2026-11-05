"""Strict support and resume identity tests without a model or simulator."""
import ast
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rl_vla.diagnostics.structural_na_resume import (
    StructuralNA,
    condition_summary,
    episode_identity,
    strict_support,
    validate_amendment,
    write,
)

ROOT = Path(__file__).resolve().parents[1]


def raw_masks(target_count=3, eligible_count=4):
    masks = {}
    for camera in ("image", "wrist_image"):
        target = np.zeros((4, 5), bool)
        target.ravel()[:target_count] = True
        allowed = np.zeros_like(target)
        allowed.ravel()[target_count:target_count + eligible_count] = True
        masks[camera + "__target"] = target
        masks[camera + "__unowned_background"] = allowed
    return {"diagnostic_masks": masks}


def test_available_control_exact_and_rng_untouched():
    source = ROOT / "src/rl_vla/diagnostics/interventions.py"
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "matched_area_control")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102 - trusted repository AST, no external input
    original = namespace["matched_area_control"]
    raw = raw_masks()
    before = copy.deepcopy(raw)
    for seed in (100000, 170000, 180007):
        expected = original(raw["diagnostic_masks"]["image__target"], raw["diagnostic_masks"]["image__unowned_background"], seed=seed)
        support = strict_support(raw, "random:mosaic", "target")
        actual = original(raw["diagnostic_masks"]["image__target"], raw["diagnostic_masks"]["image__unowned_background"], seed=seed)
        assert np.array_equal(actual, expected) and actual.sum() == 3
        assert all(item["exact_match_possible"] for item in support)
    assert all(np.array_equal(value, before["diagnostic_masks"][key]) for key, value in raw["diagnostic_masks"].items())


@pytest.mark.parametrize("condition", ["random:background_fill", "random:mosaic"])
def test_only_insufficient_unique_background_is_na(condition):
    raw = raw_masks(4, 3)
    with pytest.raises(StructuralNA) as error:
        strict_support(raw, condition, "target")
    assert error.value.support[0] == {"camera": "image", "requested_pixels": 4, "available_pixels": 3, "exact_match_possible": False}
    assert strict_support(raw, "target:mosaic", "target") == []
    assert strict_support(raw, "none", "target") == []
    raw["diagnostic_masks"]["image__unowned_background"] = np.zeros((2, 2), bool)
    with pytest.raises(ValueError, match="Malformed"):
        strict_support(raw, condition, "target")


def test_target_overlap_cannot_create_false_support():
    raw = raw_masks(4, 3)
    raw["diagnostic_masks"]["image__unowned_background"] |= raw["diagnostic_masks"]["image__target"]
    with pytest.raises(StructuralNA) as error:
        strict_support(raw, "random:mosaic", "target")
    assert error.value.support[0]["available_pixels"] == 3


def test_zero_area_control_remains_available():
    assert all(x["exact_match_possible"] for x in strict_support(raw_masks(0, 0), "random:mosaic", "target"))


def rows():
    return [{"episode": i, "success": i % 2 == 0} for i in range(10)]


def test_one_na_makes_whole_primary_condition_unavailable():
    data = rows()
    data[8].update(status="structural_NA", success=None, structural_evidence={"path": "proof", "sha256": "a"})
    result = condition_summary("random:mosaic", data, amendment={}, reuse_manifest={}, elapsed=1)
    assert result["complete"] and result["execution_coverage_complete"]
    assert result["completed_outcome_cases"] == 9 and result["structural_na_cases"] == 1
    assert not result["primary_outcome_available"]
    assert result["successes"] is result["primary_success_rate"] is result["primary_paired_comparison"] is None
    assert result["episodes"] == data


def test_no_na_keeps_existing_success_count():
    result = condition_summary("none", rows(), amendment={}, reuse_manifest={}, elapsed=1)
    assert result["successes"] == 5 and result["primary_success_rate"] == .5
    with pytest.raises(ValueError):
        condition_summary("none", rows()[:-1], amendment={}, reuse_manifest={}, elapsed=1)


def test_arbitrary_missing_or_false_outcome_cannot_be_na():
    data = rows()
    data[8]["success"] = None
    with pytest.raises(ValueError):
        condition_summary("none", data, amendment={}, reuse_manifest={}, elapsed=1)
    data[8].update(status="structural_NA", success=False, structural_evidence={})
    with pytest.raises(ValueError):
        condition_summary("none", data, amendment={}, reuse_manifest={}, elapsed=1)


def test_nonzero_original_identity_is_required():
    row = {"episode": 8, "seed": 100008, "initial_state_id": 158,
           "reset": {"initial_state_id": 158}, "condition": "random:background_fill",
           "checkpoint": "saved", "selection_sha256": "selection", "technical_smoke": False}
    episode_identity(row, 8, "random:background_fill", "saved", "selection")
    for key, value in (("episode", 0), ("seed", 100000), ("initial_state_id", 150)):
        changed = dict(row, **{key: value})
        with pytest.raises(ValueError):
            episode_identity(changed, 8, "random:background_fill", "saved", "selection")


def test_candidate_amendment_cannot_launch_research():
    path = ROOT / "runs/vla_trace_objects_20261031/compiler_profile_integration/causal_structural_na_v1/amendment_candidate_v1.json"
    value = json.loads(path.read_text())
    validate_amendment(value, execution=False)
    with pytest.raises(ValueError, match="approved"):
        validate_amendment(value, execution=True)
    value["control_semantics"]["no_area_capping"] = False
    with pytest.raises(ValueError):
        validate_amendment(value, execution=False)


def test_native_numerical_calls_and_key_schedule_are_identical():
    def function(path):
        return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_condition")
    old = function(ROOT / "scripts/run_vla_trace_interventions_gpu6_v1.py")
    new = function(ROOT / "scripts/run_vla_trace_interventions_gpu6_resume_v1.py")
    names = {"sample_with_intervention", "native_sample", "execute_chunk", "transformed_observation"}

    def calls(node):
        result = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = ast.unparse(child.func)
                if name in names or name in {"flow.prepare", "flow.decode", "flow.sample", "jax.random.PRNGKey", "jax.random.split", "env.reset"}:
                    result.append(ast.dump(child, include_attributes=False))
        return sorted(result)
    assert calls(old) == calls(new)
    new_source = ast.unparse(new)
    assert 'reset != reused[episode]["reset"]' in new_source or "reset != reused[episode]['reset']" in new_source
    assert new_source.index("env.reset(") < new_source.index("episode < len(reused)")
    assert "seed=args.seed + episode * 10000 + decision" in new_source


def test_real_resume_loop_resets_skips_and_continues_after_na(tmp_path):
    """Execute the actual evaluator AST with deterministic model/env doubles."""
    path = ROOT / "scripts/run_vla_trace_interventions_gpu6_resume_v1.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_condition")
    resets, applied_seeds, sampled_keys, events = [], [], [], []

    class Env:
        def reset(self, *, seed):
            index = len(resets)
            resets.append(seed)
            raw = raw_masks(4, 3 if index == 8 else 4)
            return raw, {"initial_state_id": 150 + index, "backend_seed": seed + 3}

        def close(self):
            pass

    env = Env()
    actions = np.zeros((1, 10, 32), np.float32)
    flow = SimpleNamespace(horizon=10, action_dim=32, frozen={},
        prepare=lambda observations: {"prefix_mask": np.ones((1, 1), bool)},
        decode=lambda *args: np.zeros((1, 10, 7), np.float32),
        sample=lambda *args, **kwargs: (actions.copy(), None))

    def sample(*args, **kwargs):
        sampled_keys.append(args[3])
        return actions.copy(), None

    def mask(raw, condition, *, target, distractor, seed):
        applied_seeds.append(seed)
        return raw, {}

    def execute(env, decoded, gamma, record_step):
        for _ in range(5):
            record_step(np.zeros(7), {}, 0, False, False, {"object_telemetry": {"simulator_state": [1, 2, 3]}})
        return {}, 0, True, False, {"executed_steps": 5, "is_success": True}

    def evidence(*args):
        events.append({"episode": args[1], "decision": args[2], "key": args[7], "support": args[11]})
        return {"path": "retained-witness", "sha256": "binding"}

    namespace = dict(np=np, copy=copy, json=json, time=time, Path=Path,  # noqa: C408 - readable evaluator namespace
        jax=SimpleNamespace(jit=lambda f: f, random=SimpleNamespace(PRNGKey=lambda seed: seed,
            split=lambda key: (key + 1, key + 2), normal=lambda key, shape: np.zeros(shape))),
        make_primitive_env=lambda _: env, evaluation_frame=lambda _: None,
        execute_chunk=execute, sample_with_intervention=sample,
        native_sample=lambda *a, **k: (actions.copy(), None),
        _token_metadata=lambda *a: {"prefix_tokens": 1},
        transformed_observation=lambda *a: None, strict_support=strict_support,
        StructuralNA=StructuralNA, save_structural_evidence=evidence,
        masked_observation=mask, sha=lambda _: "selection", condition_summary=condition_summary, write=write)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - trusted evaluator AST
    reused = [{"episode": i, "seed": 100000 + i, "initial_state_id": 150 + i,
               "reset": {"initial_state_id": 150 + i, "backend_seed": 100003 + i}, "success": True}
              for i in range(8)]
    args = SimpleNamespace(episodes=10, seed=100000, record_episodes=0, technical_smoke=False,
        checkpoint=None, selection="original", amendment_record={}, reuse_manifest_record={},
        reuse_condition={"rows": reused, "raw_lines": [json.dumps(r) for r in reused],
                         "episode_artifacts": [], "original_directory": str(tmp_path / "old")})
    summary = namespace["evaluate_condition"](flow, {}, {"env": {"kwargs": {}},
        "rldt": {"execute_steps": 5, "ode_steps": 10, "gamma": .99}}, args,
        "random:background_fill", tmp_path / "new", {"text_span": [0, 1]}, "target", "distractor")
    assert resets == list(range(100000, 100010))
    assert applied_seeds == [190000]
    assert sampled_keys == [100011, 100011]  # action plus explicit native no-op, episode9.
    assert events[0]["episode"] == 8 and events[0]["decision"] == 0
    assert events[0]["key"] == 100010
    assert summary["episodes"][:8] == reused
    assert summary["episodes"][8]["success"] is None
    assert summary["episodes"][9]["success"] is True
    assert summary["completed_outcome_cases"] == 9 and summary["structural_na_cases"] == 1
    assert summary["successes"] is None
