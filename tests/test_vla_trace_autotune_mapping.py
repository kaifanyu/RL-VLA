"""Synthetic parser/normalization controls for offline compiler configuration mapping."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("autotune_mapper", Path(__file__).resolve().parents[1] / "scripts/map_vla_trace_autotune.py")
mapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)


def test_proto_quoted_braces_do_not_change_message_boundaries():
    text = 'version: 3\nresults { hlo: "ROOT f32[1]{0}\\n" result { run_time { nanos: 12 } triton { split_k: 8 } } }'
    result = mapper.single(mapper.proto_fields(text), "results")
    assert mapper.single(result["children"], "hlo")["value"] == "ROOT f32[1]{0}\n"
    kernel = mapper.single(mapper.single(result["children"], "result")["children"], "triton")
    assert text[kernel["start"]:kernel["end"]] == "triton { split_k: 8 }"
    with pytest.raises(ValueError, match="exactly one"):
        mapper.single([kernel, kernel], "triton")


def test_hlo_normalization_preserves_shapes_and_dot_dimensions():
    hlo = '''  %left = f32[2,3]{1,0} parameter(0)
  %right = f32[3,4]{1,0} parameter(1)
  ROOT %product = f32[2,4]{1,0} dot(%left, %right), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="fixture"}
'''
    renamed = hlo.replace("left", "other_a").replace("right", "other_b").replace("fixture", "different_name")
    assert mapper.hlo_expression(hlo) == mapper.hlo_expression(renamed)
    typed = hlo.replace("dot(%left, %right)", "dot(f32[2,3]{1,0} left, f32[3,4]{1,0} right)")
    assert mapper.hlo_expression(hlo) == mapper.hlo_expression(typed)
    assert mapper.hlo_expression(hlo) != mapper.hlo_expression(hlo.replace("rhs_contracting_dims={0}", "rhs_contracting_dims={1}"))


def test_only_declared_projection_split_shape_can_be_reversed():
    expr = {"shape": "f32[8,10,32]{2,1,0}", "op": "dot",
            "attrs": "lhs_batch_dims={1},lhs_contracting_dims={2},rhs_batch_dims={0},rhs_contracting_dims={1}",
            "args": [{"shape": "f32[10,8,128]{2,1,0}", "op": "bitcast", "args": [{"fixture": "left"}]},
                     {"shape": "f32[8,128,32]{2,1,0}", "op": "bitcast", "args": [{"fixture": "right"}]}]}
    assert mapper.canonical_unsplit_projection(expr, 8)["shape"] == "f32[10,32]{1,0}"
    with pytest.raises(ValueError, match="split-K output"):
        mapper.canonical_unsplit_projection(expr, 16)


def test_autotune_constant_omission_keeps_type_and_records_payload():
    canonical = mapper.hlo_expression("ROOT value = bf16[] constant(0.0625)")
    omitted = mapper.hlo_expression("ROOT value = bf16[] constant({...})")
    assert canonical != omitted
    assert mapper.autotune_key_expression(canonical) == mapper.autotune_key_expression(omitted)
    assert mapper.constant_payloads(canonical)[0]["payload"] == [{"literal": "0.0625"}]
    different_type = mapper.hlo_expression("ROOT value = f32[] constant({...})")
    assert mapper.autotune_key_expression(canonical) != mapper.autotune_key_expression(different_type)
