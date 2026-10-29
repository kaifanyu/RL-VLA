"""Development screens must preserve published tests for later evaluation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def screening(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "benchmark_screen_preparation", scripts / "prepare_benchmark_screen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_pro_tasks_have_both_spatial_and_object_perturbations(screening, monkeypatch, tmp_path):
    monkeypatch.setattr(screening, "source", lambda _: tmp_path)
    maps = {f"{suite}_{suffix}": [f"task_{index}" for index in range(10)]
            for suite in screening.HORIZONS for suffix in ("swap", "object")}
    monkeypatch.setattr(screening, "mapping", lambda _: maps)
    monkeypatch.setattr(screening, "download", lambda path: (tmp_path / path, {"path": path}))
    cells = screening.pro_cells()
    assert len(cells) == len({cell["case_id"] for cell in cells}) == 40
    for suite in screening.HORIZONS:
        for category in ("Position", "Object"):
            selected = [cell for cell in cells if cell["suite"] == suite and cell["category"] == category]
            assert [cell["task_id"] for cell in selected] == list(range(10))


def test_plan_uses_fresh_development_files_and_disjoint_seed_ranges(screening, monkeypatch, tmp_path):
    cells = [{"case_id": f"case_{i}", "benchmark": "pro", "published_states": {"path": "/published"}}
             for i in range(3)]
    monkeypatch.setattr(screening, "pro_cells", lambda: cells)
    monkeypatch.setattr(screening, "plus_cells", list)
    screening.prepare_plan(tmp_path)
    plan = json.loads((tmp_path / "screen_plan.json").read_text())
    assert all(cell["split"] == "development_only" for cell in plan["cases"])
    assert all(cell["states"] != cell["published_states"]["path"] for cell in plan["cases"])
    assert all(cell["episodes"] == 5 for cell in plan["cases"])
    seeds = [set(range(cell["generation_seed"], cell["generation_seed"] + 100)) for cell in plan["cases"]]
    assert len(set.union(*seeds)) == 300
    plan["episodes_per_cell"] = 99
    (tmp_path / "screen_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="frozen screening plan"):
        screening.prepare_plan(tmp_path)


def test_excludes_published_and_matching_adaptation_states(screening, monkeypatch, tmp_path):
    monkeypatch.setattr(screening, "ROOT", tmp_path)
    published = tmp_path / "published.npy"
    combined = tmp_path / "combined.npy"
    np.save(published, [[0., 1., 2.], [0., 3., 4.]])
    np.save(combined, [[0., 1., 2.], [0., 3., 4.], [0., 5., 6.]])
    protocol = tmp_path / "runs/pro_multimethod_20261027/tasks/spatial_1/protocol.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_text(json.dumps({"tasks": {
        "spatial_1": {"pro": {"bddl": {"sha256": "matching"},
                               "states": {"path": str(combined)}}}}}))
    cell = {"benchmark": "pro", "published_states": {"path": str(published)},
            "bddl_record": {"sha256": "matching"}}
    values, records = screening.exclusion_states(cell)
    assert len(values) == 3
    assert len(records) == 2
    cell["bddl_record"]["sha256"] = "different_condition"
    values, records = screening.exclusion_states(cell)
    assert len(values) == 2
    assert len(records) == 1


def test_archive_prefix_removed_and_unsafe_paths_rejected(screening):
    assert str(screening.archive_relative_path("build/machine/assets/scenes/table.xml")) == (
        "assets/scenes/table.xml")
    for path in ("/assets/file", "prefix/assets/../../escape", "unrelated/file"):
        with pytest.raises(ValueError, match="Unsafe or unexpected"):
            screening.archive_relative_path(path)


def test_asset_subset_traces_selected_bddl_object_types(screening, monkeypatch, tmp_path):
    bddl = tmp_path / "case.bddl"
    bddl.write_text("(:objects thing_1 - selected_thing other_1 - base_object) (:goal example)")
    monkeypatch.setattr(screening, "plus_cells", lambda: [{"real_bddl": str(bddl)}])
    classes = tmp_path / "libero/libero/envs/objects/custom_objects.py"
    classes.parent.mkdir(parents=True)
    classes.write_text('''class Selected:
    def __init__(self, obj_name="selected_thing"):
        path = "assets/new_objects/selected_category/instance/usd/MJCF/model.xml"
class Unused:
    def __init__(self, obj_name="unused_thing"):
        path = "assets/new_objects/unneeded_category/instance/usd/MJCF/model.xml"
''')
    assert screening.required_new_object_categories(tmp_path) == ["selected_category"]
