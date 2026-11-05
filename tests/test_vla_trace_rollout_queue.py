"""Phase-bank completion must be published atomically after every fixed case."""

import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/run_vla_trace_rollout_bank.py"
SPEC = importlib.util.spec_from_file_location("rollout_queue", SOURCE)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


@pytest.mark.parametrize("existing", [False, True])
def test_marker_publication_keeps_old_or_absent_destination_until_replace(tmp_path, monkeypatch, existing):
    path = tmp_path / "summary.json"
    old = {"complete": False, "pending": ["pi05_libero"]}
    new = {"complete": True, "pending": []}
    if existing:
        path.write_text(json.dumps(old))
    original = Path.replace
    observed = []

    def replace(temporary, destination):
        assert temporary != destination
        assert json.loads(temporary.read_text()) == new
        assert (json.loads(destination.read_text()) if destination.exists() else None) == (old if existing else None)
        observed.append(True)
        return original(temporary, destination)

    monkeypatch.setattr(Path, "replace", replace)
    queue.write_json(path, new)
    assert observed == [True]
    assert json.loads(path.read_text()) == new
    assert not path.with_suffix(".json.tmp").exists()


def test_campaign_publishes_only_all_five_cases_and_resolves_selected_alias(tmp_path, monkeypatch):
    method = tmp_path / "methods/rldt/object_0"
    method.mkdir(parents=True)
    (method / "selection.json").write_text(json.dumps({"iteration": 25}))
    (tmp_path / "moving_mask_alignment_gate.json").write_text('{"passed": true}')
    episodes = [{"reset": {"initial_state_id": i}, "trajectory": f"episode_{i}"} for i in range(150, 200)]
    for policy, filename in queue.POLICIES.items():
        if policy == "rldt_000025":
            filename = "selected_test.json"
        (method / filename).write_text(json.dumps({"episodes": episodes, "checkpoint": None if policy == "pi05_libero" else f"checkpoint_{policy}"}))
        for state in range(150, 155):
            folder = tmp_path / "rollout_observation_bank/object_0" / policy / f"state_{state:06d}"
            folder.mkdir(parents=True)
            (folder / "summary.json").write_text('{"complete": true}')

    def unexpected(*args, **kwargs):
        raise AssertionError("All completed phase exports should be reused")

    monkeypatch.setattr(queue.subprocess, "run", unexpected)
    publications = []
    original = queue.write_json

    def publish(path, value):
        if path.name == "summary.json":
            assert len(value["banks"]) == 5
            assert [Path(p).name for p in value["banks"]] == [f"state_{i:06d}" for i in range(150, 155)]
            assert all(json.loads((Path(p) / "summary.json").read_text())["complete"] for p in value["banks"])
            publications.append(value["policy"])
        original(path, value)

    monkeypatch.setattr(queue, "write_json", publish)
    queue.run(tmp_path, "object_0")
    assert set(publications) == set(queue.POLICIES)
    status = json.loads((tmp_path / "rollout_observation_bank/object_0/status.json").read_text())
    assert status == {"complete": True, "pending": []}
