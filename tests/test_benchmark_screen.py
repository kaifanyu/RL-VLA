"""Development-screen identity, trajectory-video accounting and ranking checks."""

import hashlib
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
from benchmark_bridge import environment, screen

from rl_vla.trajectory_recording import TrajectoryRecorder


def _ready(tmp_path):
    states = tmp_path / "states.npy"
    np.save(states, np.arange(10, dtype=float).reshape(5, 2), allow_pickle=False)
    bddl = tmp_path / "real.bddl"
    bddl.write_text("real benchmark task")
    case = {"ready": True, "split": "development_only", "case_id": "plus_camera", "benchmark": "plus",
            "suite": "libero_spatial", "task_id": 1, "name": "task_view_0_50", "category": "Camera",
            "difficulty": 3, "states": str(states), "real_bddl": str(bddl),
            "bddl": str(tmp_path / "virtual_view_0_50.bddl"), "episodes": 5,
            "states_sha256": hashlib.sha256(states.read_bytes()).hexdigest(),
            "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(), "reset_checks": [{}] * 5}
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(case))
    return path, case


def test_case_checks_real_bddl_preserves_virtual_camera_name_and_rejects_asset_changes(tmp_path):
    path, case = _ready(tmp_path)
    checked = environment.checked_case(path)
    assert checked["bddl"] == case["bddl"]
    assert not Path(checked["bddl"]).exists()
    Path(case["real_bddl"]).write_text("changed benchmark")
    with pytest.raises(ValueError, match="asset changed"):
        environment.checked_case(path)


def test_sequential_wrapper_selects_each_dev_state_once_with_explicit_reset_seed(tmp_path, monkeypatch):
    _, case = _ready(tmp_path)
    instances = []

    class Native:
        closed = False

        def reset(self, *, seed, options):
            self.seed, self.options = seed, options
            return {"state": np.ones(8)}, {"initial_state_id": options["initial_state_id"]}

        def close(self):
            self.closed = True

    def factory(got_case, seed, output, index):
        assert got_case["bddl"] == case["bddl"]
        assert seed == 100000 + index
        native = Native()
        instances.append(native)
        return native, index

    monkeypatch.setattr(environment, "make_case_env", factory)
    env = object.__new__(environment.SequentialCaseEnv)
    env.case, env.output, env.index, env.native = case, tmp_path, 0, None
    for index in range(5):
        _, info = env.reset(seed=100000 + index)
        assert info["initial_state_id"] == index
        assert info["experiment_split"] == "development_only"
    assert all(native.closed for native in instances[:-1])
    with pytest.raises(ValueError, match="more than"):
        env.reset(seed=100005)
    env.close()
    assert instances[-1].closed


def _observation(value):
    return {"state": np.array([value]), "openpi": {
        "observation/image": np.full((16, 16, 3), value, np.uint8),
        "observation/wrist_image": np.full((16, 16, 3), value + 1, np.uint8),
    }}


def test_primitive_video_has_one_reset_frame_plus_every_executed_step(tmp_path, monkeypatch):
    import imageio.v2 as imageio

    frames = []

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def append_data(self, frame):
            frames.append(frame.copy())

    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: Writer())
    recorder = TrajectoryRecorder(tmp_path / "baseline.trajectories", {})
    recorder.start_episode(_observation(0), {})
    for start, end in ((0, 2), (2, 3)):
        recorder.begin_chunk(_observation(start), np.zeros((5, 32)), np.zeros((5, 7)), {})
        for value in range(start + 1, end + 1):
            recorder.record_step(np.zeros(7), _observation(value), float(value == 3),
                                 value == 3, False, {})
        recorder.end_chunk({})
    directory = recorder.end_episode({})
    recorder.close()
    report = tmp_path / "baseline.json"
    report.write_text(json.dumps({"episodes": [{"primitive_steps": 3, "trajectory": directory,
                                               "video": "old-chunk-video.mp4"}]}))
    updated = screen.render_primitive_videos(report)
    assert [frame[0, 0, 0] for frame in frames] == [0, 1, 2, 3]
    assert updated["episodes"][0]["chunk_boundary_video"] == "old-chunk-video.mp4"
    assert "primitive_videos" in updated["episodes"][0]["video"]


def test_ranking_retains_failures_without_mislabeling_execution_errors_as_policy_failures(tmp_path):
    _, case = _ready(tmp_path)
    second = case | {"case_id": "broken"}
    output = tmp_path / "results" / case["case_id"]
    output.mkdir(parents=True)
    (output / "status.json").write_text(json.dumps({"state": "complete"}))
    episodes = [{"seed": i, "reset": {"initial_state_id": i}, "success": i < 2,
                 "primitive_steps": 50, "video": f"video{i}.mp4", "trajectory": f"episode{i}"}
                for i in range(5)]
    (output / "baseline.json").write_text(json.dumps({"episodes": episodes, "success_rate": .4}))
    broken = tmp_path / "results/broken"
    broken.mkdir()
    (broken / "status.json").write_text(json.dumps({"state": "failed", "error": "renderer failed"}))
    ranking = screen.rank_results(tmp_path, {"cases": [case, second], "episodes_per_cell": 5})
    assert ranking["completed_cells"] == 1
    assert ranking["ranked_cells"][0]["failures"] == 3
    assert ranking["mixed_success_candidates"] == [case["case_id"]]
    assert len(ranking["errors"]) == 1


def test_toml_roundtrip_and_frozen_source_validation(tmp_path):
    target = tmp_path / "config.toml"
    data = {"algorithm": "rldt", "env": {"kwargs": {"filename": "path with spaces"}},
            "recording": {"enabled": True}, "rldt": {"gamma": .999, "execute_steps": 5}}
    screen.write_toml(target, data)
    with target.open("rb") as stream:
        assert tomllib.load(stream) == data
    snapshot = tmp_path / "execution"
    snapshot.mkdir()
    (tmp_path / "screen_plan.json").write_text("{}")
    source = snapshot / "example.py"
    source.write_text("pass\n")
    (snapshot / "contract.json").write_text(json.dumps({"files": {"example.py": screen.digest(source)},
        "screen_plan_sha256": screen.digest(tmp_path / "screen_plan.json")}))
    screen.verify_execution(snapshot, tmp_path)
    source.write_text("raise RuntimeError()\n")
    with pytest.raises(ValueError, match="source changed"):
        screen.verify_execution(snapshot, tmp_path)
