"""Independent scheduling/resource regression tests; no simulator or GPU backend."""

import copy
import fcntl
import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def queue(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "independent_ood_opportunistic_review", scripts / "run_vla_trace_ood_opportunistic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.native, "readiness", lambda *_: {"ready": True})
    return module


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def marker(path):
    save(path, {"complete": True, "child_exit": {
        "pid": 12345, "returncode": 0, "proc_start_ticks": 67890}})


@pytest.fixture
def fixture(queue, tmp_path, monkeypatch):
    output = tmp_path / "ood/object_0"
    output.mkdir(parents=True)
    marker_path = tmp_path / "released.json"
    resources = [
        {"gpu": "4", "cpus": [40, 41, 42]},
        {"gpu": "0", "cpus": [0, 1, 2]},
        {"gpu": "5", "cpus": [80, 81, 82], "release_gate": str(marker_path)},
    ]
    monkeypatch.setattr(queue.subprocess, "check_output", lambda command, **_: f"GPU-{command[2]}\n")
    monkeypatch.setattr(queue.native, "process_start_ticks", lambda _: None)
    lanes = []

    def lane_factory(resource):
        lane = queue.Lane(resource, lock_directory=tmp_path / "locks")
        lanes.append(lane)
        return lane

    configs = {variant: tmp_path / f"{variant}.toml" for variant in queue.native.VARIANTS}
    checkpoints = {policy: None if iteration is None else f"/frozen/iteration_{iteration:06d}"
                   for policy, iteration in queue.native.POLICIES.items()}
    calls = []
    seen = set()
    mutex = threading.Lock()
    contract = {"sources": {}, "inputs": {}, "checkpoint_stats": {}}

    def evaluator(config, destination, checkpoint, **kwargs):
        identity = (destination.parent.name, destination.name)
        with mutex:
            assert identity not in seen
            seen.add(identity)
            calls.append({"cell": identity, "gpu": kwargs["gpu"], "contract": kwargs["contract"]})
        assert config == configs[identity[1]]
        assert checkpoint == checkpoints[identity[0]]
        assert kwargs["task"] == "object_0"
        assert kwargs["variant"] == identity[1]
        assert kwargs["cpus"] == next(r["cpus"] for r in resources if r["gpu"] == kwargs["gpu"])
        destination.mkdir(parents=True, exist_ok=False)
        # Give both primary threads an opportunity to run without relying on
        # ordering between their completions.
        time.sleep(.005)
        result = {"complete": True}
        save(destination / "summary.json", result)
        return result

    def run(**kwargs):
        return queue.run_cells(tmp_path, "object_0", output, resources, configs, checkpoints,
                               contract, lane_factory=lane_factory, evaluator=kwargs.pop("evaluator", evaluator),
                               poll_seconds=.001, **kwargs)

    return SimpleNamespace(output=output, marker=marker_path, resources=resources, lanes=lanes,
                           configs=configs, checkpoints=checkpoints, calls=calls, seen=seen,
                           mutex=mutex, contract=contract, evaluator=evaluator, run=run,
                           lane_factory=lane_factory, campaign=tmp_path)


def assert_exact_coverage(queue, fixture, result):
    expected = {(policy, variant) for policy in queue.native.POLICIES for variant in queue.native.VARIANTS}
    assert fixture.seen == expected
    assert len(result["claims"]) == len(result["completed"]) == 16
    claims = [json.loads(Path(row["path"]).read_text()) for row in result["claims"]]
    assert [(row["policy"], row["variant"]) for row in claims] == [
        (policy, variant) for policy in queue.native.POLICIES for variant in queue.native.VARIANTS]
    assert [row["sequence"] for row in claims] == list(range(16))
    for row in claims:
        assert row["state_ids"] == list(range(50))
        assert row["noise_seeds"] == list(range(100000, 100050))
        assert (row["episodes"], row["native_shards"], row["primitive_step_limit"]) == (50, 3, 280)
        assert row["checkpoint"] == fixture.checkpoints[row["policy"]]
    assert all(lane.stream is None for lane in fixture.lanes)


def test_missing_optional_release_never_blocks_primary_completion(queue, fixture):
    original = copy.deepcopy(fixture.contract)
    result = fixture.run()
    assert_exact_coverage(queue, fixture, result)
    assert {row["gpu"] for row in fixture.calls} == {"4", "0"}
    assert not result["optional_release_guards"]
    assert not (fixture.output / "release_guards").exists()
    assert fixture.contract == original


def test_valid_but_busy_optional_gpu_never_blocks_primary_completion(queue, fixture):
    marker(fixture.marker)
    locks = fixture.campaign / "locks"
    locks.mkdir()
    with (locks / "GPU-5.lock").open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = fixture.run()
    assert_exact_coverage(queue, fixture, result)
    assert {row["gpu"] for row in fixture.calls} == {"4", "0"}
    assert not result["optional_release_guards"]


def test_late_optional_release_takes_only_unclaimed_cells_and_keeps_base_contract(queue, fixture):
    original = copy.deepcopy(fixture.contract)
    count = 0
    count_lock = threading.Lock()

    def evaluator(*args, **kwargs):
        nonlocal count
        result = fixture.evaluator(*args, **kwargs)
        with count_lock:
            count += 1
            if count == 3:
                marker(fixture.marker)
        return result

    result = fixture.run(evaluator=evaluator)
    assert_exact_coverage(queue, fixture, result)
    assert {row["gpu"] for row in fixture.calls} == {"4", "0", "5"}
    assert len(result["optional_release_guards"]) == 1
    assert fixture.contract == original
    assert fixture.calls[0]["gpu"] in {"4", "0"}
    optional_claims = [row for row in result["claims"] if row["lane"] == 2]
    assert min(json.loads(Path(row["path"]).read_text())["sequence"] for row in optional_claims) >= 3
    for call in fixture.calls:
        if call["gpu"] == "5":
            assert call["contract"] is not fixture.contract
            assert call["contract"]["inputs"][str(fixture.marker)] == queue.native.digest(fixture.marker)
            assert "optional_release_guard" in call["contract"]
        else:
            assert call["contract"] is fixture.contract
            assert "optional_release_guard" not in call["contract"]


def test_available_optional_lane_still_claims_after_both_primaries(queue, fixture):
    marker(fixture.marker)
    result = fixture.run()
    assert_exact_coverage(queue, fixture, result)
    assert [row["lane"] for row in result["claims"][:3]] == [0, 1, 2]


def test_changed_release_marker_after_lock_is_rejected_and_lock_released(queue, fixture, monkeypatch):
    marker(fixture.marker)
    lane = queue.Lane(fixture.resources[2], lock_directory=fixture.campaign / "locks")
    real_flock = fcntl.flock

    def mutate_after_lock(fd, operation):
        real_flock(fd, operation)
        save(fixture.marker, {"complete": False})

    monkeypatch.setattr(queue.fcntl, "flock", mutate_after_lock)
    with pytest.raises(ValueError, match="release changed"):
        lane.acquire()
    assert lane.stream is None
    monkeypatch.setattr(queue.fcntl, "flock", real_flock)
    with (fixture.campaign / "locks/GPU-5.lock").open("a+") as stream:
        real_flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_mandatory_readiness_is_rechecked_after_gpu_acquisition(queue, fixture, monkeypatch):
    monkeypatch.setattr(queue.native, "readiness", lambda *_: {"ready": False})
    with pytest.raises(RuntimeError, match="Mandatory OOD gates changed"):
        fixture.run()
    assert not fixture.calls
    assert not (fixture.output / "dispatch_claims").exists()
    assert all(lane.stream is None for lane in fixture.lanes)


def test_failure_waits_for_other_worker_cleanup_before_releasing_any_lock(queue, fixture):
    both_started = threading.Barrier(2, timeout=3)
    stopped = threading.Event()

    def evaluator(*args, **kwargs):
        both_started.wait()
        if kwargs["gpu"] == "4":
            raise RuntimeError("injected native failure")
        assert kwargs["stop"].wait(timeout=3)
        assert all(lane.stream is not None for lane in fixture.lanes[:2])
        time.sleep(.01)  # Represents owned native child cleanup before return.
        stopped.set()
        raise RuntimeError("other lane cleaned up")

    with pytest.raises(RuntimeError, match="injected native failure"):
        fixture.run(evaluator=evaluator)
    assert stopped.is_set()
    assert all(lane.stream is None for lane in fixture.lanes)
    assert not (fixture.output / "queue_complete.json").exists()


def test_submit_interruption_waits_for_unregistered_worker_cleanup(queue, fixture, monkeypatch):
    entered = threading.Event()
    cleaned = threading.Event()
    original_executor = queue.ThreadPoolExecutor

    class InterruptedExecutor(original_executor):
        def submit(self, *args, **kwargs):
            super().submit(*args, **kwargs)
            assert entered.wait(timeout=3)
            raise RuntimeError("injected submit interruption before future registration")

    def evaluator(*args, **kwargs):
        entered.set()
        assert kwargs["stop"].wait(timeout=3)
        assert fixture.lanes[0].stream is not None
        time.sleep(.01)
        cleaned.set()
        raise RuntimeError("unregistered worker cleaned up")

    monkeypatch.setattr(queue, "ThreadPoolExecutor", InterruptedExecutor)
    with pytest.raises(RuntimeError, match="submit interruption"):
        fixture.run(evaluator=evaluator)
    assert cleaned.is_set()
    assert all(lane.stream is None for lane in fixture.lanes)
    assert not (fixture.output / "queue_complete.json").exists()


def test_used_optional_release_guard_is_immutable_through_completion(queue, fixture):
    marker(fixture.marker)
    altered = threading.Event()

    def evaluator(*args, **kwargs):
        result = fixture.evaluator(*args, **kwargs)
        if kwargs["gpu"] == "5" and not altered.is_set():
            altered.set()
            changed = json.loads(fixture.marker.read_text())
            changed["unexpected_change"] = True
            save(fixture.marker, changed)
        return result

    with pytest.raises(ValueError, match="Frozen OOD evaluation input changed"):
        fixture.run(evaluator=evaluator)
    assert altered.is_set()
    assert all(lane.stream is None for lane in fixture.lanes)
    assert not (fixture.output / "queue_complete.json").exists()


def test_partial_existing_cell_is_preserved_without_claim_or_overwrite(queue, fixture):
    policy = next(iter(queue.native.POLICIES))
    variant = next(iter(queue.native.VARIANTS))
    existing = fixture.output / policy / variant / "partial.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("preserve diagnostic evidence")
    with pytest.raises(FileExistsError, match="Preserving existing"):
        fixture.run()
    assert existing.read_text() == "preserve diagnostic evidence"
    assert not fixture.calls
    assert not (fixture.output / "dispatch_claims").exists()


def test_noncomplete_native_result_cannot_publish_completion(queue, fixture):
    def evaluator(*args, **kwargs):
        fixture.evaluator(*args, **kwargs)
        return {"complete": False}

    with pytest.raises(ValueError, match="without a complete cell"):
        fixture.run(evaluator=evaluator)
    assert not (fixture.output / "queue_complete.json").exists()
    assert all(lane.stream is None for lane in fixture.lanes)


def test_pinned_predecessor_is_rechecked_after_mandatory_wait(queue, tmp_path, monkeypatch):
    native_path = Path(queue.native.__file__).resolve()
    original_digest = queue.native.digest
    phase = {"waiting": False}

    def fake_digest(path):
        if Path(path).resolve() == native_path:
            return "changed" if phase["waiting"] else queue.NATIVE_QUEUE_SHA256
        return original_digest(path)

    def ready(*_):
        phase["waiting"] = True
        return {"ready": True}

    monkeypatch.setattr(queue.native, "digest", fake_digest)
    monkeypatch.setattr(queue.native, "readiness", ready)
    monkeypatch.setattr(queue.native, "queue_resources", lambda *_: [{"gpu": "4", "cpus": [40, 41, 42]}])
    monkeypatch.setattr(queue.native, "freeze", lambda *_: pytest.fail("Changed imported source must fail before freeze"))
    args = SimpleNamespace(run=tmp_path, task="object_0", poll_seconds=.001)
    with pytest.raises(ValueError, match="changed during the mandatory gate wait"):
        queue.run(args)
    assert not (tmp_path / "ood/object_0/queue_complete.json").exists()
