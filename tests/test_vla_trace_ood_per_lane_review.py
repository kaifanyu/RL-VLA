"""Independent CPU-only review of dynamic OOD claims and physical-resource leases."""

import copy
import fcntl
import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_vla.diagnostics import ood_lane_release_v1 as gate


@pytest.fixture
def queue(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "independent_ood_per_lane_review", scripts / "run_vla_trace_ood_per_lane_v1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "scientific_readiness", lambda *_: {"ready": True})
    return module


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def publish_release(resource):
    role = resource["role"]
    if role == "main":
        value = {"stage": "action_evaluations_complete", "state": "completed"}
    elif role == "causal":
        value = {"complete": True, "passed": True, "technical_only": False,
                 "task": resource["task"], "episodes_per_condition": 10,
                 "policies": ["pi05_libero", "rldt_25", "rldt_50", "rldt_100"],
                 "conditions": ["none", "no_image", "no_text", "target:background_fill",
                                "distractor:background_fill", "random:background_fill",
                                "target:mosaic", "distractor:mosaic", "random:mosaic"]}
    else:
        children = [{"pid": 555 + index, "proc_start_ticks": 5550 + index, "returncode": 0}
                    for index in range(2)]
        value = {"complete": True, "passed": True, "tasks": ["object_0", "object_1", "object_2"],
                 "policies": ["pi05_libero", "rldt_000025", "rldt_000050", "rldt_000100"],
                 "jobs": [{"stage": stage, "child_exit": child}
                          for stage, child in zip(("baseline", "trained"), children, strict=True)],
                 "child_exits": children, "child_exit": children[-1]}
    save(resource["release_gate"], value)


@pytest.fixture
def fixture(queue, tmp_path):
    output = tmp_path / "ood/object_0"
    output.mkdir(parents=True)
    resources = []
    for index, (role, gpu, cpuset) in enumerate(gate.RESOURCES["object_0"]):
        parent = {"pid": 100 + index, "proc_start_ticks": 1000 + index,
                  "ppid": 1, "state": "S", "command": []}
        resources.append({"role": role, "task": "object_0", "gpu": gpu,
                          "cpuset": cpuset, "cpus": gate.cpus(cpuset),
                          "release_gate": str(tmp_path / f"{role}.json"),
                          "parent": parent, "observed_processes": []})
    publish_release(resources[0])
    lanes, tables, gpu_processes = [], {}, {}

    def lane_factory(resource):
        gpu = resource["gpu"]
        lane = gate.Lane(resource, lock_directory=tmp_path / "locks",
                         table_reader=lambda: tables.get(gpu, {}),
                         uuid_reader=lambda _: f"GPU-{gpu}",
                         gpu_process_reader=lambda _: gpu_processes.get(gpu, []))
        lanes.append(lane)
        return lane

    configs = {variant: tmp_path / f"{variant}.toml" for variant in queue.native.VARIANTS}
    checkpoints = {policy: None if iteration is None else f"/frozen/iteration_{iteration:06d}"
                   for policy, iteration in queue.native.POLICIES.items()}
    calls, seen, mutex = [], set(), threading.Lock()
    contract = {"sources": {}, "inputs": {}, "checkpoint_stats": {}}

    def evaluator(config, destination, checkpoint, **kwargs):
        identity = (destination.parent.name, destination.name)
        with mutex:
            assert identity not in seen
            seen.add(identity)
            calls.append({"cell": identity, "gpu": kwargs["gpu"], "contract": kwargs["contract"]})
        assert config == configs[identity[1]]
        assert checkpoint == checkpoints[identity[0]]
        assert kwargs["task"] == "object_0" and kwargs["variant"] == identity[1]
        assert kwargs["cpus"] == next(r["cpus"] for r in resources if r["gpu"] == kwargs["gpu"])
        destination.mkdir(parents=True, exist_ok=False)
        time.sleep(.008)
        value = {"complete": True}
        save(destination / "summary.json", value)
        return value

    def run(**kwargs):
        return queue.run_cells(tmp_path, "object_0", output, resources, configs, checkpoints,
                               contract, lane_factory=lane_factory,
                               evaluator=kwargs.pop("evaluator", evaluator), poll_seconds=.001, **kwargs)

    return SimpleNamespace(output=output, resources=resources, lanes=lanes, tables=tables,
                           gpu_processes=gpu_processes, configs=configs, checkpoints=checkpoints,
                           calls=calls, seen=seen, contract=contract, evaluator=evaluator,
                           run=run, lane_factory=lane_factory, campaign=tmp_path)


def assert_exact_coverage(queue, fixture, result):
    expected = [(policy, variant) for policy in queue.native.POLICIES for variant in queue.native.VARIANTS]
    assert fixture.seen == set(expected)
    assert len(result["claims"]) == len(result["completed"]) == 16
    claims = [json.loads(Path(row["path"]).read_text()) for row in result["claims"]]
    assert [(row["policy"], row["variant"]) for row in claims] == expected
    assert [row["sequence"] for row in claims] == list(range(16))
    for row in claims:
        assert row["state_ids"] == list(range(50))
        assert row["noise_seeds"] == list(range(100000, 100050))
        assert (row["episodes"], row["native_shards"], row["primitive_step_limit"]) == (50, 3, 280)
        assert row["checkpoint"] == fixture.checkpoints[row["policy"]]
        assert row["config"] == str(fixture.configs[row["variant"]])
        assert row["resource"] == fixture.resources[row["lane"]]
        assert Path(row["lane_release_guard"]["path"]).is_file()
    assert [(row["policy"], row["variant"]) for row in result["completed"]] == expected
    assert all(lane.stream is None for lane in fixture.lanes)


def test_main_alone_finishes_all16_without_causal_or_trace(queue, fixture):
    original = copy.deepcopy(fixture.contract)
    result = fixture.run()
    assert_exact_coverage(queue, fixture, result)
    assert {row["gpu"] for row in fixture.calls} == {"0"}
    assert len(result["lane_release_guards"]) == 1
    assert fixture.contract == original
    for call in fixture.calls:
        assert call["contract"] is not fixture.contract
        assert call["contract"]["inputs"][fixture.resources[0]["release_gate"]] == queue.native.digest(
            fixture.resources[0]["release_gate"])


@pytest.mark.parametrize("busy", ["lock", "compute_process", "owned_process"])
def test_busy_optional_resources_do_not_stall_released_main(queue, fixture, busy):
    held = []
    for resource in fixture.resources[1:]:
        publish_release(resource)
        if busy == "compute_process":
            fixture.gpu_processes[resource["gpu"]] = [4321]
        elif busy == "owned_process":
            parent = resource["parent"]
            fixture.tables[resource["gpu"]] = {parent["pid"]: parent}
        else:
            lock = fixture.campaign / "locks" / f"GPU-{resource['gpu']}.lock"
            lock.parent.mkdir(exist_ok=True)
            stream = lock.open("a+")
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held.append(stream)
    try:
        result = fixture.run()
    finally:
        for stream in held:
            stream.close()
    assert_exact_coverage(queue, fixture, result)
    assert {row["gpu"] for row in fixture.calls} == {"0"}


def test_optional_lanes_join_midwork_without_reassigning_claims(queue, fixture):
    original = copy.deepcopy(fixture.contract)
    counter = [0]
    lock = threading.Lock()

    def evaluator(*args, **kwargs):
        result = fixture.evaluator(*args, **kwargs)
        with lock:
            counter[0] += 1
            if counter[0] == 2:
                publish_release(fixture.resources[1])
            if counter[0] == 5:
                publish_release(fixture.resources[2])
        return result

    result = fixture.run(evaluator=evaluator)
    assert_exact_coverage(queue, fixture, result)
    assert {row["gpu"] for row in fixture.calls} == {"0", "4", "5"}
    assert fixture.contract == original
    assert [row["lane"] for row in result["claims"][:2]] == [0, 0]
    assert len(result["lane_release_guards"]) == 3
    for call in fixture.calls:
        marker = next(r["release_gate"] for r in fixture.resources if r["gpu"] == call["gpu"])
        assert call["contract"]["inputs"][marker] == queue.native.digest(marker)
        assert set(call["contract"]["inputs"]) == {marker, call["contract"]["lane_release_guard"]["path"]}


def test_all_available_lanes_claim_main_first(queue, fixture):
    for resource in fixture.resources[1:]:
        publish_release(resource)
    result = fixture.run()
    assert_exact_coverage(queue, fixture, result)
    assert [row["lane"] for row in result["claims"][:3]] == [0, 1, 2]


def test_released_causal_can_proceed_while_main_owner_finishes_cleanup(queue, fixture):
    parent = fixture.resources[0]["parent"]
    fixture.tables["0"] = {parent["pid"]: parent}
    publish_release(fixture.resources[1])
    result = fixture.run()
    assert_exact_coverage(queue, fixture, result)
    assert {row["gpu"] for row in fixture.calls} == {"4"}


def test_gate_rechecked_after_gpu_acquisition_before_claim(queue, fixture, monkeypatch):
    monkeypatch.setattr(queue, "scientific_readiness", lambda *_: {"ready": False})
    with pytest.raises(RuntimeError, match="Mandatory OOD gates changed"):
        fixture.run()
    assert not fixture.calls and not (fixture.output / "dispatch_claims").exists()
    assert all(lane.stream is None for lane in fixture.lanes)


def test_marker_mutation_after_lock_cannot_be_blessed(queue, fixture, monkeypatch):
    lane = fixture.lane_factory(fixture.resources[0])
    real_flock = fcntl.flock

    def changed(fd, operation):
        real_flock(fd, operation)
        save(fixture.resources[0]["release_gate"], {"stage": "test", "state": "running"})

    monkeypatch.setattr(gate.fcntl, "flock", changed)
    with pytest.raises(ValueError, match="release changed"):
        lane.acquire()
    assert lane.stream is None
    monkeypatch.setattr(gate.fcntl, "flock", real_flock)
    with (fixture.campaign / "locks/GPU-0.lock").open("a+") as stream:
        real_flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_failure_cleans_other_workers_before_releasing_any_lane(queue, fixture):
    for resource in fixture.resources[1:]:
        publish_release(resource)
    started = threading.Barrier(3, timeout=5)
    cleaned = set()

    def evaluator(*args, **kwargs):
        started.wait()
        if kwargs["gpu"] == "0":
            raise RuntimeError("injected main evaluator failure")
        assert kwargs["stop"].wait(timeout=5)
        assert all(lane.stream is not None for lane in fixture.lanes)
        time.sleep(.01)
        cleaned.add(kwargs["gpu"])
        raise RuntimeError("native children cleaned")

    with pytest.raises(RuntimeError, match="injected main evaluator failure"):
        fixture.run(evaluator=evaluator)
    assert cleaned == {"4", "5"}
    assert all(lane.stream is None for lane in fixture.lanes)
    assert not (fixture.output / "queue_complete.json").exists()


def test_submit_interruption_keeps_lock_until_unregistered_worker_cleanup(queue, fixture, monkeypatch):
    entered, cleaned = threading.Event(), threading.Event()
    original_executor = queue.ThreadPoolExecutor

    class InterruptedExecutor(original_executor):
        def submit(self, *args, **kwargs):
            super().submit(*args, **kwargs)
            assert entered.wait(timeout=5)
            raise RuntimeError("submit interrupted before registration")

    def evaluator(*args, **kwargs):
        entered.set()
        assert kwargs["stop"].wait(timeout=5)
        assert fixture.lanes[0].stream is not None
        cleaned.set()
        raise RuntimeError("unregistered worker cleaned")

    monkeypatch.setattr(queue, "ThreadPoolExecutor", InterruptedExecutor)
    with pytest.raises(RuntimeError, match="submit interrupted"):
        fixture.run(evaluator=evaluator)
    assert cleaned.is_set()
    assert all(lane.stream is None for lane in fixture.lanes)


def test_cache_monitor_failure_stops_running_native_workers_before_unlock(queue, fixture):
    entered, cleaned = threading.Event(), threading.Event()

    def monitor():
        if entered.is_set():
            raise ValueError("native cache binary changed")

    def evaluator(*args, **kwargs):
        entered.set()
        assert kwargs["stop"].wait(timeout=5)
        assert fixture.lanes[0].stream is not None
        cleaned.set()
        raise RuntimeError("cache failure cleanup")

    with pytest.raises(ValueError, match="native cache binary changed"):
        fixture.run(evaluator=evaluator, input_monitor=monitor)
    assert cleaned.is_set()
    assert all(lane.stream is None for lane in fixture.lanes)


def test_changed_used_marker_rejected_before_complete(queue, fixture):
    changed = threading.Event()

    def evaluator(*args, **kwargs):
        result = fixture.evaluator(*args, **kwargs)
        if not changed.is_set():
            path = Path(fixture.resources[0]["release_gate"])
            value = json.loads(path.read_text())
            value["unexpected_change"] = True
            save(path, value)
            changed.set()
        return result

    with pytest.raises(ValueError, match="Frozen OOD evaluation input changed"):
        fixture.run(evaluator=evaluator)
    assert all(lane.stream is None for lane in fixture.lanes)


def test_existing_partial_cell_preserved_before_claim(queue, fixture):
    path = fixture.output / next(iter(queue.native.POLICIES)) / next(iter(queue.native.VARIANTS)) / "partial.txt"
    path.parent.mkdir(parents=True)
    path.write_text("preserve")
    with pytest.raises(FileExistsError, match="Preserving existing"):
        fixture.run()
    assert path.read_text() == "preserve"
    assert not fixture.calls and not (fixture.output / "dispatch_claims").exists()


def test_noncomplete_native_result_cannot_be_published(queue, fixture):
    def evaluator(*args, **kwargs):
        fixture.evaluator(*args, **kwargs)
        return {"complete": False}

    with pytest.raises(ValueError, match="without a complete cell"):
        fixture.run(evaluator=evaluator)
    assert all(lane.stream is None for lane in fixture.lanes)
    assert not (fixture.output / "queue_complete.json").exists()


def test_grid_release_matches_actual_producer_scope_and_all_children_exit(tmp_path):
    children = [{"pid": 600 + index, "proc_start_ticks": 6000 + index, "returncode": 0}
                for index in range(12)]
    identities = [(task, stage) for task in ("object_0", "object_1", "object_2")
                  for stage in ("baseline", "trained", "baseline_projection", "aggregate")]
    jobs = [{"task": task, "stage": stage, "child_exit": child}
            for (task, stage), child in zip(identities, children, strict=True)]
    value = {"complete": True, "completed": True, "tasks": ["object_0", "object_1", "object_2"],
             "count_per_task": 50, "jobs": jobs, "child_exits": children, "child_exit": children[-2]}
    path = tmp_path / "grid.json"
    save(path, value)
    resource = {"role": "trace", "task": "object_2", "release_gate": str(path)}
    assert gate.validate_marker(resource, dict)
    # Even a nonterminal child's continued lifetime blocks release.
    live = {**children[0], "state": "S"}
    assert not gate.validate_marker(resource, lambda: {live["pid"]: live})
    live["proc_start_ticks"] += 1
    assert gate.validate_marker(resource, lambda: {live["pid"]: live})
    value["count_per_task"] = 5
    save(path, value)
    with pytest.raises(ValueError, match="twelve-stage fifty-case"):
        gate.validate_marker(resource, dict)


def test_rollout_release_rejects_partial_scope_and_nonterminal_live_child(fixture):
    resource = fixture.resources[2]
    publish_release(resource)
    path = Path(resource["release_gate"])
    value = json.loads(path.read_text())
    live = {**value["child_exits"][0], "state": "S"}
    assert not gate.validate_marker(resource, lambda: {live["pid"]: live})
    value["policies"] = ["pi05_libero"]
    save(path, value)
    with pytest.raises(ValueError, match="two-stage four-policy"):
        gate.validate_marker(resource, dict)
