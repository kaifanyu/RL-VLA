import copy
import json

import pytest

from rl_vla.diagnostics import ood_lane_release_v1 as gate


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def process(pid=10, start=100, ppid=1, state="S"):
    return {"pid": pid, "proc_start_ticks": start, "ppid": ppid, "state": state, "command": []}


@pytest.fixture
def ready(tmp_path):
    task = "object_0"
    method = tmp_path / "methods/rldt" / task
    layout = tmp_path / "diagnostic_layouts" / task
    save(method / "selection.json", {"iteration": 25, "locked": True, "test_results_seen": False})
    save(method / "status.json", {"stage": "action_evaluations_complete", "state": "completed"})
    save(method / "checkpoint_audit.json", {"passed": True})
    save(layout / "kinematic_audit.json", {"passed": True})
    save(layout / "summary.json", {"complete": True, "task": task, "variants": list(gate.VARIANTS), "states_per_variant": 50})
    return tmp_path, task


def test_main_readiness_does_not_read_or_require_causal_and_trace(ready):
    root, task = ready
    save(root / "interventions" / task / "queue_complete.json", {"complete": False})
    value = gate.scientific_readiness(root, task)
    assert value["ready"]
    assert set(value["gate_paths"]) == {"selection", "main", "checkpoint_audit", "kinematics", "layouts"}


@pytest.mark.parametrize("field,value", [("locked", False), ("test_results_seen", True), ("iteration", 75)])
def test_invalid_selection_cannot_be_blessed(ready, field, value):
    root, task = ready
    path = root / "methods/rldt" / task / "selection.json"
    data = json.loads(path.read_text())
    data[field] = value
    save(path, data)
    with pytest.raises(ValueError, match="selection"):
        gate.scientific_readiness(root, task)


def test_running_main_waits_failed_main_raises(ready):
    root, task = ready
    path = root / "methods/rldt" / task / "status.json"
    save(path, {"stage": "test", "state": "running"})
    assert not gate.scientific_readiness(root, task)["ready"]
    save(path, {"state": "failed"})
    with pytest.raises(RuntimeError):
        gate.scientific_readiness(root, task)


def test_tracker_remembers_child_after_parent_exit_and_does_not_follow_reused_pid():
    table = {10: process(), 11: process(11, 101, 10)}
    tracker = gate.ProcessTracker(table[10], [], reader=lambda: table)
    assert not tracker.released()
    table.pop(10)
    table[11]["ppid"] = 1
    assert not tracker.released()
    table[10] = process(10, 999)
    table.pop(11)
    assert tracker.released()
    assert len(tracker.observed) == 2


def test_tracker_finds_grandchildren_and_treats_zombies_as_exited():
    table = {10: process(), 11: process(11, 101, 10), 12: process(12, 102, 11)}
    tracker = gate.ProcessTracker(table[10], [], reader=lambda: table)
    tracker.observe()
    assert len(tracker.observed) == 3
    for value in table.values():
        value["state"] = "Z"
    assert tracker.released()


def causal_marker():
    return {"complete": True, "passed": True, "technical_only": False, "task": "object_0",
            "policies": ["pi05_libero", "rldt_25", "rldt_50", "rldt_100"], "episodes_per_condition": 10,
            "conditions": ["none", "no_image", "no_text", "target:background_fill", "distractor:background_fill",
                           "random:background_fill", "target:mosaic", "distractor:mosaic", "random:mosaic"]}


@pytest.mark.parametrize("mutation", ["technical", "task", "policies", "conditions", "count"])
def test_causal_marker_rejects_wrong_research_coverage(tmp_path, mutation):
    marker = causal_marker()
    field, value = {"technical": ("technical_only", True), "task": ("task", "object_1"),
                    "policies": ("policies", ["pi05_libero"]), "conditions": ("conditions", ["none"]),
                    "count": ("episodes_per_condition", 2)}[mutation]
    marker[field] = value
    path = tmp_path / "marker.json"
    save(path, marker)
    with pytest.raises(ValueError):
        gate.validate_marker({"role": "causal", "task": "object_0", "release_gate": str(path)})


def test_trace_child_still_running_waits_and_pid_reuse_releases(tmp_path):
    path = tmp_path / "trace.json"
    children = [{"pid": 19, "proc_start_ticks": 199, "returncode": 0},
                {"pid": 20, "proc_start_ticks": 200, "returncode": 0}]
    save(path, {"complete": True, "passed": True, "tasks": ["object_0", "object_1", "object_2"],
                "policies": ["pi05_libero", "rldt_000025", "rldt_000050", "rldt_000100"],
                "jobs": [{"stage": stage, "child_exit": child} for stage, child in
                         zip(["baseline", "trained"], children, strict=True)],
                "child_exits": children, "child_exit": children[-1]})
    resource = {"role": "trace", "task": "object_0", "release_gate": str(path)}
    assert not gate.validate_marker(resource, lambda: {20: process(20, 200)})
    assert gate.validate_marker(resource, lambda: {20: process(20, 201)})


def test_lane_requires_parent_and_child_exit_then_empty_gpu(tmp_path):
    path = tmp_path / "main.json"
    save(path, {"stage": "action_evaluations_complete", "state": "completed"})
    parent, child = process(), process(11, 101, 10)
    table = {10: parent, 11: child}
    resident = [555]
    resource = {"role": "main", "gpu": "0", "parent": parent, "observed_processes": [child], "release_gate": str(path)}
    lane = gate.Lane(resource, lock_directory=tmp_path / "locks", table_reader=lambda: table,
                     uuid_reader=lambda _: "GPU-test", gpu_process_reader=lambda _: resident)
    assert not lane.acquire()
    table.pop(10)
    assert not lane.acquire()
    table.clear()
    assert not lane.acquire()
    assert lane.stream is None
    resident.clear()
    assert lane.acquire()
    assert lane.release_record["owned_processes_exited"]
    assert lane.release_record["gpu_compute_processes_at_acquisition"] == []
    lane.release()


def test_real_gpu6_oracle_contract_and_environment(monkeypatch):
    root = gate.REPO / "runs/vla_trace_objects_20261031"
    if not root.exists():
        pytest.skip("Research evidence not mounted")
    value = gate.gpu6_cache_contract(root)
    assert len(value["binary_sha256"]) == 3
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", value["cache_root"])
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.delenv("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", raising=False)
    gate.verify_cache_environment(value)
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_load_autotune_results_from=/diagnostic/profile")
    with pytest.raises(ValueError, match="original cache"):
        gate.verify_cache_environment(value)


def test_plan_rejects_resource_reassignment(tmp_path, monkeypatch):
    table = {10: process(), 20: process(20, 200), 30: process(30, 300)}
    table[10]["command"] = ["python", "run_vla_trace_objects.py", str(tmp_path), "object_0"]
    table[20]["command"] = ["python", "queue_vla_trace_diagnostics.py", "interventions", str(tmp_path), "object_0", "--gpu", "4"]
    table[30]["command"] = ["python", "queue_vla_trace_diagnostics.py", "rollout", str(tmp_path), "--gpu", "5"]
    monkeypatch.setattr(gate, "process_table", lambda: table)
    plan = gate.capture_release_plan(tmp_path, "object_0", 10, 20, 30)
    assert len(gate.validate_release_plan(plan, tmp_path, "object_0")) == 3
    changed = copy.deepcopy(plan)
    changed["resources"][0]["gpu"] = "6"
    with pytest.raises(ValueError, match="physical GPUs"):
        gate.validate_release_plan(changed, tmp_path, "object_0")


@pytest.mark.parametrize("task", ["object_0", "object_1", "object_2"])
def test_actual_campaign_selection_timestamp_accepted(ready, task):
    root, target = ready
    actual = gate.REPO / "runs/vla_trace_objects_20261031/methods/rldt" / task / "selection.json"
    if not actual.exists():
        pytest.skip("Campaign evidence not mounted")
    selection = json.loads(actual.read_text())
    assert isinstance(selection["locked"], str) and "T" in selection["locked"]
    save(root / "methods/rldt" / target / "selection.json", selection)
    assert gate.scientific_readiness(root, target)["ready"]


def test_final_plan_binds_preserved_capture(tmp_path, monkeypatch):
    table = {10: process(), 20: process(20, 200), 30: process(30, 300)}
    table[10]["command"] = ["python", "run_vla_trace_objects.py", str(tmp_path), "object_0"]
    table[20]["command"] = ["python", "queue_vla_trace_diagnostics.py", "interventions", str(tmp_path), "object_0", "--gpu", "4"]
    table[30]["command"] = ["python", "queue_vla_trace_diagnostics.py", "rollout", str(tmp_path), "--gpu", "5"]
    monkeypatch.setattr(gate, "process_table", lambda: table)
    original = gate.capture_release_plan(tmp_path, "object_0", 10, 20, 30)
    original["helper_sha256"] = "prior-capture-helper-hash"
    path = tmp_path / "capture.json"
    save(path, original)
    final = {**original, "helper_sha256": gate.digest(gate.__file__), "process_capture": {
        "path": str(path), "sha256": gate.digest(path), "helper_sha256": original["helper_sha256"]}}
    gate.validate_release_plan(final, tmp_path, "object_0")
    final["resources"][0]["parent"]["proc_start_ticks"] += 1
    with pytest.raises(ValueError, match="captured process"):
        gate.validate_release_plan(final, tmp_path, "object_0")


def test_full_dispatch_freeze_binds_amendment_and_helper_without_native_execution(tmp_path, monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(gate.REPO / "scripts"))
    source = gate.REPO / "scripts/run_vla_trace_ood_per_lane_v1.py"
    spec = importlib.util.spec_from_file_location("per_lane_full_freeze_test", source)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    resources = [{"role": role, "task": "object_0", "gpu": gpu, "cpuset": pool, "cpus": gate.cpus(pool),
                  "parent": process(100000 + i, 100 + i), "observed_processes": [],
                  "release_gate": str(gate.marker_for(tmp_path, "object_0", role))}
                 for i, (role, gpu, pool) in enumerate(gate.RESOURCES["object_0"])]
    plan = {"version": gate.VERSION, "campaign": str(tmp_path), "task": "object_0",
            "helper_sha256": gate.digest(gate.__file__), "resources": resources}
    plan_path = tmp_path / "plan.json"
    save(plan_path, plan)
    amendment = tmp_path / "amendment.json"
    save(amendment, {"version": gate.VERSION, "approved_for_scheduling": True,
         "native_source_sha256": queue.NATIVE_QUEUE_SHA256, "cases_unchanged": True,
         "outcome_dependent_scheduling": False, "release_plan_sha256": {"object_0": gate.digest(plan_path)},
         "source_sha256": {str(source): gate.digest(source)}, "release_helper_sha256": gate.digest(gate.__file__)})
    monkeypatch.setattr(queue, "scientific_readiness", lambda *_: {"ready": True, "gate_paths": {}})
    checkpoints = {policy: str(iteration) for policy, iteration in queue.native.POLICIES.items()}

    def freeze(campaign, task, output, gates):
        inputs = {"sources": {str(queue.native.__file__): queue.NATIVE_QUEUE_SHA256},
                  "inputs": {}, "checkpoint_stats": {}}
        save(output / "input_manifest.json", inputs)
        (output / "source_snapshot.tar.gz").write_bytes(b"test source archive")
        return inputs, {}, checkpoints

    monkeypatch.setattr(queue.native, "freeze", freeze)

    def no_gpu_cells(campaign, task, output, resource_arg, configs, checkpoint_arg, contract, **kwargs):
        assert resource_arg == resources and checkpoint_arg == checkpoints
        assert contract["inputs"][str(amendment)] == gate.digest(amendment)
        assert contract["inputs"][str(plan_path)] == gate.digest(plan_path)
        assert contract["sources"][str(output / "release_helper_source.py")] == gate.digest(gate.__file__)
        return {"claims": [], "completed": [], "lane_release_guards": []}

    monkeypatch.setattr(queue, "run_cells", no_gpu_cells)
    monkeypatch.setattr(queue.native, "summarize_task", lambda *_: None)
    args = SimpleNamespace(run=tmp_path, task="object_0", release_plan=plan_path, amendment=amendment, poll_seconds=.001)
    queue.run(args)
    complete = json.loads((tmp_path / "ood/object_0/queue_complete.json").read_text())
    assert complete["scheduling_amendment_sha256"] == gate.digest(amendment)
    assert complete["release_plan_sha256"] == gate.digest(plan_path)
    with pytest.raises(FileExistsError):
        queue.run(args)
