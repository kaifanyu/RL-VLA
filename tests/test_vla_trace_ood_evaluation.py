"""OOD queues must wait for real completion and preserve every state/noise pair."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from rl_vla.diagnostics.ood_evaluation import VARIANTS, merge_shards, readiness


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def complete_gates(root):
    method = root / "methods/rldt/object_0"
    save(method / "selection.json", {"iteration": 25, "locked": "before-any-test", "test_results_seen": False})
    save(method / "status.json", {"stage": "action_evaluations_complete", "state": "completed"})
    save(method / "checkpoint_audit.json", {"passed": True})
    save(root / "diagnostic_layouts/object_0/kinematic_audit.json", {"passed": True})
    save(root / "interventions/object_0/queue_complete.json", {
        "complete": True, "passed": True, "task": "object_0", "technical_only": False})
    save(root / "diagnostic_layouts/object_0/summary.json", {
        "complete": True, "task": "object_0", "variants": list(VARIANTS), "states_per_variant": 50})


def test_queue_requires_all_main_intervention_and_layout_gates(tmp_path):
    assert readiness(tmp_path, "object_0")["ready"] is False
    complete_gates(tmp_path)
    assert readiness(tmp_path, "object_0")["ready"] is True
    save(tmp_path / "methods/rldt/object_0/status.json", {"stage": "selected_test", "state": "running"})
    result = readiness(tmp_path, "object_0")
    assert not result["ready"] and "main_action_evaluations_complete" in result["waiting_for"]


def test_intervention_smoke_cannot_release_ood_queue(tmp_path):
    complete_gates(tmp_path)
    save(tmp_path / "interventions/object_0/queue_complete.json", {
        "complete": True, "passed": True, "task": "object_0", "technical_only": True})
    with pytest.raises(ValueError, match="technical smoke"):
        readiness(tmp_path, "object_0")


def test_kinematic_audit_is_mandatory(tmp_path):
    complete_gates(tmp_path)
    (tmp_path / "diagnostic_layouts/object_0/kinematic_audit.json").unlink()
    assert "kinematics" in readiness(tmp_path, "object_0")["waiting_for"]
    save(tmp_path / "diagnostic_layouts/object_0/kinematic_audit.json", {"passed": False})
    assert "kinematic_audit_passed" in readiness(tmp_path, "object_0")["waiting_for"]


def test_failure_or_unlocked_selection_never_silently_evaluates(tmp_path):
    complete_gates(tmp_path)
    save(tmp_path / "methods/rldt/object_0/status.json", {"stage": "train", "state": "failed"})
    with pytest.raises(RuntimeError, match="experiment failed"):
        readiness(tmp_path, "object_0")
    complete_gates(tmp_path)
    save(tmp_path / "methods/rldt/object_0/selection.json", {"iteration": 75, "locked": "now", "test_results_seen": False})
    with pytest.raises(ValueError, match="predeclared"):
        readiness(tmp_path, "object_0")


def shard_reports():
    specs, reports = [], []
    for index, (start, count) in enumerate(((0, 17), (17, 17), (34, 16))):
        config = {"shard_index": index}
        specs.append({"start": start, "count": count, "config": config})
        rows = [{"seed": 100000 + i, "reset": {"initial_state_id": i, "experiment_split": VARIANTS[0],
                 "task_id": 0, "object_telemetry": {}}, "success": i % 2 == 0, "primitive_steps": 280,
                 "trajectory": f"/shard{index}/episode{i}", "video": f"/shard{index}/episode{i}.mp4"}
                for i in range(start, start + count)]
        reports.append({"config": config, "checkpoint": "/checkpoint25", "episodes": rows,
                        "success_rate": sum(row["success"] for row in rows) / count})
    return reports, specs


def test_merge_exact_fifty_cases_without_modifying_episode_artifacts():
    reports, specs = shard_reports()
    before = copy.deepcopy(reports)
    result = merge_shards(reports, specs, config={"original": True}, checkpoint="/checkpoint25",
                          variant=VARIANTS[0], task="object_0")
    assert result["complete"] is True and result["technical_only"] is False
    assert result["successes"] == 25 and result["episode_count"] == 50
    assert result["episodes"][34]["seed"] == 100034
    assert reports == before


@pytest.mark.parametrize("corruption", ("state", "seed", "missing", "budget", "telemetry", "checkpoint", "task", "variant"))
def test_merge_rejects_any_case_or_contract_mismatch(corruption):
    reports, specs = shard_reports()
    row = reports[1]["episodes"][0]
    if corruption == "state":
        row["reset"]["initial_state_id"] = 0
    elif corruption == "seed":
        row["seed"] = 100000
    elif corruption == "missing":
        reports[1]["episodes"].pop()
    elif corruption == "budget":
        row["primitive_steps"] = 281
    elif corruption == "telemetry":
        del row["reset"]["object_telemetry"]
    elif corruption == "checkpoint":
        reports[1]["checkpoint"] = "/checkpoint50"
    elif corruption == "task":
        row["reset"]["task_id"] = 2
    else:
        row["reset"]["experiment_split"] = VARIANTS[1]
    with pytest.raises(ValueError):
        merge_shards(reports, specs, config={}, checkpoint="/checkpoint25", variant=VARIANTS[0], task="object_0")


def test_shard_config_preserves_native_policy_and_sampler(monkeypatch):
    import tomllib

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("ood_queue", scripts / "run_vla_trace_ood_evaluations.py")
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    text = '''[policy]
kind = "pi05"
model_config = "pi05_libero"
[env]
factory = "rl_vla.diagnostics.observation_bank:make_ood_env"
[env.kwargs]
protocol = "/immutable/layout.json"
[rldt]
ode_steps = 10
execute_steps = 5
[recording]
enabled = true
'''
    before = tomllib.loads(text)
    after = tomllib.loads(queue.make_shard_config(text, 34, 16))
    assert after["policy"] == before["policy"]
    assert after["rldt"] == before["rldt"]
    assert after["env"]["kwargs"] == {"protocol": "/immutable/layout.json", "shard_start": 34, "shard_count": 16}


@pytest.fixture
def queue(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("ood_parallel_queue", scripts/"run_vla_trace_ood_evaluations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_gpu_plan_has_eight_disjoint_cells_each(queue):
    assignments = queue.cell_assignments(2)
    sets = [{(row["policy"], row["variant"]) for row in cells} for cells in assignments]
    assert len(sets[0]) == len(sets[1]) == 8
    assert not sets[0].intersection(sets[1])
    assert sets[0] | sets[1] == {(p, v) for p in queue.POLICIES for v in queue.VARIANTS}
    assert len(queue.cell_assignments(1)[0]) == 16


def complete_cells(root, queue):
    checkpoints = {policy: (None if iteration is None else f"/iteration_{iteration:06d}")
                   for policy, iteration in queue.POLICIES.items()}
    for policy in queue.POLICIES:
        for variant in queue.VARIANTS:
            episodes = [{"reset": {"initial_state_id": i}, "seed": 100000+i, "success": i % 2 == 0}
                        for i in range(50)]
            save(root/policy/variant/"summary.json", {"complete": True, "technical_only": False,
                 "task": "object_0", "variant": variant, "checkpoint": checkpoints[policy],
                 "episode_count": 50, "episodes": episodes, "successes": 25, "success_rate": .5,
                 "wilson95": queue.wilson(25, 50)})
    return checkpoints


def test_task_summary_requires_all_sixteen_complete_paired_cells(queue, tmp_path):
    checkpoints = complete_cells(tmp_path, queue)
    missing = tmp_path/"rldt_100"/queue.VARIANTS[-1]/"summary.json"
    contents = missing.read_text()
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        queue.summarize_task(tmp_path, "object_0", checkpoints)
    assert not any((tmp_path/policy/"summary.json").exists() for policy in queue.POLICIES)
    missing.write_text(contents)
    queue.summarize_task(tmp_path, "object_0", checkpoints)
    for policy in queue.POLICIES:
        result = json.loads((tmp_path/policy/"summary.json").read_text())
        assert result["complete"] and result["episodes"] == 200
        assert len(result["variants"]) == 4
    assert len((tmp_path/"events.jsonl").read_text().splitlines()) == 16


@pytest.mark.parametrize("workers", [2, 3])
def test_parallel_workers_keep_static_cell_output_names_and_locks(queue, monkeypatch, tmp_path, workers):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager

    barrier = threading.Barrier(workers)
    lock_calls, calls = [], []
    @contextmanager
    def lock(gpu, stop):
        lock_calls.append(gpu)
        barrier.wait(timeout=5)
        yield f"GPU-{gpu}"
    def evaluate(config, destination, checkpoint, **kwargs):
        destination.mkdir(parents=True, exist_ok=False)
        calls.append((kwargs["gpu"], destination.parts[-2], destination.name, checkpoint))
    monkeypatch.setattr(queue, "gpu_lock", lock)
    monkeypatch.setattr(queue, "readiness", lambda *args: {"ready": True})
    monkeypatch.setattr(queue, "evaluate_variant", evaluate)
    checkpoints = {p: str(i) for p, i in queue.POLICIES.items()}
    configs = {v: Path(v) for v in queue.VARIANTS}
    assignments = queue.cell_assignments(workers)
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(queue.run_worker, i, assignment, {"gpu": str(i), "cpus": list(range(3))},
                   tmp_path, "object_0", tmp_path, configs, checkpoints, {}, stop)
                   for i, assignment in enumerate(assignments)]
        assert [f.result() for f in futures] == assignments
    assert sorted(lock_calls) == list(map(str, range(workers)))
    assert len(calls) == 16 and len({(p, v) for _, p, v, _ in calls}) == 16
    assert all((tmp_path/p/v).is_dir() for _, p, v, _ in calls)


def test_worker_failure_sets_shared_stop_and_preserves_partial(queue, monkeypatch, tmp_path):
    import threading
    from contextlib import contextmanager

    @contextmanager
    def lock(gpu, stop):
        yield "GPU-test"
    def evaluate(config, destination, checkpoint, **kwargs):
        destination.mkdir(parents=True)
        raise RuntimeError("native child failed")
    monkeypatch.setattr(queue, "gpu_lock", lock)
    monkeypatch.setattr(queue, "readiness", lambda *args: {"ready": True})
    monkeypatch.setattr(queue, "evaluate_variant", evaluate)
    stop = threading.Event()
    assignment = queue.cell_assignments(2)[0]
    with pytest.raises(RuntimeError, match="native child failed"):
        queue.run_worker(0, assignment, {"gpu": "0", "cpus": [0, 1, 2]}, tmp_path, "object_0", tmp_path,
                         {v: Path(v) for v in queue.VARIANTS}, {p: None for p in queue.POLICIES}, {}, stop)
    assert stop.is_set()
    assert (tmp_path/assignment[0]["policy"]/assignment[0]["variant"]).is_dir()
    assert json.loads((tmp_path/"workers/worker_00/queue_status.json").read_text())["state"] == "failed"
    assert not (tmp_path/"queue_complete.json").exists()


def test_cleanup_ignores_exited_group_and_still_stops_later_groups(queue, monkeypatch):
    from types import SimpleNamespace
    calls = []
    def killpg(pid, signum):
        calls.append((pid, signum))
        if pid == 111:
            raise ProcessLookupError
    monkeypatch.setattr(queue.os, "killpg", killpg)
    processes = [SimpleNamespace(pid=pid, poll=lambda: 0, wait=lambda timeout: 0) for pid in (111, 222)]
    queue.stop_process_groups(processes, grace_seconds=0)
    assert (111, queue.signal.SIGTERM) in calls
    assert (222, queue.signal.SIGTERM) in calls
    assert (222, queue.signal.SIGKILL) in calls


def test_cleanup_stops_descendant_after_group_leader_exited(queue):
    import subprocess
    import sys
    leader = subprocess.Popen([sys.executable, "-c",
        "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True)"],
        start_new_session=True, stdout=subprocess.PIPE, text=True)
    child_pid = int(leader.stdout.readline())
    leader.wait(timeout=5)
    try:
        queue.stop_process_groups([leader], grace_seconds=.25)
        child_stat = Path(f"/proc/{child_pid}/stat")
        assert not child_stat.exists() or child_stat.read_text().split()[2] == "Z"
    finally:
        leader.stdout.close()
        queue.stop_process_groups([leader], grace_seconds=0)


def three_lane_args(tmp_path, **updates):
    from types import SimpleNamespace
    values = dict(run=tmp_path, task="object_0", gpu="4", cpuset="24-35", extra_gpu="0", extra_cpuset="0-7",
                  third_gpu="5", third_cpuset="68-71,82-85", poll_seconds=1)
    values.update(updates)
    return SimpleNamespace(**values)


def test_third_lane_has_exact_six_five_five_partition_without_changing_original_plans(queue, tmp_path):
    cells = [{"policy": p, "variant": v} for p in queue.POLICIES for v in queue.VARIANTS]
    assert queue.cell_assignments(1) == [cells]
    assert queue.cell_assignments(2) == [cells[0::2], cells[1::2]]
    assert queue.cell_assignments(3) == [cells[0::3], cells[1::3], cells[2::3]]
    assert list(map(len, queue.cell_assignments(3))) == [6, 5, 5]
    resources = queue.queue_resources(three_lane_args(tmp_path), tmp_path)
    assert resources[:2] == [{"gpu": "4", "cpus": list(range(24, 36)), "cpuset": "24-35"},
                             {"gpu": "0", "cpus": list(range(8)), "cpuset": "0-7"}]
    assert resources[2]["release_gate"] == str(tmp_path / "rollout_trace_queue_complete.json")
    resources2 = queue.queue_resources(three_lane_args(tmp_path, task="object_2", gpu="7", extra_gpu="2",
        third_gpu="3", third_cpuset="48-55"), tmp_path)
    assert resources2[2]["release_gate"] == str(tmp_path / "attention_grid_queue_complete.json")
    assert queue.queue_resources(three_lane_args(tmp_path, third_gpu=None, third_cpuset=None), tmp_path) == resources[:2]


@pytest.mark.parametrize("updates", [{"task": "object_1"}, {"third_gpu": "3"},
    {"third_gpu": "4"}, {"third_cpuset": "24-27"}, {"third_cpuset": "0,68,69"},
    {"third_cpuset": "68-69"}, {"third_cpuset": None}, {"extra_gpu": None, "extra_cpuset": None}])
def test_third_lane_rejects_unassigned_or_overlapping_resources(queue, tmp_path, updates):
    with pytest.raises(ValueError):
        queue.queue_resources(three_lane_args(tmp_path, **updates), tmp_path)


def release_marker(path, pid=12345, ticks=67890, **changes):
    child = {"pid": pid, "returncode": 0, "proc_start_ticks": ticks}
    child.update(changes)
    save(path, {"complete": True, "child_exit": child})


def test_release_gate_distinguishes_live_child_reaped_child_and_reused_pid(queue, monkeypatch, tmp_path):
    import os
    marker = tmp_path / "release.json"
    assert queue.released_diagnostic_gate(marker) is False
    ticks = queue.process_start_ticks(os.getpid())
    assert type(ticks) is int and ticks > 0
    release_marker(marker, pid=os.getpid(), ticks=ticks)
    assert queue.released_diagnostic_gate(marker) is False
    monkeypatch.setattr(queue, "process_start_ticks", lambda pid: None)
    assert queue.released_diagnostic_gate(marker) is True
    monkeypatch.setattr(queue, "process_start_ticks", lambda pid: ticks + 1)
    assert queue.released_diagnostic_gate(marker) is True


@pytest.mark.parametrize("changes", [{"pid": 0}, {"pid": True}, {"returncode": 1},
    {"returncode": False}, {"proc_start_ticks": 0}, {"proc_start_ticks": "123"}])
def test_completed_release_marker_requires_actual_successful_child_identity(queue, tmp_path, changes):
    marker = tmp_path / "release.json"
    release_marker(marker, **changes)
    with pytest.raises(ValueError):
        queue.released_diagnostic_gate(marker)


def test_third_lane_release_gate_is_required_before_freeze_and_in_input_paths(queue, monkeypatch, tmp_path):
    complete_gates(tmp_path)
    resources = queue.queue_resources(three_lane_args(tmp_path), tmp_path)
    gate = Path(resources[-1]["release_gate"])
    pending = queue.queue_readiness(tmp_path, "object_0", resources)
    assert pending["ready"] is False
    assert "third_gpu_diagnostic_child_exit" in pending["waiting_for"]
    release_marker(gate)
    monkeypatch.setattr(queue, "process_start_ticks", lambda pid: None)
    ready = queue.queue_readiness(tmp_path, "object_0", resources)
    assert ready["ready"] is True
    assert str(gate) in ready["gate_paths"].values()


def test_changed_release_marker_after_exclusive_lock_stops_before_any_cell(queue, monkeypatch, tmp_path):
    import threading
    from contextlib import contextmanager
    resources = queue.queue_resources(three_lane_args(tmp_path), tmp_path)
    marker = Path(resources[-1]["release_gate"])
    release_marker(marker)
    contract = {"inputs": {str(marker): queue.digest(marker)}}
    @contextmanager
    def lock(gpu, stop):
        release_marker(marker, ticks=99999)
        yield "GPU-test"
    monkeypatch.setattr(queue, "gpu_lock", lock)
    monkeypatch.setattr(queue, "readiness", lambda *args: {"ready": True})
    monkeypatch.setattr(queue, "evaluate_variant", lambda *args, **kwargs: pytest.fail("A cell launched before gate verification"))
    with pytest.raises(ValueError, match="input changed"):
        queue.run_worker(2, queue.cell_assignments(3)[2], resources[2], tmp_path, "object_0", tmp_path,
                         {}, {}, contract, threading.Event())


def test_existing_output_is_never_repartitioned_or_resumed(queue, monkeypatch, tmp_path):
    existing = tmp_path / "ood/object_0"
    existing.mkdir(parents=True)
    sentinel = existing / "queue_plan.json"
    sentinel.write_text("immutable prior two-lane plan")
    monkeypatch.setattr(queue, "queue_resources", lambda *args: pytest.fail("Existing output was reconsidered"))
    with pytest.raises(FileExistsError):
        queue.run(three_lane_args(tmp_path))
    assert sentinel.read_text() == "immutable prior two-lane plan"
