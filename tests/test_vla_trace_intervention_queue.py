"""Deterministic parallel plans and fail-fast preservation of independent jobs."""

import fcntl
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def queue():
    path = Path(__file__).resolve().parents[1]/"scripts/queue_vla_trace_diagnostics.py"
    spec = importlib.util.spec_from_file_location("diagnostic_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_cpu_lanes_cover_exact_original_pool(queue):
    assert queue.cpu_lanes("24-35") == ["24,25,26,27", "28,29,30,31", "32,33,34,35"]
    assert queue.cpu_lanes("72-79") == ["72,73,74", "75,76,77", "78,79"]
    with pytest.raises(ValueError):
        queue.cpu_lanes("1-2")
    with pytest.raises(ValueError):
        queue.cpu_lanes("1-3,2")


def concurrency_summary(queue):
    return {"complete": True, "passed": True, "workers": 4, "memory_fraction_per_worker": .2,
        "conditions": [{"condition": condition, "four_workers_finite": True,
            "baseline_bit_exact_to_previous_three_worker_reference":
                True if condition in {"none", "no_image", "target:mosaic"} else None,
            "native_no_op": [{"passed": True, "max_abs_difference": 0.0,
                "persisted_noise_recurrence_max_abs_difference": 0.0} for _ in range(4)]}
            for condition in queue.CONDITIONS]}


def test_concurrency_accepts_rich_per_condition_results(queue):
    queue.validate_concurrency_summary(concurrency_summary(queue))


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(conditions=list(reversed(s["conditions"]))),
    lambda s: s.update(conditions=[r["condition"] for r in s["conditions"]]),
    lambda s: s["conditions"][0].update(four_workers_finite=False),
    lambda s: s["conditions"][0]["native_no_op"].pop(),
    lambda s: s["conditions"][0]["native_no_op"][0].update(max_abs_difference=1e-8),
    lambda s: s["conditions"][0]["native_no_op"][0].update(persisted_noise_recurrence_max_abs_difference=float("nan")),
    lambda s: s["conditions"][6].update(baseline_bit_exact_to_previous_three_worker_reference=False),
])
def test_concurrency_rejects_incomplete_or_inexact_results(queue, mutation):
    summary = copy.deepcopy(concurrency_summary(queue))
    mutation(summary)
    with pytest.raises(ValueError):
        queue.validate_concurrency_summary(summary)


def test_policy_plan_preserves_exact_conditions_seed_and_outputs(queue, tmp_path):
    args = SimpleNamespace(run=tmp_path, task="object_1", gpu="6")
    policies = {"pi05_libero": None, **{f"rldt_{i}": tmp_path/f"iteration_{i:06d}" for i in (25, 50, 100)}}
    jobs = queue.policy_jobs(args, policies, tmp_path/"selection.json", tmp_path/"output")
    assert [j["name"] for j in jobs] == list(policies)
    for job in jobs:
        command = job["command"]
        assert command[command.index("--conditions") + 1].split(",") == list(queue.CONDITIONS)
        assert command[command.index("--episodes") + 1] == "10"
        assert command[command.index("--record-episodes") + 1] == "10"
        assert command[command.index("--seed") + 1] == "100000"
        assert job["environment"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.25"
        assert job["destination"] == str(tmp_path/"output"/job["name"])
        assert "--technical-smoke" not in command


def test_scheduler_runs_four_jobs_once_without_output_collisions(queue, tmp_path):
    cpu = str(min(os.sched_getaffinity(0)))
    jobs = []
    for index in range(4):
        destination = tmp_path/f"result{index}"
        code = "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.mkdir(); (p/'done').write_text('yes')"
        jobs.append({"name": str(index), "destination": str(destination),
                     "command": [sys.executable, "-c", code, str(destination)]})
    completed = queue.run_jobs(jobs, [cpu]*3, tmp_path/"logs")
    assert sorted(completed) == ["0", "1", "2", "3"]
    assert all((Path(j["destination"])/"done").read_text() == "yes" for j in jobs)
    with pytest.raises(FileExistsError, match="Preserving"):
        queue.run_jobs(jobs, [cpu]*3, tmp_path/"logs2")


def test_failed_child_terminates_other_owned_process_groups(queue, tmp_path):
    cpu = str(min(os.sched_getaffinity(0)))
    pid_file = tmp_path/"waiting.pid"
    wait_code = "import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    jobs = [{"name": "waiting", "destination": str(tmp_path/"waiting"),
             "command": [sys.executable, "-c", wait_code, str(pid_file)]},
            {"name": "fails", "destination": str(tmp_path/"fails"),
             "command": [sys.executable, "-c", "import time; time.sleep(.2); raise SystemExit(7)"]}]
    with pytest.raises(RuntimeError, match="exited 7"):
        queue.run_jobs(jobs, [cpu]*2, tmp_path/"logs")
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    status = json.loads((tmp_path/"logs/queue_status.json").read_text())
    assert status["state"] == "running"  # caller publishes failure; never a completion gate


def test_checkpoint_weights_match_main_audit_and_stat_guards(queue, tmp_path):
    audit = {"passed": True, "checkpoints": {}}
    for iteration in (25, 50, 100):
        folder = tmp_path/"training"/f"iteration_{iteration:06d}"
        folder.mkdir(parents=True)
        files = {}
        for name in ("expert.npz", "rldt_pi05.json"):
            path = folder/name
            path.write_bytes(f"{iteration}/{name}".encode())
            files[name] = {"bytes": path.stat().st_size, "sha256": queue.file_hash(path)}
        audit["checkpoints"][str(iteration)] = {"files": files}
    contract = queue.checkpoint_contract(tmp_path, audit)
    assert len(contract) == 6
    queue.verify_checkpoint_contract(contract)
    queue.verify_checkpoint_contract(contract, full_hash=True)
    path = Path(next(iter(contract)))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="stat changed"):
        queue.verify_checkpoint_contract(contract)
    with pytest.raises(ValueError, match="main audit"):
        queue.checkpoint_contract(tmp_path, audit)


def test_cleanup_continues_after_an_owned_group_already_exited(queue, monkeypatch):
    calls = []
    def killpg(pid, signum):
        calls.append((pid, signum))
        if pid == 111:
            raise ProcessLookupError
    monkeypatch.setattr(queue.os, "killpg", killpg)
    processes = [SimpleNamespace(pid=pid, poll=lambda: 0, wait=lambda timeout: 0) for pid in (111, 222)]
    queue.stop_children(processes, grace_seconds=0)
    assert (222, queue.signal.SIGTERM) in calls and (222, queue.signal.SIGKILL) in calls


def overflow_lane(queue, monkeypatch, tmp_path, task="object_0"):
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *args, **kwargs: "GPU-test\n")
    return queue.GridReleasedOverflow(tmp_path, task, lock_directory=tmp_path/"locks")


def test_overflow_requires_complete_grid_and_distinct_shared_slots(queue, monkeypatch, tmp_path):
    lanes = [overflow_lane(queue, monkeypatch, tmp_path, task) for task in queue.OVERFLOW_CPU_LANES]
    assert [lane.cpuset for lane in lanes] == ["48,49,50", "51,52,53", "54,55"]
    assert not lanes[0].try_acquire()
    gate = tmp_path/"attention_grid_queue_complete.json"
    gate.write_text('{"complete":false}')
    assert not lanes[0].try_acquire()
    gate.write_text('{"complete":')
    assert not lanes[0].try_acquire()
    gate.write_text('{"complete":true}')
    try:
        assert all(lane.try_acquire() for lane in lanes)
        duplicate = overflow_lane(queue, monkeypatch, tmp_path)
        assert not duplicate.try_acquire()
        with (tmp_path/"locks/GPU-test.lock").open("a") as exclusive, pytest.raises(BlockingIOError):
            fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lanes[0].mark_launched({"name": "rldt_100"}, SimpleNamespace(pid=12345))
        lanes[0].release()
        assert not lanes[0].try_acquire()  # never more than one overflow policy per task
    finally:
        for lane in lanes:
            lane.release()
    with (tmp_path/"locks/GPU-test.lock").open("a") as exclusive:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not lanes[1].try_acquire()


def fake_scheduler(queue, monkeypatch, tmp_path):
    starts, processes, stopped, validated = [], {}, [], []
    class Process:
        def __init__(self, name):
            self.name, self.code = name, None
            self.pid = 900000 + len(processes)
        def poll(self):
            return self.code
    def popen(command, **kwargs):
        name = command[-1]
        process = Process(name)
        processes[name] = process
        starts.append({"name": name, "gpu": kwargs["env"]["CUDA_VISIBLE_DEVICES"],
                       "cpuset": kwargs["env"]["RLDT_CPUSET"],
                       "memory": kwargs["env"]["XLA_PYTHON_CLIENT_MEM_FRACTION"]})
        return process
    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    monkeypatch.setattr(queue, "stop_children", lambda children: stopped.extend(p.name for p in children))
    jobs = [{"name": f"policy{index}", "destination": str(tmp_path/f"policy{index}"),
             "command": ["fake-native-policy", f"policy{index}"],
             "environment": {"CUDA_VISIBLE_DEVICES": "4", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.25"}}
            for index in range(4)]
    return jobs, starts, processes, stopped, validated


@pytest.mark.parametrize("grid_state", (None, {"complete": False}, "incomplete-json"))
def test_grid_delay_or_failure_never_blocks_fourth_policy_on_primary(queue, monkeypatch, tmp_path, grid_state):
    overflow = overflow_lane(queue, monkeypatch, tmp_path)
    if grid_state is not None:
        (tmp_path/"attention_grid_queue_complete.json").write_text(
            grid_state if isinstance(grid_state, str) else json.dumps(grid_state))
    jobs, starts, processes, _, validated = fake_scheduler(queue, monkeypatch, tmp_path)
    def tick(_):
        if len(starts) == 3:
            processes["policy0"].code = 0
        else:
            for process in processes.values():
                process.code = 0
    monkeypatch.setattr(queue.time, "sleep", tick)
    result = queue.run_jobs(jobs, ["0", "1", "2"], tmp_path/"logs", overflow=overflow,
                            validate=lambda job: validated.append(job["name"]))
    assert sorted(result) == sorted(validated) == [f"policy{i}" for i in range(4)]
    assert [row["gpu"] for row in starts] == ["4"]*4
    assert starts[-1]["cpuset"] == "0" and not overflow.used


def test_grid_completion_opens_fourth_policy_once_on_gpu3(queue, monkeypatch, tmp_path):
    overflow = overflow_lane(queue, monkeypatch, tmp_path)
    jobs, starts, processes, _, validated = fake_scheduler(queue, monkeypatch, tmp_path)
    immutable_calls = []
    def tick(_):
        if len(starts) == 3:
            (tmp_path/"attention_grid_queue_complete.json").write_text('{"complete":true}')
        else:
            for process in processes.values():
                process.code = 0
    monkeypatch.setattr(queue.time, "sleep", tick)
    result = queue.run_jobs(jobs, ["0", "1", "2"], tmp_path/"logs", overflow=overflow,
        validate=lambda job: validated.append(job["name"]), immutable_check=lambda: immutable_calls.append(True))
    assert result == validated == [f"policy{i}" for i in range(4)]
    assert len(starts) == 4 and len({row["name"] for row in starts}) == 4
    assert [row["gpu"] for row in starts] == ["4", "4", "4", "3"]
    assert starts[-1]["cpuset"] == "48,49,50" and starts[-1]["memory"] == "0.25"
    assert overflow.used and not overflow._locks and len(immutable_calls) >= 3
    record = json.loads((tmp_path/"logs/overflow_policy.json").read_text())
    assert record["policy"] == "policy3" and record["grid_completion_sha256"] == queue.file_hash(overflow.gate)


def test_primary_finishing_during_overflow_acquisition_still_wins(queue, monkeypatch, tmp_path):
    overflow = overflow_lane(queue, monkeypatch, tmp_path)
    (tmp_path/"attention_grid_queue_complete.json").write_text('{"complete":true}')
    jobs, starts, processes, _, _ = fake_scheduler(queue, monkeypatch, tmp_path)
    acquire = overflow.try_acquire
    def acquire_then_primary_finishes():
        result = acquire()
        processes["policy0"].code = 0
        return result
    monkeypatch.setattr(overflow, "try_acquire", acquire_then_primary_finishes)
    def tick(_):
        for process in processes.values():
            process.code = 0
    monkeypatch.setattr(queue.time, "sleep", tick)
    queue.run_jobs(jobs, ["0", "1", "2"], tmp_path/"logs", overflow=overflow)
    assert [row["gpu"] for row in starts] == ["4"]*4
    assert starts[-1]["cpuset"] == "0" and not overflow.used and not overflow._locks


def test_overflow_failure_stops_all_four_registered_groups_and_releases_locks(queue, monkeypatch, tmp_path):
    overflow = overflow_lane(queue, monkeypatch, tmp_path)
    (tmp_path/"attention_grid_queue_complete.json").write_text('{"complete":true}')
    jobs, starts, processes, stopped, _ = fake_scheduler(queue, monkeypatch, tmp_path)
    def tick(_):
        processes["policy3"].code = 9
    monkeypatch.setattr(queue.time, "sleep", tick)
    with pytest.raises(RuntimeError, match="policy3 exited 9"):
        queue.run_jobs(jobs, ["0", "1", "2"], tmp_path/"logs", overflow=overflow)
    assert len(starts) == 4 and set(stopped) == {f"policy{i}" for i in range(4)}
    assert not overflow._locks
    replacement = overflow_lane(queue, monkeypatch, tmp_path)
    assert replacement.try_acquire()
    replacement.release()


def test_trained_profile_cannot_implicitly_compile_untouched_baseline(queue, monkeypatch, tmp_path):
    args = SimpleNamespace(run=tmp_path, task="object_0", gpu="4", workers=3)
    trained = {"manifest": {"profile_id": "native_trained_v2"}, "manifest_path": "/trained/manifest.json"}
    monkeypatch.setattr(queue, "child_environment", lambda *args: pytest.fail("Baseline reached the trained profile environment"))
    with pytest.raises(ValueError, match="separate native compiler profile"):
        queue.policy_jobs(args, {"pi05_libero": None}, tmp_path / "selection.json", tmp_path / "output",
                          compiler_profile=trained)


def test_four_worker_policy_plan_changes_only_memory_and_cpu_partition(queue, monkeypatch, tmp_path):
    profiles = {kind: {"manifest_path": str(tmp_path / kind / "manifest.json"),
        "manifest": {"profile_id": identifier, "isolated_cache_root": str(tmp_path / "cache" / kind)}}
        for kind, identifier in (("baseline", "native_baseline_v1"), ("trained", "native_trained_v2"))}
    def environment(profile, cache):
        env = {"XLA_FLAGS": "profile=" + profile["manifest"]["profile_id"],
               "JAX_COMPILATION_CACHE_DIR": str(cache), "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": ""}
        return env, {"profile_id": profile["manifest"]["profile_id"], "environment": env}
    monkeypatch.setattr(queue, "child_environment", environment)
    args = SimpleNamespace(run=tmp_path, task="object_0", gpu="4", workers=3)
    policies = {"pi05_libero": None, **{f"rldt_{iteration}": tmp_path / f"iteration_{iteration:06d}"
                                       for iteration in (25, 50, 100)}}
    three = queue.policy_jobs(args, policies, tmp_path / "selection.json", tmp_path / "output",
                              compiler_profile=profiles["trained"], baseline_profile=profiles["baseline"])
    args.workers = 4
    four = queue.policy_jobs(args, policies, tmp_path / "selection.json", tmp_path / "output",
                             compiler_profile=profiles["trained"], baseline_profile=profiles["baseline"])
    assert queue.cpu_lanes("24-31", workers=4) == ["24,25", "26,27", "28,29", "30,31"]
    assert [job["name"] for job in four] == list(policies)
    for previous, job in zip(three, four, strict=True):
        assert {key: value for key, value in job.items() if key != "environment"} == {
            key: value for key, value in previous.items() if key != "environment"}
        assert job["environment"] == {**previous["environment"], "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.20"}
        assert previous["environment"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.25"
        command = job["command"]
        assert command[command.index("--conditions") + 1].split(",") == list(queue.CONDITIONS)
        assert len(queue.CONDITIONS) == 9
        assert command[command.index("--config") + 1] == str(tmp_path / "tasks/object_0/test.toml")
        assert command[command.index("--episodes") + 1] == command[command.index("--record-episodes") + 1] == "10"
        assert command[command.index("--seed") + 1] == "100000"
        assert job["compiler_profile"]["profile_id"] == (
            "native_baseline_v1" if job["name"] == "pi05_libero" else "native_trained_v2")
