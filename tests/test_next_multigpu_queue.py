"""The dispatcher fails closed and never touches a real GPU in these tests."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def queue(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "SLURM_JOB_GPUS",
                 "SLURM_STEP_GPUS", "SLURM_JOB_ID"):
        monkeypatch.delenv(name, raising=False)
    spec = importlib.util.spec_from_file_location("next_queue_test", scripts / "run_next_multigpu_queue.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dependencies_require_success_and_the_first_run_must_really_start(queue, tmp_path):
    first = {"job_id": "E00", "state": "blocked_dependency"}
    smoke = {"job_id": "smoke", "state": "running"}
    artifact = tmp_path / "smoke.json"
    job = {"job_id": "E01", "command_argv": ["python"], "dependencies": ["smoke"],
           "required_started_jobs": ["E00"], "required_artifacts": [str(artifact)]}
    assert len(queue.unresolved(job, [first, smoke, job])) == 3
    first.update(state="running", started_at="2026-10-31T10:00:00Z")
    smoke["state"] = "completed"
    artifact.write_text("{}")
    assert queue.unresolved(job, [first, smoke, job]) == []
    first["state"] = "failed"
    assert "mandatory first production" in queue.unresolved(job, [first, smoke, job])[0]


def test_prerequisites_missing_malformed_and_unknown_never_pass(queue, tmp_path):
    path = tmp_path / "prerequisite.json"
    job = {"job_id": "E00", "command_argv": ["python"], "prerequisites": [
        {"path": str(path), "kind": "json_equals", "key": "scope.passed", "value": True}]}
    assert queue.unresolved(job, [job])
    path.write_text('{"scope": {"passed": false}}')
    assert queue.unresolved(job, [job])
    path.write_text('{"scope": {"passed": true}}')
    assert not queue.unresolved(job, [job])
    job["prerequisites"][0]["kind"] = "typo"
    assert "unsupported prerequisite kind" in queue.unresolved(job, [job])[0]
    job["gpu_count"] = 2
    assert "exactly one GPU" in queue.unresolved(job, [job])[-1]


def test_gpu_allocations_intersect_and_explicit_empty_disables(queue, monkeypatch):
    rows = [{"index": i, "uuid": f"GPU-{i}"} for i in range(5)]
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert queue.allocated_gpus(rows) == []
    monkeypatch.setenv("SLURM_JOB_GPUS", "1-3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-2,GPU-3,GPU-4")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "all")
    assert [r["index"] for r in queue.allocated_gpus(rows)] == [2, 3]
    monkeypatch.setenv("SLURM_STEP_GPUS", "3")
    assert [r["index"] for r in queue.allocated_gpus(rows)] == [3]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert queue.allocated_gpus(rows) == []


@pytest.mark.parametrize("message,expected", [
    ("CUDA_ERROR_OUT_OF_MEMORY", True),
    ("Connection reset by peer", True),
    ("RESOURCE_EXHAUSTED: CUDA out of memory", True),
    ("CUDA out of memory\nValueError: invalid simulator contract", False),
    ("CUDA out of memory\nnon-finite actor gradient", False),
    ("CUDA out of memory\ncheckpoint hash mismatch", False),
    ("All task successes were zero", False),
])
def test_only_transient_execution_failures_are_retryable(queue, tmp_path, message, expected):
    path = tmp_path / "attempt.log"
    path.write_text(message)
    assert queue.transient_failure(path) is expected


def test_immutable_input_guards_after_initial_hash(queue, tmp_path):
    path = tmp_path / "spec.json"
    path.write_text('{"seed": 0}')
    contract = {str(path): queue.sha256(path)}
    versions = queue.verify_input_contract(contract)
    assert queue.verify_input_contract(contract, versions) == versions
    path.write_text('{"seed": 1}')
    with pytest.raises(RuntimeError, match="changed after verification"):
        queue.verify_input_contract(contract, versions)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        queue.verify_input_contract(contract)


def scheduler_fixture(queue, monkeypatch, tmp_path, jobs, outcomes, gate=True):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    config = {"repository": str(tmp_path), "source_snapshot": str(tmp_path),
              "campaign": str(campaign), "current_run": str(tmp_path / "current")}
    (campaign / "queue_config.json").write_text(json.dumps(config))
    (campaign / "input_hashes.json").write_text("{}")
    for job in jobs:
        job.setdefault("state", "ready")
        job.setdefault("priority", 0)
        job.setdefault("output_directory", str(campaign / job["job_id"]))
        job.setdefault("command_argv", ["python", "worker.py", "--output", "unused"])
    queue.save_jobs(campaign / "jobs.jsonl", jobs)
    row = {"index": 0, "uuid": "GPU-test", "total_mib": 65536, "used_mib": 0,
           "utilization": 0, "has_compute_process": False}
    monkeypatch.setattr(queue, "query_gpus", lambda: [row])
    monkeypatch.setattr(queue, "cpu_groups", lambda count: [[0, 1]])
    monkeypatch.setattr(queue, "memory_available_gib", lambda: 300)
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(queue, "make_environment", lambda *args: {})
    monkeypatch.setattr(queue, "inspect_current_gate", lambda path: {"ready": gate})
    monkeypatch.setattr(queue.signal, "signal", lambda *args: None)
    monkeypatch.setattr(queue.time, "sleep", lambda delay: None)
    launched = []

    def launch(argv, **kwargs):
        output = Path(argv[argv.index("--output") + 1])
        job_id = output.parent.name
        launched.append((job_id, output.name))
        code, text, completed_id = outcomes.pop(0)
        if code == "launch_error":
            raise OSError("executable unavailable")
        kwargs["stdout"].write(text)
        kwargs["stdout"].flush()
        if completed_id:
            output.mkdir()
            (output / "completion.json").write_text(json.dumps({"completed": True, "job_id": completed_id}))
        return SimpleNamespace(pid=os.getpid(), poll=lambda: code)

    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    return campaign, launched


def test_closed_gate_never_launches_even_with_free_gpu(queue, monkeypatch, tmp_path):
    campaign, launched = scheduler_fixture(queue, monkeypatch, tmp_path, [{"job_id": "E00"}], [], gate=False)
    queue.run(campaign, once=True)
    assert launched == []
    job = json.loads((campaign / "jobs.jsonl").read_text())
    assert job["state"] == "blocked_dependency"
    assert "current LIBERO" in job["waiting_for"][0]


def test_authorized_independent_jobs_launch_without_claiming_prior_analysis_complete(queue, monkeypatch, tmp_path):
    campaign, launched = scheduler_fixture(queue, monkeypatch, tmp_path,
        [{"job_id": "independent"}], [(0, "", "independent")], gate=False)
    path = campaign / "queue_config.json"
    config = json.loads(path.read_text())
    config.update(wait_for_current_campaign=False, scheduling_amendment="user-approved.json")
    path.write_text(json.dumps(config))
    queue.run(campaign)
    assert launched == [("independent", "attempt_1")]
    gate = json.loads((campaign / "dispatch_gate.json").read_text())
    assert gate["ready"] and not gate["current_campaign_complete"]


def test_explicit_cpu_assignments_are_disjoint_and_inside_allocation(queue, monkeypatch):
    monkeypatch.setattr(queue.os, "sched_getaffinity", lambda pid: set(range(8)))
    rows = [{"index": 0, "uuid": "GPU-a"}, {"index": 4, "uuid": "GPU-b"}]
    config = {"cpu_assignments_by_gpu_index": {"0": [0, 1], "4": [4, 5]}}
    assert queue.worker_cpu_sets(rows, config) == {"GPU-a": [0, 1], "GPU-b": [4, 5]}
    for bad in ([1, 4], [8, 9], [4, 4]):
        config["cpu_assignments_by_gpu_index"]["4"] = bad
        with pytest.raises(ValueError, match="CPU assignment"):
            queue.worker_cpu_sets(rows, config)


def test_transient_retries_once_in_fresh_output_then_blocks_only_that_job(queue, monkeypatch, tmp_path):
    jobs = [{"job_id": "first", "priority": 2}, {"job_id": "second", "priority": 10}]
    campaign, launched = scheduler_fixture(queue, monkeypatch, tmp_path, jobs, [
        (1, "CUDA out of memory", None), (1, "CUDA out of memory", None), (0, "", "second")])
    queue.run(campaign)
    assert launched == [("first", "attempt_1"), ("first", "attempt_2"), ("second", "attempt_1")]
    result = [json.loads(line) for line in (campaign / "jobs.jsonl").read_text().splitlines()]
    assert [j["state"] for j in result] == ["failed", "completed"]
    assert result[0]["attempts"] == 2


def test_wrong_completion_id_and_launch_failure_do_not_complete_or_abort_other_jobs(queue, monkeypatch, tmp_path):
    jobs = [{"job_id": "a"}, {"job_id": "b"}, {"job_id": "c"}]
    campaign, launched = scheduler_fixture(queue, monkeypatch, tmp_path, jobs, [
        (0, "", "someone-else"), ("launch_error", "", None), (0, "", "c")])
    queue.run(campaign)
    result = [json.loads(line) for line in (campaign / "jobs.jsonl").read_text().splitlines()]
    assert [j["state"] for j in result] == ["failed", "failed", "completed"]
    assert len(launched) == 3


def test_restart_refuses_to_duplicate_a_previously_running_job(queue, monkeypatch, tmp_path):
    campaign, launched = scheduler_fixture(queue, monkeypatch, tmp_path,
                                          [{"job_id": "E00", "state": "running"}], [])
    with pytest.raises(RuntimeError, match="Reconcile prior running job"):
        queue.run(campaign)
    assert launched == []


def test_transient_classifier_reads_native_stage_logs(queue, tmp_path):
    outer = tmp_path / "attempt_1.log"
    outer.write_text("subprocess.CalledProcessError: train exited with code 1")
    attempt = tmp_path / "attempt_1"
    attempt.mkdir()
    stage = attempt / "train.log"
    stage.write_text("XlaRuntimeError: RESOURCE_EXHAUSTED: memory allocation failed")
    assert queue.transient_failure(outer, attempt)
    stage.write_text("Nonfinite training metrics; RESOURCE_EXHAUSTED during cleanup")
    assert not queue.transient_failure(outer, attempt)


def test_coordinator_lock_prevents_two_dispatchers(queue, monkeypatch, tmp_path):
    campaign, launched = scheduler_fixture(queue, monkeypatch, tmp_path, [{"job_id": "E00"}], [])
    with (campaign / "coordinator.lock").open("a+") as owner:
        queue.fcntl.flock(owner, queue.fcntl.LOCK_EX | queue.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            queue.run(campaign)
    assert launched == []


def test_results_only_include_completed_attempts(queue, tmp_path):
    jobs = []
    for job_id, state in (("done", "completed"), ("partial", "failed"), ("active", "running")):
        path = tmp_path / job_id
        path.mkdir()
        (path / "results.csv").write_text(f"job_id,successes,episodes\n{job_id},0,50\n")
        jobs.append({"job_id": job_id, "state": state, "active_output": str(path)})
    assert queue.aggregate_results(tmp_path, jobs) == []
    with (tmp_path / "results.csv").open() as stream:
        rows = list(queue.csv.DictReader(stream))
    assert [(row["job_id"], row["successes"]) for row in rows] == [("done", "0")]


def test_finite_queue_reports_unresolved_blockers_and_releases_resources(queue, monkeypatch, tmp_path):
    campaign, launched = scheduler_fixture(queue, monkeypatch, tmp_path,
        [{"job_id": "E00", "blockers": ["LoRA bridge unavailable"]}], [])
    queue.run(campaign)
    assert launched == []
    final = json.loads((campaign / "final_status.json").read_text())
    assert final["state"] == "finished_with_blockers"
    assert final["blocked_jobs"] == [{"job_id": "E00", "reasons": ["LoRA bridge unavailable"]}]
    assert final["resources_released"] is True
