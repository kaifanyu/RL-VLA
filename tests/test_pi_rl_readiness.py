"""Readiness gates fail closed on incomplete, unsuccessful, or missing evidence."""

import fcntl
import hashlib
import importlib.util
import json
import select
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location(
    "pi_rl_readiness_test", Path(__file__).resolve().parents[1] / "scripts/run_pi_rl_readiness_grasp.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def _smoke(directory, method, *, finished=True):
    directory.mkdir(parents=True, exist_ok=True)
    cfg = {"runner": {"max_epochs": 1, "max_steps": 1,
                      "logger": {"log_path": str(directory), "experiment_name": method}},
           "env": {"train": {"max_steps_per_rollout_epoch": 10}},
           "actor": {"model": {"openpi": {"noise_method": method}}}}
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg))
    if finished:
        worker.write_json(directory / "summary.json", {"completed": True, "iterations": 1, "decisions": 2})
        path = directory / method / "checkpoints/global_step_1/actor/model_state_dict/full_weights.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture-checkpoint")
        (directory / "iteration_metrics.jsonl").write_text('{"iteration": 1}\n')
        (directory / "trajectories").mkdir()
        worker.write_json(directory / "trajectories/summary.json", {"chunks": 2})


def test_artifact_contract_requires_saved_actor_and_final_partial_recording(tmp_path):
    _smoke(tmp_path, "flow_sde")
    artifacts = worker.smoke_artifacts(tmp_path, "flow_sde")
    assert artifacts["checkpoint"].name == "full_weights.pt"
    (tmp_path / "trajectories/summary.json").unlink()
    assert worker.smoke_artifacts(tmp_path, "flow_sde") is None
    worker.write_json(tmp_path / "summary.json", {"completed": True, "iterations": 1, "decisions": 1})
    with pytest.raises(ValueError, match="interaction budget"):
        worker.smoke_artifacts(tmp_path, "flow_sde")


@pytest.mark.parametrize("state,code", [("FAILED", "1:0"), ("TIMEOUT", "0:15"),
                                        ("CANCELLED", "0:15"), ("COMPLETED", "1:0")])
def test_failed_slurm_steps_never_count_as_success(state, code):
    with pytest.raises(RuntimeError, match="Slurm step failed"):
        worker.assert_step_not_failed({"state": state, "exit_code": code})
    assert not worker.assert_step_not_failed(None)
    assert not worker.assert_step_not_failed({"state": "RUNNING", "exit_code": "0:0"})
    assert worker.assert_step_not_failed({"state": "COMPLETED", "exit_code": "0:0"})


@pytest.mark.parametrize("missing", worker.GATES)
def test_every_gate_requires_explicit_true(missing):
    values = {name: name != missing for name in worker.GATES}
    with pytest.raises(RuntimeError, match="unpassed checks"):
        worker.readiness_document(values["installation_complete"],
                                  {"parity_passed": values["parity_passed"]},
                                  {method: {"smoke_passed": values[f"{method}_smoke_passed"]}
                                   for method in ("flow_sde", "flow_noise")}, {})


def _models():
    return {method: {"parameter_counts": {
        "action_expert": 430_098_464, "frozen_vlm": 2_923_335_408,
        "noise_head": 141_536 if method == "flow_noise" else 0,
    }} for method in ("flow_sde", "flow_noise")}


def test_deep_cpu_model_check_requires_both_official_variants():
    assert worker.validate_models(_models())
    with pytest.raises(ValueError, match="flow_noise"):
        worker.validate_models({"flow_sde": _models()["flow_sde"]})


def test_successful_worker_launches_only_after_smokes_gpu_parity_and_install_checks(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    _smoke(suite / "smoke/flow_sde", "flow_sde")
    _smoke(suite / "smoke/flow_noise", "flow_noise", finished=False)
    models = tmp_path / "models.json"
    worker.write_json(models, _models())
    parity = tmp_path / "parity"
    parity.mkdir()
    args = SimpleNamespace(suite=suite, allocation=1470624, timeout=30, sde_step="1470624.32",
                           parity_step="1470624.36", parity_directory=parity, model_report=models,
                           native_log=tmp_path / "native.log")
    monkeypatch.setenv("SLURM_JOB_ID", "1470624")
    monkeypatch.setattr(worker, "step_state", lambda step: {"state": "COMPLETED", "exit_code": "0:0"})
    monkeypatch.setattr(worker, "timed_gpu_lock", lambda *args: nullcontext())
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(worker, "verify_converted_checkpoint", lambda *args: {"weights": "verified"})
    monkeypatch.setattr(worker, "verify_gpu_parity_provenance", lambda *args: {"gpu": "verified"})
    order = []

    def verify(directory, method, artifacts, work, deadline):
        assert not (suite / "pi_rl_readiness.json").exists()
        order.append(method + "_verification")
        result = {"smoke_passed": True}
        worker.write_json(work / f"{method}_verification.json", result)
        return result

    def run_logged(command, log, deadline, *, cpu=False):
        assert not (suite / "pi_rl_readiness.json").exists()
        log.write_text("passed\n")
        if "integrations/rlinf_bridge/train.py" in command:
            order.append("noise_train")
            _smoke(suite / "smoke/flow_noise", "flow_noise")
            assert not cpu
        elif "compare" in command:
            order.append("gpu_parity")
            worker.write_json(Path(command[-1]), {"parity_passed": True})
            assert cpu
        else:
            order.append("installation")
            assert cpu

    def launch(suite, gates, allocation, work):
        assert all(json.loads(gates.read_text())[name] is True for name in worker.GATES)
        order.append("launch_six_runs")
        return {"pid": 123}

    monkeypatch.setattr(worker, "verify_smoke", verify)
    monkeypatch.setattr(worker, "run_logged", run_logged)
    monkeypatch.setattr(worker, "launch_suite", launch)
    worker.run(args)
    assert order == ["flow_sde_verification", "noise_train", "flow_noise_verification", "gpu_parity",
                     "installation", "installation", "installation", "installation", "launch_six_runs"]
    assert json.loads((suite / "pi_rl_readiness/status.json").read_text())["state"] == "complete"


def test_slurm_failure_records_false_gates_and_never_starts_noise(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    _smoke(suite / "smoke/flow_sde", "flow_sde", finished=False)
    args = SimpleNamespace(suite=suite, allocation=1470624, timeout=30, sde_step="1470624.32",
                           parity_step="1470624.36", parity_directory=tmp_path, model_report=tmp_path / "model.json")
    monkeypatch.setenv("SLURM_JOB_ID", "1470624")
    monkeypatch.setattr(worker, "step_state", lambda step: {"state": "FAILED", "exit_code": "1:0"})
    monkeypatch.setattr(worker, "run_logged", lambda *args, **kwargs: pytest.fail("Must not run Noise"))
    with pytest.raises(RuntimeError, match="Slurm step failed"):
        worker.run(args)
    result = json.loads((suite / "pi_rl_readiness.json").read_text())
    assert all(result[name] is False for name in worker.GATES)
    assert json.loads((suite / "pi_rl_readiness/status.json").read_text())["state"] == "failed"


def test_converted_weights_are_checked_against_the_actual_manifest(tmp_path):
    weights, stats = tmp_path / "model.safetensors", tmp_path / "physical-intelligence/libero/norm_stats.json"
    stats.parent.mkdir(parents=True)
    weights.write_bytes(b"original weights")
    stats.write_text("{}")
    worker.write_json(tmp_path / "conversion_manifest.json", {
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "rlinf_normalization_sha256": hashlib.sha256(stats.read_bytes()).hexdigest(),
    })
    assert worker.verify_converted_checkpoint(tmp_path, worker.Deadline(30))["weights"]["bytes"] == 16
    weights.write_bytes(b"modified weights")
    with pytest.raises(ValueError, match="SHA256"):
        worker.verify_converted_checkpoint(tmp_path, worker.Deadline(30))


def test_suite_launcher_keeps_worker_alive_until_child_exits(tmp_path, monkeypatch):
    waited = []

    class Process:
        pid = 123

        def wait(self):
            waited.append(True)
            record = json.loads((tmp_path / "pi_rl_launch.json").read_text())
            assert record["status"] == "running"
            _completed_queue(tmp_path)
            return 0

    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: Process())
    result = worker.launch_suite(tmp_path, tmp_path / "gates.json", 1470624, tmp_path)
    assert waited == [True] and result["status"] == "completed"
    assert "srun" not in result["command"]


def _completed_queue(suite):
    directory = suite / "pi_rl_queue"
    directory.mkdir()
    worker.write_json(directory / "progress.json", {"failures": [], "remaining": 0, "completed": [
        {"task": task, "method": method, "exit_code": 0}
        for task in ("spatial_1", "spatial_2", "object_0") for method in ("flow_sde", "flow_noise")]})
    worker.write_json(directory / "status.json", {"state": "completed"})


def test_zero_exit_does_not_mask_partial_or_failed_queue(tmp_path):
    _completed_queue(tmp_path)
    worker.validate_queue_completion(tmp_path)
    path = tmp_path / "pi_rl_queue/progress.json"
    progress = json.loads(path.read_text())
    progress["failures"].append(progress["completed"].pop())
    worker.write_json(path, progress)
    with pytest.raises(RuntimeError, match="six successful"):
        worker.validate_queue_completion(tmp_path)


def test_gpu_parity_requires_actual_cuda_execution_and_bound_outputs(tmp_path):
    parity, converted = tmp_path / "parity", tmp_path / "converted"
    parity.mkdir()
    converted.mkdir()
    for name in ("native.npz", "torch.npz", "fixture.npz"):
        (parity / name).write_bytes(name.encode())
    manifest = converted / "conversion_manifest.json"
    worker.write_json(manifest, {"weights_sha256": "expected converted weights digest"})
    record = {"backend": "torch", "actual_device": "cuda:0",
              "fixture": worker.file_record(parity / "fixture.npz"),
              "output_artifact": worker.file_record(parity / "torch.npz"),
              "checkpoint": {"path": str(converted), "conversion_manifest_path": str(manifest),
                             "conversion_manifest_sha256": worker.file_record(manifest)["sha256"],
                             "model_sha256": "expected converted weights digest"}}
    metadata = parity / "torch.json"
    worker.write_json(metadata, record)
    native_log = tmp_path / "native.log"
    worker.write_json(native_log, {"backend": "jax", "devices": ["cuda:0"],
                                  "output": str(parity / "native.npz")})
    # Native stdout is a one-line JSON record, unlike a pretty-printed report.
    native_log.write_text(json.dumps({"backend": "jax", "devices": ["cuda:0"],
                                     "output": str(parity / "native.npz")}) + "\n")
    assert worker.verify_gpu_parity_provenance(parity, native_log, converted)["actual_torch_device"] == "cuda:0"
    worker.write_json(metadata, record | {"actual_device": "cpu"})
    with pytest.raises(ValueError, match="executed on the GPU"):
        worker.verify_gpu_parity_provenance(parity, native_log, converted)
    worker.write_json(metadata, record)
    (parity / "torch.npz").write_bytes(b"substituted output")
    with pytest.raises(ValueError, match="output_artifact identity"):
        worker.verify_gpu_parity_provenance(parity, native_log, converted)


def test_readiness_deadline_is_bounded():
    with pytest.raises(TimeoutError, match="deadline"):
        worker.Deadline(0).remaining()


@pytest.fixture
def gpu_lock_path(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    path = tmp_path / "runs/gpu_allocation_test/experiment.lock"
    path.parent.mkdir(parents=True)
    return path


@pytest.fixture
def isolated_alarm():
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    received = []

    def handler(signum, frame):
        received.append(signum)

    signal.signal(signal.SIGALRM, handler)
    try:
        yield handler, received
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


@contextmanager
def _other_process_holds_lock(path):
    """Hold a real flock until asked to release it after a short delay."""
    script = (
        "import fcntl,sys,time\n"
        "with open(sys.argv[1], 'a') as lock:\n"
        " fcntl.flock(lock, fcntl.LOCK_EX)\n"
        " print('locked', flush=True)\n"
        " sys.stdin.readline()\n"
        " time.sleep(.1)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", script, str(path)], text=True,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert select.select([process.stdout], [], [], 10)[0], "Lock holder failed to start"
        assert process.stdout.readline() == "locked\n"
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=5)


def test_gpu_lock_wait_times_out_without_leaking_alarm(gpu_lock_path, isolated_alarm):
    handler, received = isolated_alarm
    with _other_process_holds_lock(gpu_lock_path):
        started = time.monotonic()
        with (
            pytest.raises(TimeoutError, match="deadline.*GPU lock"),
            worker.timed_gpu_lock("test", worker.Deadline(.15)),
        ):
            pytest.fail("Another process still holds the GPU lock")
        assert time.monotonic() - started < 2
        assert signal.getsignal(signal.SIGALRM) is handler
        assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
        assert received == []
    # A timed-out waiter leaves neither an open held lock nor a stale alarm.
    with worker.timed_gpu_lock("test", worker.Deadline(1)):
        assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_gpu_lock_joins_waiters_and_restores_previous_alarm(gpu_lock_path, isolated_alarm):
    handler, received = isolated_alarm
    with _other_process_holds_lock(gpu_lock_path) as holder:
        holder.stdin.write("release\n")
        holder.stdin.flush()
        signal.setitimer(signal.ITIMER_REAL, 5, 7)
        # The old 15-second polling path would miss this release and time out.
        with worker.timed_gpu_lock("test", worker.Deadline(.5)):
            assert signal.getsignal(signal.SIGALRM) is handler
            remaining, interval = signal.getitimer(signal.ITIMER_REAL)
            assert 0 < remaining < 4.95
            assert interval == 7
            with gpu_lock_path.open("a") as other, pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # The readiness wait timer must be cancelled before yielding.
            time.sleep(.55)
        with gpu_lock_path.open("a") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert signal.getsignal(signal.SIGALRM) is handler
        assert signal.getitimer(signal.ITIMER_REAL)[1] == 7
        assert received == []


def test_gpu_lock_body_exception_releases_lock_and_cancels_timer(gpu_lock_path, isolated_alarm):
    handler, received = isolated_alarm
    with (
        pytest.raises(RuntimeError, match="body failed"),
        worker.timed_gpu_lock("test", worker.Deadline(.1)),
    ):
        raise RuntimeError("body failed")
    with gpu_lock_path.open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    time.sleep(.15)
    assert signal.getsignal(signal.SIGALRM) is handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert received == []
