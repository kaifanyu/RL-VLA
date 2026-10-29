"""Verify real PPO smoke runs and GPU parity before launching the six-run queue.

Run inside the existing Slurm allocation. This worker does not create another
allocation, and its Noise smoke shares the experiment GPU lock. A failed or
missing check never produces a true readiness gate.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GATES = ("installation_complete", "parity_passed", "flow_sde_smoke_passed", "flow_noise_smoke_passed")
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "RESIZING"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def file_record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class Deadline:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def remaining(self):
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("PPO readiness exceeded its configured deadline")
        return remaining

    def sleep(self):
        time.sleep(min(15, self.remaining()))


def step_state(step):
    result = subprocess.run(["sacct", "-j", step, "--format=JobIDRaw,State,ExitCode",
                             "--parsable2", "--noheader"], capture_output=True, text=True,
                            check=False, timeout=20)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[0] == step:
            return {"step": step, "state": fields[1].split()[0].rstrip("+"), "exit_code": fields[2]}
    return None


def assert_step_not_failed(record):
    if record is None or record["state"] in ACTIVE_STATES:
        return False
    if record["state"] != "COMPLETED" or record["exit_code"] != "0:0":
        raise RuntimeError(f"Prerequisite Slurm step failed: {record}")
    return True


def fatal_log_error(path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 64 * 1024))
        text = stream.read().decode(errors="replace")
    patterns = (r"^Error executing job with overrides:.*$",
                r"^srun: error:.*(?:Exited with exit code|Killed|Out Of Memory).*$")
    return next((match.group(0) for pattern in patterns
                 if (match := re.search(pattern, text, re.MULTILINE))), None)


def smoke_artifacts(directory, method):
    cfg = yaml.safe_load((directory / "config.yaml").read_text())
    if (cfg["runner"]["max_epochs"], cfg["runner"]["max_steps"],
        cfg["env"]["train"]["max_steps_per_rollout_epoch"],
        cfg["actor"]["model"]["openpi"]["noise_method"]) != (1, 1, 10, method):
        raise ValueError("Readiness requires the declared one-iteration/two-decision smoke")
    if Path(cfg["runner"]["logger"]["log_path"]).resolve() != directory.resolve():
        raise ValueError("Smoke logger directory differs from the monitored directory")
    summary = read_json(directory / "summary.json")
    if summary is None:
        return None
    if (summary.get("completed"), summary.get("iterations"), summary.get("decisions")) != (True, 1, 2):
        raise ValueError(f"Smoke did not finish its declared interaction budget: {summary}")
    checkpoint = (directory / cfg["runner"]["logger"]["experiment_name"] / "checkpoints"
                  / "global_step_1/actor/model_state_dict/full_weights.pt")
    metrics = directory / "iteration_metrics.jsonl"
    recording = directory / "trajectories/summary.json"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (checkpoint, metrics, recording)):
        return None
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    if len(rows) != 1 or rows[0].get("iteration") != 1:
        raise ValueError("Smoke requires its actual iteration-one diagnostics")
    recorded = read_json(recording)
    if recorded is None or recorded.get("chunks") != 2:
        raise ValueError("The smoke's full trajectory recording was not finalized")
    return {"checkpoint": checkpoint, "metrics": metrics, "summary": summary}


def validate_models(report):
    for method in ("flow_sde", "flow_noise"):
        counts = report.get(method, {}).get("parameter_counts", {})
        if counts.get("action_expert") != 430_098_464 or counts.get("frozen_vlm") != 2_923_335_408:
            raise ValueError(f"Deep CPU model-construction check missing or wrong for {method}")
        if counts.get("noise_head") != (141_536 if method == "flow_noise" else 0):
            raise ValueError(f"Wrong official noise-head configuration for {method}")
    return True


def verify_converted_checkpoint(directory, deadline):
    manifest_path = directory / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    weights = directory / "model.safetensors"
    if weights.stat().st_size != manifest["weights_bytes"]:
        raise ValueError("Converted checkpoint size differs from its conversion manifest")
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            deadline.remaining()
            digest.update(block)
    if digest.hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Converted checkpoint SHA256 differs from its conversion manifest")
    normalization = file_record(directory / "physical-intelligence/libero/norm_stats.json")
    if normalization["sha256"] != manifest["rlinf_normalization_sha256"]:
        raise ValueError("Converted normalization differs from its conversion manifest")
    return {"manifest": file_record(manifest_path), "weights": {"path": str(weights),
            "sha256": digest.hexdigest(), "bytes": weights.stat().st_size},
            "normalization": normalization}


def verify_gpu_parity_provenance(directory, native_log, converted):
    def resolved(path):
        path = Path(path)
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    native, torch_output, fixture = (directory / name for name in ("native.npz", "torch.npz", "fixture.npz"))
    torch_metadata = directory / "torch.json"
    metadata = json.loads(torch_metadata.read_text())
    if metadata.get("backend") != "torch" or not str(metadata.get("actual_device", "")).startswith("cuda"):
        raise ValueError("Authoritative Torch parity must have executed on the GPU")
    for name, path in (("output_artifact", torch_output), ("fixture", fixture)):
        record = metadata.get(name, {})
        if resolved(record.get("path", "")) != path.resolve() or record.get("sha256") != file_record(path)["sha256"]:
            raise ValueError(f"Torch GPU parity {name} identity does not match the actual compared files")
    checkpoint = metadata.get("checkpoint", {})
    manifest_path = converted / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (resolved(checkpoint.get("path", "")) != converted.resolve()
            or resolved(checkpoint.get("conversion_manifest_path", "")) != manifest_path.resolve()
            or checkpoint.get("conversion_manifest_sha256") != file_record(manifest_path)["sha256"]
            or checkpoint.get("model_sha256") != manifest["weights_sha256"]):
        raise ValueError("Torch GPU parity did not use the declared converted checkpoint")
    native_runs = []
    for line in native_log.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (record.get("backend") == "jax" and resolved(record.get("output", "")) == native.resolve()
                and any(str(device).lower().startswith(("cuda", "gpu")) for device in record.get("devices", []))):
            native_runs.append(record)
    if not native_runs:
        raise ValueError("Native reference has no matching GPU execution evidence")
    return {"torch_metadata": file_record(torch_metadata), "torch_output": file_record(torch_output),
            "native_output": file_record(native), "native_gpu_log": file_record(native_log),
            "fixture": file_record(fixture), "actual_torch_device": metadata["actual_device"]}


def readiness_document(installation, parity, smokes, evidence):
    passed = {
        "installation_complete": installation is True,
        "parity_passed": parity.get("parity_passed") is True,
        "flow_sde_smoke_passed": smokes.get("flow_sde", {}).get("smoke_passed") is True,
        "flow_noise_smoke_passed": smokes.get("flow_noise", {}).get("smoke_passed") is True,
    }
    missing = [name for name, value in passed.items() if not value]
    if missing:
        raise RuntimeError(f"Cannot publish readiness with unpassed checks: {missing}")
    return passed | {"evidence": evidence, "verified_at": datetime.now(UTC).isoformat()}


def stop_process_tree(process):
    # Capture Ray descendants before their launcher can exit/reparent them.
    listing = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True)
    parents = {int(pid): int(parent) for pid, parent in (line.split() for line in listing.splitlines())}
    descendants = {process.pid}
    while True:
        expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        for pid in descendants:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(2)
    process.wait(timeout=20)


def run_logged(command, log, deadline, *, cpu=False):
    environment = os.environ.copy()
    if cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["JAX_PLATFORMS"] = "cpu"
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=deadline.remaining())
        except BaseException:
            stop_process_tree(process)
            raise
    if code:
        stop_process_tree(process)
        raise RuntimeError(f"Readiness command exited {code}; see {log}")


@contextmanager
def timed_gpu_lock(allocation, deadline):
    path = ROOT / "runs" / f"gpu_allocation_{allocation}" / "experiment.lock"
    with path.open("a") as lock:
        remaining = deadline.remaining()
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        started = time.monotonic()

        def timed_out(signum, frame):
            raise TimeoutError("PPO readiness exceeded its configured deadline waiting for the GPU lock")

        try:
            # Blocking flock joins the kernel waiters instead of racing the
            # active queues every 15 seconds. This caller runs on the main thread.
            signal.signal(signal.SIGALRM, timed_out)
            try:
                signal.setitimer(signal.ITIMER_REAL, remaining)
                fcntl.flock(lock, fcntl.LOCK_EX)
                deadline.remaining()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_timer[0]:
                    # Preserve a pre-existing alarm's deadline and repeat period;
                    # an alarm that elapsed during the wait is delivered promptly.
                    delay = max(previous_timer[0] - (time.monotonic() - started), 1e-6)
                    signal.setitimer(signal.ITIMER_REAL, delay, previous_timer[1])
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def verify_smoke(directory, method, artifacts, work, deadline):
    report = work / f"{method}_verification.json"
    initial = ROOT / "checkpoints/openpi/pi05_libero_rlinf/model.safetensors"
    command = ["bash", "scripts/run_pi_rl.sh", "integrations/pi_rl/verify_smoke.py",
               "--initial", str(initial), "--checkpoint", str(artifacts["checkpoint"]),
               "--metrics", str(artifacts["metrics"]), "--output", str(report)]
    run_logged(command, work / f"{method}_verification.log", deadline, cpu=True)
    result = read_json(report)
    if result is None or result.get("smoke_passed") is not True:
        raise RuntimeError(f"Actual {method} checkpoint verification failed")
    if Path(result["checkpoint"]).resolve() != artifacts["checkpoint"].resolve():
        raise ValueError("Smoke report does not identify the actual saved actor checkpoint")
    return result


def launch_suite(suite, gates, allocation, work):
    if (suite / "pi_rl_queue").exists() or (suite / "pi_rl_launch.json").exists():
        raise FileExistsError("PPO experiment queue already exists; refusing a duplicate launch")
    command = ["bash", str(ROOT / "scripts/run_pi_rl_suite_grasp.sh"), str(suite), str(gates)]
    with (work / "suite_launcher.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": process.pid, "command": command, "gates": str(gates),
              "status": "running", "allocation": allocation, "step_id": os.environ.get("SLURM_STEP_ID"),
              "launched": datetime.now(UTC).isoformat()}
    write_json(suite / "pi_rl_launch.json", record)
    # Keep this Slurm step alive for the entire suite. Its cgroup would terminate
    # a detached descendant if readiness exited immediately after spawning it.
    # The eight-hour deadline applies only to readiness, not the full experiments.
    try:
        code = process.wait()
    except BaseException:
        stop_process_tree(process)
        raise
    try:
        if code:
            raise RuntimeError(f"The six-experiment queue exited {code}; see {work / 'suite_launcher.log'}")
        validate_queue_completion(suite)
    except BaseException as error:
        record.update(status="failed", exit_code=code, error=repr(error),
                      finished=datetime.now(UTC).isoformat())
        write_json(suite / "pi_rl_launch.json", record)
        raise
    record.update(status="completed", exit_code=code, finished=datetime.now(UTC).isoformat())
    write_json(suite / "pi_rl_launch.json", record)
    return record


def validate_queue_completion(suite):
    progress = read_json(suite / "pi_rl_queue/progress.json")
    expected = {(task, method) for task in ("spatial_1", "spatial_2", "object_0")
                for method in ("flow_sde", "flow_noise")}
    if progress is None or progress.get("failures") != [] or progress.get("remaining") != 0:
        raise RuntimeError("PPO queue exited without six successful experiments")
    completed = progress.get("completed", [])
    actual = {(row.get("task"), row.get("method")) for row in completed}
    if len(completed) != 6 or actual != expected or any(row.get("exit_code") != 0 for row in completed):
        raise RuntimeError("PPO queue completion does not cover all six unique experiments")
    status = read_json(suite / "pi_rl_queue/status.json")
    if status is None or status.get("state") != "completed":
        raise RuntimeError("PPO queue has not published successful completion")


def run(args):
    allocation = os.environ.get("SLURM_JOB_ID")
    if allocation != str(args.allocation):
        raise RuntimeError("Run readiness inside the already assigned Slurm allocation")
    suite = args.suite.resolve()
    work, gates = suite / "pi_rl_readiness", suite / "pi_rl_readiness.json"
    work.mkdir(exist_ok=False)
    if gates.exists():
        raise FileExistsError("Readiness gates already exist; preserve prior evidence")
    deadline = Deadline(args.timeout)
    published = False

    def status(state, **details):
        write_json(work / "status.json", {"state": state, "allocation": args.allocation,
                   "step_id": os.environ.get("SLURM_STEP_ID"), "host": socket.gethostname(),
                   "updated": datetime.now(UTC).isoformat(), **details})

    try:
        write_json(work / "plan.json", {"sde_step": args.sde_step, "parity_step": args.parity_step,
                   "timeout_seconds": args.timeout, "model_report": str(args.model_report),
                   "parity_directory": str(args.parity_directory), "suite": str(suite)})
        sde = suite / "smoke/flow_sde"
        while True:
            record = step_state(args.sde_step)
            status("waiting_for_sde_smoke", prerequisite=record)
            complete = assert_step_not_failed(record)
            error = fatal_log_error(sde / "launch.log")
            if error:
                raise RuntimeError(error)
            artifacts = smoke_artifacts(sde, "flow_sde")
            if complete:
                if artifacts is None:
                    raise RuntimeError("SDE Slurm step completed without all actual smoke artifacts")
                break
            deadline.sleep()
        status("verifying_sde_checkpoint")
        smokes = {"flow_sde": verify_smoke(sde, "flow_sde", artifacts, work, deadline)}

        noise = suite / "smoke/flow_noise"
        if not (noise / "config.yaml").exists():
            sys.path.insert(0, str(ROOT / "integrations"))
            from rlinf_bridge.configure import configuration

            noise.mkdir(parents=True, exist_ok=True)
            config = configuration(suite / "tasks/spatial_1", noise, "flow_noise", 1, 2)
            (noise / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        artifacts = smoke_artifacts(noise, "flow_noise")
        if artifacts is None:
            if (noise / "launch.json").exists():
                raise RuntimeError("An unfinished Noise smoke already exists; refusing a duplicate")
            command = ["bash", "scripts/run_pi_rl.sh", "integrations/rlinf_bridge/train.py",
                       "--config-path", str(noise), "--config-name", "config"]
            status("waiting_for_noise_gpu")
            with timed_gpu_lock(args.allocation, deadline):
                status("running_noise_smoke")
                write_json(noise / "launch.json", {"command": command, "readiness_worker": str(work),
                           "allocation": args.allocation, "step_id": os.environ.get("SLURM_STEP_ID")})
                run_logged(command, noise / "launch.log", deadline)
            artifacts = smoke_artifacts(noise, "flow_noise")
            if artifacts is None:
                raise RuntimeError("Noise training exited without all actual smoke artifacts")
        status("verifying_noise_checkpoint")
        smokes["flow_noise"] = verify_smoke(noise, "flow_noise", artifacts, work, deadline)

        parity_dir = args.parity_directory.resolve()
        while True:
            record = step_state(args.parity_step)
            status("waiting_for_gpu_parity", prerequisite=record)
            if assert_step_not_failed(record):
                break
            error = fatal_log_error(ROOT / "runs/setup/pirl_parity_torch.log")
            if error:
                raise RuntimeError(error)
            deadline.sleep()
        parity = parity_dir / "comparison.json"
        temporary = parity.with_suffix(".readiness.json")
        parity_provenance = verify_gpu_parity_provenance(
            parity_dir, args.native_log, ROOT / "checkpoints/openpi/pi05_libero_rlinf")
        command = [str(ROOT / ".venv/bin/python"), "integrations/pi_rl/check_parity.py", "compare",
                   "--native", str(parity_dir / "native.npz"), "--torch", str(parity_dir / "torch.npz"),
                   "--output", str(temporary)]
        status("verifying_gpu_parity")
        run_logged(command, work / "parity_comparison.log", deadline, cpu=True)
        parity_result = read_json(temporary)
        if parity_result is None or parity_result.get("parity_passed") is not True:
            raise RuntimeError("GPU inference parity did not pass")
        temporary.replace(parity)

        status("checking_installation")
        while (models := read_json(args.model_report)) is None:
            deadline.sleep()
        validate_models(models)
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("The existing uv executable is unavailable for dependency verification")
        run_logged([uv, "pip", "check", "--python", str(ROOT / ".venv-rlinf/bin/python")],
                   work / "dependency_check.log", deadline, cpu=True)
        run_logged([uv, "pip", "freeze", "--python", str(ROOT / ".venv-rlinf/bin/python")],
                   work / "rlinf_freeze.txt", deadline, cpu=True)
        run_logged([uv, "pip", "freeze", "--python", str(ROOT / ".venv-pro/bin/python")],
                   work / "simulator_freeze.txt", deadline, cpu=True)
        imports = ("import json,torch,openpi,rlinf,transformers,pyarrow; "
                   "assert torch.version.cuda == '12.8'; "
                   "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
                   "'transformers':transformers.__version__,'pyarrow':pyarrow.__version__}))")
        run_logged(["bash", "scripts/run_pi_rl.sh", "-c", imports],
                   work / "import_check.log", deadline, cpu=True)
        status("verifying_converted_checkpoint_identity")
        converted = verify_converted_checkpoint(ROOT / "checkpoints/openpi/pi05_libero_rlinf", deadline)
        evidence = {"cpu_models": file_record(args.model_report), "gpu_parity": file_record(parity),
                    "gpu_parity_provenance": parity_provenance,
                    "dependency_check": file_record(work / "dependency_check.log"),
                    "environment_freeze": file_record(work / "rlinf_freeze.txt"),
                    "simulator_environment_freeze": file_record(work / "simulator_freeze.txt"),
                    "import_check": file_record(work / "import_check.log"),
                    "converted_checkpoint": converted,
                    "smokes": {method: file_record(work / f"{method}_verification.json") for method in smokes}}
        document = readiness_document(True, parity_result, smokes, evidence)
        write_json(gates, document)
        published = True
        status("running_six_experiment_queue", gates=str(gates))
        launch = launch_suite(suite, gates, args.allocation, work)
        status("complete", gates=str(gates), suite_launch=launch)
    except BaseException as error:
        if not published:
            write_json(gates, {name: False for name in GATES} | {"error": repr(error)})
        status("failed", error=repr(error), readiness_published=published)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--allocation", type=int, default=1470624)
    parser.add_argument("--sde-step", default="1470624.32")
    parser.add_argument("--parity-step", default="1470624.36")
    parser.add_argument("--parity-directory", type=Path, default=ROOT / "runs/setup/pirl_parity")
    parser.add_argument("--model-report", type=Path, default=ROOT / "runs/setup/pirl_model_cpu.json")
    parser.add_argument("--native-log", type=Path, default=ROOT / "runs/setup/pirl_parity_native.log")
    parser.add_argument("--timeout", type=int, default=8 * 3600)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
