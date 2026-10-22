"""Persistent, resumable, single-simulator robustness evaluation queue."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from collections import defaultdict, deque
from importlib.metadata import distributions
from pathlib import Path

from rl_vla.baseline import atomic_json, wilson_interval
from rl_vla.robustness import ROOT, sha256


def schedule(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[(case["benchmark"], case["category"])].append(case)
    pending = {}
    for key, members in groups.items():
        # Cover every task before repeating it on its next initial state.
        pending[key] = deque((case, index) for index in range(max(c["episodes"] for c in members))
                             for case in members if index < case["episodes"])
    plus = sorted(k for k in pending if k[0] == "plus")
    pro = sorted(k for k in pending if k[0] == "pro")
    order = []
    for index in range(max(len(plus), len(pro))):
        if index < len(plus):
            order.append(plus[index])
        if index < len(pro):
            order.append(pro[index])
    while any(pending.values()):
        for key in order:
            if pending[key]:
                yield pending[key].popleft()


def episode_id(case, index):
    return f"{case['case_id']}_init{index:03d}"


def summary(rows, cases, status):
    planned = defaultdict(int)
    grouped = defaultdict(list)
    for case in cases:
        for key in (case["benchmark"], f"{case['benchmark']}/{case['category']}",
                    f"{case['benchmark']}/{case['category']}/{case['suite']}"):
            planned[key] += case["episodes"]
    for row in rows:
        for key in (row["benchmark"], f"{row['benchmark']}/{row['category']}",
                    f"{row['benchmark']}/{row['category']}/{row['suite']}"):
            grouped[key].append(row)
    metrics = {}
    for key, total in planned.items():
        group = grouped[key]
        wins = sum(r["success"] for r in group)
        metrics[key] = {"episodes": len(group), "planned_episodes": total, "successes": wins,
                        "success_rate": wins / len(group) if group else None,
                        "success_95pct_wilson": wilson_interval(wins, len(group)) if group else None,
                        "status": "complete" if len(group) == total else "partial"}
    return {"status": status, "updated_unix_time": time.time(), "episodes": len(rows),
            "planned_episodes": sum(c["episodes"] for c in cases), "metrics": metrics,
            "note": "Partial results are progress indicators. PRO Environment uses locally generated "
            "states and includes five unchanged Long controls; see protocol.json. "
            "A pooled Plus+PRO success rate is deliberately not reported."}


def storage_check(output):
    # Apparent bytes is conservative for reflinked / hardlinked package caches.
    used = int(subprocess.check_output(["du", "-sb", "/workspace"], text=True,
                                      timeout=120).split()[0])
    disk = os.statvfs("/workspace")
    free = disk.f_bavail * disk.f_frsize
    atomic_json(output / "storage.json", {"unix_time": time.time(), "workspace_bytes": used,
                "free_bytes": free, "stop_workspace_bytes": 190_000_000_000})
    if used >= 190_000_000_000 or free < 10_000_000_000:
        raise RuntimeError("Storage guard: keep workspace below 190 GB and filesystem free above 10 GB")


def terminate(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def run(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT.parent / ".cache/rl-vla-robustness.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((output / "protocol.json").read_text())
    cases = plan["cases"]
    verified_files = {}
    for case in cases:
        for path, expected_hash in [(case["real_bddl"], case["bddl_sha256"]),
                                    (case["states"], case.get("states_sha256"))]:
            if path is None:
                continue
            if path not in verified_files:
                verified_files[path] = sha256(path)
            if verified_files[path] != expected_hash:
                raise ValueError(f"Benchmark file changed after plan creation: {path}")
    code_files = ["src/rl_vla/robustness.py", "src/rl_vla/envs/libero.py",
                  "src/rl_vla/env_adapter.py", "src/rl_vla/base_policy.py",
                  "scripts/serve_baseline.py", "scripts/robustness_queue.py",
                  "src/rl_vla/baseline.py", "scripts/env.sh", "configs/libero_sac.toml"]
    contract = {"protocol_sha256": sha256(output / "protocol.json"),
                "code_sha256": {p: sha256(ROOT / p) for p in code_files},
                "libero_configs": {name: sha256(ROOT.parent / ".cache" /
                    ("rl-vla-libero-" + name) / "config.yaml") for name in ("plus", "pro")}}
    for benchmark in ("plus", "pro"):
        source = ROOT / "third_party" / ("libero_" + benchmark)
        expected_commit = next(c["source_commit"] for c in cases if c["benchmark"] == benchmark)
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if commit != expected_commit or subprocess.check_output(
                ["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True).strip():
            raise ValueError(f"Upstream {benchmark} source differs from pinned clean commit")
    contract_path = output / "queue_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Refusing to mix results from different code or protocols")
    atomic_json(contract_path, contract)
    atomic_json(output / "environment.json", {
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
        "environment": {k: os.environ.get(k) for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM",
        "NUMBA_DISABLE_JIT", "LP_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE")}})
    storage_check(output)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 8001))
    free = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    if free < 32768:
        raise RuntimeError("Need 32 GiB free GPU memory before starting")
    rows = {}
    for path in (output / "episodes").glob("*.json"):
        row = json.loads(path.read_text())
        rows[row["episode_id"]] = row
    expected = {episode_id(c, i) for c in cases for i in range(c["episodes"])}
    if not rows.keys() <= expected:
        raise ValueError("Output contains episodes outside the frozen plan")
    (output / "launcher.pid").write_text(str(os.getpid()) + "\n")
    server = monitor = worker = None
    status, code = "running", 1
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def pids():
        values = [os.getpid()] + [p.pid for p in (server, worker) if p and p.poll() is None]
        path = output / "pids.txt"
        temporary = path.with_suffix(".tmp")
        temporary.write_text("\n".join(map(str, values)) + "\n")
        temporary.replace(path)
    def update(active=None):
        atomic_json(output / "summary.json", summary(list(rows.values()), cases, status))
        atomic_json(output / "queue.json", {"status": status, "active": active,
                    "completed_episodes": len(rows), "planned_episodes": len(expected),
                    "updated_unix_time": time.time(), "pid": os.getpid()})
    try:
        update()
        with (output / "server.log").open("a") as log:
            server = subprocess.Popen([str(ROOT / "third_party/openpi/.venv/bin/python"),
                str(ROOT / "scripts/serve_baseline.py"), "--checkpoint",
                str(ROOT / "checkpoints/openpi/pi05_libero"), "--model-config", "pi05_libero",
                "--port", "8001", "--manifest", str(output / "server_manifest.json")],
                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        pids()
        with (output / "monitor.log").open("a") as log:
            monitor = subprocess.Popen([str(ROOT / ".venv-robustness/bin/python"),
                str(ROOT / "scripts/monitor_baseline.py"), "--pids", str(output / "pids.txt"),
                "--output", str(output / "telemetry.jsonl")], stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1200
        while True:
            if server.poll() is not None or monitor.poll() is not None:
                raise RuntimeError("Server or safety monitor exited during startup; see logs")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8001/healthz", timeout=2):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Policy server startup timed out")
                time.sleep(5)
        manifest = json.loads((output / "server_manifest.json").read_text())
        fingerprint = output / "checkpoint_fingerprint.json"
        checkpoint = {"checkpoint_sha256": manifest["checkpoint_sha256"]}
        if fingerprint.exists() and json.loads(fingerprint.read_text()) != checkpoint:
            raise ValueError("Checkpoint differs from previous queue execution")
        atomic_json(fingerprint, checkpoint)
        last_storage_check = time.monotonic()
        videos = set()
        for case, index in schedule(cases):
            key = episode_id(case, index)
            if key in rows:
                continue
            if time.monotonic() - last_storage_check > 3600:
                storage_check(output)
                last_storage_check = time.monotonic()
            atomic_json(output / "active_case.json", case)
            active = {"episode_id": key, "benchmark": case["benchmark"], "suite": case["suite"],
                      "category": case["category"], "initial_state_id": index,
                      "started_unix_time": time.time()}
            update(active)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(ROOT / "third_party" / ("libero_" + case["benchmark"]))
            env["LIBERO_CONFIG_PATH"] = str(ROOT.parent / ".cache" / ("rl-vla-libero-" + case["benchmark"]))
            command = [str(ROOT / ".venv-robustness/bin/python"), "-m", "rl_vla.robustness",
                       "--case", str(output / "active_case.json"), "--output", str(output),
                       "--start-index", str(index), "--limit", str(index + 1)]
            video_key = (case["benchmark"], case["category"], case["suite"])
            if video_key not in videos and index == 0:
                command.append("--video")
                videos.add(video_key)
            # One log per category avoids tens of thousands of files; stdout includes episode IDs.
            log_path = output / "logs" / f"{case['benchmark']}_{case['category']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"START {key} ({case['category']}); {len(rows)}/{len(expected)} complete", flush=True)
            with log_path.open("a") as log:
                log.write(f"\nSTART {key}\n")
                log.flush()
                worker = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            pids()
            while worker.poll() is None:
                if server.poll() is not None or monitor.poll() is not None:
                    raise RuntimeError("Server or safety monitor exited; see logs")
                heartbeat = output / "heartbeat.json"
                recent = max(active["started_unix_time"], heartbeat.stat().st_mtime if heartbeat.exists() else 0)
                if time.time() - recent > 1800:
                    raise TimeoutError(f"Worker made no progress for 30 minutes: {key}")
                time.sleep(5)
            if worker.returncode:
                raise RuntimeError(f"Worker exited {worker.returncode}: {key}; see {log_path}")
            row = json.loads((output / "episodes" / (key + ".json")).read_text())
            if row["episode_id"] != key or row["case_id"] != case["case_id"]:
                raise ValueError("Worker returned an unexpected episode")
            rows[key] = row
            pids()
            update()
            print(f"DONE {key} success={row['success']} steps={row['primitive_steps']}", flush=True)
            time.sleep(2)
        status, code = "complete", 0
    except KeyboardInterrupt as exc:
        status, code = "interrupted", 130
        atomic_json(output / "interruption.json", {"message": str(exc), "unix_time": time.time()})
    except Exception as exc:
        status = "error"
        atomic_json(output / "error.json", {"error": repr(exc), "unix_time": time.time()})
        raise
    finally:
        for process in (worker, server, monitor):
            terminate(process)
        update()
        (output / "exit_code").write_text(str(code) + "\n")
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
