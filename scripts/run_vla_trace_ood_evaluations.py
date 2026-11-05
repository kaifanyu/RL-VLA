"""Durable post-selection OOD queue, after main evaluations and interventions."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import tarfile
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from evaluate_vla_trace_sharded import parse_cpuset, partition

from rl_vla.diagnostics.metrics import paired_success, wilson
from rl_vla.diagnostics.ood_evaluation import (
    POLICIES,
    SHARD_FACTORY,
    SOURCE_FACTORY,
    VARIANTS,
    merge_shards,
    readiness,
)

ROOT = Path(__file__).resolve().parents[1]
THIRD_LANES = {
    "object_0": ("5", "rollout_trace_queue_complete.json"),
    "object_2": ("3", "attention_grid_queue_complete.json"),
}


def now():
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify(contract):
    for name, expected in contract.items():
        if digest(name) != expected:
            raise ValueError(f"Frozen OOD evaluation input changed: {name}")


def status(output, state, **extra):
    write_json(output / "queue_status.json", {"state": state, "updated": now(), **extra})


@contextmanager
def gpu_lock(gpu, stop=None):
    identity = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not identity.startswith("GPU-") or "\n" in identity:
        raise ValueError("Expected one physical GPU")
    directory = ROOT / "runs/.vla_trace_gpu_locks"
    directory.mkdir(exist_ok=True)
    with (directory / f"{identity}.lock").open("a+") as stream:
        while True:
            if stop is not None and stop.is_set():
                raise RuntimeError("Other OOD worker failed while waiting for GPU lock")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(1)
        yield identity


def make_shard_config(text, start, count):
    original = tomllib.loads(text)
    if original["env"]["factory"] != SOURCE_FACTORY or set(original["env"]["kwargs"]) != {"protocol"}:
        raise ValueError("Expected the original immutable OOD factory configuration")
    if original["policy"]["kind"] != "pi05" or original["policy"]["model_config"] != "pi05_libero":
        raise ValueError("OOD actions require the compatible native LIBERO policy")
    if original["rldt"]["ode_steps"] != 10 or original["rldt"]["execute_steps"] != 5 or not original["recording"]["enabled"]:
        raise ValueError("OOD sampler, action execution, or trajectory-recording contract differs")
    expression = r'(?m)^factory[ \t]*=[ \t]*[\"\']' + re.escape(SOURCE_FACTORY) + r'[\"\'][ \t]*$'
    result, changed = re.subn(expression, f'factory = "{SHARD_FACTORY}"', text)
    if changed != 1 or result.count("[env.kwargs]") != 1:
        raise ValueError("Expected one conventional OOD TOML environment declaration")
    result = result.replace("[env.kwargs]", f"[env.kwargs]\nshard_start = {start}\nshard_count = {count}\n", 1)
    return result


def freeze(campaign, task, output, gates):
    method = campaign / "methods/rldt" / task
    inherited = json.loads((method / "input_contract.json").read_text())
    verify(inherited)
    source_paths = [Path(__file__), ROOT / "scripts/evaluate_vla_trace_sharded.py",
                    ROOT / "src/rl_vla/diagnostics/ood_evaluation.py",
                    ROOT / "src/rl_vla/diagnostics/observation_bank.py",
                    ROOT / "src/rl_vla/diagnostics/metrics.py"]
    source_paths += [Path(name) for name in inherited if Path(name).suffix in {".py", ".sh"}]
    source_paths = sorted({path.resolve() for path in source_paths})
    source_contract = {str(path): digest(path) for path in source_paths}
    inputs = {str(Path(path).resolve()): digest(path) for path in gates["gate_paths"].values()}
    inputs[str((method / "input_contract.json").resolve())] = digest(method / "input_contract.json")
    configs = {}
    for variant in VARIANTS:
        directory = campaign / "diagnostic_layouts" / task / variant
        path = directory / "evaluate.toml"
        config = tomllib.loads(path.read_text())
        protocol_path = Path(config["env"]["kwargs"]["protocol"])
        protocol = json.loads(protocol_path.read_text())
        if protocol["task"] != task or protocol["variant"] != variant or protocol["state_ids"] != list(range(50)) or protocol["max_steps"] != 280:
            raise ValueError("OOD protocol differs from the predetermined task, condition, IDs, or budget")
        inputs[str(path.resolve())] = digest(path)
        inputs[str(protocol_path.resolve())] = digest(protocol_path)
        for field in ("bddl", "states", "source_protocol", "generation", "predeclaration"):
            item = protocol[field]
            inputs[item["path"]] = item["sha256"]
        generation = json.loads(Path(protocol["generation"]["path"]).read_text())
        if len(generation["accepted"]) != 50:
            raise ValueError("OOD generation lacks fifty accepted physical states")
        configs[variant] = path
    checkpoints = {}
    checkpoint_audit = json.loads((method / "checkpoint_audit.json").read_text())
    for policy, iteration in POLICIES.items():
        checkpoint = method / "training" / f"iteration_{iteration:06d}" if iteration is not None else None
        checkpoints[policy] = str(checkpoint.resolve()) if checkpoint else None
        if checkpoint:
            for name in ("expert.npz", "rldt_pi05.json"):
                path = checkpoint / name
                expected = checkpoint_audit["checkpoints"][str(iteration)]["files"][name]
                if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
                    raise ValueError("OOD checkpoint differs from passed main checkpoint audit")
                inputs[str(path.resolve())] = expected["sha256"]
    verify(inputs)
    contract = {"sources": source_contract, "inputs": inputs,
                "inherited_training_contract": str(method / "input_contract.json"),
                "checkpoints": checkpoints, "frozen_at": now(), "gate_records": gates["gate_paths"]}
    contract["checkpoint_stats"] = {name: file_stat(name) for name in inputs
                                     if name.endswith(("/expert.npz", "/rldt_pi05.json"))}
    write_json(output / "input_manifest.json", contract)
    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in source_paths:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    return contract, configs, checkpoints


def file_stat(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def guard_checkpoints(contract):
    if any(file_stat(name) != expected for name, expected in contract.get("checkpoint_stats", {}).items()):
        raise ValueError("Frozen OOD checkpoint stat changed")


def stop_process_groups(processes, grace_seconds=10):
    """Terminate every registered owned group, including orphaned descendants."""
    def send(process, signum):
        try:
            os.killpg(process.pid, signum)
            return True
        except ProcessLookupError:
            return False
    for process in processes:
        send(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        for process in processes:
            process.poll()  # reap leaders before checking surviving process groups
        if not any(send(process, 0) for process in processes):
            break
        time.sleep(.1)
    for process in processes:
        send(process, signal.SIGKILL)
    for process in processes:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            send(process, signal.SIGKILL)


def evaluate_variant(config_path, destination, checkpoint, *, task, variant, cpus, gpu, contract, stop=None):
    destination.mkdir(parents=True, exist_ok=False)
    verify(contract["sources"])
    guard_checkpoints(contract)
    # Inputs include large expert files, verified once at freeze and once at final
    # queue completion. Small protocol/asset/source files are checked per stage.
    verify({name: value for name, value in contract["inputs"].items() if not name.endswith("/expert.npz")})
    text = config_path.read_text()
    config = tomllib.loads(text)
    specs, processes, handles = [], [], []
    started = time.monotonic()
    try:
        for index, ((start, count), (cpu_start, cpu_count)) in enumerate(zip(partition(50, 3), partition(len(cpus), 3), strict=True)):
            directory = destination / "shards" / f"shard_{index:02d}"
            directory.mkdir(parents=True)
            path = directory / "config.toml"
            path.write_text(make_shard_config(text, start, count))
            report_path = directory / "report.json"
            cpuset = ",".join(map(str, cpus[cpu_start:cpu_start + cpu_count]))
            command = ["bash", str(ROOT / "scripts/run_rldt.sh"), "evaluate", "--config", str(path),
                       "--output", str(report_path), "--episodes", str(count), "--seed", str(100000 + start),
                       "--record-episodes", str(count)]
            if checkpoint is not None:
                command += ["--checkpoint", checkpoint]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, RLDT_CPUSET=cpuset,
                       XLA_PYTHON_CLIENT_MEM_FRACTION="0.25", XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       LP_NUM_THREADS=str(min(4, cpu_count)), RL_VLA_OBJECT_TELEMETRY="1",
                       LIBERO_CONFIG_PATH=str(ROOT / ".cache/libero-pro"))
            handle = (directory / "native_evaluate.log").open("x")
            handles.append(handle)
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            specs.append({"index": index, "start": start, "count": count, "config": tomllib.loads(path.read_text()),
                          "command": command, "pid": process.pid, "cpuset": cpuset, "report": str(report_path)})
        write_json(destination / "shards.json", {"task": task, "variant": variant, "checkpoint": checkpoint,
                   "state_ids": list(range(50)), "seeds": list(range(100000, 100050)), "shards": specs})
        while True:
            if stop is not None and stop.is_set():
                raise RuntimeError("Other OOD worker failed; stopping owned native shards")
            guard_checkpoints(contract)
            codes = [process.poll() for process in processes]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f"Native OOD evaluator failed: {codes}")
            if all(code is not None for code in codes):
                break
            time.sleep(2)
        reports = [json.loads(Path(spec["report"]).read_text()) for spec in specs]
        result = merge_shards(reports, specs, config=config, checkpoint=checkpoint, variant=variant, task=task)
        result["wilson95"] = wilson(result["successes"], 50)
        result["wall_seconds"] = time.monotonic() - started
        result["source_config"] = str(config_path)
        result["source_config_sha256"] = digest(config_path)
        result["input_manifest"] = str(destination.parents[1] / "input_manifest.json")
        result["episodes_file"] = str(destination / "episodes.jsonl")
        verify(contract["sources"])
        guard_checkpoints(contract)
        verify({name: value for name, value in contract["inputs"].items() if not name.endswith("/expert.npz")})
        temporary = destination / "episodes.jsonl.tmp"
        with temporary.open("x") as stream:
            for episode in result["episodes"]:
                # Keep native videos and complete telemetry-bearing trajectory shards.
                if not Path(episode["video"]).exists() or not Path(episode["trajectory"]).exists():
                    raise ValueError("Native OOD episode artifacts are missing")
                stream.write(json.dumps(episode, allow_nan=False) + "\n")
        temporary.replace(destination / "episodes.jsonl")
        write_json(destination / "summary.json", result)
        return result
    except BaseException as exc:
        stop_process_groups(processes)
        write_json(destination / "failed.json", {"complete": False, "error": repr(exc), "updated": now()})
        raise
    finally:
        for handle in handles:
            handle.close()


def cell_assignments(workers):
    if workers not in (1, 2, 3):
        raise ValueError("OOD supports one, two, or three independent physical GPUs per task")
    cells = [{"policy": policy, "variant": variant} for policy in POLICIES for variant in VARIANTS]
    assignments = [cells[index::workers] for index in range(workers)]
    if sum(len(part) for part in assignments) != 16 or len({(c["policy"], c["variant"]) for part in assignments for c in part}) != 16:
        raise ValueError("OOD static partition must cover all16cells exactly once")
    return assignments


def process_start_ticks(pid):
    """Linux process identity; the command field may contain spaces or parentheses."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[19])


def released_diagnostic_gate(path):
    """Require a completed, reaped child; a reused PID is a different process."""
    path = Path(path)
    if not path.exists():
        return False
    marker = json.loads(path.read_text())
    if marker.get("complete") is not True:
        return False
    child = marker.get("child_exit", {})
    if (type(child.get("pid")) is not int or child["pid"] <= 0
            or type(child.get("returncode")) is not int or child["returncode"] != 0
            or type(child.get("proc_start_ticks")) is not int or child["proc_start_ticks"] <= 0):
        raise ValueError("Diagnostic GPU release marker lacks a successful child process identity")
    return process_start_ticks(child["pid"]) != child["proc_start_ticks"]


def queue_resources(args, campaign):
    """Preserve existing one/two-lane plans; a third lane has an explicit release gate."""
    cpus = parse_cpuset(args.cpuset)
    if len(cpus) < 3:
        raise ValueError("Allocate at least three CPU IDs for three native evaluator shards")
    resources = [{"gpu": args.gpu, "cpus": cpus, "cpuset": args.cpuset}]
    if args.extra_gpu:
        extra_cpus = parse_cpuset(args.extra_cpuset)
        if args.extra_gpu == args.gpu or len(extra_cpus) < 3 or set(cpus).intersection(extra_cpus):
            raise ValueError("Paired GPUs need distinct physical IDs and disjoint CPU pools of at least3")
        resources.append({"gpu": args.extra_gpu, "cpus": extra_cpus, "cpuset": args.extra_cpuset})
    third_gpu = getattr(args, "third_gpu", None)
    third_cpuset = getattr(args, "third_cpuset", None)
    if bool(third_gpu) != bool(third_cpuset):
        raise ValueError("Provide both third GPU and its CPU pool")
    if third_gpu:
        if not args.extra_gpu or args.task not in THIRD_LANES or third_gpu != THIRD_LANES[args.task][0]:
            raise ValueError("Third GPU requires the predefined task-specific released diagnostic lane")
        third_cpus = parse_cpuset(third_cpuset)
        allocated = {cpu for resource in resources for cpu in resource["cpus"]}
        if third_gpu in {resource["gpu"] for resource in resources} or len(third_cpus) < 3 or allocated.intersection(third_cpus):
            raise ValueError("Third GPU and its CPU pool must be distinct from both existing lanes")
        resources.append({"gpu": third_gpu, "cpus": third_cpus, "cpuset": third_cpuset,
                          "release_gate": str((campaign / THIRD_LANES[args.task][1]).resolve())})
    return resources


def queue_readiness(campaign, task, resources):
    gates = readiness(campaign, task)
    for resource in resources:
        path = resource.get("release_gate")
        if path:
            name = "third_gpu_diagnostic_child_exit"
            gates["gate_paths"][name] = path
            if not released_diagnostic_gate(path):
                gates["ready"] = False
                gates["waiting_for"].append(name)
    return gates


def summarize_task(output, task, checkpoints):
    """Publish policy summaries only after verifying the complete sixteen-cell set."""
    results = {}
    for policy in POLICIES:
        for variant in VARIANTS:
            path = output/policy/variant/"summary.json"
            result = json.loads(path.read_text())
            if (result.get("complete") is not True or result.get("technical_only") is not False or
                    result.get("task") != task or result.get("variant") != variant or
                    result.get("checkpoint") != checkpoints[policy] or result.get("episode_count") != 50):
                raise ValueError("OOD cell missing or differs from complete frozen task/checkpoint contract")
            episodes = result["episodes"]
            if ([r["reset"]["initial_state_id"] for r in episodes] != list(range(50)) or
                    [r["seed"] for r in episodes] != list(range(100000, 100050)) or
                    any(type(r["success"]) is not bool for r in episodes)):
                raise ValueError("OOD cell outcomes or exact state/noise pairing differ")
            if result["successes"] != sum(r["success"] for r in episodes):
                raise ValueError("OOD cell outcome summary differs")
            if abs(result["success_rate"] - result["successes"]/50) > 1e-12 or list(result["wilson95"]) != list(wilson(result["successes"], 50)):
                raise ValueError("OOD cell success rate or Wilson interval differs from outcomes")
            results[policy, variant] = result
    events = []
    for policy, iteration in POLICIES.items():
        rows = {}
        for variant in VARIANTS:
            result = results[policy, variant]
            row = {"variant": variant, "successes": result["successes"], "episodes": 50,
                   "success_rate": result["success_rate"], "wilson95": result["wilson95"],
                   "summary": str(output/policy/variant/"summary.json")}
            if iteration is not None:
                row["paired_vs_pi05_libero"] = paired_success(results["pi05_libero", variant]["episodes"], result["episodes"])
            rows[variant] = row
            events.append({"updated": now(), "policy": policy, **row})
        write_json(output/policy/"summary.json", {"complete": True, "technical_only": False,
                   "task": task, "policy": policy, "checkpoint": checkpoints[policy],
                   "variants": rows, "episodes": 200, "input_manifest": str(output/"input_manifest.json"),
                   "scope": "Four separately reported custom novel-layout conditions; no pooled official-PRO score",
                   "failure_labels": "Raw object telemetry and videos retained; semantic classification requires review"})
    with (output/"events.jsonl").open("x") as stream:
        for row in events:
            stream.write(json.dumps(row) + "\n")


def run_worker(index, assignment, resource, campaign, task, output, configs, checkpoints, contract, stop):
    worker = output/"workers"/f"worker_{index:02d}"
    worker.mkdir(parents=True, exist_ok=False)
    completed = []
    try:
        status(worker, "waiting_for_gpu", gpu=resource["gpu"], cells=assignment)
        with gpu_lock(resource["gpu"], stop) as uuid:
            if not readiness(campaign, task)["ready"]:
                raise RuntimeError("Required queue gates changed while acquiring worker GPU")
            if resource.get("release_gate"):
                verify({resource["release_gate"]: contract["inputs"][resource["release_gate"]]})
                if not released_diagnostic_gate(resource["release_gate"]):
                    raise RuntimeError("Diagnostic process remains active after acquiring its released GPU")
            for cell in assignment:
                if stop.is_set():
                    raise RuntimeError("Other OOD worker failed")
                policy, variant = cell["policy"], cell["variant"]
                status(worker, "evaluating", gpu=uuid, cell=cell, completed=completed)
                evaluate_variant(configs[variant], output/policy/variant, checkpoints[policy],
                    task=task, variant=variant, cpus=resource["cpus"], gpu=resource["gpu"],
                    contract=contract, stop=stop)
                completed.append(cell)
        status(worker, "completed", cells=completed, gpu=resource["gpu"])
        return completed
    except BaseException as exc:
        stop.set()
        status(worker, "failed", complete=False, completed=completed, error=repr(exc))
        raise


def run(args):
    campaign = args.run.resolve()
    output = campaign / "ood" / args.task
    output.mkdir(parents=True, exist_ok=False)
    resources = queue_resources(args, campaign)
    assignments = cell_assignments(len(resources))
    plan = {"task": args.task, "policies": list(POLICIES), "variants": list(VARIANTS),
            "episodes_per_cell": 50, "primitive_step_limit": 280,
            "state_ids": list(range(50)), "noise_seeds": list(range(100000, 100050)),
            "selection_can_change": False, "gpu": args.gpu, "cpuset": args.cpuset,
            "resources": resources, "static_cell_assignments": assignments,
            "required_order": "Locked selection, all main action evaluations, all closed-loop interventions, then OOD",
            "created": now(), "pid": os.getpid(), "source": str(Path(__file__).resolve())}
    write_json(output / "queue_plan.json", plan)
    raw = output / "pi05_base"
    raw.mkdir()
    write_json(raw / "summary.json", {"complete": True, "applicable": False, "status": "not_applicable",
               "task": args.task, "policy": "pi05_base", "reason": "N/A — incompatible action contract",
               "episodes": 0, "note": "Raw base receives compatible representation diagnostics; no LIBERO action score is invented"})
    started = time.monotonic()
    try:
        while True:
            gates = queue_readiness(campaign, args.task, resources)
            if gates["ready"]:
                break
            status(output, "waiting_for_required_gates", **gates)
            time.sleep(args.poll_seconds)
        contract, configs, checkpoints = freeze(campaign, args.task, output, gates)
        stop = threading.Event()
        with ThreadPoolExecutor(max_workers=len(resources)) as executor:
            try:
                futures = [executor.submit(run_worker, index, assignment, resource, campaign, args.task,
                           output, configs, checkpoints, contract, stop)
                           for index, (assignment, resource) in enumerate(zip(assignments, resources, strict=True))]
                while not all(f.done() for f in futures):
                    failed = [f for f in futures if f.done() and f.exception() is not None]
                    if failed:
                        stop.set()
                        failed[0].result()
                    status(output, "evaluating", resources=resources,
                           worker_status=[str(output/"workers"/f"worker_{i:02d}"/"queue_status.json") for i in range(len(resources))])
                    time.sleep(2)
                for future in futures:
                    future.result()
            except BaseException:
                stop.set()
                raise
        verify(contract["sources"])
        guard_checkpoints(contract)
        verify(contract["inputs"])
        summarize_task(output, args.task, checkpoints)
        write_json(output / "queue_complete.json", {"complete": True, "passed": True,
                   "technical_only": False, "task": args.task, "policies": list(POLICIES),
                   "variants": list(VARIANTS), "episodes": 800, "finished": now(),
                   "resources": resources, "static_cell_assignments": assignments,
                   "elapsed_seconds_including_gate_wait": time.monotonic() - started})
        status(output, "completed", episodes=800, policies=list(POLICIES), variants=list(VARIANTS))
    except BaseException as exc:
        status(output, "failed", complete=False, error=repr(exc))
        raise


if __name__ == "__main__":
    def terminate(signum, frame):
        raise SystemExit(f"Received signal {signum}; stopping owned OOD evaluators")
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=("object_0", "object_1", "object_2"), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpuset", required=True)
    parser.add_argument("--extra-gpu")
    parser.add_argument("--extra-cpuset")
    parser.add_argument("--third-gpu")
    parser.add_argument("--third-cpuset")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if bool(args.extra_gpu) != bool(args.extra_cpuset):
        parser.error("--extra-gpu and --extra-cpuset must be provided together")
    if bool(args.third_gpu) != bool(args.third_cpuset):
        parser.error("--third-gpu and --third-cpuset must be provided together")
    run(args)
