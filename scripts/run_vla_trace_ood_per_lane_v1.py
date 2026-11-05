"""Versioned OOD dispatcher with independent main, causal, and trace releases.

The scientific evaluator and its three native shards are imported unchanged
from the frozen predecessor. One scheduler owns all sixteen cell claims.
"""

import argparse
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_vla_trace_ood_evaluations as native

VERSION = "per_lane_release_v1"
NATIVE_QUEUE_SHA256 = "864a3187d7fa49634c2682e451eea4969bd74ce07f9afe422081ca9e74ad2af6"


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


# Release/process/cache helpers are isolated from every frozen predecessor.
from rl_vla.diagnostics.ood_lane_release_v1 import (
    Lane,
    capture_release_plan,
    gpu6_cache_contract,
    scientific_readiness,
    validate_release_plan,
    verify_cache_environment,
)


def lane_contract(base, lane, output, index):
    if not lane.release_record:
        raise ValueError("Lane lacks its successfully acquired release guard")
    record = dict(lane.release_record)
    native.verify({record["path"]: record["sha256"]})
    path = output / "release_guards" / f"lane_{index:02d}.json"
    immutable_json(path, {"version": VERSION, "lane": index, "resource": lane.resource, **record})
    # Copy the dictionary: primary workers may be hashing their immutable base
    # contract concurrently. No optional gate is ever added to that dictionary.
    return {**base, "inputs": {**base["inputs"], record["path"]: record["sha256"],
                              str(path): native.digest(path)},
            "lane_release_guard": {"path": str(path), "sha256": native.digest(path)}}


def run_cells(campaign, task, output, resources, configs, checkpoints, contract, *, lane_factory=Lane,
              evaluator=native.evaluate_variant, poll_seconds=1, input_monitor=None):
    """Exactly one owner dequeues cells; unavailable optional resources never wait."""
    output = Path(output)
    cells = [{"policy": policy, "variant": variant} for policy in native.POLICIES for variant in native.VARIANTS]
    pending, active, completed, claims = list(cells), {}, {}, []
    lanes = [lane_factory(resource) for resource in resources]
    lane_contracts = {}
    stop = threading.Event()
    sequence = 0

    def execute(cell, lane, inputs):
        if stop.is_set():
            raise RuntimeError("OOD queue already stopped before this claimed cell started")
        return evaluator(configs[cell["variant"]], output / cell["policy"] / cell["variant"],
            checkpoints[cell["policy"]], task=task, variant=cell["variant"], cpus=lane.resource["cpus"],
            gpu=lane.resource["gpu"], contract=inputs, stop=stop)

    with ThreadPoolExecutor(max_workers=len(resources)) as executor:
        def launch(index):
            nonlocal sequence
            lane = lanes[index]
            if not scientific_readiness(campaign, task)["ready"]:
                raise RuntimeError("Mandatory OOD gates changed after acquiring a worker GPU")
            if index not in lane_contracts:
                lane_contracts[index] = lane_contract(contract, lane, output, index)
            inputs = lane_contracts[index]
            cell = pending[0]
            destination = output / cell["policy"] / cell["variant"]
            if destination.exists():
                raise FileExistsError(f"Preserving existing complete or partial OOD cell: {destination}")
            key = (cell["policy"], cell["variant"])
            claim_path = output / "dispatch_claims" / f"{cell['policy']}__{cell['variant']}.json"
            claim = {"version": VERSION, "sequence": sequence, "task": task, **cell,
                "lane": index, "resource": lane.resource, "gpu_uuid": lane.uuid,
                "checkpoint": checkpoints[cell["policy"]], "config": str(configs[cell["variant"]]),
                "state_ids": list(range(50)), "noise_seeds": list(range(100000, 100050)),
                "episodes": 50, "native_shards": 3, "primitive_step_limit": 280,
                "input_manifest": str(output / "input_manifest.json"),
                "lane_release_guard": inputs.get("lane_release_guard"), "claimed": native.now()}
            immutable_json(claim_path, claim)
            claims.append({"path": str(claim_path), "sha256": native.digest(claim_path), **cell, "lane": index})
            sequence += 1
            pending.pop(0)
            future = executor.submit(execute, cell, lane, inputs)
            active[index] = (future, key, claim_path)

        def reap():
            for index, (future, key, claim_path) in list(active.items()):
                if not future.done():
                    continue
                result = future.result()
                if result.get("complete") is not True:
                    raise ValueError("Native OOD evaluator returned without a complete cell")
                summary = output / key[0] / key[1] / "summary.json"
                completed[key] = {"policy": key[0], "variant": key[1], "lane": index,
                    "claim": str(claim_path), "claim_sha256": native.digest(claim_path),
                    "summary": str(summary), "summary_sha256": native.digest(summary)}
                immutable_json(output / "dispatch_results" / f"{key[0]}__{key[1]}.json", completed[key])
                del active[index]

        def fill_primaries():
            for index, lane in enumerate(lanes):
                if pending and not lane.optional and index not in active and lane.acquire():
                    launch(index)

        try:
            while pending or active:
                if input_monitor is not None:
                    input_monitor()
                for lane in lanes:
                    lane.observe()
                reap()
                fill_primaries()
                for index, lane in enumerate(lanes):
                    if pending and lane.optional and index not in active and lane.acquire():
                        # A primary may finish while the optional lock/gate is
                        # checked; give it first choice before claiming more work.
                        reap()
                        fill_primaries()
                        if pending:
                            launch(index)
                        else:
                            lane.release()
                native.status(output, "evaluating", version=VERSION, completed_cells=len(completed),
                              pending_cells=len(pending), active_lanes=list(active),
                              optional_lanes_used=[index for index in lane_contracts if lanes[index].optional])
                if pending or active:
                    time.sleep(poll_seconds)
            if set(completed) != {(cell["policy"], cell["variant"]) for cell in cells} or len(claims) != 16:
                raise ValueError("OOD dispatcher failed to cover all sixteen unique cells")
            native.verify({claim["path"]: claim["sha256"] for claim in claims})
            for inputs in lane_contracts.values():
                if "lane_release_guard" in inputs:
                    native.verify({path: expected for path, expected in inputs["inputs"].items()
                                   if path not in contract["inputs"]})
            return {"claims": claims, "completed": [completed[(cell["policy"], cell["variant"])] for cell in cells],
                    "lane_release_guards": [inputs["lane_release_guard"] for inputs in lane_contracts.values()
                                                if "lane_release_guard" in inputs]}
        except BaseException:
            stop.set()  # Existing native evaluator cleans up all owned shard process groups.
            raise
        finally:
            # Do not release a GPU while a failed/interrupted native evaluator
            # still owns children. This also covers an interrupt after submit()
            # started a worker but before its Future entered the active map.
            executor.shutdown(wait=True, cancel_futures=stop.is_set())
            for lane in lanes:
                lane.release()


def run(args):
    source = Path(__file__).resolve()
    native_source = Path(native.__file__).resolve()
    if native.digest(native_source) != NATIVE_QUEUE_SHA256:
        raise ValueError("Frozen native OOD evaluator predecessor changed")
    campaign = args.run.resolve()
    output = campaign / "ood" / args.task
    output.mkdir(parents=True, exist_ok=False)
    release_plan = json.loads(args.release_plan.read_text())
    resources = validate_release_plan(release_plan, campaign, args.task)
    amendment = json.loads(args.amendment.read_text())
    if (amendment.get("version") != VERSION or amendment.get("approved_for_scheduling") is not True
            or amendment.get("native_source_sha256") != NATIVE_QUEUE_SHA256
            or amendment.get("cases_unchanged") is not True
            or amendment.get("outcome_dependent_scheduling") is not False
            or amendment.get("release_plan_sha256", {}).get(args.task) != native.digest(args.release_plan)
            or amendment.get("source_sha256", {}).get(str(source)) != native.digest(source)
            or amendment.get("release_helper_sha256") != release_plan["helper_sha256"]):
        raise ValueError("Expected immutable pre-scoring scheduling amendment")
    release_inputs = {str(args.release_plan.resolve()): native.digest(args.release_plan),
                      str(args.amendment.resolve()): native.digest(args.amendment)}
    if release_plan.get("process_capture"):
        release_inputs[release_plan["process_capture"]["path"]] = release_plan["process_capture"]["sha256"]
    cache_contract = gpu6_cache_contract(campaign) if args.task == "object_1" else None
    if cache_contract:
        verify_cache_environment(cache_contract)
        release_inputs.update(cache_contract["inputs"])
        immutable_json(output / "gpu6_native_cache_binding.json", {**cache_contract,
            "parent_pid": os.getpid(), "parent_start_ticks": native.process_start_ticks(os.getpid()),
            "version": VERSION, "scope": "Read-only native shared-cache binding; no diagnostic profile"})
        release_inputs[str(output / "gpu6_native_cache_binding.json")] = native.digest(output / "gpu6_native_cache_binding.json")
    trackers = [Lane(resource) for resource in resources]
    cells = [{"policy": policy, "variant": variant} for policy in native.POLICIES for variant in native.VARIANTS]
    plan = {"version": VERSION, "task": args.task, "policies": list(native.POLICIES), "variants": list(native.VARIANTS),
        "predeclared_cells": cells, "episodes_per_cell": 50, "native_shards_per_cell": 3,
        "primitive_step_limit": 280, "state_ids": list(range(50)), "noise_seeds": list(range(100000, 100050)),
        "selection_can_change": False, "resources": resources, "pid": os.getpid(), "created": native.now(),
        "source": str(source), "source_sha256": native.digest(source), "native_source_sha256": NATIVE_QUEUE_SHA256,
        "scheduling": "Single dequeue owner; primary lanes preferred; optional lane never blocks primary claims",
        "required_order": "Locked selection, complete main tests, frozen layouts and reachability; each lane independently requires its successful owner marker, process exit and exclusive physical GPU lock"}
    immutable_json(output / "queue_plan.json", plan)
    raw = output / "pi05_base"
    raw.mkdir()
    immutable_json(raw / "summary.json", {"complete": True, "applicable": False, "status": "not_applicable",
        "task": args.task, "policy": "pi05_base", "reason": "N/A — incompatible action contract", "episodes": 0})
    started = time.monotonic()
    try:
        while True:
            for lane in trackers:
                lane.observe()
            native.verify(release_inputs)
            gates = scientific_readiness(campaign, args.task)
            if gates["ready"]:
                break
            native.status(output, "waiting_for_required_gates", version=VERSION, **gates)
            time.sleep(args.poll_seconds)
        if native.digest(native_source) != NATIVE_QUEUE_SHA256:
            raise ValueError("Pinned native OOD predecessor changed during the mandatory gate wait")
        # Re-read semantics before freezing hashes; a changed failed marker cannot be blessed.
        if not scientific_readiness(campaign, args.task)["ready"]:
            raise ValueError("Scientific gate changed before native freeze")
        preparation = output / "native_freeze"
        preparation.mkdir()
        contract, configs, checkpoints = native.freeze(campaign, args.task, preparation, gates)
        if contract["sources"].get(str(native_source)) != NATIVE_QUEUE_SHA256:
            raise ValueError("Native OOD predecessor changed while freezing its source contract")
        snapshot = output / "dispatcher_source.py"
        with snapshot.open("xb") as stream:
            stream.write(source.read_bytes())
        if native.digest(source) != plan["source_sha256"] or native.digest(snapshot) != plan["source_sha256"]:
            raise ValueError("Dispatcher source changed before its first cell")
        contract["sources"] = {**contract["sources"], str(source): plan["source_sha256"],
                               str(snapshot): plan["source_sha256"]}
        contract["dispatcher_version"] = VERSION
        contract["dispatcher_source_snapshot"] = str(snapshot)
        contract["native_freeze_manifest"] = str(preparation / "input_manifest.json")
        archive = output / "source_snapshot.tar.gz"
        os.link(preparation / "source_snapshot.tar.gz", archive)
        contract["native_source_archive"] = str(archive)
        contract["inputs"] = {**contract["inputs"], **release_inputs, str(archive): native.digest(archive),
                              str(preparation / "input_manifest.json"): native.digest(preparation / "input_manifest.json")}
        helper = Path(__import__("rl_vla.diagnostics.ood_lane_release_v1", fromlist=["x"]).__file__).resolve()
        if native.digest(helper) != release_plan["helper_sha256"]:
            raise ValueError("Release helper changed after lease-plan preparation")
        helper_snapshot = output / "release_helper_source.py"
        with helper_snapshot.open("xb") as stream:
            stream.write(helper.read_bytes())
        if native.digest(helper_snapshot) != release_plan["helper_sha256"]:
            raise ValueError("Release helper changed during source snapshot")
        contract["sources"].update({str(helper): release_plan["helper_sha256"],
                                    str(helper_snapshot): release_plan["helper_sha256"]})
        immutable_json(output / "input_manifest.json", contract)
        lane_by_gpu = {lane.resource["gpu"]: lane for lane in trackers}
        last_cache_check = [0.0]
        cache_checks = [0]

        def monitor_cache():
            if cache_contract and time.monotonic() - last_cache_check[0] >= 60:
                verify_cache_environment(cache_contract)
                native.verify(cache_contract["inputs"])
                cache_checks[0] += 1
                native.write_json(output / "gpu6_native_cache_status.json", {
                    "passed": True, "state": "watching", "parent_pid": os.getpid(),
                    "parent_start_ticks": native.process_start_ticks(os.getpid()),
                    "checks_performed": cache_checks[0], "checked": native.now(),
                    "binary_sha256": cache_contract["binary_sha256"]})
                last_cache_check[0] = time.monotonic()
        result = run_cells(campaign, args.task, output, resources, configs, checkpoints, contract,
                           lane_factory=lambda resource: lane_by_gpu[resource["gpu"]], input_monitor=monitor_cache)
        if cache_contract:
            verify_cache_environment(cache_contract)
        native.verify(contract["sources"])
        native.guard_checkpoints(contract)
        native.verify(contract["inputs"])
        native.summarize_task(output, args.task, checkpoints)
        if cache_contract:
            immutable_json(output / "gpu6_native_cache_completion.json", {
                "passed": True, "complete": True, "parent_pid": os.getpid(),
                "parent_start_ticks": native.process_start_ticks(os.getpid()),
                "checks_performed": cache_checks[0], "input_sha256": cache_contract["inputs"]})
        native.write_json(output / "queue_complete.json", {"complete": True, "passed": True,
            "technical_only": False, "version": VERSION, "task": args.task, "policies": list(native.POLICIES),
            "variants": list(native.VARIANTS), "episodes": 800, "finished": native.now(),
            "resources": resources, "release_plan_sha256": native.digest(args.release_plan),
            "scheduling_amendment_sha256": native.digest(args.amendment), **result, "elapsed_seconds_including_gate_wait": time.monotonic() - started})
        native.status(output, "completed", version=VERSION, episodes=800)
    except BaseException as exc:
        if cache_contract:
            native.write_json(output / "gpu6_native_cache_status.json", {
                "passed": False, "complete": False, "state": "queue_failed",
                "parent_pid": os.getpid(), "error": repr(exc), "checked": native.now()})
        native.status(output, "failed", complete=False, version=VERSION, error=repr(exc))
        raise


if __name__ == "__main__":
    def terminate(signum, frame):
        raise SystemExit(f"Received signal {signum}; stopping owned OOD evaluators")
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=("object_0", "object_1", "object_2"), required=True)
    parser.add_argument("--release-plan", type=Path, required=True)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--prepare-release-plan", action="store_true")
    parser.add_argument("--main-parent-pid", type=int)
    parser.add_argument("--causal-parent-pid", type=int)
    parser.add_argument("--trace-parent-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.prepare_release_plan:
        immutable_json(args.release_plan, capture_release_plan(args.run.resolve(), args.task,
            args.main_parent_pid, args.causal_parent_pid, args.trace_parent_pid))
    else:
        if not args.amendment:
            parser.error("--amendment is mandatory before research scoring")
        run(args)
