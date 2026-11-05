"""Versioned OOD dispatcher: primary lanes proceed while an optional GPU is busy.

The scientific evaluator and its three native shards are imported unchanged
from the frozen predecessor. One scheduler owns all sixteen cell claims.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_vla_trace_ood_evaluations as native

VERSION = "opportunistic_third_lane_v1"
NATIVE_QUEUE_SHA256 = "864a3187d7fa49634c2682e451eea4969bd74ce07f9afe422081ca9e74ad2af6"


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Lane:
    """Nonblocking physical GPU acquisition; one optional immutable release proof."""

    def __init__(self, resource, *, lock_directory=None):
        self.resource = resource
        self.optional = bool(resource.get("release_gate"))
        self.lock_directory = Path(lock_directory) if lock_directory else native.ROOT / "runs/.vla_trace_gpu_locks"
        self.stream = None
        self.uuid = None
        self.release_record = None

    def acquire(self):
        if self.stream is not None:
            return True
        marker_bytes = None
        gate = self.resource.get("release_gate")
        if gate:
            if not native.released_diagnostic_gate(gate):
                return False
            marker_bytes = Path(gate).read_bytes()
        if self.uuid is None:
            self.uuid = subprocess.check_output(["nvidia-smi", "-i", self.resource["gpu"],
                "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
            if not self.uuid.startswith("GPU-") or "\n" in self.uuid:
                raise ValueError("Expected exactly one physical GPU UUID")
        self.lock_directory.mkdir(parents=True, exist_ok=True)
        stream = (self.lock_directory / f"{self.uuid}.lock").open("a+")
        try:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            if gate:
                if Path(gate).read_bytes() != marker_bytes or not native.released_diagnostic_gate(gate):
                    raise ValueError("Optional diagnostic release changed while acquiring its GPU")
                self.release_record = {"path": str(Path(gate).resolve()),
                    "sha256": hashlib.sha256(marker_bytes).hexdigest(),
                    "child_exit": json.loads(marker_bytes)["child_exit"], "gpu_uuid": self.uuid}
            self.stream = stream
            return True
        finally:
            if self.stream is not stream:
                stream.close()

    def release(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def lane_contract(base, lane, output, index):
    if not lane.optional:
        return base
    if not lane.release_record:
        raise ValueError("Optional lane lacks its successfully acquired release guard")
    record = dict(lane.release_record)
    native.verify({record["path"]: record["sha256"]})
    path = output / "release_guards" / f"lane_{index:02d}.json"
    immutable_json(path, {"version": VERSION, "lane": index, "resource": lane.resource, **record})
    # Copy the dictionary: primary workers may be hashing their immutable base
    # contract concurrently. No optional gate is ever added to that dictionary.
    return {**base, "inputs": {**base["inputs"], record["path"]: record["sha256"],
                              str(path): native.digest(path)},
            "optional_release_guard": {"path": str(path), "sha256": native.digest(path)}}


def run_cells(campaign, task, output, resources, configs, checkpoints, contract, *, lane_factory=Lane,
              evaluator=native.evaluate_variant, poll_seconds=1):
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
            if not native.readiness(campaign, task)["ready"]:
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
                "optional_release_guard": inputs.get("optional_release_guard"), "claimed": native.now()}
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
                if "optional_release_guard" in inputs:
                    native.verify({path: expected for path, expected in inputs["inputs"].items()
                                   if path not in contract["inputs"]})
            return {"claims": claims, "completed": [completed[(cell["policy"], cell["variant"])] for cell in cells],
                    "optional_release_guards": [inputs["optional_release_guard"] for inputs in lane_contracts.values()
                                                if "optional_release_guard" in inputs]}
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
    resources = native.queue_resources(args, campaign)
    cells = [{"policy": policy, "variant": variant} for policy in native.POLICIES for variant in native.VARIANTS]
    plan = {"version": VERSION, "task": args.task, "policies": list(native.POLICIES), "variants": list(native.VARIANTS),
        "predeclared_cells": cells, "episodes_per_cell": 50, "native_shards_per_cell": 3,
        "primitive_step_limit": 280, "state_ids": list(range(50)), "noise_seeds": list(range(100000, 100050)),
        "selection_can_change": False, "resources": resources, "pid": os.getpid(), "created": native.now(),
        "source": str(source), "source_sha256": native.digest(source), "native_source_sha256": NATIVE_QUEUE_SHA256,
        "scheduling": "Single dequeue owner; primary lanes preferred; optional lane never blocks primary claims",
        "required_order": "Each task's locked selection, all main action evaluations and all causal interventions; optional GPU additionally requires its own diagnostic child exit and exclusive lock"}
    immutable_json(output / "queue_plan.json", plan)
    raw = output / "pi05_base"
    raw.mkdir()
    immutable_json(raw / "summary.json", {"complete": True, "applicable": False, "status": "not_applicable",
        "task": args.task, "policy": "pi05_base", "reason": "N/A — incompatible action contract", "episodes": 0})
    started = time.monotonic()
    try:
        while True:
            gates = native.readiness(campaign, args.task)  # Only the task's mandatory scientific gates.
            if gates["ready"]:
                break
            native.status(output, "waiting_for_required_gates", version=VERSION, **gates)
            time.sleep(args.poll_seconds)
        if native.digest(native_source) != NATIVE_QUEUE_SHA256:
            raise ValueError("Pinned native OOD predecessor changed during the mandatory gate wait")
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
        contract["inputs"] = {**contract["inputs"], str(archive): native.digest(archive),
                              str(preparation / "input_manifest.json"): native.digest(preparation / "input_manifest.json")}
        immutable_json(output / "input_manifest.json", contract)
        result = run_cells(campaign, args.task, output, resources, configs, checkpoints, contract)
        native.verify(contract["sources"])
        native.guard_checkpoints(contract)
        native.verify(contract["inputs"])
        native.summarize_task(output, args.task, checkpoints)
        native.write_json(output / "queue_complete.json", {"complete": True, "passed": True,
            "technical_only": False, "version": VERSION, "task": args.task, "policies": list(native.POLICIES),
            "variants": list(native.VARIANTS), "episodes": 800, "finished": native.now(),
            "resources": resources, **result, "elapsed_seconds_including_gate_wait": time.monotonic() - started})
        native.status(output, "completed", version=VERSION, episodes=800)
    except BaseException as exc:
        native.status(output, "failed", complete=False, version=VERSION, error=repr(exc))
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
    if bool(args.extra_gpu) != bool(args.extra_cpuset) or bool(args.third_gpu) != bool(args.third_cpuset):
        parser.error("Every optional GPU requires its corresponding CPU pool")
    run(args)
