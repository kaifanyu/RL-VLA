"""Restore saved native learner states on CPU without training or checkpoint writes."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from rl_vla.diagnostics.learner_restore import file_hash

TASKS = ("object_0", "object_1", "object_2")
ITERATIONS = (25, 50, 100)
SCOPE = "Serialization restore only: actor/reference/critic/target/optimizer/PRNG arrays. No optimizer step, rollout, replay restoration or environment-state resume is performed or established."
REPO = Path(__file__).resolve().parents[1]
SOURCES = ("scripts/audit_vla_trace_learner_restore.py", "src/rl_vla/diagnostics/learner_restore.py",
           "src/rl_vla/rldt.py", "src/rl_vla/rldt_pi05.py",
           "third_party/openpi/src/openpi/models/pi0.py", "third_party/openpi/src/openpi/models/gemma.py")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def memory():
    limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    current = int(Path("/sys/fs/cgroup/memory.current").read_text())
    return {"limit_bytes": None if limit == "max" else int(limit), "current_bytes": current,
            "headroom_bytes": None if limit == "max" else int(limit) - current}


def source_hashes():
    return {name: file_hash(REPO / name) for name in SOURCES}


def checked_sources(root):
    destination = root / "learner_restore_audit/sources.json"
    current = source_hashes()
    if destination.exists():
        if json.loads(destination.read_text())["source_hashes"] != current:
            raise ValueError("Pinned restore-audit sources changed")
    else:
        for name, sha in current.items():
            output = root / "learner_restore_audit/source" / sha / Path(name).name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes((REPO / name).read_bytes())
        atomic_json(destination, {"source_hashes": current, "scope": SCOPE})
    return current


def snapshot_files(checkpoint):
    paths = [checkpoint / name for name in ("learner.npz", "expert.npz", "rldt_pi05.json")]
    paths += [checkpoint.parent / name for name in ("config.json", "manifest.json")]
    return {str(path): {"bytes": path.stat().st_size, "sha256": file_hash(path)} for path in paths}


def audit(root, task, iteration):
    import jax
    from rl_vla.rldt import Learner
    from rl_vla.rldt_pi05 import Pi05Flow
    from rl_vla.diagnostics.learner_restore import check_restored_state, expected_updates, tree_records, tree_schema

    if os.environ.get("JAX_PLATFORMS") != "cpu" or any(d.platform != "cpu" for d in jax.devices()):
        raise RuntimeError("Native restore audit requires explicit JAX_PLATFORMS=cpu")
    destination = root / "learner_restore_audit" / task / f"iteration_{iteration:06d}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable audit: {destination}")
    started = time.monotonic()
    sources = checked_sources(root)
    before_memory = memory()
    if before_memory["headroom_bytes"] is not None and before_memory["headroom_bytes"] < 48 * 1024**3:
        raise MemoryError("Need at least48GiB cgroup headroom before native loading")
    method = root / "methods/rldt" / task
    checkpoint = method / "training" / f"iteration_{iteration:06d}"
    config = json.loads((checkpoint.parent / "config.json").read_text())
    manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    if config != manifest["config"] or manifest["init_from"] is not None or manifest["frozen_vlm"] is not True:
        raise ValueError("Training initialization/config manifest mismatch")
    contract_path = method / "input_contract.json"
    contract = json.loads(contract_path.read_text())
    if not all(file_hash(path) == sha for path, sha in contract.items()):
        raise ValueError("Frozen training/input contract changed")
    files_before = snapshot_files(checkpoint)
    print(json.dumps({"stage": "loading_untouched_native_flow", "task": task, "iteration": iteration,
                      "memory": before_memory}), flush=True)
    flow = Pi05Flow.from_config(config["policy"])
    if flow.metadata != manifest["policy_metadata"]:
        raise ValueError("Untouched native policy metadata differs from training initialization")
    original = tree_records(flow.params)
    frozen = tree_records(flow.frozen)
    if not all(row["finite"] for row in original + frozen):
        raise ValueError("Untouched native model contains nonfinite values")
    learner = Learner(flow, flow.critic_feature_dim, config)
    template = tree_schema(learner.state())
    print(json.dumps({"stage": "calling_native_Learner.restore", "template_leaves": len(template),
                      "feature_dim": flow.critic_feature_dim, "memory": memory()}), flush=True)
    learner.restore(checkpoint)
    expected = expected_updates(config, iteration)
    metrics = [json.loads(line) for line in (checkpoint.parent / "metrics.jsonl").read_text().splitlines(keepends=True)
               if line.endswith("\n")]
    matching = [row for row in metrics if row["iterations"] == iteration]
    if len(matching) != 1 or any(matching[0][kind + "_updates"] != value for kind, value in expected.items()):
        raise ValueError("Recorded native training counters disagree with configured optimizer schedule")
    checks = check_restored_state(learner, checkpoint, template, original, expected)
    if tree_records(flow.params) != original or tree_records(flow.frozen) != frozen:
        raise ValueError("Untouched base expert or frozen backbone changed during restore")
    files_after = snapshot_files(checkpoint)
    if files_before != files_after or sources != source_hashes():
        raise ValueError("Checkpoint or source bytes changed during restore audit")
    result = {"complete": True, "passed": True, "task": task, "iteration": iteration,
              "checkpoint": str(checkpoint), "scope": SCOPE, "native_restore_invoked": True,
              "no_optimizer_or_policy_execution": True, "checkpoint_files_unchanged": True,
              "untouched_base_expert_unchanged": True, "frozen_backbone_unchanged": True,
              "base_expert_leaves": original, "frozen_backbone_leaves": frozen,
              "feature_dim": flow.critic_feature_dim, "template": template, **checks,
              "source_hashes": sources, "files": files_after,
              "input_contract_sha256": file_hash(contract_path), "policy_metadata": flow.metadata,
              "training_counter_record": matching[0], "jax_version": jax.__version__,
              "devices": [str(d) for d in jax.devices()], "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "memory_before": before_memory, "memory_after": memory(),
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "elapsed_seconds": time.monotonic() - started}
    atomic_json(destination, result)
    print(json.dumps({"passed": True, "artifact": str(destination), "seconds": result["elapsed_seconds"],
                      "peak_rss_gib": result["peak_process_rss_bytes"] / 1024**3}), flush=True)


def campaign(root):
    checked_sources(root)
    destination = root / "learner_restore_audit"
    while True:
        rows = []
        for task in TASKS:
            for iteration in ITERATIONS:
                output = destination / task / f"iteration_{iteration:06d}.json"
                if not output.exists():
                    validation = root / "methods/rldt" / task / f"validation_{iteration:06d}.json"
                    headroom = memory()["headroom_bytes"]
                    if validation.exists() and (headroom is None or headroom >= 48 * 1024**3):
                        command = [sys.executable, str(Path(__file__).resolve()), "--run", str(root),
                                   "--task", task, "--iteration", str(iteration)]
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with output.with_suffix(".log").open("a") as stream:
                            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
                        if result.returncode:
                            raise RuntimeError(f"Native restore child failed: {output.with_suffix('.log')}")
                if output.exists():
                    value = json.loads(output.read_text())
                    rows.append({"task": task, "iteration": iteration, "passed": value["passed"],
                                 "path": str(output), "sha256": file_hash(output)})
                else:
                    rows.append({"task": task, "iteration": iteration, "passed": None, "path": str(output)})
        complete = all(row["passed"] is True for row in rows)
        atomic_json(destination / "summary.json", {"complete": complete, "passed": True if complete else None,
            "scope": SCOPE, "required_checkpoints": 9, "completed_checkpoints": sum(r["passed"] is True for r in rows),
            "checkpoints": rows, "sources_sha256": file_hash(destination / "sources.json"),
            "native_models_retained_while_waiting": False})
        if complete:
            return
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--iteration", type=int, choices=ITERATIONS)
    args = parser.parse_args()
    try:
        if args.task is None and args.iteration is None:
            campaign(args.run.resolve())
        elif args.task is not None and args.iteration is not None:
            audit(args.run.resolve(), args.task, args.iteration)
        else:
            parser.error("Provide both --task and --iteration, or neither for the waiting campaign")
    except Exception:
        failure = args.run.resolve() / "learner_restore_audit"
        if args.task:
            failure = failure / args.task / f"iteration_{args.iteration:06d}.failed.json"
        else:
            failure = failure / "campaign_failed.json"
        if not failure.exists():
            atomic_json(failure, {"complete": True, "passed": False, "scope": SCOPE, "error": traceback.format_exc()})
        raise
