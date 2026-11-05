"""Completed native OOD artifacts: semantic/lineage mutations beyond a Boolean marker."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_vla.diagnostics import ood_completion as consumer
from rl_vla.diagnostics.metrics import wilson


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def build(root, task="object_0", monkeypatch=None):
    root = root.resolve()
    output = root / "ood" / task
    review = root / "ood_per_lane_release_review_v1"
    output.mkdir(parents=True)
    source = root / "scripts/dispatcher.py"
    source.parent.mkdir()
    source.write_text("# frozen CPU fixture, never a policy result\n")
    sources = {str(source): sha(source)}
    resources = [{"role": "main", "task": task, "gpu": task[-1], "cpus": [24, 25, 26], "parent": {"pid": 100, "proc_start_ticks": 1000}}]
    release = review / "release_plans" / f"{task}.json"
    save(release, {"resources": resources})
    acceptance = save(review / "acceptance.json", {"passed": True})
    independent = save(review / "independent.json", {"passed": True})
    amendment = review / "scheduling_amendment_v1.json"
    save(amendment, {"root_acceptance": acceptance, "independent_review": independent,
         "approved_for_scheduling": True, "cases_unchanged": True, "outcome_dependent_scheduling": False,
         "version": consumer.VERSION, "release_plan_sha256": {task: sha(release)}})
    parent = {"pid": 200, "start_ticks": 2000}
    save(review / "launch_manifest_v1.json", {"complete": True, "scheduling_amendment": str(amendment),
         "scheduling_amendment_sha256": sha(amendment), "new_queues": {task: {"parent": parent}}, "source_sha256": sources})
    save(output / "queue_plan.json", {"pid": 200, "version": consumer.VERSION, "resources": resources,
         "state_ids": list(range(50)), "noise_seeds": list(range(100000, 100050)), "selection_can_change": False})
    protocol = root / "layout_protocol.json"
    save(protocol, {"states": list(range(50)), "task": task})
    contract = {"sources": sources, "inputs": {str(release): sha(release), str(amendment): sha(amendment), str(protocol): sha(protocol)},
                "dispatcher_version": consumer.VERSION, "checkpoint_stats": {}}
    for iteration in (25, 50, 100):
        expert = root / "methods/rldt" / task / "training" / f"iteration_{iteration:06d}" / "expert.npz"
        expert.parent.mkdir(parents=True)
        expert.write_bytes(f"fixture expert {iteration}".encode())
        stat = expert.stat()
        contract["inputs"][str(expert)] = sha(expert)
        contract["checkpoint_stats"][str(expert)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
    marker = root / "main_complete.json"
    save(marker, {"stage": "action_evaluations_complete", "state": "completed"})
    guard = save(output / "release_guards/lane_00.json", {"path": str(marker), "sha256": sha(marker),
                 "owned_processes_exited": True, "gpu_compute_processes_at_acquisition": [], "lane": 0,
                 "resource": resources[0], "gpu_uuid": "GPU-fixture"})
    claims, completed = [], []
    for policy in consumer.POLICIES:
        checkpoint = None if policy == "pi05_libero" else str(root / "methods/rldt" / task / "training" / f"iteration_{int(policy.split('_')[1]):06d}")
        for variant in consumer.VARIANTS:
            destination = output / policy / variant
            config = root / "diagnostic_layouts" / task / variant / "evaluate.toml"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(f"# frozen fixture {variant}\n")
            contract["inputs"][str(config)] = sha(config)
            claim_value = {"policy": policy, "variant": variant, "task": task, "state_ids": list(range(50)),
                           "noise_seeds": list(range(100000, 100050)), "episodes": 50, "native_shards": 3,
                           "primitive_step_limit": 280, "checkpoint": checkpoint, "config": str(config),
                           "input_manifest": str(output / "input_manifest.json"), "lane_release_guard": guard,
                           "lane": 0, "resource": resources[0], "gpu_uuid": "GPU-fixture"}
            claim = save(output / "dispatch_claims" / f"{policy}__{variant}.json", claim_value)
            claims.append({**claim, "policy": policy, "variant": variant, "lane": 0})
            episodes = []
            for state in range(50):
                video = destination / f"episode_{state:03d}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"technical fixture video")
                trajectory = destination / f"episode_{state:03d}_trajectory"
                trajectory.mkdir()
                episodes.append({"seed": 100000 + state, "success": state < 20, "primitive_steps": 200,
                    "reset": {"initial_state_id": state, "experiment_split": variant, "task_id": int(task[-1]),
                              "backend_seed": 12000 + state, "object_telemetry": {"object_ids": ["target", "distractor"]}},
                    "video": str(video), "trajectory": str(trajectory)})
            report = {"complete": True, "technical_only": False, "task": task, "variant": variant,
                      "config": {}, "checkpoint": checkpoint, "source_config": str(config),
                      "source_config_sha256": sha(config), "episodes": episodes, "successes": 20,
                      "episode_count": 50, "success_rate": .4, "wilson95": wilson(20, 50),
                      "input_manifest": str(output / "input_manifest.json"), "episodes_file": str(destination / "episodes.jsonl")}
            (destination / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
            summary = save(destination / "summary.json", report)
            result = {"policy": policy, "variant": variant, "lane": 0, "claim": claim["path"], "claim_sha256": claim["sha256"],
                      "summary": summary["path"], "summary_sha256": summary["sha256"]}
            save(output / "dispatch_results" / f"{policy}__{variant}.json", result)
            completed.append(result)
    if task == "object_1":
        cache_inputs, binaries = {}, {}
        for number in range(3):
            file = root / f"native_cache/binary_{number}-cache"
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(f"native binary {number}".encode())
            binaries[str(file)] = cache_inputs[str(file)] = sha(file)
        oracle = root / "compiler_profile_integration/gpu6_native_cache_oracle_v1/summary.json"
        save(oracle, {"passed": True, "complete": True, "original_native_cache_compatibility_verified": True})
        if monkeypatch is not None:
            monkeypatch.setattr(consumer, "ACCEPTED_GPU6_ORACLE_SHA256", sha(oracle))
        cache_inputs[str(oracle)] = sha(oracle)
        binding = {"parent_pid": 200, "parent_start_ticks": 2000, "inputs": cache_inputs,
                   "binary_sha256": binaries, "oracle": str(oracle), "oracle_sha256": sha(oracle)}
        binding_path = output / "gpu6_native_cache_binding.json"
        save(binding_path, binding)
        contract["inputs"][str(binding_path)] = sha(binding_path)
        save(output / "gpu6_native_cache_completion.json", {"complete": True, "passed": True, "parent_pid": 200,
             "parent_start_ticks": 2000, "checks_performed": 2, "input_sha256": cache_inputs})
    save(output / "input_manifest.json", contract)
    queue_path = output / "queue_complete.json"
    save(queue_path, {"complete": True, "passed": True, "technical_only": False, "version": consumer.VERSION,
         "task": task, "episodes": 800, "policies": list(consumer.POLICIES), "variants": list(consumer.VARIANTS),
         "release_plan_sha256": sha(release), "scheduling_amendment_sha256": sha(amendment), "resources": resources,
         "claims": claims, "completed": completed, "lane_release_guards": [guard]})
    return SimpleNamespace(root=root, task=task, output=output, queue_path=queue_path, protocol=protocol, guard=guard)


@pytest.fixture
def full(tmp_path):
    value = build(tmp_path)
    assert consumer.completion(value.root, value.task)["passed"] is True
    return value


def rebind_report(value, mutate):
    queue = read(value.queue_path)
    row = queue["completed"][0]
    report = read(row["summary"])
    mutate(report)
    save(row["summary"], report)
    row["summary_sha256"] = sha(row["summary"])
    save(value.output / "dispatch_results" / f"{row['policy']}__{row['variant']}.json", row)
    save(value.queue_path, queue)


def test_full_native_matrix_and_sources(full):
    result = consumer.completion(full.root, full.task)
    assert result["passed"] is True and result["cells"] == 16 and result["episodes"] == 800
    assert str(full.protocol) in result["sources"]


@pytest.mark.parametrize("field", ["claims", "completed"])
def test_missing_or_duplicate_matrix_row_fails(full, field):
    queue = read(full.queue_path)
    queue[field][-1] = copy.deepcopy(queue[field][0])
    save(full.queue_path, queue)
    assert consumer.completion(full.root, full.task)["passed"] is False


@pytest.mark.parametrize("mutation", ["seed", "state", "task", "variant", "outcome", "steps", "count", "rate", "wilson", "telemetry"])
def test_rehashed_report_semantic_mutations_fail(full, mutation):
    def mutate(report):
        episode = report["episodes"][0]
        if mutation == "seed": episode["seed"] += 1
        elif mutation == "state": episode["reset"]["initial_state_id"] += 1
        elif mutation == "task": episode["reset"]["task_id"] = 2
        elif mutation == "variant": episode["reset"]["experiment_split"] = "source"
        elif mutation == "outcome": episode["success"] = 1
        elif mutation == "steps": episode["primitive_steps"] = 281
        elif mutation == "count": report["successes"] += 1
        elif mutation == "rate": report["success_rate"] = .6
        elif mutation == "wilson": report["wilson95"] = [0, 1]
        elif mutation == "telemetry": del episode["reset"]["object_telemetry"]
    rebind_report(full, mutate)
    assert consumer.completion(full.root, full.task)["passed"] is False


@pytest.mark.parametrize("kind", ["protocol", "dispatch", "video", "trajectory", "release_marker", "expert"])
def test_missing_or_changed_evidence_fails(full, kind):
    queue = read(full.queue_path)
    row = queue["completed"][0]
    report = read(row["summary"])
    if kind == "protocol": full.protocol.write_text('{"changed":true}')
    elif kind == "dispatch": (full.output / "dispatch_results" / f"{row['policy']}__{row['variant']}.json").unlink()
    elif kind == "video": Path(report["episodes"][0]["video"]).unlink()
    elif kind == "trajectory": Path(report["episodes"][0]["trajectory"]).rmdir()
    elif kind == "release_marker": Path(read(full.guard["path"])["path"]).write_text('{}')
    elif kind == "expert":
        expert = next((full.root / "methods").rglob("expert.npz"))
        expert.write_bytes(b"changed expert")
    assert consumer.completion(full.root, full.task)["passed"] is False


def test_pending_and_failed_are_separate(tmp_path):
    assert consumer.completion(tmp_path, "object_0")["passed"] is None
    save(tmp_path / "ood/object_0/queue_status.json", {"state": "failed"})
    assert consumer.completion(tmp_path, "object_0")["passed"] is False


@pytest.mark.parametrize("mutation", [None, "parent", "start", "missing", "binary", "inputs", "oracle", "oracle_path"])
def test_gpu6_native_cache_completion_and_superseded_parent(tmp_path, mutation, monkeypatch):
    value = build(tmp_path, task="object_1", monkeypatch=monkeypatch)
    assert consumer.completion(value.root, value.task)["passed"] is True
    path = value.output / "gpu6_native_cache_completion.json"
    data = read(path)
    if mutation == "parent": data["parent_pid"] = 199
    elif mutation == "start": data["parent_start_ticks"] = 1999
    elif mutation == "inputs": data["input_sha256"] = {}
    elif mutation == "missing": path.unlink()
    elif mutation == "binary": next((tmp_path / "native_cache").glob("*-cache")).write_bytes(b"mutated")
    elif mutation in ("oracle", "oracle_path"):
        binding_path = value.output / "gpu6_native_cache_binding.json"
        binding = read(binding_path)
        old_oracle = binding["oracle"]
        oracle = Path(old_oracle) if mutation == "oracle" else tmp_path / "substituted_oracle.json"
        payload = read(old_oracle)
        if mutation == "oracle": payload["substituted"] = True
        save(oracle, payload)
        binding["oracle"], binding["oracle_sha256"] = str(oracle), sha(oracle)
        del binding["inputs"][old_oracle]
        binding["inputs"][str(oracle)] = sha(oracle)
        save(binding_path, binding)
        data["input_sha256"] = binding["inputs"]
        contract_path = value.output / "input_manifest.json"
        contract = read(contract_path)
        contract["inputs"][str(binding_path)] = sha(binding_path)
        save(contract_path, contract)
    if mutation in ("parent", "start", "inputs", "oracle", "oracle_path"): save(path, data)
    result = consumer.completion(value.root, value.task)
    assert result["passed"] is (mutation is None)
