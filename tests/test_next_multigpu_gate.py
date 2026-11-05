"""The next campaign must wait for actual outputs, not idle GPUs or dead PIDs."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import next_multigpu_gate as gate


def put(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def complete(root, relative, **fields):
    put(root, relative, dict(complete=True, **fields))


def episodes(ids):
    return [{"reset": {"initial_state_id": state}, "seed": 100000 + i,
             "success": False, "primitive_steps": 280} for i, state in enumerate(ids)]


def reports(root):
    put(root, "analysis.json", {"diagnostics": []})
    for name in ("report.md", "runbook.md", "tables/success.csv", "tables/paired_success.csv",
                 "tables/diagnostic_status.csv", "resource_cost.json",
                 "figures/task_comparison/validation_by_steps.png", "figures/task_comparison/success_and_retention.png"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("nonempty test artifact\n")


@pytest.fixture
def finished(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_active_processes", lambda root: [])
    source = tmp_path / "frozen_runtime.py"
    source.write_text("original runtime\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    for task in gate.TASKS:
        method = f"methods/rldt/{task}"
        put(tmp_path, f"{method}/status.json", {"state": "completed", "stage": "action_evaluations_complete"})
        put(tmp_path, f"{method}/training/summary.json", {"iterations": 100, "decisions": 3200,
                                                         "actor_updates": 392, "critic_updates": 6400})
        rows = [{"iterations": i, "decisions": 32 * i, "critic_updates": 64 * i,
                 "actor_updates": 4 * max(i - 2, 0), "primitive_steps": 160 * i} for i in range(1, 101)]
        (tmp_path / method / "training/metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        checkpoints = {}
        for iteration in gate.ITERATIONS:
            files = {}
            for filename in ("expert.npz", "learner.npz", "rldt_pi05.json"):
                path = tmp_path / method / "training" / f"iteration_{iteration:06d}" / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("immutable checkpoint test bytes")
                files[filename] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            checkpoints[str(iteration)] = {"files": files}
        put(tmp_path, f"{method}/checkpoint_audit.json", {"passed": True, "checkpoints": checkpoints})
        put(tmp_path, f"{method}/input_contract.json", {str(source): digest})
        splits = {"validation": list(range(100, 150)), "test": list(range(150, 200)), "source": list(range(50))}
        put(tmp_path, f"tasks/{task}/protocol.json", {"tasks": {task: {f"{s}_state_ids": ids for s, ids in splits.items()}}})
        for iteration in (0, *gate.ITERATIONS):
            name = "baseline_validation" if iteration == 0 else f"validation_{iteration:06d}"
            put(tmp_path, f"{method}/{name}.json", {"episodes": episodes(splits["validation"])})
        put(tmp_path, f"{method}/selection.json", {"iteration": 25, "locked": "timestamp", "test_results_seen": False,
                                                   "validation_counts": {"25": 0, "50": 0, "100": 0}})
        for split in ("test", "source"):
            for name in ("baseline", "selected", "checkpoint_000050", "checkpoint_000100"):
                put(tmp_path, f"{method}/{name}_{split}.json", {"episodes": episodes(splits[split])})
        for directory in ("traces", "trace_analysis"):
            complete(tmp_path, f"{directory}/{task}/summary.json", task=task, states=50)
        complete(tmp_path, f"interventions/{task}/queue_complete.json", passed=True, technical_only=False,
                 task=task, episodes_per_condition=10)
        complete(tmp_path, f"ood/{task}/queue_complete.json", passed=True, technical_only=False, task=task,
                 episodes=800, variants=list(gate.VARIANTS))
        complete(tmp_path, f"rollout_observation_bank/{task}/status.json", pending=[])
        complete(tmp_path, f"rollout_trace_analysis/fixed_input_qualitative/{task}/summary.json", task=task)
        for model in ("pi05_base", "pi05_libero"):
            for directory in ("probes", "probes_counterfactual"):
                complete(tmp_path, f"{directory}/{task}/{model}/metrics.json")
        for policy in ("pi05_base", *gate.POLICIES):
            complete(tmp_path, f"traces/{task}/{policy}/text_summary.json", task=task, policy=policy, states=50)
            if policy == "pi05_base":
                continue
            complete(tmp_path, f"traces/{task}/{policy}/expert_summary.json", task=task, policy=policy, states=50)
            complete(tmp_path, f"causal_analysis/{task}/{policy}/summary.json", task=task, policy=policy,
                     closed_loop_conditions=9, paired_episodes_per_condition=10, open_loop_conditions=15, open_loop_states=50)
            for directory in ("rollout_observation_bank", "rollout_traces", "rollout_trace_analysis"):
                complete(tmp_path, f"{directory}/{task}/{policy}/summary.json", task=task, policy=policy,
                         episodes=5, **({"passed": True} if directory == "rollout_traces" else {}))
            ood_policy = f"rldt_{int(policy.split('_')[-1])}" if policy.startswith("rldt_") else policy
            complete(tmp_path, f"ood/{task}/{ood_policy}/summary.json", task=task, policy=ood_policy, episodes=200)
            for variant in gate.VARIANTS:
                complete(tmp_path, f"ood/{task}/{ood_policy}/{variant}/summary.json", task=task, variant=variant,
                         technical_only=False, episode_count=50, episodes=episodes(list(range(50))))
    complete(tmp_path, "attention_grid_queue_complete.json", count_per_task=50)
    complete(tmp_path, "probe_campaign_status.json", pending=[])
    complete(tmp_path, "trace_analysis_campaign_status.json", pending=[], causal_pending=[], counterfactual_probe_pending=[])
    complete(tmp_path, "rollout_traces/status.json", pending=[])
    complete(tmp_path, "rollout_traces/summary.json", passed=True)
    complete(tmp_path, "rollout_trace_analysis/status.json", rollout_pending=[], fixed_input_pending=[])
    complete(tmp_path, "rollout_trace_analysis/summary.json")
    reports(tmp_path)
    return tmp_path


def test_computations_can_finish_without_manual_acceptance(finished):
    result = gate.inspect_current_gate(finished)
    assert result["ready"], result["blockers"]
    assert not result["scientific_review_passed"]
    assert not result["acceptance_report_complete"]
    assert not (finished / "fixed_input_qualitative").exists()
    assert any("actual renderer destination" in warning for warning in result["warnings"])


def test_idle_or_disappeared_workers_do_not_establish_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_active_processes", lambda root: [])
    result = gate.inspect_current_gate(tmp_path)
    assert not result["ready"]
    assert any("100-iteration" in row for row in result["blockers"])


def test_active_finite_worker_blocks_finished_artifacts(finished, monkeypatch):
    monkeypatch.setattr(gate, "_active_processes", lambda root: [{"pid": 123, "argv": ["worker"]}])
    result = gate.inspect_current_gate(finished)
    assert not result["ready"]
    assert result["active_processes"][0]["pid"] == 123


@pytest.mark.parametrize("relative,change", [
    ("methods/rldt/object_0/baseline_test.json", lambda d: d["episodes"][1]["reset"].update(initial_state_id=150)),
    ("methods/rldt/object_0/selection.json", lambda d: d.update(iteration=50)),
    ("ood/object_1/pi05_libero/both_new_regions/summary.json", lambda d: d.update(successes=5)),
    ("trace_analysis_campaign_status.json", lambda d: d.update(causal_pending=[["object_0", "pi05_libero"]])),
])
def test_inconsistent_complete_artifacts_block(finished, relative, change):
    value = json.loads((finished / relative).read_text())
    change(value)
    put(finished, relative, value)
    reports(finished)
    assert not gate.inspect_current_gate(finished)["ready"]


def test_current_failure_blocks_but_historical_failure_does_not(finished):
    put(finished, "old_failed_attempt/status.json", {"state": "failed"})
    assert gate.inspect_current_gate(finished)["ready"]
    put(finished, "ood/object_2/queue_status.json", {"state": "failed", "error": "worker exited 1"})
    result = gate.inspect_current_gate(finished)
    assert not result["ready"]
    assert any("worker exited 1" in error for error in result["failures"])


def test_final_hash_check_detects_changed_checkpoint(finished):
    assert gate.inspect_current_gate(finished)["ready"]
    (finished / "methods/rldt/object_2/training/iteration_000050/expert.npz").write_text("changed")
    result = gate.inspect_current_gate(finished)
    assert not result["ready"]
    assert any("hash mismatch" in row for row in result["blockers"])


def test_report_must_refresh_after_analysis(finished):
    complete(finished, "rollout_trace_analysis/summary.json")
    result = gate.inspect_current_gate(finished)
    assert not result["ready"]
    assert any("report refresh" in row for row in result["blockers"])
    reports(finished)
    assert gate.inspect_current_gate(finished)["ready"]


def test_truncated_or_malformed_json_blocks_without_throwing(finished):
    (finished / "trace_analysis_campaign_status.json").write_text('{"complete":')
    (finished / "methods/rldt/object_0/validation_000025.json").write_text('{"episodes": "invalid"}')
    result = gate.inspect_current_gate(finished)
    assert not result["ready"]
    assert any("unreadable" in row for row in result["blockers"])


def test_process_scan_excludes_permanent_monitor_and_unrelated_python(tmp_path, monkeypatch):
    run = tmp_path / "repo/runs/current"
    proc = tmp_path / "proc"
    for pid, script, target in ((100, "monitor_vla_trace_objects.py", run),
                                (101, "run_vla_trace_analysis_campaign.py", run),
                                (102, "run_vla_trace_objects.py", tmp_path / "unrelated")):
        folder = proc / str(pid)
        folder.mkdir(parents=True)
        (folder / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in
                                         ["/env/bin/python", f"scripts/{script}", "--run", str(target)]) + b"\0")
    original = gate.Path
    monkeypatch.setattr(gate, "Path", lambda value: proc if str(value) == "/proc" else original(value))
    assert [row["pid"] for row in gate._active_processes(run)] == [101]
