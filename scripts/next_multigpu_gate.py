"""Read-only computational completion gate for the current LIBERO campaign.

This is a scheduling dependency, not the handoff's scientific acceptance audit.
The latter additionally needs a manually authored scientific_review.json. No
upstream file is created, rewritten, repaired, or treated as complete by this
module. Missing outputs and failed workers both keep the scheduling gate shut.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

TASKS = ("object_0", "object_1", "object_2")
ITERATIONS = (25, 50, 100)
POLICIES = ("pi05_libero", "rldt_000025", "rldt_000050", "rldt_000100")
VARIANTS = ("both_new_regions", "continuous_new_offsets", "target_moved_distractor_fixed",
            "distractor_moved_target_fixed")
PRODUCERS = frozenset({
    "run_vla_trace_objects.py", "evaluate_vla_trace_sharded.py", "run_vla_trace_probe_campaign.py",
    "run_vla_trace_probes.py", "run_vla_trace_analysis_campaign.py", "analyze_vla_trace_grid.py",
    "analyze_vla_trace_causal.py", "score_vla_trace_counterfactual_probes.py",
    "run_vla_trace_ood_evaluations.py", "run_vla_trace_rollout_bank.py",
    "extract_vla_trace_rollout_bank.py", "trace_pi05_rollout_bank.py",
    "queue_vla_trace_diagnostics.py", "run_vla_trace_grid.py", "run_vla_trace_interventions.py",
    "plot_vla_trace_qualitative.py", "analyze_vla_trace_objects.py",
})


def _active_processes(run: Path) -> list[dict]:
    """Inspect finite producers, excluding the permanent progress monitor."""
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [arg.decode(errors="replace") for arg in (entry / "cmdline").read_bytes().split(b"\0") if arg]
            if not argv or "python" not in Path(argv[0]).name:
                continue
            producer = any(Path(arg).name in PRODUCERS for arg in argv[1:])
            producer |= any(arg in {"rl_vla.rldt", "rl_vla.rldt_env"} for arg in argv[1:])
            if not producer:
                continue
            text = " ".join(argv)
            belongs = str(run) in text
            if not belongs and run.name in text:
                belongs = (entry / "cwd").resolve() == run.parents[1]
            if belongs:
                result.append({"pid": int(entry.name), "argv": argv})
        except (OSError, RuntimeError):
            continue
    return sorted(result, key=lambda item: item["pid"])


@lru_cache(maxsize=2048)
def _hash_version(path: str, inode: int, size: int, mtime_ns: int, ctime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    current = Path(path).stat()
    if (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (inode, size, mtime_ns, ctime_ns):
        raise ValueError("Artifact changed while hashing")
    return digest.hexdigest()


def _sha(path: Path) -> str:
    stat = path.stat()
    return _hash_version(str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class _Inspection:
    def __init__(self, root):
        self.root = root
        self.blockers = []
        self.failures = []
        self.checked = 0
        self.latest_result_ns = 0

    def require(self, value, description):
        if not value:
            self.blockers.append(description)
        return bool(value)

    def read(self, relative, *, required=True):
        path = self.root / relative
        self.checked += 1
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")
        except (OSError, ValueError) as exc:
            if required:
                self.blockers.append(f"{relative}: missing or unreadable ({type(exc).__name__})")
            return {}
        if data.get("state") in {"failed", "cancelled"}:
            error = f"{relative}: {data['state']}: {data.get('error', 'no error detail')}"
            self.blockers.append(error)
            self.failures.append(error)
        return data

    def complete(self, relative, **fields):
        data = self.read(relative)
        if not data:
            return data
        self.require(data.get("complete", data.get("completed")) is True, f"{relative}: incomplete")
        for key, expected in fields.items():
            self.require(data.get(key) == expected, f"{relative}: expected {key}={expected!r}")
        if data.get("complete", data.get("completed")) is True:
            self.latest_result_ns = max(self.latest_result_ns, (self.root / relative).stat().st_mtime_ns)
        return data

    def episodes(self, relative, ids):
        data = self.read(relative)
        if not data:
            return data
        rows = data.get("episodes", [])
        valid = isinstance(rows, list) and len(rows) == 50 and len(ids) == 50
        if valid:
            valid = all(isinstance(row, dict) and isinstance(row.get("reset"), dict) for row in rows)
        if valid:
            valid = ([row["reset"].get("initial_state_id") for row in rows] == ids and
                     [row.get("seed") for row in rows] == list(range(100000, 100050)) and
                     all(type(row.get("success")) is bool and isinstance(row.get("primitive_steps"), int) and
                         0 < row["primitive_steps"] <= 280 for row in rows))
        self.require(valid, f"{relative}: expected 50 unique paired state/noise outcomes and valid step counts")
        if valid:
            count = sum(row["success"] for row in rows)
            if "successes" in data:
                self.require(data["successes"] == count, f"{relative}: inconsistent success count")
            if "success_rate" in data:
                self.require(isinstance(data["success_rate"], (float, int)) and
                             abs(data["success_rate"] - count / 50) < 1e-12,
                             f"{relative}: inconsistent success rate")
        return data


def _inspect_task(check, task):
    method = f"methods/rldt/{task}"
    state = check.read(f"{method}/status.json")
    check.require((state.get("state"), state.get("stage")) == ("completed", "action_evaluations_complete"),
                  f"{method}: training and final action evaluations are not complete")
    summary = check.read(f"{method}/training/summary.json")
    check.require(tuple(summary.get(key) for key in ("iterations", "decisions", "actor_updates", "critic_updates")) ==
                  (100, 3200, 392, 6400), f"{method}: final training summary counters are incomplete")
    try:
        rows = [json.loads(line) for line in (check.root / method / "training/metrics.jsonl").read_text().splitlines()]
        valid = len(rows) == 100 and all(
            (r.get("iterations"), r.get("decisions"), r.get("critic_updates"), r.get("actor_updates")) ==
            (i, 32 * i, 64 * i, 4 * max(i - 2, 0)) and 0 < r.get("primitive_steps", 0) <= 160 * i
            for i, r in enumerate(rows, 1))
    except (OSError, ValueError, AttributeError, TypeError):
        valid = False
    check.require(valid, f"{method}: missing or inconsistent 100-iteration training counters")
    audit = check.read(f"{method}/checkpoint_audit.json")
    check.require(audit.get("passed") is True and set(audit.get("checkpoints", {})) == {"25", "50", "100"},
                  f"{method}: incomplete checkpoint audit")
    for iteration, record in audit.get("checkpoints", {}).items():
        check.require({"expert.npz", "learner.npz", "rldt_pi05.json"}.issubset(record.get("files", {})),
                      f"{method}: checkpoint {iteration} audit lacks complete checkpoint files")
    protocol = check.read(f"tasks/{task}/protocol.json").get("tasks", {}).get(task, {})
    validation = {}
    for iteration in (0, *ITERATIONS):
        name = "baseline_validation" if iteration == 0 else f"validation_{iteration:06d}"
        report = check.episodes(f"{method}/{name}.json", protocol.get("validation_state_ids", []))
        if iteration and len(report.get("episodes", [])) == 50:
            validation[str(iteration)] = sum(row.get("success") is True for row in report["episodes"])
    selection = check.read(f"{method}/selection.json")
    selected = selection.get("iteration")
    valid_selection = (bool(selection.get("locked")) and selection.get("test_results_seen") is False and
                       selected in ITERATIONS and len(validation) == 3 and
                       selection.get("validation_counts") == validation and
                       selected == min(ITERATIONS, key=lambda i: (-validation[str(i)], i)))
    check.require(valid_selection, f"{method}: selection is not the locked earliest best validation checkpoint")
    for split in ("test", "source"):
        ids = protocol.get(f"{split}_state_ids", [])
        check.episodes(f"{method}/baseline_{split}.json", ids)
        check.episodes(f"{method}/selected_{split}.json", ids)
        for iteration in ITERATIONS:
            if iteration != selected:
                check.episodes(f"{method}/checkpoint_{iteration:06d}_{split}.json", ids)
    check.complete(f"traces/{task}/summary.json", task=task, states=50)
    check.complete(f"trace_analysis/{task}/summary.json", task=task, states=50)
    check.complete(f"interventions/{task}/queue_complete.json", passed=True, technical_only=False,
                   task=task, episodes_per_condition=10)
    check.complete(f"ood/{task}/queue_complete.json", passed=True, technical_only=False, task=task,
                   episodes=800, variants=list(VARIANTS))
    check.complete(f"rollout_observation_bank/{task}/status.json", pending=[])
    check.complete(f"rollout_trace_analysis/fixed_input_qualitative/{task}/summary.json", task=task)
    for status in (f"interventions/{task}/queue_status.json", f"ood/{task}/queue_status.json"):
        check.read(status, required=False)
    for model in ("pi05_base", "pi05_libero"):
        check.complete(f"probes/{task}/{model}/metrics.json")
        check.complete(f"probes_counterfactual/{task}/{model}/metrics.json")
    for policy in ("pi05_base", *POLICIES):
        check.complete(f"traces/{task}/{policy}/text_summary.json", task=task, policy=policy, states=50)
        if policy == "pi05_base":
            continue
        check.complete(f"traces/{task}/{policy}/expert_summary.json", task=task, policy=policy, states=50)
        check.complete(f"causal_analysis/{task}/{policy}/summary.json", task=task, policy=policy,
                       closed_loop_conditions=9, paired_episodes_per_condition=10, open_loop_conditions=15,
                       open_loop_states=50)
        check.complete(f"rollout_observation_bank/{task}/{policy}/summary.json", task=task, policy=policy, episodes=5)
        check.complete(f"rollout_traces/{task}/{policy}/summary.json", passed=True, task=task, policy=policy, episodes=5)
        check.complete(f"rollout_trace_analysis/{task}/{policy}/summary.json", task=task, policy=policy, episodes=5)
        ood_policy = f"rldt_{int(policy.split('_')[-1])}" if policy.startswith("rldt_") else policy
        check.complete(f"ood/{task}/{ood_policy}/summary.json", task=task, policy=ood_policy, episodes=200)
        for variant in VARIANTS:
            relative = f"ood/{task}/{ood_policy}/{variant}/summary.json"
            check.complete(relative, task=task, variant=variant, technical_only=False, episode_count=50)
            check.episodes(relative, list(range(50)))
    return audit


def _verify_final_contracts(check, audits):
    """Hash immutable source and published checkpoints only when all work is done."""
    for task, audit in audits.items():
        method = f"methods/rldt/{task}"
        contract = check.read(f"{method}/input_contract.json")
        check.require(bool(contract), f"{method}: empty frozen input contract")
        expected = dict(contract)
        for iteration, record in audit.get("checkpoints", {}).items():
            for filename, meta in record.get("files", {}).items():
                path = check.root / method / "training" / f"iteration_{int(iteration):06d}" / filename
                expected[str(path)] = meta.get("sha256")
        for filename, digest in expected.items():
            try:
                matches = isinstance(digest, str) and _sha(Path(filename)) == digest
            except (OSError, ValueError):
                matches = False
            check.require(matches, f"Immutable source/checkpoint hash mismatch: {filename}")


def inspect_current_gate(run: Path) -> dict:
    """Return a read-only, fail-closed snapshot of computational readiness."""
    root = Path(run).resolve()
    check = _Inspection(root)
    audits = {}
    for task in TASKS:
        try:
            audits[task] = _inspect_task(check, task)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            check.blockers.append(f"{task}: malformed or changing completion evidence ({type(exc).__name__}: {exc})")
    for relative, fields in (
        ("attention_grid_queue_complete.json", {"count_per_task": 50}),
        ("probe_campaign_status.json", {"pending": []}),
        ("trace_analysis_campaign_status.json", {"pending": [], "causal_pending": [], "counterfactual_probe_pending": []}),
        ("rollout_traces/status.json", {"pending": []}),
        ("rollout_traces/summary.json", {"passed": True}),
        ("rollout_trace_analysis/status.json", {"rollout_pending": [], "fixed_input_pending": []}),
        ("rollout_trace_analysis/summary.json", {}),
    ):
        check.complete(relative, **fields)
    check.read("queue_status.json", required=False)
    check.read("analysis.json")
    for relative in ("report.md", "runbook.md", "tables/success.csv", "tables/paired_success.csv",
                     "tables/diagnostic_status.csv", "resource_cost.json", "figures/task_comparison/validation_by_steps.png",
                     "figures/task_comparison/success_and_retention.png"):
        path = root / relative
        check.require(path.is_file() and path.stat().st_size > 0, f"{relative}: missing or empty final report artifact")
    for relative in ("report.md", "analysis.json", "tables/success.csv", "tables/paired_success.csv", "tables/diagnostic_status.csv"):
        path = root / relative
        if path.exists():
            check.require(path.stat().st_mtime_ns >= check.latest_result_ns,
                          f"{relative}: waiting for report refresh after the latest completed analysis")
    active = _active_processes(root)
    if active:
        check.blockers.append(f"Current finite experiment/analysis producers remain active: {[item['pid'] for item in active]}")
    # Large immutable checkpoint hashes are not repeatedly read while jobs run.
    if not check.blockers:
        try:
            _verify_final_contracts(check, audits)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            check.blockers.append(f"Invalid or changing immutable artifact contract: {type(exc).__name__}: {exc}")
        active = _active_processes(root)
        if active:
            check.blockers.append("Current producers became active during final immutable-artifact verification")
    def optional_evidence(name):
        try:
            data = json.loads((root / name).read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}
    acceptance = optional_evidence("acceptance_gates.json")
    review = optional_evidence("scientific_review.json")
    warnings = ["Scheduling gate covers completed computations; scientific_review.json and full scientific acceptance remain separate."]
    if not (root / "fixed_input_qualitative").exists():
        warnings.append("Existing analyzer reads fixed_input_qualitative at the run root, but the renderer writes rollout_trace_analysis/fixed_input_qualitative; this gate checks the actual renderer destination. Upstream diagnostic acceptance can remain pending until the analyzer path is repaired.")
    return {"ready": not check.blockers, "run": str(root), "blockers": list(dict.fromkeys(check.blockers)),
            "failures": list(dict.fromkeys(check.failures)), "active_processes": active,
            "checked_artifacts": check.checked, "warnings": warnings,
            "acceptance_report_complete": acceptance.get("complete") is True,
            "scientific_review_passed": review.get("passed") is True,
            "scope": "computational_completion_not_scientific_acceptance"}
