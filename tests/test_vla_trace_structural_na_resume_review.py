"""Independent adversarial review of exact resume proofs and cleanup."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rl_vla.diagnostics import structural_na_resume as resume

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return resume.record(path)


def function(name, source, namespace):
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return namespace[name]


@pytest.fixture
def proof(tmp_path):
    run = tmp_path / "run"
    required = [ROOT / "scripts/run_vla_trace_interventions_gpu6_resume_v1.py",
                Path(resume.__file__).resolve(), ROOT / "scripts/run_vla_trace_interventions_gpu6_v1.py"]
    source_hashes = {str(p): resume.sha(p) for p in required}
    comparisons = []
    for policy in resume.POLICIES:
        cases = [(c, 0) for c in resume.CONDITIONS] + [(resume.NONZERO_PROOFS[policy][1], resume.NONZERO_PROOFS[policy][0])]
        checkpoint = "pi05_libero" if policy == "pi05_libero" else str(run / "methods/rldt/object_1/training" / f"iteration_{int(policy.split('_')[1]):06d}")
        for condition, index in cases:
            row = {"policy": policy, "condition": condition, "episode_index": index}
            for kind in ("candidate", "reference"):
                if kind == "candidate":
                    folder = run / "technical" / str(index) / policy / condition.replace(":", "_")
                elif index == 0:
                    folder = run / "compiler_profile_integration/gpu6_concurrency_v1" / f"worker_{resume.POLICIES.index(policy)}" / condition.replace(":", "_")
                else:
                    folder = run / "interventions/object_1" / policy / condition.replace(":", "_")
                folder.mkdir(parents=True, exist_ok=True)
                actions = folder / f"episode_{index:03d}_actions.npz"
                np.savez(actions, normalized=np.zeros((2, 10, 32), np.float32))
                telemetry = [{"primitive_steps": i + 1, "simulator_state": [i, 1]} for i in range(10)]
                episode = {"episode": index, "initial_state_id": 150 + index, "seed": 100000 + index,
                           "checkpoint": checkpoint, "condition": condition,
                           "success": True if kind == "reference" and index else None,
                           "technical_smoke": not (kind == "reference" and index),
                           "reset": {"initial_state_id": 150 + index, "backend_seed": 12 + index},
                           "intervention_severity": [{"pixels": 3}, {"pixels": 4}]}
                plan = {"policy": policy, "checkpoint": checkpoint, "technical_smoke": True,
                        "state_ids": [150 + index], "seeds": [100000 + index],
                        "reset_progression_state_ids": list(range(150, 151 + index)),
                        "source_sha256": source_hashes,
                        "compiler_profile": {"profile_id": "native_baseline_gpu6_v1" if policy == "pi05_libero" else "native_trained_gpu6_v1"}}
                artifacts = {"actions": resume.record(actions),
                    "telemetry": dump(folder / f"episode_{index:03d}_telemetry.json", telemetry),
                    "no_op": dump(folder / "no_op.json", {"passed": True, "max_abs_difference": 0., "persisted_noise_recurrence_max_abs_difference": 0.}),
                    "case_plan": dump(folder.parent / "case_plan.json", plan)}
                if kind == "reference" and index:
                    log = folder / "episodes.jsonl"
                    log.write_text("\n".join(json.dumps({} if i < index else episode) for i in range(index + 1)) + "\n")
                    artifacts["episode_log"] = resume.record(log)
                else:
                    artifacts["summary"] = dump(folder / "summary.json", {"technical_smoke": True, "episodes": [episode]})
                row[kind] = artifacts
            comparisons.append(row)
    # Shared per-policy plans are rewritten for every condition; bind their final identities.
    for row in comparisons:
        for kind in ("candidate", "reference"):
            row[kind]["case_plan"] = resume.record(row[kind]["case_plan"]["path"])
    value = {"complete": True, "passed": True, "technical_only": True, "decisions": 2,
             "run": str(run), "source_input_sha256": source_hashes, "comparisons": comparisons}
    path = run / "proof.json"
    dump(path, value)
    resume.verify_resume_proof(path, run=run)
    return path, run, value


def test_complete_independent_proof_fixture(proof):
    path, run, _ = proof
    assert len(resume.verify_resume_proof(path, run=run)) > 300


@pytest.mark.parametrize("mutation", ["actions", "telemetry", "reset", "severity", "state", "seed", "progression", "profile", "noop", "reference_path", "duplicate_case", "source"])
def test_rehashed_false_proof_is_rejected(proof, mutation):
    path, run, value = proof
    row = next(r for r in value["comparisons"] if r["policy"] == "rldt_100" and r["episode_index"] == 7)
    artifacts = row["candidate"]
    if mutation == "actions":
        target = Path(artifacts["actions"]["path"])
        data = np.zeros((2, 10, 32), np.float32)
        data[1, 0, 0] = 1
        np.savez(target, normalized=data)
        artifacts["actions"] = resume.record(target)
    elif mutation == "telemetry":
        target = Path(artifacts["telemetry"]["path"])
        data = json.loads(target.read_text()); data[9]["simulator_state"][0] += 1
        artifacts["telemetry"] = dump(target, data)
    elif mutation in {"reset", "severity", "state", "seed"}:
        target = Path(artifacts["summary"]["path"])
        data = json.loads(target.read_text()); episode = data["episodes"][0]
        if mutation == "reset":
            episode["reset"]["backend_seed"] += 1
        elif mutation == "severity":
            episode["intervention_severity"][1]["pixels"] += 1
        else:
            episode["initial_state_id" if mutation == "state" else "seed"] += 1
        artifacts["summary"] = dump(target, data)
    elif mutation in {"progression", "profile"}:
        target = Path(artifacts["case_plan"]["path"])
        data = json.loads(target.read_text())
        if mutation == "progression":
            data["reset_progression_state_ids"] = [157]
        else:
            data["compiler_profile"]["profile_id"] = "native_trained_v2"
        artifacts["case_plan"] = dump(target, data)
    elif mutation == "noop":
        target = Path(artifacts["no_op"]["path"])
        data = json.loads(target.read_text()); data["max_abs_difference"] = .001
        artifacts["no_op"] = dump(target, data)
    elif mutation == "reference_path":
        old = Path(row["reference"]["actions"]["path"])
        substitute = old.with_name("substituted_actions.npz")
        substitute.write_bytes(old.read_bytes())
        row["reference"]["actions"] = resume.record(substitute)
    elif mutation == "duplicate_case":
        value["comparisons"][-1] = copy.deepcopy(value["comparisons"][0])
    else:
        value["source_input_sha256"][str(ROOT / "scripts/run_vla_trace_interventions_gpu6_resume_v1.py")] = "0" * 64
    dump(path, value)
    with pytest.raises(ValueError):
        resume.verify_resume_proof(path, run=run)


def test_malformed_masks_are_not_structural_missingness():
    masks = {f"{cam}__{role}": np.zeros((224, 224), bool)
             for cam in ("image", "wrist_image") for role in ("target", "unowned_background")}
    masks["wrist_image__unowned_background"] = np.zeros((1, 224), bool)
    with pytest.raises(ValueError, match="Malformed"):
        resume.strict_support({"diagnostic_masks": masks}, "random:background_fill", "target")
    del masks["wrist_image__target"]
    with pytest.raises(KeyError):
        resume.strict_support({"diagnostic_masks": masks}, "random:mosaic", "target")


def test_both_cameras_require_distinct_background_support():
    masks = {f"{cam}__{role}": np.zeros((224, 224), bool)
             for cam in ("image", "wrist_image") for role in ("target", "unowned_background")}
    for cam in ("image", "wrist_image"):
        masks[cam + "__target"].ravel()[:3] = True
        masks[cam + "__unowned_background"].ravel()[3:6] = True
    resume.strict_support({"diagnostic_masks": masks}, "random:mosaic", "target")
    masks["wrist_image__unowned_background"].ravel()[5] = False
    with pytest.raises(resume.StructuralNA) as error:
        resume.strict_support({"diagnostic_masks": masks}, "random:mosaic", "target")
    assert error.value.support[0]["exact_match_possible"] is True
    assert error.value.support[1]["requested_pixels"] == 3
    assert error.value.support[1]["available_pixels"] == 2


def test_malformed_journal_and_parent_cleanup_error_do_not_skip_groups(tmp_path):
    import os
    import signal
    import time

    output = tmp_path / "queue"
    output.mkdir()
    destination = output / "policy"
    process = SimpleNamespace(pid=712345, poll=lambda: None)
    groups = {712345, 712346}
    killed, streams = [], []
    identity = {"pid": 712345, "start_ticks": 1, "pgid": 712345, "ppid": 5, "state": "S"}

    def popen(*args, **kwargs):
        streams.append(kwargs["stdout"])
        destination.mkdir()
        descendant = {"pid": 712346, "start_ticks": 2, "pgid": 712346, "ppid": 712345, "state": "S"}
        (destination / "owned_processes.jsonl").write_text("{malformed}\n" + json.dumps(descendant) + "\n")
        return process

    def killpg(group, sig):
        killed.append((group, sig))
        groups.discard(group)

    def failing_stop(_):
        raise RuntimeError("An earlier parent cleanup failed")

    namespace = {"Path": Path, "json": json, "time": time, "ROOT": ROOT, "signal": signal,
        "os": SimpleNamespace(environ=os.environ, getpgrp=os.getpgrp, killpg=killpg),
        "subprocess": SimpleNamespace(Popen=popen, STDOUT=-2),
        "prior": SimpleNamespace(native=SimpleNamespace(stop_children=failing_stop)),
        "process_identity": lambda pid: identity if pid == process.pid else None,
        "group_gone": lambda group: group not in groups, "write": resume.write}
    run_children = function("run_children", ROOT / "scripts/queue_vla_trace_gpu6_resume_v1.py", namespace)
    job = {"name": "policy", "destination": str(destination), "environment": {}, "command": ["never-executed"]}
    with pytest.raises(ValueError):
        run_children([job], ["8,9"], output, lambda: None, lambda _: None)
    assert not groups
    assert {group for group, _ in killed} == {712345, 712346}
    assert all(stream.closed for stream in streams)


def test_unjournaled_child_escalates_and_reaps_before_raising():
    import io
    import os
    import subprocess

    events = []

    class Journal:
        def open(self, mode):
            if mode == "x":
                return io.StringIO()
            raise OSError("journal write unavailable")

    class Output:
        def __truediv__(self, name):
            return Journal()

    def wait(timeout):
        events.append(("wait", timeout))
        if timeout == 15:
            raise subprocess.TimeoutExpired("owned child", timeout)
        return 0

    child = SimpleNamespace(pid=os.getpid(), terminate=lambda: events.append("terminate"),
        kill=lambda: events.append("kill"), wait=wait)
    subprocess_double = SimpleNamespace(Popen=lambda *a, **k: child, TimeoutExpired=subprocess.TimeoutExpired)
    namespace = {"subprocess": subprocess_double, "Path": Path, "json": json, "os": os}
    install = function("install_process_journal", ROOT / "scripts/run_vla_trace_interventions_gpu6_resume_v1.py", namespace)
    install(Output())
    with pytest.raises(OSError, match="journal"):
        subprocess_double.Popen(["never-executed"])
    assert events == ["terminate", ("wait", 15), "kill", ("wait", 5)]
