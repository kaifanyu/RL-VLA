"""Synthetic consumer fixtures; these are not measured model evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

import queue_vla_trace_gpu6_interventions_v1 as queue


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    campaign = tmp_path / "campaign"
    directory = campaign / "smoke"
    profiles = {role: {"manifest_path": str(campaign / "compiler_profiles" / role / "manifest.json"),
                       "manifest_sha256": role + "manifest", "profile_sha256": role + "profile",
                       "helper_sha256": "helper"} for role in ("baseline", "trained")}
    monkeypatch.setattr(queue, "observed_device", lambda: {"uuid": "test-device"})
    checkpoints = ["pi05_libero", *[str(campaign / "methods/rldt/object_1/training" / f"iteration_{i:06d}")
                                    for i in (25, 50, 100)]]
    sources = [queue.CHILD, queue.ROOT / "src/rl_vla/diagnostics/pi05_attention.py",
               queue.ROOT / "src/rl_vla/diagnostics/interventions.py",
               queue.ROOT / "src/rl_vla/diagnostics/compiler_profile_gpu6_v1.py"]
    write(directory / "plan.json", {"source_input_sha256": {str(p): queue.native.file_hash(p) for p in sources}})
    release = {"physical_gpu_lock_acquired_after_smoke": True, "live_worker_pids": []}
    write(directory / "release_verification.json", release)
    noop = {"passed": True, "max_abs_difference": 0., "persisted_noise_recurrence_max_abs_difference": 0.}
    reset = {"object_telemetry": {"simulator_state": [0., 0., 0.]}}
    native_replays = []
    for worker, checkpoint in enumerate(checkpoints):
        trajectory = campaign / "native" / str(worker)
        trajectory.mkdir(parents=True)
        chunks = []
        for step in (1, 2):
            chunk = trajectory / f"chunk_{step:06d}.npz"
            np.savez(chunk, normalized_actions=np.full((10, 32), worker, np.float32),
                     executed_actions=np.zeros((5, 7), np.float32),
                     metadata_json=json.dumps({"step_infos": [{"object_telemetry": {"simulator_state": [0., 0., 0.]}}] * 5}))
            chunks.append({"path": str(chunk), "sha256": queue.native.file_hash(chunk)})
        report = trajectory / "report.json"
        write(report, {"checkpoint": None if worker == 0 else checkpoint,
                       "episodes": [{"trajectory": str(trajectory), "reset": reset}]})
        native_replays.append({"worker": worker, "policy": ("pi05_libero", "rldt_25", "rldt_50", "rldt_100")[worker],
                               "checks": {key: True for key in ("normalized_actions_exact", "executed_actions_exact", "simulator_states_exact", "reset_metadata_exact")},
                               "recorded_report": str(report), "report_sha256": queue.native.file_hash(report),
                               "source_chunks": chunks})
        write(directory / f"worker_{worker}/case_plan.json", {
            "checkpoint": checkpoint, "state_ids": [150], "seeds": [100000],
            "conditions": list(queue.native.CONDITIONS), "technical_smoke": True,
            "smoke_decisions": 2, "source_sha256": {}})
    rows = []
    for condition in queue.native.CONDITIONS:
        hashes = []
        for worker, checkpoint in enumerate(checkpoints):
            folder = directory / f"worker_{worker}" / condition.replace(":", "_")
            folder.mkdir(parents=True)
            action = folder / "episode_000_actions.npz"
            np.savez(action, normalized=np.full((2, 10, 32), worker, np.float32))
            hashes.append(queue.native.file_hash(action))
            write(folder / "no_op.json", noop)
            write(folder / "episode_000_telemetry.json", [{"action": [0.] * 7, "simulator_state": [0.] * 3}] * 10)
            video = folder / "episode_000.mp4"
            video.write_bytes(b"synthetic test placeholder, never a research video")
            write(folder / "summary.json", {"completed": True, "technical_smoke": True,
                "episodes": [{"success": None, "initial_state_id": 150, "seed": 100000,
                              "primitive_steps": 10, "checkpoint": checkpoint, "reset": reset, "video": str(video)}]})
        rows.append({"condition": condition, "four_workers_finite": True, "native_no_op": [noop] * 4,
                     "action_artifact_sha256": hashes,
                     "none_recorded_native_replay": native_replays if condition == "none" else []})
    write(directory / "summary.json", {"complete": True, "passed": True, "workers": 4,
         "memory_fraction_per_worker": .2, "peak_gpu_processes": 4, "physical_gpu_lock_released": True,
         "gpu_uuid": "test-device", "compiler_profiles": profiles, "loaded_experts": checkpoints,
         "release_verification_sha256": queue.native.file_hash(directory / "release_verification.json"),
         "conditions": rows})
    return directory, profiles


@pytest.mark.parametrize("mutation", ["actions", "no_op", "release", "telemetry"])
def test_saved_concurrency_output_mutations_fail_closed(evidence, mutation):
    directory, profiles = evidence
    bindings = queue.concurrency_contract(directory, profiles)
    folder = directory / "worker_0/none"
    assert str(folder / "episode_000_actions.npz") in bindings
    assert str(folder / "episode_000_telemetry.json") in bindings
    if mutation == "actions":
        with (folder / "episode_000_actions.npz").open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "no_op":
        write(folder / "no_op.json", {"passed": False})
    elif mutation == "release":
        path = directory / "release_verification.json"
        value = json.loads(path.read_text())
        value["changed_after_proof"] = True
        write(path, value)
    else:
        path = folder / "episode_000_telemetry.json"
        value = json.loads(path.read_text())
        value[0]["simulator_state"][0] = .25
        write(path, value)
    with pytest.raises(ValueError):
        queue.concurrency_contract(directory, profiles)
