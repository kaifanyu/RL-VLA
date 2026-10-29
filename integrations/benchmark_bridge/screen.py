"""Launch/resume a native-JAX baseline screen, releasing the GPU after every cell."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
import tomllib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_toml(path, data):
    lines = []

    def section(value, prefix=()):
        if prefix:
            lines.extend(["", "[" + ".".join(prefix) + "]"])
        for key, item in value.items():
            if not isinstance(item, dict):
                lines.append(f"{key} = {json.dumps(item)}")
        for key, item in value.items():
            if isinstance(item, dict):
                section(item, prefix + (key,))

    section(data)
    Path(path).write_text("\n".join(lines) + "\n")


def freeze_execution(repo, screen):
    snapshot = screen / "execution"
    snapshot.mkdir(exist_ok=False)
    shutil.copytree(repo / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(repo / "integrations/benchmark_bridge", snapshot / "integrations/benchmark_bridge",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for directory in ("scripts", "configs"):
        (snapshot / directory).mkdir()
    for name in ("scripts/env.sh", "scripts/run_rldt.sh", "configs/rldt_pi05_libero.toml"):
        shutil.copy2(repo / name, snapshot / name)
    for name in (".venv", ".cache", "third_party", "checkpoints"):
        (snapshot / name).symlink_to(repo / name, target_is_directory=True)
    files = [path for path in snapshot.rglob("*") if path.is_file() and not path.is_symlink()]
    contract = {"files": {str(path.relative_to(snapshot)): digest(path) for path in files},
                "screen_plan_sha256": digest(screen / "screen_plan.json"),
                "source_git_head": subprocess.check_output(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()}
    write_json(snapshot / "contract.json", contract)
    return snapshot


def verify_execution(snapshot, screen):
    contract = json.loads((snapshot / "contract.json").read_text())
    if digest(screen / "screen_plan.json") != contract["screen_plan_sha256"]:
        raise ValueError("Screen plan changed after launch")
    for name, expected in contract["files"].items():
        if digest(snapshot / name) != expected:
            raise ValueError(f"Frozen screening source changed: {name}")


def make_config(snapshot, screen, case, ready_file):
    with (snapshot / "configs/rldt_pi05_libero.toml").open("rb") as stream:
        config = tomllib.load(stream)
    plan = json.loads((screen / "screen_plan.json").read_text())
    config["policy"] = {"kind": "pi05", "checkpoint": plan["checkpoint"],
                        "model_config": "pi05_libero", "projection_only": False}
    config["env"] = {"kind": "external", "python": case["python"],
                     "factory": "benchmark_bridge.environment:make_env", "kwargs": {
                         "ready_file": str(ready_file), "output": str(screen),
                     }}
    config["rldt"].update(ode_steps=10, execute_steps=5, gamma=.999)
    config["recording"] = {"enabled": True}
    config["screen"] = {"case_id": case["case_id"], "split": "development_only",
                        "ready_sha256": digest(ready_file), "native_jax_baseline": True}
    return config


def _frames(path, *, skip_initial=False):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        schema = metadata["observation_schema"]
        names = [next(name for name, keys in schema.items() if keys == ["openpi", key])
                 for key in ("observation/image", "observation/wrist_image")]
        left, right = archive[names[0]], archive[names[1]]
        for index in range(int(skip_initial), len(left)):
            yield np.concatenate((left[index], right[index]), axis=1)


def render_primitive_videos(report_path):
    """Reconstruct actual 20Hz videos from lossless NPZs after releasing the GPU."""
    import imageio.v2 as imageio

    report = json.loads(report_path.read_text())
    target = report_path.with_suffix(".primitive_videos")
    target.mkdir(exist_ok=True)
    for index, episode in enumerate(report["episodes"], 1):
        trajectory = Path(episode["trajectory"])
        video = target / f"episode_{index:03d}.mp4"
        frame_count = 0
        with imageio.get_writer(str(video), fps=20, codec="libx264", quality=8,
                                macro_block_size=None, ffmpeg_params=["-threads", "1"]) as writer:
            for frame in _frames(trajectory / "reset.npz"):
                writer.append_data(frame)
                frame_count += 1
            for chunk in sorted(trajectory.glob("chunk_*.npz")):
                for frame in _frames(chunk, skip_initial=True):
                    writer.append_data(frame)
                    frame_count += 1
        if frame_count != episode["primitive_steps"] + 1:
            raise ValueError("Rendered video does not contain every executed primitive step")
        episode["chunk_boundary_video"] = episode.get("video")
        episode["video"] = str(video)
        episode["video_note"] = "Every actual 20Hz front/wrist policy observation; reset plus "
        episode["video_note"] += "all executed steps, reconstructed from lossless trajectory NPZs."
    write_json(report_path, report)
    path = report_path.with_suffix(".episodes.jsonl")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n"
                                 for row in report["episodes"]))
    temporary.replace(path)
    return report


def rank_results(screen, plan):
    rows, errors = [], []
    for case in plan["cases"]:
        directory = screen / "results" / case["case_id"]
        status_path = directory / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status.get("state") == "failed":
            errors.append({"case_id": case["case_id"], "error": status.get("error"),
                           "log": str(directory / "evaluation.log")})
        report_path = directory / "baseline.json"
        if status.get("state") != "complete" or not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        episodes = report["episodes"]
        failures = [episode for episode in episodes if not episode["success"]]
        rows.append({
            "case_id": case["case_id"], "benchmark": case["benchmark"], "suite": case["suite"],
            "task_id": case["task_id"], "name": case["name"], "category": case["category"],
            "difficulty": case["difficulty"], "successes": len(episodes) - len(failures),
            "episodes": len(episodes), "success_rate": report["success_rate"],
            "failures": len(failures), "mean_primitive_steps": float(np.mean(
                [episode["primitive_steps"] for episode in episodes])), "report": str(report_path),
            "failed_episodes": [{"seed": episode["seed"], "reset": episode["reset"],
                                 "video": episode["video"], "trajectory": episode["trajectory"]}
                                for episode in failures],
        })
    rows.sort(key=lambda row: (row["success_rate"], row["case_id"]))
    result = {"purpose": "Development-only condition selection, not an official benchmark score",
              "episodes_per_cell": plan["episodes_per_cell"], "completed_cells": len(rows),
              "total_cells": len(plan["cases"]), "ranked_cells": rows, "errors": errors,
              "mixed_success_candidates": [row["case_id"] for row in rows
                                           if 0 < row["successes"] < row["episodes"]],
              "caveat": "Five fresh states per cell give preliminary screening evidence only. "
              "Selection requires new independently frozen adaptation/test states. "
              "No fine-tuning is launched automatically by this queue."}
    write_json(screen / "ranking.json", result)
    return result


def run_queue(repo, screen, allocation, ready_timeout=3600, case_timeout=3600):
    snapshot = screen / "execution"
    plan = json.loads((screen / "screen_plan.json").read_text())
    verify_execution(snapshot, screen)
    remaining = {case["case_id"]: case for case in plan["cases"]}
    if len(remaining) != len(plan["cases"]):
        raise ValueError("Duplicate screen case IDs")
    lock_path = repo / "runs" / f"gpu_allocation_{allocation}" / "experiment.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    (screen / "results").mkdir(exist_ok=True)
    wait_start = time.monotonic()

    def status(state, **details):
        write_json(screen / "screen_status.json", {
            "state": state, "remaining_cells": len(remaining), "allocation": allocation,
            "step_id": os.environ.get("SLURM_STEP_ID"), "host": socket.gethostname(),
            "updated": datetime.now(ZoneInfo("America/New_York")).isoformat(), **details,
        })

    while remaining:
        ready = [case for key, case in remaining.items()
                 if (screen / "cases" / key / "ready.json").is_file()]
        if not ready:
            status("waiting_for_prepared_cells")
            if time.monotonic() - wait_start > ready_timeout:
                status("failed", error="No additional prepared screen cells arrived within timeout")
                raise TimeoutError("Screen preparation wait exceeded configured timeout")
            time.sleep(15)
            continue
        case = ready[0]
        key = case["case_id"]
        directory = screen / "results" / key
        directory.mkdir(exist_ok=True)
        local_status = directory / "status.json"
        if local_status.exists() and json.loads(local_status.read_text()).get("state") in {
            "complete", "failed",
        }:
            del remaining[key]
            continue
        ready_file = screen / "cases" / key / "ready.json"
        prepared = json.loads(ready_file.read_text())
        if any(prepared.get(name) != case.get(name) for name in case):
            raise ValueError(f"Prepared case differs from frozen screening plan: {key}")
        config_path = directory / "baseline.toml"
        write_toml(config_path, make_config(snapshot, screen, prepared, ready_file))
        report_path = directory / "baseline.json"
        command = ["bash", str(snapshot / "scripts/run_rldt.sh"), "evaluate", "--config",
                   str(config_path), "--output", str(report_path), "--episodes", str(case["episodes"]),
                   "--seed", "100000", "--record-episodes", str(case["episodes"])]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(snapshot / "integrations") + ":" + str(snapshot / "src")
        environment["LIBERO_CONFIG_PATH"] = case["config_path"]
        environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.55"
        try:
            status("waiting_for_gpu", case_id=key, lock=str(lock_path))
            write_json(local_status, {"state": "waiting_for_gpu", "command": command})
            with lock_path.open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                verify_execution(snapshot, screen)
                status("evaluating", case_id=key)
                write_json(local_status, {"state": "evaluating", "command": command})
                with (directory / "evaluation.log").open("x") as log:
                    subprocess.run(command, cwd=snapshot, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=case_timeout)
            # The native learner process has exited and GPU memory is gone before
            # another job can acquire the lock. Video encoding needs only the CPU.
            status("rendering_primitive_videos", case_id=key)
            report = render_primitive_videos(report_path)
            write_json(local_status, {"state": "complete", "success_rate": report["success_rate"],
                                      "episodes": len(report["episodes"]), "command": command})
        except Exception as error:  # noqa: BLE001 - isolate failures to one development condition
            write_json(local_status, {"state": "failed", "error": repr(error), "command": command})
        del remaining[key]
        wait_start = time.monotonic()
        rank_results(screen, plan)
    result = rank_results(screen, plan)
    status("complete" if not result["errors"] else "complete_with_errors",
           completed_cells=result["completed_cells"], errors=len(result["errors"]))


def launch(repo, screen, allocation):
    launch_path = screen / "screen_launch.json"
    if launch_path.exists():
        raise FileExistsError(launch_path)
    snapshot = freeze_execution(repo, screen)
    command = ["srun", f"--jobid={allocation}", "--overlap", "--nodes=1", "--ntasks=1",
               "--gres=gpu:1", "--cpus-per-task=8", "--time=1-00:00:00", "--job-name=baseline-screen",
               "/usr/bin/env", f"PYTHONPATH={snapshot / 'integrations'}:{snapshot / 'src'}",
               str(repo / ".venv/bin/python"), "-m", "benchmark_bridge.screen", "queue",
               "--repo", str(repo), "--screen", str(screen), "--allocation", str(allocation)]
    with (screen / "screen_launcher.log").open("x") as log:
        process = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"allocation": allocation, "launcher_pid": process.pid, "command": command,
              "snapshot": str(snapshot), "plan_sha256": digest(screen / "screen_plan.json"),
              "gpu_lock_scope": "One case; policy process exits before unlocking", "episodes_per_cell": 5}
    write_json(launch_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["launch", "queue", "rank"])
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--allocation", type=int, default=1470624)
    parser.add_argument("--ready-timeout", type=int, default=3600)
    parser.add_argument("--case-timeout", type=int, default=3600)
    args = parser.parse_args()
    repo, screen = args.repo.resolve(), args.screen.resolve()
    if args.command == "launch":
        print(json.dumps(launch(repo, screen, args.allocation), indent=2), flush=True)
    elif args.command == "queue":
        run_queue(repo, screen, args.allocation, args.ready_timeout, args.case_timeout)
    else:
        print(json.dumps(rank_results(screen, json.loads((screen / "screen_plan.json").read_text())),
                         indent=2), flush=True)


if __name__ == "__main__":
    main()
