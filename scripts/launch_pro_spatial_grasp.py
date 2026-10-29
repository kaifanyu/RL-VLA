"""Freeze checked inputs and start a detached Slurm step in an existing allocation."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def launch(experiment, allocation):
    if (experiment / "launch.json").exists() or (experiment / "status.json").exists():
        raise FileExistsError("Experiment already launched")
    report = json.loads((experiment / "setup_check.json").read_text())
    if report["states_checked"] != 250:
        raise ValueError("All 250 train/validation/test/source resets must pass")
    paths = list((ROOT / "src").rglob("*.py"))
    paths += list((ROOT / "third_party/openpi/src/openpi").rglob("*.py"))
    paths += list((ROOT / "third_party/libero_pro/libero").rglob("*.py"))
    paths += [ROOT / name for name in (
        "scripts/env.sh", "scripts/run_rldt.sh", "scripts/run_pro_spatial_grasp.py",
        "scripts/run_pro_spatial_grasp.sh", "scripts/analyze_rldt.py",
        ".cache/libero-pro/config.yaml", "runs/setup/pi05_download.json",
        "checkpoints/openpi/pi05_libero/assets/physical-intelligence/libero/norm_stats.json")]
    paths += [experiment / name for name in ("protocol.json", "pro_states.npy", "generation.json",
                                             "assets.json", "setup_check.json")]
    paths += list(experiment.glob("*.toml"))
    repositories = ["third_party/openpi", "third_party/libero_pro",
                    "third_party/openpi/third_party/libero"]
    contract = {"files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(paths)},
                "repositories": {p: subprocess.check_output(
                    ["git", "-C", str(ROOT / p), "rev-parse", "HEAD"], text=True).strip()
                    for p in repositories}}
    (experiment / "launch_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    subprocess.run(["tar", "--exclude=__pycache__", "--exclude=*.pyc", "-czf",
                    str(experiment / "source_snapshot.tar.gz"), "src", "scripts", "configs",
                    "tests", "pyproject.toml"], cwd=ROOT, check=True)
    for name, command in (("source_commit.txt", ["git", "rev-parse", "HEAD"]),
                          ("source_changes.patch", ["git", "diff", "--binary"])):
        (experiment / name).write_bytes(subprocess.check_output(command, cwd=ROOT))
    command = ["srun", f"--jobid={allocation}", "--overlap", "--nodes=1", "--ntasks=1",
               "--gres=gpu:1", "--cpus-per-task=8", "--time=24:00:00",
               "--job-name=pro-spatial0-rldt", "bash", "scripts/run_pro_spatial_grasp.sh",
               str(experiment)]
    with (experiment / "launcher.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"allocation_job_id": allocation, "launcher_pid": process.pid,
              "experiment": str(experiment), "command": command}
    (experiment / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--allocation", required=True, type=int)
    args = parser.parse_args()
    launch(args.experiment.resolve(), args.allocation)
