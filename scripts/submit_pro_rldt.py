"""Freeze execution inputs and submit baseline -> RLDT job arrays to PARCC."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def submit(experiment):
    experiment = experiment.resolve()
    if (experiment / "jobs.json").exists():
        raise FileExistsError("Jobs already submitted for this experiment; inspect jobs.json")
    report = json.loads((experiment / "setup_check/report.json").read_text())
    if not report.get("passed") or len(report["checks"]) != 6:
        raise RuntimeError("All six real simulator checks must pass before submission")
    paths = list((ROOT / "src").rglob("*.py"))
    paths += list((ROOT / "third_party/openpi/src/openpi").rglob("*.py"))
    paths += list((ROOT / "third_party/libero_pro/libero").rglob("*.py"))
    paths += [ROOT / relative for relative in (
        "scripts/env.sh", "scripts/run_rldt.sh", "scripts/run_pro_rldt.py",
        "scripts/run_pro_rldt.slurm", ".cache/libero-pro/config.yaml",
        "runs/setup/pi05_download.json",
        "checkpoints/openpi/pi05_libero/assets/physical-intelligence/libero/norm_stats.json",
    )]
    paths += [experiment / "protocol.json"] + list(experiment.glob("*/*.toml"))
    contract = {"files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(paths)}}
    (experiment / "launch_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    jobs = {"experiment": str(experiment), "tasks": {"0": "spatial_0", "1": "object_0"}}

    def launch(phase, options):
        command = ["sbatch", "--parsable", *options, "scripts/run_pro_rldt.slurm",
                   str(experiment), phase]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        (experiment / f"submit_{phase}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Submission failed: {result.stdout}{result.stderr}")
        ids = [line.split(";")[0] for line in result.stdout.splitlines()
               if re.fullmatch(r"\d+(;[^\s]+)?", line)]
        if len(ids) != 1:
            raise RuntimeError(f"Could not identify submitted job: {result.stdout}")
        jobs[phase] = {"array_job_id": ids[0], "command": command}
        (experiment / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        print(f"Submitted {phase} array {ids[0]}", flush=True)
        return ids[0]

    baseline = launch("baseline", ["--job-name=pi05-pro-base", "--partition=b200-mig45",
                                   "--cpus-per-task=6", "--mem-per-cpu=8G", "--time=04:00:00"])
    launch("train", ["--job-name=pi05-pro-rldt", f"--dependency=aftercorr:{baseline}",
                     "--kill-on-invalid-dep=yes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    submit(parser.parse_args().experiment)
