"""Persist live progress and GPU telemetry; refresh reports from completed data."""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rl_vla.diagnostics.causal_support_overlay_v1 import operational_snapshot


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def lines(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def snapshot(root):
    tasks = {}
    for key in ("object_0", "object_1", "object_2"):
        task, method = root/"tasks"/key, root/"methods/rldt"/key
        metrics = lines(method/"training/metrics.jsonl")
        status = read(method/"status.json", {"stage": "preparation_or_smoke", "state": "pending"})
        current = status.get("stage", "")
        episodes = lines(method/f"{current}.episodes.jsonl")
        if not episodes:
            episodes = [episode for shard in (method/f"{current}.shards").glob("shard_*/report.episodes.jsonl")
                        for episode in lines(shard)]
        online = read(method/"checkpoint_validation_status.json")
        if online and online.get("state") == "running":
            iteration = online["iteration"]
            online["episodes_completed"] = sum(len(lines(shard)) for shard in
                (method/f"validation_{iteration:06d}.shards").glob("shard_*/report.episodes.jsonl"))
        tasks[key] = {"preparation": read(task/"preparation_complete.json"),
                      "generation": read(task/"generation_progress.json"),
                      "smoke": read(root/"smoke"/key/"verification.json", {}).get("passed", False),
                      "status": status,
                      "completed_iterations": metrics[-1]["iterations"] if metrics else 0,
                      "training_counters": metrics[-1] if metrics else None,
                      "current_evaluation_episodes_completed": len(episodes),
                      "online_validation": online,
                      "selection": read(method/"selection.json")}
    return {"updated": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "tasks": tasks, "gate_b": read(root/"gate_b_actual/gate_summary.json"),
            "operational_health": operational_snapshot(root),
            "report": str(root/"report.md")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.run.resolve()
    last_analysis = 0.
    previous_alerts = None
    while True:
        value = snapshot(root)
        temporary = root/"progress.tmp"
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
        temporary.replace(root/"progress.json")
        alerts = value["operational_health"]["failures"]
        signature = json.dumps(alerts, sort_keys=True)
        if signature != previous_alerts:
            alert = {"updated": value["updated"], "failed_queues": len(alerts), "failures": alerts}
            temporary = root/"operational_alerts.tmp"
            temporary.write_text(json.dumps(alert, indent=2)+"\n")
            temporary.replace(root/"operational_alerts.json")
            with (root/"operational_alerts.jsonl").open("a") as stream:
                stream.write(json.dumps(alert)+"\n")
            print(json.dumps({"operational_alert": alert}), flush=True)
            previous_alerts = signature
        telemetry = subprocess.check_output(["nvidia-smi", "--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw", "--format=csv,noheader"], text=True)
        with (root/"gpu_telemetry.csv").open("a") as stream:
            stream.write(telemetry)
        if time.monotonic()-last_analysis > 300 or args.once:
            result = subprocess.run([str(Path(__file__).resolve().parents[1]/".venv/bin/python"),
                str(Path(__file__).with_name("analyze_vla_trace_objects.py")), "--run", str(root)],
                capture_output=True, text=True, check=False)
            with (root/"report_refresh.log").open("a") as stream:
                stream.write(value["updated"]+"\n"+result.stdout+result.stderr)
                audit = subprocess.run([str(Path(__file__).resolve().parents[1]/".venv/bin/python"),
                    str(Path(__file__).with_name("audit_vla_trace_completion.py")), "--run", str(root)],
                    capture_output=True, text=True, check=False)
                stream.write(audit.stdout+audit.stderr)
            last_analysis = time.monotonic()
        if args.once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
