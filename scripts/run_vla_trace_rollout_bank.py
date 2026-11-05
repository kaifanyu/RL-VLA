"""Export fixed first-five-episode phase banks after checkpoint selection is locked."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

POLICIES={"pi05_libero":"baseline_test.json","rldt_000025":"checkpoint_000025_test.json",
          "rldt_000050":"checkpoint_000050_test.json","rldt_000100":"checkpoint_000100_test.json"}


def write_json(path, value):
    """Consumers may observe the destination only after a complete JSON write."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(root,task):
    root=Path(root).resolve()
    destination=root/"rollout_observation_bank"/task
    destination.mkdir(parents=True,exist_ok=True)
    pending=set(POLICIES)
    while pending:
        for policy in sorted(pending):
            summary=destination/policy/"summary.json"
            if summary.exists() and json.loads(summary.read_text()).get("complete"):
                pending.remove(policy);continue
            selection=root/"methods/rldt"/task/"selection.json"
            gate=root/"moving_mask_alignment_gate.json"
            if not selection.exists() or not gate.exists():continue
            if not json.loads(gate.read_text()).get("passed"):continue
            selected=json.loads(selection.read_text())
            filename=POLICIES[policy]
            if policy.startswith("rldt_") and int(policy.split("_")[1])==selected["iteration"]:
                filename="selected_test.json"
            report=root/"methods/rldt"/task/filename
            if not report.exists():continue
            data=json.loads(report.read_text())
            if len(data.get("episodes",[]))!=50:raise ValueError("Final test report must contain all50 episodes")
            episodes=data["episodes"][:5]
            if [row["reset"]["initial_state_id"] for row in episodes]!=list(range(150,155)):
                raise ValueError("Predetermined first5 held-out IDs changed")
            outputs=[]
            for row in episodes:
                output=destination/policy/f"state_{row['reset']['initial_state_id']:06d}"
                if not (output/"summary.json").exists():
                    output.parent.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,"scripts/extract_vla_trace_rollout_bank.py","--protocol",str(root/"tasks"/task/"protocol.json"),
                             "--task",task,"--split","test","--episode",row["trajectory"],"--output",str(output),"--policy",policy]
                    if data["checkpoint"]:command.extend(["--checkpoint",str(data["checkpoint"])])
                    with output.with_suffix(".log").open("x") as stream:
                        subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
                value=json.loads((output/"summary.json").read_text())
                if not value.get("complete"):raise ValueError("Phase export is incomplete")
                outputs.append(str(output))
            write_json(summary,{"complete":True,"task":task,"policy":policy,"episodes":5,"banks":outputs,
                "selection_sha256":hashlib.sha256(selection.read_bytes()).hexdigest(),"selection":selected,
                "selection_rule":"First5 test IDs150–154, no outcome or attention filtering; phases fixed by predeclaration",
                "report_sha256":hashlib.sha256(report.read_bytes()).hexdigest()})
            pending.remove(policy)
        write_json(destination/"status.json",{"complete":not pending,"pending":sorted(pending)})
        if pending:time.sleep(30)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--task",required=True,choices=["object_0","object_1","object_2"])
    args=parser.parse_args();run(args.run,args.task)
