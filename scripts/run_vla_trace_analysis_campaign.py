"""Run substantive attention analysis once each complete trace grid is available."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from analyze_vla_trace_causal import intervention_policy


def completed(path):
    if not path.exists():return False
    data=json.loads(path.read_text())
    return bool(data.get("complete",data.get("completed",False)))


def run(root):
    root=Path(root).resolve()
    pending={"object_0","object_1","object_2"}
    causal_pending={(task,policy) for task in pending for policy in ("pi05_libero","rldt_000025","rldt_000050","rldt_000100")}
    probe_pending={(task,model) for task in pending for model in ("pi05_base","pi05_libero")}
    while pending or causal_pending or probe_pending:
        for task in sorted(pending):
            if completed(root/"trace_analysis"/task/"summary.json"):
                pending.remove(task);continue
            ready=root/"traces"/task/"summary.json"
            if not ready.exists():continue
            summary=json.loads(ready.read_text())
            if not summary.get("completed",summary.get("complete")):continue
            command=[sys.executable,"scripts/analyze_vla_trace_grid.py","--run",str(root),"--task",task]
            with (root/f"trace_analysis_{task}.log").open("x") as stream:
                subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
            pending.remove(task)
        for task,policy in sorted(causal_pending):
            if completed(root/"causal_analysis"/task/policy/"summary.json"):
                causal_pending.remove((task,policy));continue
            if not completed(root/"trace_analysis"/task/"summary.json") or not completed(root/"interventions"/task/intervention_policy(policy)/"summary.json"):continue
            command=[sys.executable,"scripts/analyze_vla_trace_causal.py","--run",str(root),"--task",task,"--policy",policy]
            with (root/f"causal_analysis_{task}_{policy}.log").open("x") as stream:
                subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
            causal_pending.remove((task,policy))
        for task,model in sorted(probe_pending):
            if completed(root/"probes_counterfactual"/task/model/"metrics.json"):
                probe_pending.remove((task,model));continue
            if not completed(root/"traces"/task/"summary.json") or not completed(root/"probes"/task/model/"metrics.json"):continue
            command=[sys.executable,"scripts/score_vla_trace_counterfactual_probes.py","--run",str(root),"--task",task,"--model",model]
            with (root/f"counterfactual_probes_{task}_{model}.log").open("x") as stream:
                subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
            probe_pending.remove((task,model))
        path=root/"trace_analysis_campaign_status.json"
        temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"complete":not pending and not causal_pending and not probe_pending,"pending":sorted(pending),"causal_pending":sorted(causal_pending),"counterfactual_probe_pending":sorted(probe_pending)},indent=2)+"\n")
        temporary.replace(path)
        if pending or causal_pending or probe_pending:time.sleep(15)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    args=parser.parse_args();run(args.run)
