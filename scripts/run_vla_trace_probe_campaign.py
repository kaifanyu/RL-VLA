"""Wait for immutable feature banks and run diagnostic probes once per task/model."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(root):
    root=Path(root).resolve()
    pending={(task,model) for task in ("object_0","object_1","object_2") for model in ("pi05_base","pi05_libero")}
    while pending:
        for task,model in sorted(pending):
            output=root/"probes"/task/model
            if (output/"metrics.json").exists():
                result=json.loads((output/"metrics.json").read_text())
                if not result["complete"]:
                    raise ValueError(f"Existing incomplete probe output needs inspection: {output}")
                pending.remove((task,model));continue
            features=root/"features"/model/task
            source=root/"features_retention"/model/task/"source"
            required=[features/s/"summary.json" for s in ("train","validation","test")]+[source/"summary.json"]
            required.append(root/"methods/rldt"/task/"selection.json")
            if not all(path.exists() for path in required):
                continue
            if output.exists():
                raise FileExistsError(f"Incomplete previous probe attempt preserved: {output}")
            command=[sys.executable,"scripts/run_vla_trace_probes.py","--task",task,"--model",model,
                     "--features",str(features),"--bank",str(root/"observation_bank"/task),
                     "--source-features",str(source),"--output",str(output)]
            print(json.dumps({"stage":"fitting_probes","task":task,"model":model,"command":command}),flush=True)
            log_path=root/f"probes_{task}_{model}.log"
            with log_path.open("x") as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            pending.remove((task,model))
        temporary=root/"probe_campaign_status.tmp"
        temporary.write_text(json.dumps({"complete":not pending,"pending":sorted(pending)},indent=2)+"\n")
        temporary.replace(root/"probe_campaign_status.json")
        if pending:
            time.sleep(15)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    args=parser.parse_args()
    run(args.run)
