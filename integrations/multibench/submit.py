"""Submit six isolated sequential GPU jobs only after the frozen readiness gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from multibench.environment import ROOT
from multibench.run_experiment import verify_contract, write


def submit(campaign, contract, dependency=None):
    campaign,contract=Path(campaign).resolve(),Path(contract).resolve()
    verify_contract(contract)
    receipt=campaign/'submission.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    jobs=[]
    plan=json.loads((campaign/'plan.json').read_text())
    for directory in plan['experiments']:
        output=Path(directory)
        script=output/'job.sh'
        command=[str(ROOT/'.venv/bin/python'),'-m','multibench.run_experiment',
                 '--experiment',str(output),'--contract',str(contract)]
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'
            +'export PYTHONPATH='+shlex.quote(str(ROOT/'integrations/multibench/core')+':'+str(ROOT/'integrations'))+'\n'
            +'exec '+shlex.join(command)+'\n')
        args=['sbatch','--parsable','--job-name',f'mb-{output.name}', '--account=gu-account',
              '--partition=gu-compute','--qos=gu-low','--nodes=1','--ntasks=1',
              '--gres=gpu:nvidia_rtx_pro_6000_blackwell_server_edition:1','--constraint=b6000',
              '--cpus-per-task=8','--mem=128G','--time=4-00:00:00','--no-requeue',
              '--chdir',str(ROOT),'--output',str(output/'slurm_%j.log')]
        if dependency:
            args+=['--dependency',f'afterany:{dependency}']
        job=subprocess.check_output(args+[str(script)],text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError(f'Unexpected sbatch response: {job}')
        jobs.append({'job_id':job,'experiment':str(output),'dependency':dependency})
        write(receipt,{'state':'submitting','jobs':jobs})
        dependency=job
    write(receipt,{'state':'submitted','jobs':jobs})
    return jobs

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True)
    p.add_argument('--contract',required=True)
    p.add_argument('--dependency')
    a=p.parse_args()
    print(json.dumps(submit(a.campaign,a.contract,a.dependency),indent=2))
