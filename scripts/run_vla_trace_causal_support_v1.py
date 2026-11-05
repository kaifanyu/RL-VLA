"""Wait for accepted strict-support raw coverage, then publish CPU-only analysis."""
import argparse
import os
import time
from pathlib import Path

from rl_vla.diagnostics import causal_support_v1 as analysis
from rl_vla.diagnostics.causal_structural_validation_v1 import publish_execution, validate_execution


def run(root, wait, poll):
    root=Path(root).resolve()
    sources=[Path(__file__).resolve(),Path(analysis.__file__).resolve(),
             Path(analysis.__file__).with_name('causal_structural_validation_v1.py'),
             Path(analysis.__file__).with_name('metrics.py'),
             Path(__file__).with_name('analyze_vla_trace_causal.py')]
    contract={str(path):analysis.sha(path) for path in sources}
    output=root/analysis.ANALYSIS
    output.mkdir(parents=True,exist_ok=True)
    launch=output/'consumer_launch.json'
    if launch.exists():raise FileExistsError(launch)
    analysis.atomic_json(launch,{'pid':os.getpid(),'source_sha256':contract,'scope':'CPU-only support-aware execution audit and causal figures; original producers/artifacts unchanged'})
    while True:
        if any(analysis.sha(path)!=digest for path,digest in contract.items()):raise ValueError('Live support-aware consumer source changed')
        certificate=validate_execution(root)
        if certificate is None:
            analysis.atomic_json(output/'status.json',{'complete':False,'state':'waiting_for_raw_execution','passed':None})
            if not wait:return
            time.sleep(poll)
            continue
        certificate=publish_execution(root,certificate)
        for policy in analysis.POLICIES:
            status=analysis.verify_policy(root,policy,full_hash=True)
            if status['status']=='failed':raise ValueError(status)
            if not status['complete']:analysis.analyze_policy(root,policy,certificate)
        cells=[analysis.verify_policy(root,policy,full_hash=True) for policy in analysis.POLICIES]
        if not all(cell['complete'] for cell in cells):raise ValueError(cells)
        analysis.atomic_json(output/'summary.json',{'complete':True,'passed':True,'task':'object_1','cells':cells,
                             'execution_coverage':analysis.binding(root/'compiler_profile_integration/causal_structural_na_v1/execution_coverage.json'),
                             'source_sha256':contract})
        analysis.atomic_json(output/'status.json',{'complete':True,'passed':True,'state':'completed'})
        return


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--poll',type=float,default=30.)
    args=parser.parse_args()
    try:run(args.run,args.wait,args.poll)
    except BaseException as error:
        analysis.atomic_json(args.run/analysis.ANALYSIS/'status.json',{'complete':False,'passed':False,'state':'failed','error':repr(error)})
        raise
