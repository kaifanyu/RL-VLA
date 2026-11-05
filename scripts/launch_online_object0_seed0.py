#!/usr/bin/env python
"""Seed0-only three-method pilot admission/queue. Dry-run is the default."""
import argparse, json, os, pathlib, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1]


def plan(gpus, concurrency, cpus):
    if len(set(gpus))!=len(gpus) or not gpus:raise ValueError('Assign distinct GPU IDs')
    if not 1<=concurrency<=min(3,len(gpus)):raise ValueError('Concurrency must fit the assigned GPUs')
    if len(set(cpus))!=len(cpus):raise ValueError('CPU slots must not overlap')
    if len(cpus)<8*concurrency:raise ValueError('Need eight distinct allowed CPUs per concurrent job')
    slots=[{'gpu':gpus[i],'cpus':','.join(map(str,cpus[i*8:(i+1)*8]))} for i in range(concurrency)]
    return {'seed':0,'methods':['dice','rldt','flashsac'],'slots':slots,'max_concurrent':concurrency,'optional_retention':False,'optional_best_of16':False,'host_ram_reservation_gib':96+48+32 if concurrency==3 else (144 if concurrency==2 else 96),'workspace_growth_reservation_gib':100,'gpu_per_job':1}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpus',required=True);p.add_argument('--max-concurrent',type=int,default=1);p.add_argument('--output-root',type=pathlib.Path,required=True);p.add_argument('--execute',action='store_true',help='Explicit production launch; omit for planning only');p.add_argument('--cpus',help='Comma-separated assigned CPU IDs, otherwise current affinity');a=p.parse_args()
    cpus=list(map(int,a.cpus.split(','))) if a.cpus else sorted(os.sched_getaffinity(0))
    if not set(cpus)<=os.sched_getaffinity(0):p.error('CPU assignment exceeds current affinity')
    result=plan(a.gpus.split(','),a.max_concurrent,cpus)
    result['command_template']=[str(ROOT/'third_party/openpi/.venv/bin/python'),str(ROOT/'scripts/launch_online_object0.py'),'--method','METHOD','--seed','0','--gpu','ASSIGNED_GPU','--output-root',str(a.output_root.resolve())]
    print(json.dumps(result,indent=2),flush=True)
    if not a.execute:return
    readiness=json.loads((ROOT/'docs/online-object0/readiness.json').read_text())
    if not readiness['production_ready']:raise RuntimeError('Production readiness gate is closed; see docs/online-object0/readiness.json')
    for slot in result['slots']:
        used=int(subprocess.check_output(['nvidia-smi','--id',slot['gpu'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>1024:raise RuntimeError(f"Assigned GPU{slot['gpu']} is occupied ({used}MiB); preserving that job")
    used=int(subprocess.check_output(['du','-sb','/workspace'],text=True).split()[0])/2**30
    import shutil
    available=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/2**20
    if used+result['workspace_growth_reservation_gib']>=2300 or shutil.disk_usage(a.output_root.parent).free<100*2**30:raise RuntimeError('Insufficient output/atomic-save headroom below existing guard')
    if available<result['host_ram_reservation_gib']:raise RuntimeError('Insufficient shared host RAM; request fewer concurrent jobs')
    a.output_root.mkdir(parents=True,exist_ok=False)
    (a.output_root/'schedule.json').write_text(json.dumps(result,indent=2)+'\n')
    queue=list(result['methods']);active=[];outcomes=[]
    while queue or active:
        for slot in result['slots']:
            if not queue or any(x['slot']==slot for x in active):continue
            method=queue.pop(0);env=os.environ.copy();env.update(RLDT_CPUSET=slot['cpus'],DICE_CPUSET=slot['cpus'],RL_VLA_CPUSET=slot['cpus'],LP_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            cmd=[str(ROOT/'third_party/openpi/.venv/bin/python'),str(ROOT/'scripts/launch_online_object0.py'),'--method',method,'--seed','0','--gpu',slot['gpu'],'--output-root',str(a.output_root)]
            log=(a.output_root/f'{method}_launcher.log').open('x');proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active.append({'method':method,'slot':slot,'process':proc,'log':log,'pid':proc.pid})
        for item in list(active):
            code=item['process'].poll()
            if code is not None:
                item['log'].close();outcomes.append({'method':item['method'],'pid':item['pid'],'gpu':item['slot']['gpu'],'exit_code':code});active.remove(item)
                (a.output_root/'outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n')
        time.sleep(5)
    if any(x['exit_code'] for x in outcomes):raise SystemExit(1)


if __name__=='__main__':main()
