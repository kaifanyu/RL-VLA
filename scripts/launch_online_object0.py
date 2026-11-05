#!/usr/bin/env python
"""One fresh uninterrupted benchmark attempt. Never resumes or launches nine runs."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rl_vla.benchmark_contract import ROOT, MILESTONES, resolve, choose, canonical_hash
from rl_vla.attempt_accounting import append_durable, summarize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['rldt','flashsac','dice']);p.add_argument('--seed',type=int,choices=[0],default=0);p.add_argument('--gpu');p.add_argument('--output-root',type=Path);p.add_argument('--dry-run',action='store_true');p.add_argument('--monitor',type=Path);p.add_argument('--retention',action='store_true');p.add_argument('--best-of-16',action='store_true');a=p.parse_args()
    if a.monitor:
        print(json.dumps(summarize(a.monitor),indent=2));return
    if a.method is None or a.seed is None or not a.gpu or a.output_root is None:p.error('method, seed, allocated gpu and output-root are required')
    if a.best_of_16 and a.method!='dice':p.error('Best-of16 is a separate DICE-only supplement')
    path,config=resolve(a.method,a.seed)
    attempt_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex
    out=a.output_root.resolve()/f'{a.method}_seed{a.seed}'/attempt_id
    if a.dry_run:
        print(json.dumps({'attempt_directory_pattern':str(out),'method':a.method,'seed':a.seed,'config':str(path),'resolved_config':config,'resolved_config_sha256':canonical_hash(config),'training':training_command(a.method,path,out),'milestones':MILESTONES,'retention':a.retention,'best_of16':a.best_of_16},indent=2));return
    readiness=json.loads((ROOT/'docs/online-object0/readiness.json').read_text())
    if not readiness['production_ready']:
        raise RuntimeError('Production blocked by unresolved verification: '+ '; '.join(readiness['blockers']))
    # Keep the existing guard; reserve sufficient headroom for primary compact outputs.
    used=int(subprocess.check_output(['du','-sb','/workspace'],text=True).split()[0])/2**30
    if used+30 >=2300:raise RuntimeError('Need30GiB atomic-save/output headroom below the unchanged2300GiB storage guard')
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONPATH=str(ROOT/'src'),RL_VLA_ATTEMPT_DIR=str(out),RL_VLA_ATTEMPT_ID=attempt_id,MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LIBERO_CONFIG_PATH=str(ROOT/'.cache/libero-pro'))
    sources={str(x.relative_to(ROOT)):__import__('hashlib').sha256(x.read_bytes()).hexdigest() for folder in [ROOT/'src/rl_vla',ROOT/'configs/online_object0'] for x in sorted(folder.rglob('*')) if x.is_file() and '__pycache__' not in str(x)}
    meta={'created_time_ns':time.time_ns(),'launcher_pid':os.getpid(),'attempt_id':attempt_id,'method':a.method,'seed':a.seed,'status':'created','resolved_config':config,'resolved_config_sha256':canonical_hash(config),'sources':sources,'attempt_policy':'fresh uninterrupted; never restart in this directory','test_scope':'reused states; no blind confirmation','retention':a.retention,'best_of16':a.best_of_16}
    def save_meta():
        tmp=out/'attempt.tmp';tmp.write_text(json.dumps(meta,indent=2)+'\n');tmp.replace(out/'attempt.json')
    save_meta()
    def run(phase,command):
        phase_env={**env,'RL_VLA_ATTEMPT_PHASE':phase}
        start=time.monotonic();code=None
        with (out/f'{phase}.log').open('x') as log:
            process=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=phase_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            append_durable(out/'phases.jsonl',{'event':'start','phase':phase,'command':list(map(str,command)),'pid':process.pid,'gpu':a.gpu,'time_ns':time.time_ns()})
            try:
                while process.poll() is None:
                    try:code=process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        append_durable(out/'phases.jsonl',{'event':'heartbeat','phase':phase,'pid':process.pid,'seconds':time.monotonic()-start,'time_ns':time.time_ns()})
                code=process.returncode
            except BaseException:
                import signal
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                process.wait();raise
            finally:
                append_durable(out/'phases.jsonl',{'event':'complete' if code==0 else 'failed','phase':phase,'seconds':time.monotonic()-start,'returncode':code,'gpu_reservation_seconds':time.monotonic()-start,'time_ns':time.time_ns()})
            if code:raise subprocess.CalledProcessError(code,command)
    try:
        meta['status']='training';save_meta();run('training',training_command(a.method,path,out))
        accounting=summarize(out);counts=accounting['scopes'].get('training_controller',{})
        if not accounting['bounds_valid'] or counts.get('completed')!=16000 or counts.get('uncertain')!=0:raise RuntimeError('Training physical command accounting is not exactly16000 acknowledged commands')
        candidates=[]
        if a.method=='flashsac':
            import csv
            with (out/'training/logs/validation_metrics.csv').open() as stream:
                bystep={int(r['env_steps']):r for r in csv.DictReader(stream) if r['variant']=='residual'}
            for step in MILESTONES:
                r=bystep[step]
                ordered=json.loads((out/'training/logs'/f'validation_episodes_step{step:08d}.json').read_text())
                if [x['initial_state_id'] for x in ordered]!=list(range(100,150)):raise ValueError('Flash validation state order changed')
                candidates.append({'controller_commands':step,'successes':int(r['successes']),'episodes':int(r['episodes']),'inference_k':1,'checkpoint':str(out/'training/checkpoints'/f'step_{step:08d}.pt')})
        else:
            for step in MILESTONES:
                ck=checkpoint(a.method,out,step);dest=out/f'validation_{step:06d}'
                run(f'validation_{step:06d}',eval_command(a.method,path,ck,'validation',dest))
                r=json.loads((dest if a.method=='rldt' else dest/'summary.json').read_text())
                if a.method=='rldt' and ([x['reset']['initial_state_id'] for x in r['episodes']]!=list(range(100,150)) or [x['seed'] for x in r['episodes']]!=list(range(100000,100050))):raise ValueError('RLDT validation order/reset seed changed')
                successes=sum(x['success'] for x in r['episodes']) if a.method=='rldt' else r['successes']
                episodes=len(r['episodes']) if a.method=='rldt' else r['episodes']
                candidates.append({'controller_commands':step,'successes':successes,'episodes':episodes,'inference_k':1,'checkpoint':str(ck)})
        winner=choose(candidates);selection={'selected':winner,'candidates':candidates,'test_results_used':False,'profile':'K1','test_scope':'reused'}
        # Durable selection must precede the first test process.
        append_durable(out/'selection.jsonl',selection);(out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        meta['status']='selected';save_meta()
        if a.method=='flashsac':
            run('test', ['bash','scripts/run_flashsac_pro.sh','evaluate','--run',out/'training','--split','test','--checkpoint',winner['checkpoint'],'--heldout','--no-record','--eval-id','primary_k1_test'])
        else:
            for label,ck in [('selected',winner['checkpoint']),('prior',None)]:run(f'test_{label}_k1',eval_command(a.method,path,ck,'test',out/f'test_{label}_k1'))
        if a.retention:
            if a.method=='flashsac':run('retention',['bash','scripts/run_flashsac_pro.sh','evaluate','--run',out/'training','--split','retention','--checkpoint',winner['checkpoint'],'--no-record','--eval-id','optional_retention'])
            else:
                for label,ck in [('selected',winner['checkpoint']),('prior',None)]:run(f'retention_{label}',eval_command(a.method,path,ck,'source',out/f'retention_{label}'))
        if a.best_of_16:
            for split in ['validation','test']:run(f'supplement_{split}_k16',eval_command('dice',path,winner['checkpoint'],split,out/f'supplement_{split}_k16',k=16))
        meta['status']='complete'
    except BaseException as exc:
        meta['status']='failed_or_interrupted';meta['error']=repr(exc);raise
    finally:
        meta['accounting']=summarize(out);save_meta()


def training_command(method,path,out):
    script={'rldt':'run_rldt.sh','flashsac':'run_flashsac_pro.sh','dice':'run_dice.sh'}[method]
    return ['bash',f'scripts/{script}','train','--config',str(path),'--output',str(out/'training')]


def checkpoint(method,out,step):
    return out/'training'/(f'step_{step:06d}' if method=='rldt' else f'step_{step:06d}.pt')


def eval_command(method,path,ck,split,dest,k=1):
    if method=='rldt':
        cmd=['bash','scripts/run_rldt.sh','evaluate','--config',path,'--env-split',split,'--episodes','50','--seed','100000','--output',dest]
        if ck:cmd+=['--checkpoint',ck]
    else:
        cmd=['bash','scripts/run_dice.sh','evaluate','--config',path,'--split',split,'--episodes','50','--seed','100000','--evaluation-k',str(k),'--output',dest]
        cmd+=['--checkpoint',ck] if ck else ['--baseline']
    return cmd


if __name__=='__main__':main()
