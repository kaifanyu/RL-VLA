"""Durable single-GPU baseline, continuous training, and selected-checkpoint test."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
_HASH_CACHE = {}


def file_sha256(path):
    """Stream large checkpoints; reuse hashes only while file identity is unchanged."""
    path = Path(path)
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _HASH_CACHE.get(path)
    if cached and cached[0] == identity:
        return cached[1]
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(block)
    after = path.stat()
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError(f'Experiment input changed while hashing: {path}')
    result = digest.hexdigest()
    _HASH_CACHE[path] = (identity, result)
    return result


def write(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def verify_contract(path):
    contract=json.loads(Path(path).read_text())
    for name, expected in contract['sha256'].items():
        file=ROOT/name
        actual=file_sha256(file)
        if actual!=expected:
            raise RuntimeError(f'Frozen experiment source/config changed: {name}')
    if not contract.get('ready'):
        raise RuntimeError('Readiness contract is not approved by actual environment and policy checks')
    return contract


def run(output, contract):
    verify_contract(contract)
    output=Path(output).resolve()
    experiment=json.loads((output/'experiment.json').read_text())
    task=Path(experiment['task_directory'])
    benchmark,method=experiment['benchmark'],experiment['method']
    backend='native' if method=='rldt' else 'torch'
    config=task/'validation.toml' if method=='rldt' else output/'training_config.yaml'
    launcher=['bash',str(ROOT/'integrations/multibench/run.sh'),backend]
    status={'state':'running','benchmark':benchmark,'method':method,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
            'started_unix':time.time(),'stages':[]}
    if (output/'status.json').exists():
        previous=json.loads((output/'status.json').read_text())
        if previous['state']=='completed':
            return previous
        raise RuntimeError('Previous partial run requires explicit inspection before restart')
    write(output/'status.json',status)

    def stage(name,command):
        verify_contract(contract)
        start=time.monotonic()
        status['current_stage']=name
        write(output/'status.json',status)
        with (output/f'{name}.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status['stages'].append({'name':name,'elapsed_seconds':time.monotonic()-start})
        write(output/'status.json',status)

    def evaluation(name,split,checkpoint=None):
        cmd=launcher+['-m','multibench.evaluate','--method',method,'--config',str(config),
                      '--protocol',str(task/'protocol.json'),'--benchmark',benchmark,'--split',split,
                      '--output',str(output/f'{name}.json'),'--episodes','50',
                      '--seed','100000' if split=='validation' else '200000']
        if checkpoint:
            cmd+=['--checkpoint',str(checkpoint)]
        stage(name,cmd)

    try:
        evaluation('baseline_validation','validation')
        evaluation('baseline_test','test')
        if method=='rldt':
            stage('train',launcher+['-m','rl_vla.rldt','train','--config',str(task/'train.toml'),
                                   '--output',str(output/'training')])
        else:
            training=output/'training'
            training.mkdir(exist_ok=True)
            # Base inference writes model scope reports here before training.
            # Preserve those reports while rejecting any prior learner output.
            unexpected={path.name for path in training.iterdir()}-{'model_scope'}
            if unexpected:
                raise RuntimeError(f'Training directory contains prior output: {sorted(unexpected)}')
            with (training/'config.yaml').open('x') as stream:
                stream.write(config.read_text())
            stage('train',launcher+['-m','multibench.train','--config-path',str(training), '--config-name','config'])
        from rl_vla.checkpoint_validation import select_checkpoint
        selected=select_checkpoint(output)
        evaluation('selected_test','test',selected['checkpoint'])
        status.update(state='completed',completed_unix=time.time(),selected_checkpoint=selected)
    except BaseException as error:
        status.update(state='failed',error=repr(error),failed_unix=time.time())
        raise
    finally:
        write(output/'status.json',status)
        from multibench.summarize import summarize
        summarize(output.parents[1])
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True)
    p.add_argument('--contract',required=True)
    a=p.parse_args()
    print(json.dumps(run(a.experiment,a.contract),indent=2))
