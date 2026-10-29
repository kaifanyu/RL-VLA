"""Freeze a completed readiness audit and the inputs used by queued jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from multibench.environment import ROOT
from multibench.run_experiment import file_sha256, write


def simulator_runtime_files():
    """Collect tracked runtime inputs without walking downloaded asset trees."""
    paths = set()
    suffixes = {'.py', '.json', '.yaml', '.yml', '.xml', '.urdf', '.srdf'}
    for repository, prefixes in (
        ('third_party/robocasa365', ('robocasa',)),
        ('third_party/robocasa365/dependencies/robosuite', ('robosuite',)),
        ('third_party/robotwin', ('envs', 'task_config')),
        ('third_party/maniskill3', ('mani_skill',)),
    ):
        directory = ROOT / repository
        names = subprocess.check_output(['git', '-C', str(directory), 'ls-files', '-z', '--', *prefixes])
        paths.update(directory / name for name in names.decode().split('\0')
                     if name and Path(name).suffix in suffixes)
    # These inputs are installed/generated outside the upstream Git indexes.
    required = (
        'third_party/robocasa365/dependencies/openpi_reference/src/openpi/policies/robocasa_policy.py',
        'third_party/robotwin/assets/embodiments/aloha-agilex/config.yml',
        'third_party/robotwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf',
        'third_party/robotwin/assets/embodiments/aloha-agilex/srdf/arx5_description_isaac.srdf',
        '.venv-robotwin/lib/python3.10/site-packages/mplib/planner.py',
    )
    for name in required:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.add(path)
    for name in (
        'third_party/robocasa365/robocasa/macros_private.py',
        'third_party/robocasa365/dependencies/robosuite/robosuite/macros_private.py',
    ):
        if (ROOT / name).is_file():
            paths.add(ROOT / name)
    paths.update((ROOT / 'third_party/rlinf/rlinf/envs/maniskill').rglob('*.py'))
    paths.update((ROOT / 'third_party/openpi/packages/openpi-client/src').rglob('*.py'))
    # Targeted embodiment is small (~120 MB). Include its mesh and texture
    # bytes, excluding generated planner scratch files changed by resets.
    embodiment = ROOT / 'third_party/robotwin/assets/embodiments/aloha-agilex'
    paths.update(path for path in embodiment.rglob('*') if path.is_file()
                 and not (path.name.startswith('curobo_') and path.name.endswith('_tmp.yml'))
                 and '__pycache__' not in path.parts)
    return paths


def freeze(campaign):
    campaign=Path(campaign).resolve()
    target=campaign/'input_contract.json'
    if target.exists():
        raise FileExistsError(target)
    reports={
        'robotwin_environment':ROOT/'integrations/multibench/robotwin/smoke/smoke.json',
        'robocasa_environment':ROOT/'integrations/multibench/robocasa/evidence/smoke/report.json',
        'maniskill_environment':ROOT/'integrations/multibench/maniskill/validation_gpu/smoke.json',
        'preprocessing':ROOT/'integrations/multibench/tests/evidence/preprocessing_parity.json',
    }
    for benchmark in ('robotwin','robocasa','maniskill'):
        for method in ('rldt','flow_sde'):
            reports[f'{benchmark}_{method}']=campaign.parent/'readiness'/f'{benchmark}_{method}_verification.json'
    reports['numerical_parity']=ROOT/'integrations/multibench/tests/evidence/model_parity/comparison.json'
    evidence={}
    for name,path in reports.items():
        data=json.loads(path.read_text())
        if data.get('ready',data.get('passed',data.get('parity_passed',False))) is not True:
            raise RuntimeError(f'Readiness has not passed: {name} ({path})')
        evidence[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    paths=set(reports.values())
    for pattern in ('**/*.py','**/*.sh','**/requirements.lock'):
        paths.update((ROOT/'integrations/multibench').glob(pattern))
    for pattern in ('tasks/*/protocol.json','tasks/*/*.toml','experiments/*/experiment.json','experiments/*/training_config.yaml','plan.json'):
        paths.update(campaign.glob(pattern))
    for name in ('robotwin/norm_stats.json','robotwin/normalization_manifest.json','robocasa/assets/norm_stats.json'):
        paths.add(ROOT/'integrations/multibench'/name)
    paths.add(ROOT/'datasets/benchmark_assets/maniskill3/normalization/physical-intelligence/maniskill/norm_stats.json')
    paths.add(ROOT/'checkpoints/openpi/pi05_base_rlinf/conversion_manifest.json')
    # Include reused adapters/scope/checkpoint code and actual upstream learner implementation.
    paths.update((ROOT/'integrations/pi_rl').glob('*.py'))
    paths.update((ROOT/'integrations/rlinf_bridge').glob('*.py'))
    for folder in ('rlinf/models/embodiment/openpi','rlinf/workers/actor','rlinf/workers/rollout',
                   'rlinf/workers/env','rlinf/runners','rlinf/utils','rlinf/hybrid_engines','rlinf/scheduler'):
        paths.update((ROOT/'third_party/rlinf'/folder).rglob('*.py'))
    for source in ('third_party/openpi/src/openpi', 'third_party/rlinf_openpi/src/openpi'):
        paths.update((ROOT/source).rglob('*.py'))
    paths.update({ROOT/'third_party/rlinf/rlinf/envs/__init__.py',ROOT/'scripts/env.sh'})
    paths.update(simulator_runtime_files())
    for pattern in ('*/setup_manifest.json', '*/assets/manifest.json', '*/assets/*provenance.json'):
        paths.update((ROOT/'integrations/multibench').glob(pattern))
    hashes={str(path.relative_to(ROOT)):file_sha256(path)
            for path in sorted(paths) if path.is_file()}
    conversion=json.loads((ROOT/'checkpoints/openpi/pi05_base_rlinf/conversion_manifest.json').read_text())
    checkpoint=Path(conversion['source_checkpoint'])
    weights={str((checkpoint/name).relative_to(ROOT)):record['sha256']
             for name,record in conversion['source_parameters'].items()}
    weights['checkpoints/openpi/pi05_base_rlinf/model.safetensors']=conversion['weights_sha256']
    for name,expected in weights.items():
        if file_sha256(ROOT/name) != expected:
            raise RuntimeError(f'Base checkpoint changed since verified conversion: {name}')
    hashes.update(weights)
    result={'schema':1,'ready':True,'frozen_unix':time.time(),'evidence':evidence,'sha256':hashes,
            'scope':'New three-benchmark raw pi05_base campaign; original object campaign is separate',
            'validation_iterations':[25,50,100]}
    write(target,result)
    return target

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True)
    a=p.parse_args()
    print(freeze(a.campaign))
