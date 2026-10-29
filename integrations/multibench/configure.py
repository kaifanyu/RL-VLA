"""Prepare frozen, paired raw-pi05-base experiments for the three requested tasks."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import yaml
from multibench.environment import ROOT, SPECS, process_config, state_id
from multibench.policy_spec import BENCHMARKS
from pi_rl.model_configs import model_config as reference_model_config

TASKS = {"robotwin": "handover_block", "robocasa": "robocasa/PickPlaceCounterToCabinet",
         "maniskill": "PutOnPlateInScene25Main-v3"}
CALIBRATION_NOTES = {
    'robotwin': 'Normalization computed from official handover_block demo_clean training demonstrations; all policy weights start from raw pi05_base, with no task SFT.',
    'robocasa': 'Official RoboCasa pi05 release mean/std normalization only; all policy weights start from raw pi05_base, with no task SFT.',
    'maniskill': 'Official physical-intelligence/maniskill quantile normalization only; all policy weights start from raw pi05_base, with no task SFT.',
}


def write_toml(path, value):
    lines = []
    def emit(mapping, prefix=''):
        scalars = {k: v for k, v in mapping.items() if not isinstance(v, dict)}
        if prefix:
            lines.append(f'[{prefix}]')
        for key, val in scalars.items():
            lines.append(f'{key} = {json.dumps(val)}')
        lines.append('')
        for key, val in mapping.items():
            if isinstance(val, dict):
                emit(val, f'{prefix}.{key}' if prefix else key)
    emit(value)
    Path(path).write_text('\n'.join(lines))


def torch_configuration(benchmark, task, output, iterations=100, decisions=32):
    upstream = ROOT / 'third_party/rlinf/examples/embodiment/config'
    definition = json.loads((task / 'protocol.json').read_text())['tasks'][benchmark]
    cfg = yaml.safe_load((upstream / 'libero_spatial_ppo_openpi_pi05.yaml').read_text())
    cfg.pop('defaults')
    cfg['hydra'].pop('searchpath', None)
    cfg['cluster'] = {'num_nodes': 1, 'component_placement': {'actor,env,rollout': 'all'}}
    cfg['runner'].update(max_epochs=iterations, max_steps=iterations, save_interval=min(25, iterations),
                         val_check_interval=-1, per_worker_log=False)
    cfg['runner']['logger'].update(log_path=str(output), project_name='multibench-base-pi05',
                                  experiment_name=f'{benchmark}_flow_sde', logger_backends=['tensorboard'])
    model = reference_model_config(ROOT / 'checkpoints/openpi/pi05_base_rlinf', 'flow_sde')
    spec = BENCHMARKS[benchmark]
    model.update(benchmark=benchmark, action_dim=spec['action_dim'], enforce_native_scope=True,
                 scope_report_directory=str(output / 'model_scope'))
    model['openpi'].update(config_name='pi05_multibench', action_horizon=50, discrete_state_input=True,
                           num_images_in_input=spec['images'], action_env_dim=spec['action_dim'])
    cfg['actor']['model'] = model
    cfg['actor'].update(micro_batch_size=1, global_batch_size=min(8, decisions), seed=0, enable_offload=False)
    cfg['actor']['fsdp_config'] = yaml.safe_load((upstream / 'training_backend/fsdp.yaml').read_text())
    cfg['actor']['fsdp_config'].update(sharding_strategy='no_shard', use_orig_params=True, save_full_model_weights=True)
    cfg['actor']['optim']['critic_warmup_steps'] = 2 if iterations >= 25 else 0
    cfg['algorithm'].update(gamma=.999**5, update_epoch=1, entropy_bonus=0.)
    cfg['rollout'].update(enable_offload=True, pipeline_stage_num=1)
    cfg['rollout']['model']['model_path'] = model['model_path']
    cfg['rollout']['unnorm_key'] = benchmark
    env = dict(env_type='pro_controlled', benchmark=benchmark, protocol=str(task/'protocol.json'),
               task_key=benchmark, split='train', simulator_python=str(ROOT/SPECS[benchmark]['python']/'bin/python'),
               seed=0, total_num_envs=1, group_size=1, rollout_epoch=1, max_steps_per_rollout_epoch=decisions*5,
               max_episode_steps=definition['max_steps'], primitive_gamma=.999, auto_reset=True,
               ignore_terminations=False, is_eval=False, recording_directory=str(output/'interactions'),
               record_trajectories=False, video_cfg={'save_video':False}, init_params={})
    cfg['env']['train'] = env
    cfg['env']['eval'] = dict(env, split='validation', is_eval=True, seed=100000,
                              recording_directory=str(output/'unused_online_validation'))
    cfg['weight_syncer'] = yaml.safe_load((upstream/'weight_syncer/patch_syncer.yaml').read_text())
    return cfg


def prepare(destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    template_path = ROOT/'runs/object_adaptation_20261028/tasks/object_2/train.toml'
    import tomllib
    template = tomllib.loads(template_path.read_text())
    result = []
    for index, benchmark in enumerate(TASKS):
        task = destination/'tasks'/benchmark
        task.mkdir(parents=True, exist_ok=False)
        rng = np.random.default_rng(20261028 + index)
        seeds, ids, seen = [], [], set()
        while len(seeds) < 200:
            seed = int(rng.integers(1000000, 2**31-1))
            identity = state_id(benchmark, seed)
            if identity not in seen:
                seeds.append(seed); ids.append(identity); seen.add(identity)
        definition = {'task_id':TASKS[benchmark], 'max_steps':SPECS[benchmark]['max_steps']}
        if benchmark == 'robotwin':
            definition.update(environment_variant='demo_clean / Aloha-AgileX', expert_filtering=False,
                score_scope='Custom paired unfiltered resets; not official expert-filtered leaderboard evaluation')
        for split, a, b in [('train',0,100),('validation',100,150),('test',150,200)]:
            definition[f'{split}_seeds'] = seeds[a:b]
            definition[f'{split}_state_ids'] = ids[a:b]
        protocol = {'schema':1, 'benchmark':benchmark, 'validation_iterations':[25,50,100],
                    'online_validation':True, 'tasks':{benchmark:definition},
                    'split_note':'Disjoint deterministic resets within the same task distribution; ManiSkill uses official train object pool, RoboCasa uses official target split.',
                    'base_checkpoint':'pi05_base', 'task_sft':False,
                    'execution_steps':5, 'ode_steps':10, 'decisions_per_iteration':32,
                    'calibration_note':CALIBRATION_NOTES[benchmark]}
        (task/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
        for split in ('train','validation','test'):
            cfg = copy.deepcopy(template)
            cfg['policy'].update(checkpoint=str(ROOT/'checkpoints/openpi/pi05_base'), model_config='pi05_base', benchmark=benchmark)
            cfg['env'] = process_config(benchmark, task/'protocol.json', split)
            cfg['recording']['enabled'] = False
            write_toml(task/f'{split}.toml', cfg)
        for method in ('rldt','flow_sde'):
            output = destination/'experiments'/f'{benchmark}_{method}'
            output.mkdir(parents=True, exist_ok=False)
            experiment = {'benchmark':benchmark, 'task_key':benchmark, 'task_directory':str(task),
                          'method':method, 'iterations':[25,50,100], 'protocol':protocol,
                          'selection':'Highest validation success; tie chooses earlier iteration. Test only selected checkpoint.',
                          'recording':'Counters, metrics, checkpoints, per-episode outcomes and selected videos; no full raw RGB training traces.'}
            (output/'experiment.json').write_text(json.dumps(experiment, indent=2)+'\n')
            if method == 'flow_sde':
                cfg = torch_configuration(benchmark, task, output/'training')
                (output/'training_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            result.append(str(output))
    (destination/'plan.json').write_text(json.dumps({'experiments':result, 'validation_iterations':[25,50,100]},indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))
