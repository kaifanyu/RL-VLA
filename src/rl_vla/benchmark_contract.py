"""Executable contract for fresh, online-only Object0 benchmark attempts."""
from pathlib import Path
import hashlib
import json
import tomllib

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path('/workspace/upenn/RL-VLA')
MILESTONES = [0, 4000, 8000, 12000, 16000]


def require_ready(readiness, method):
    """Admit a reviewed method without promoting the whole three-method pilot."""
    if readiness.get('production_ready') is True:
        return
    entry = readiness.get('methods', {}).get(method, {})
    if entry.get('production_ready') is True and not entry.get('blockers'):
        return
    blockers = entry.get('blockers') or readiness.get('blockers') or ['No readiness evidence']
    raise RuntimeError(f'Production blocked for {method}: ' + '; '.join(blockers))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resolve(method, seed):
    path = ROOT / 'configs/online_object0' / f'{method}_seed{seed}.toml'
    if method == 'flashsac':
        from rl_vla.pro_residual.spec import load_config
        config = load_config(path)
    elif method == 'dice':
        from rl_vla.dice import config_from
        config = config_from(path)
    else:
        from rl_vla.rldt import load_config
        config = load_config(path)
    if config['train']['seed'] != seed or seed not in [0, 1, 2]: raise ValueError('Seed profile mismatch')
    if method == 'rldt':
        assert config['train']['max_controller_commands'] == 16000
        assert config['train']['command_milestones'] == MILESTONES
        assert (config['train']['critic_batch_size'],config['train']['actor_batch_size'],config['rldt']['particles']) == (64,1,8)
        assert config['benchmark']['fresh_uninterrupted']
    elif method == 'dice':
        assert config['data']['mode'] == 'online_only_ablation' and not config['data']['path']
        assert config['train']['max_steps'] == 16000 and config['train']['batch_size'] == 256
        assert config['train']['milestones'] == MILESTONES[1:] and config['evaluation']['k'] == 1
        assert config['dice']['k_train'] == config['dice']['k_bootstrap'] == 16
        assert config['dice']['critic_ensemble_size'] == 10 and config['dice']['width'] == 1024
        assert config['benchmark']['fresh_uninterrupted']
    else:
        assert config['train']['total_env_steps'] == 16000 and config['train']['batch_size'] == 256
        assert config['train']['validate_at_steps'] == MILESTONES[1:] and config['train']['validate_at_start']
        assert config['contract']['budget'] == 'strict' and config['contract']['fresh_uninterrupted']
        assert config['base']['backend'] == 'pi05_flow'
    return path, config


def choose(candidates):
    if [c['controller_commands'] for c in candidates] != MILESTONES:
        raise ValueError('Selection requires exactly the five command milestones in order')
    if any(c['episodes'] != 50 or c.get('inference_k',1) != 1 for c in candidates):
        raise ValueError('Selection requires full ordered50 K1 validation')
    return max(candidates,key=lambda c:(c['successes'],-c['controller_commands']))


def write_toml(path, config):
    def scalar(value):
        if isinstance(value,bool):return str(value).lower()
        if isinstance(value,str):return json.dumps(value)
        if isinstance(value,list):return '['+', '.join(scalar(x) for x in value)+']'
        if isinstance(value,(int,float)):return repr(value)
        raise TypeError(type(value))
    lines=[]
    def table(value, prefix=''):
        if prefix:lines.append(f'[{prefix}]')
        for key,item in value.items():
            if not isinstance(item,dict) and item is not None:lines.append(f'{key} = {scalar(item)}')
        lines.append('')
        for key,item in value.items():
            if isinstance(item,dict):table(item,prefix+'.'+key if prefix else key)
    table(config);Path(path).write_text('\n'.join(lines))
