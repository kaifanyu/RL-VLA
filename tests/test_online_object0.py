"""Protocol/ledger fixtures; these do not assert real-policy task success."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pytest

from rl_vla.attempt_accounting import command_call, summarize, reject_restart
from rl_vla.benchmark_contract import MILESTONES, choose, resolve


def test_ledger_completion_and_uncertain_bounds(tmp_path, monkeypatch):
    monkeypatch.setenv('RL_VLA_ATTEMPT_DIR',str(tmp_path));monkeypatch.setenv('RL_VLA_ATTEMPT_ID','fixture_attempt');monkeypatch.setenv('RL_VLA_ATTEMPT_PHASE','training')
    assert command_call(lambda a:4,[0.]*7)==4
    assert command_call(lambda a:5,[0.]*7,settling=True)==5
    def lost(a):raise EOFError('fixture lost acknowledgement after intent')
    with pytest.raises(EOFError):command_call(lost,[0.]*7)
    r=summarize(tmp_path)['scopes']
    assert r['training_controller']['completed']==1 and r['training_controller']['uncertain']==1
    assert (r['training_controller']['physical_command_lower_bound'],r['training_controller']['physical_command_upper_bound'])==(1,2)
    assert r['training_settling']['completed']==1


def test_fresh_attempt_rejects_warm_starts():
    for section in ['benchmark','contract']:
        with pytest.raises(ValueError,match='fresh uninterrupted'):reject_restart({section:{'fresh_uninterrupted':True}},'checkpoint')


def test_seed_profiles_lock_shapes_and_independent_training_seeds():
    hashes=set()
    for method in ['rldt','flashsac','dice']:
        for seed in [0,1,2]:
            path,c=resolve(method,seed);assert c['train']['seed']==seed and path.is_file()
            from rl_vla.benchmark_contract import canonical_hash
            hashes.add(canonical_hash(c))
    assert len(hashes)==9


def test_initialization_and_earliest_ties_only_five_milestones():
    rows=[dict(controller_commands=x,successes=17,episodes=50,inference_k=1) for x in MILESTONES]
    assert choose(rows)['controller_commands']==0
    rows[2]['successes']=20;rows[4]['successes']=20
    assert choose(rows)['controller_commands']==8000
    with pytest.raises(ValueError):choose(rows[1:])
    rows[0]['inference_k']=16
    with pytest.raises(ValueError):choose(rows)


def tiny_budget_config():
    from rl_vla.rldt import load_config
    c=load_config('configs/rldt_toy.toml')
    c['train'].update(iterations=1,collect_decisions=4,critic_steps=2,actor_steps=1,critic_batch_size=4,actor_batch_size=1,critic_trajectories=1,max_controller_commands=11,command_milestones=[0,3,7,11])
    c['rldt'].update(critic_warmup_iterations=0,particles=8)
    c['benchmark']={'fresh_uninterrupted':True}
    return c


def state_hash(learner):
    import hashlib,jax
    digest=hashlib.sha256()
    for x in jax.tree.leaves(learner.state()):digest.update(np.asarray(x).tobytes())
    return digest.hexdigest()


def test_command_snapshots_preserve_pending_chunk_and_rng(tmp_path):
    from rl_vla.rldt import train, Learner, make_flow, restore_replay_state
    c=tiny_budget_config();a=train(c,str(tmp_path/'snapshots'))
    d=copy.deepcopy(c);d['train']['command_milestones']=[];b=train(d,str(tmp_path/'no_snapshots'))
    assert a['primitive_steps']==b['primitive_steps']==11
    for step in [0,3,7,11]:
        snap=json.loads((tmp_path/'snapshots'/f'step_{step:06d}'/'snapshot.json').read_text())
        assert snap['controller_commands']==step
        assert snap['phase']=='post_command_pre_chunk_replay_and_updates'
        if step:assert snap['pending']['chunk_executed_commands']>0
    f=make_flow(c['policy']);feature=f.prepare([{'state':np.zeros(4,np.float32)}])['features'].shape[-1]
    first=Learner(f,feature,c);first.restore(tmp_path/'snapshots/final')
    second=Learner(make_flow(c['policy']),feature,c);second.restore(tmp_path/'no_snapshots/final')
    assert state_hash(first)==state_hash(second)
    rng=np.random.default_rng(999);restored=restore_replay_state(tmp_path/'snapshots/final',rng)
    assert restored['counters']['primitive_steps']==11 and not restored['simulator_continuation']
    assert restored['replay'].size>0
    assert not (tmp_path/'snapshots/iteration_000001').exists()


def test_literal_precommand_guard_prevents_execution():
    from rl_vla.rldt import execute_chunk
    class Env:
        low=np.full(2,-1);high=np.full(2,1);calls=0
        def step(self,a):self.calls+=1;return {},0,False,False,{}
    env=Env()
    def guard():raise RuntimeError('exhausted')
    with pytest.raises(RuntimeError,match='exhausted'):execute_chunk(env,np.zeros((5,2)),.99,before_step=guard)
    assert env.calls==0
