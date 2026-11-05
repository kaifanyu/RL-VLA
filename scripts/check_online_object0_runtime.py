"""Bounded normal-horizon checks; diagnostic attempts, never production scores."""
import argparse, copy, json, os, pathlib, time, hashlib, resource, pickle, shutil
import numpy as np
import torch
from rl_vla.benchmark_contract import resolve
from rl_vla.attempt_accounting import summarize


def digest(value):
    import jax
    h=hashlib.sha256()
    for leaf in jax.tree.leaves(value):h.update(np.asarray(leaf).tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['rldt','flashsac','dice'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    os.environ.update(RL_VLA_ATTEMPT_DIR=str(out),RL_VLA_ATTEMPT_ID='diagnostic_'+a.method,RL_VLA_ATTEMPT_PHASE='training')
    os.environ.setdefault("NUMBA_DISABLE_JIT","1");os.environ.setdefault("LP_NUM_THREADS","1");os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    _,c=resolve(a.method,0);c=copy.deepcopy(c);start=time.monotonic()
    if a.method=='rldt':
        from rl_vla.rldt import train,Learner,make_flow,restore_replay_state
        c['train'].update(max_controller_commands=335,command_milestones=[0,101,200,335]);c['recording']['enabled']=False
        summary=train(c,out/'training')
        assert summary['primitive_steps']==335 and summary['actor_updates']>0 and summary['critic_updates']>=64
        import gc;gc.collect()
        flow=make_flow(c['policy'])
        learner=Learner(flow,flow.critic_feature_dim,c);learner.restore(out/'training/final')
        state_before=digest(learner.state());learner.restore(out/'training/final');assert digest(learner.state())==state_before
        rng=np.random.default_rng(500);recovery=restore_replay_state(out/'training/final',rng)
        assert recovery['counters']['primitive_steps']==335 and recovery['replay'].size>0
        result={'training':summary,'learner_optimizer_rng_restore':'bitwise stable after restore/re-restore','replay_rows':recovery['replay'].size,'pending':recovery['pending'],'simulator_continuation':False}
        del learner,flow
        gc.collect()
        from rl_vla.rldt import evaluate
        ev=copy.deepcopy(c);ev['env']['kwargs']['split']='validation'
        evaluation_started=time.monotonic();report=evaluate(ev,out/'frozen_validation_diagnostic.json',episodes=1,seed=100000)
        result['normal_horizon_frozen_evaluation']={'seconds':time.monotonic()-evaluation_started,'episodes':report['episodes'],'diagnostic_only':True}
    elif a.method=='dice':
        from rl_vla import dice,dice_runtime
        c['train'].update(max_steps=340,milestones=[100,200,340],save_every=1)
        original=dice_runtime.save_training
        def capture(path,learner,config,prior,counters,replay,rng):
            original(path,learner,config,prior,counters,replay,rng)
            if 0<counters['controller_steps']<340 and not (out/'precap_resume.pt').exists():shutil.copyfile(path,out/'precap_resume.pt')
        dice_runtime.save_training=capture
        dice.run(c,out/'training')
        value=torch.load(out/'training/resume.pt',weights_only=False,map_location='cpu');assert value['counters']['controller_steps']==340
        # Full updated optimizer restoration is verified in the separate batch256 check.
        from rl_vla.dice_core import DICE
        from rl_vla.dice_pi05 import FrozenPi05
        prior=FrozenPi05(c['policy'],c['dice']);learner=DICE(prior.feature_dim,prior.noise_dim,prior.action_dim,c['dice'],device='cuda')
        restored=[];rng=np.random.default_rng(99)
        episode,decisions=dice_runtime.restore_training(out/'precap_resume.pt',learner,c,prior,restored,rng)
        saved=torch.load(out/'precap_resume.pt',weights_only=False,map_location='cpu')
        assert len(restored)==len(saved['replay']) and episode==saved['counters']['episodes']
        result={'counters':value['counters'],'restored_precap_episode':episode,'restored_precap_decisions':decisions,'replay_rows_restored':len(restored),'simulator_continuation':False,'production_shape_update_evidence':'separate benchmark_dice_production check; this short collection need not reach256 completed replay rows'}
        del learner,prior
        import gc;gc.collect()
        from rl_vla.dice_eval import evaluate
        ev=copy.deepcopy(c);ev['env']['kwargs']['split']='validation'
        evaluation_started=time.monotonic();evaluate(ev,out/'frozen_validation_diagnostic',episodes=1,seed=100000,diagnostic=True)
        result['normal_horizon_frozen_evaluation']={'seconds':time.monotonic()-evaluation_started,'summary':json.loads((out/'frozen_validation_diagnostic/summary.json').read_text()),'diagnostic_only':True}
    else:
        from rl_vla.pro_residual.experiment import Experiment,prepare_new_run
        from rl_vla.pro_residual.__main__ import _sim_factory
        from rl_vla.pro_residual.train import train,_rng_state,_restore_rng
        from rl_vla.pro_residual.residual import ArrayReplay
        c['train'].update(total_env_steps=280,validate_at_steps=[100,280],validation_states=1)
        c['record'].update(train_every_episodes=0,max_train_recordings=0)
        run=out/'training';prepare_new_run(run);exp=Experiment(c,run);summary=train(exp,_sim_factory(exp),resume=False,smoke=False)
        assert summary['env_steps']==280 and summary['updates']==0 # unchanged2000-command warmup
        replay=ArrayReplay(c['train']['replay_capacity'],exp.builder.dim,exp.spec.action_dim);replay.load(run/'checkpoints/replay_latest.npz')
        learner=exp.make_learner('cuda');payload=torch.load(run/'checkpoints/final.pt',weights_only=True,map_location='cpu');learner.load_state_dict(payload['learner'])
        # Shape verification after normal collection, explicitly outside warmup schedule.
        torch.cuda.synchronize();updates=[]
        for _ in range(2):
            t=time.perf_counter();updates.append(learner.update(replay.sample(256,device='cuda')));torch.cuda.synchronize();updates[-1]['seconds']=time.perf_counter()-t
        state=copy.deepcopy(learner.state_dict());rng=np.random.default_rng(4);saved_rng=_rng_state(rng);sample_rng=copy.deepcopy(replay.rng.bit_generator.state)
        filename=out/'updated_learner.pt';torch.save({'learner':state,'rng':saved_rng},filename)
        replay.save(out/'updated_replay.npz');learner.load_state_dict(torch.load(filename,weights_only=True,map_location='cpu')['learner']);_restore_rng(saved_rng,rng)
        replay.load(out/'updated_replay.npz');assert replay.rng.bit_generator.state==sample_rng
        result={'training':summary,'diagnostic_postcollection_batch256_updates':updates,'optimizer_rng_replay_restored':True,'simulator_continuation':False,'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated()}
    import jax
    result['jax_memory_stats']=jax.devices()[0].memory_stats()
    result['child_max_rss_gib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/2**20
    result.update(method=a.method,scope='bounded diagnostic with production algorithm shapes and unchanged280-command task horizon; budgets/milestones shortened for verification only',resolved_diagnostic_config=c,elapsed_seconds=time.monotonic()-start,host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,accounting=summarize(out))
    (out/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n');print(json.dumps(result,default=str),flush=True)


if __name__=='__main__':main()
