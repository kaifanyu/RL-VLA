"""Bounded real-prior production-shape learner check; never environment training."""
import argparse, json, pathlib, time, resource, threading, subprocess, sys, inspect, hashlib, copy
import numpy as np
import torch
from rl_vla.dice import config_from, proposals, tensor
from rl_vla.dice_pi05 import FrozenPi05
from rl_vla.dice_core import DICE


def real_observations(path):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(d['metadata_json'].item())
        schema = meta.get('observation_schema', meta.get('observation_arrays'))
        if schema is None:
            for k,v in meta.items():
                if isinstance(v,dict) and 'observation_000' in v: schema=v;break
        if schema is None: raise ValueError(f'Unknown schema keys: {list(meta)}')
        observations=[]
        for i in range(min(4,len(d['observation_000']))):
            raw={}
            for name,keys in schema.items():
                cursor=raw
                for k in keys[:-1]:cursor=cursor.setdefault(k,{})
                value=d[name][i].copy();cursor[keys[-1]]=value.item() if value.ndim==0 else value
            observations.append(raw)
        return observations


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--observation',required=True);p.add_argument('--output',required=True);p.add_argument('--updates',type=int,default=2);p.add_argument('--optimized',action='store_true');p.add_argument('--microbatch-rows',type=int,default=4);p.add_argument('--distinct-observations',action='store_true');p.add_argument('--restore-equivalence',action='store_true');a=p.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    config=config_from(a.config);assert config['train']['batch_size']==256
    assert config['dice']['k_train']==config['dice']['k_bootstrap']==16
    assert config['dice']['width']==1024 and config['dice']['critic_ensemble_size']==10
    torch.set_num_threads(1);torch.manual_seed(config['train']['seed'])
    if a.optimized:
        config['train']['learner_device']='cuda';config['dice']['proposal_microbatch_rows']=a.microbatch_rows;config['dice']['prefix_cache_max_mb']=65536
    records=[];gpu=[];stop=threading.Event()
    def sample_memory():
        while not stop.is_set():
            try:gpu.append({'time':time.time(),'used_mb':int(subprocess.check_output(['nvidia-smi','--id','0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())})
            except Exception:pass
            stop.wait(1)
    thread=threading.Thread(target=sample_memory,daemon=True);thread.start()
    start=time.perf_counter();prior=FrozenPi05(config['policy'],{**config['dice'],'seed':config['train']['seed']});load_s=time.perf_counter()-start
    obs=real_observations(a.observation);start=time.perf_counter();f,z,pv,full=prior.propose(obs[0],16);cold_s=time.perf_counter()-start
    learner=DICE(prior.feature_dim,prior.noise_dim,prior.action_dim,config['dice'],device=config['train'].get('learner_device','cpu'))
    initial_cpu={k:v.detach().cpu().clone() for k,v in learner.actor.state_dict().items()} if a.restore_equivalence else None
    initial_state=copy.deepcopy(learner.state_dict()) if a.restore_equivalence else None
    equivalence=None
    original=prior.propose;calls=0;candidates=0
    def counted(raw,count):
        nonlocal calls,candidates
        calls+=1;candidates+=count
        if calls%64==0:print(json.dumps({'progress_calls':calls,'time':time.time()}),flush=True)
        return original(raw,count)
    prior.propose=counted
    if a.distinct_observations:
        bank=sorted(pathlib.Path(a.observation).parent.parent.glob('episode_*/chunk_*.npz'))[:256]
        obs=[real_observations(x)[0] for x in bank]
        assert len(obs)==256
    rows=[obs[i%len(obs)] for i in range(256)]
    source,first=inspect.getsourcelines(DICE.update);pf_line=first+next(i for i,x in enumerate(source) if 'pf, pz, pp = proposals' in x);crit_line=first+next(i for i,x in enumerate(source) if 'self.critic_optimizer.zero_grad' in x)
    for index in range(a.updates):
        before_counts=dict(prior.execution_counts)
        before_calls,before_candidates=calls,candidates;t=time.perf_counter();prop=proposals(prior,rows,16);current_s=time.perf_counter()-t
        t=time.perf_counter();nxt=proposals(prior,rows,16);next_s=time.perf_counter()-t
        batch={'features':prop[0][:,0],'actions':prop[2][:,0],'rewards':torch.zeros(256),'discounts':torch.full((256,),.99),'returns':torch.zeros(256),'expert':torch.zeros(256),'action_mask':torch.ones(256,35)}
        phase={'name':'critic','last':time.perf_counter(),'critic':0.,'actor':0.}
        def trace(frame,event,arg):
            if frame.f_code is not DICE.update.__code__:return None
            if event=='line':
                if a.optimized and frame.f_lineno in (pf_line,crit_line):torch.cuda.synchronize()
                now=time.perf_counter();phase[phase['name']]+=now-phase['last'];phase['last']=now
                if frame.f_lineno==pf_line:phase['name']='actor'
                if frame.f_lineno==crit_line:phase['name']='critic'
            return trace
        t=time.perf_counter();sys.settrace(trace)
        try: metrics=learner.update(batch,prop,nxt,step=1 if index%2==0 else 2)
        finally:sys.settrace(None)
        if a.optimized:torch.cuda.synchronize()
        elapsed=time.perf_counter()-t;phase[phase['name']]+=time.perf_counter()-phase['last']
        record={'index':index,'collection_group_clock':1 if index%2==0 else 2,'proposal_current_seconds':current_s,'proposal_next_seconds':next_s,'critic_seconds':phase['critic'],'actor_objective_seconds':phase['actor'],'update_seconds':elapsed,'prior_calls':prior.execution_counts['logical_prior_calls']-before_counts['logical_prior_calls'],'prior_candidates':prior.execution_counts['candidates']-before_counts['candidates'],'execution_counts':{k:v-before_counts[k] for k,v in prior.execution_counts.items()},'metrics':metrics,'host_peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        if a.restore_equivalence and index==0:
            reference=DICE(prior.feature_dim,prior.noise_dim,prior.action_dim,config['dice'],device='cpu')
            reference.load_state_dict(initial_state)
            oracle=reference.update(batch,prop,nxt,step=1)
            deltas={}
            for name in ['actor','critics','target']:
                deltas[name]=max(float((v.detach().cpu()-getattr(reference,name).state_dict()[key]).abs().max()) for key,v in getattr(learner,name).state_dict().items())
            loss_deltas={key:abs(metrics[key]-oracle[key]) for key in ['critic_loss','actor_loss','regularization_fraction','target_mean']}
            equivalence={'cpu_gpu_parameter_max_abs':deltas,'loss_max_abs':loss_deltas,'parameter_atol':2e-5,'loss_atol':1e-4,'gate_fraction_equal':metrics['regularization_fraction']==oracle['regularization_fraction']}
            assert max(deltas.values())<=2e-5 and max(loss_deltas.values())<=1e-4 and equivalence['gate_fraction_equal']
            del reference,initial_state
        records.append(record);(out/'progress.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
    restoration=None
    if a.restore_equivalence:
        from rl_vla.dice_runtime import save_training,restore_training
        rng=np.random.default_rng(991)
        counters={'episodes':0,'decisions':0,'controller_steps':0,'updates':learner.updates}
        replay=[{'observation':rows[0],'features':np.asarray(prop[0][0,0]),'expert':0.0}]
        save_training(out/'diagnostic_restore.pt',learner,config,prior,counters,replay,rng)
        clone=DICE(prior.feature_dim,prior.noise_dim,prior.action_dim,config['dice'],device='cuda')
        import jax
        rr=[];other=np.random.default_rng(9)
        restore_training(out/'diagnostic_restore.pt',clone,config,prior,rr,other)
        def exact(x,y):
            if isinstance(x,torch.Tensor):return torch.equal(x.cpu(),y.cpu())
            if isinstance(x,dict):return x.keys()==y.keys() and all(exact(x[k],y[k]) for k in x)
            if isinstance(x,(tuple,list)):return len(x)==len(y) and all(exact(a,b) for a,b in zip(x,y))
            return x==y
        assert exact(learner.state_dict(),clone.state_dict()) and rng.bit_generator.state==other.bit_generator.state and len(rr)==1
        payload=torch.load(out/'diagnostic_restore.pt',weights_only=False,map_location='cpu')
        assert torch.equal(torch.get_rng_state(),payload['torch_rng'])
        assert np.array_equal(jax.random.key_data(prior.key),payload['prior_rng'])
        if payload.get('torch_cuda_rng') is not None:assert all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),payload['torch_cuda_rng']))
        restoration={'torch_cpu_cuda_prior_rng_equal':True,'learner_optimizer_scheduler_bitwise_equal':True,'numpy_rng_equal':True,'replay_rows':len(rr),'simulator_continuation':False,'scope':'diagnostic updated learner; no collected production interactions'}
        del clone
    stop.set();thread.join();import jax
    result={'cpu_gpu_equivalence':equivalence,'restoration':restoration,'config':config,'observation_source':a.observation,'scope':'diagnostic replay batch of real training observations; no environment commands, no production replay export','initial_model_load_seconds':load_s,'first_K16_prepare_compile_infer_seconds':cold_s,'records':records,'peak_host_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'peak_gpu_sampled_mb':max(x['used_mb'] for x in gpu),'gpu_samples':gpu,'jax_memory_stats':jax.devices()[0].memory_stats(),'execution_counts':prior.execution_counts,'prefix_cache_bytes':prior._prefix_cache_bytes,'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.optimized else None,'prior_calls':calls,'prior_candidates':candidates,'torch_threads':torch.get_num_threads()}
    (out/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')

if __name__=='__main__':main()
