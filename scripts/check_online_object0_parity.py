"""Capture real canonical observations; compare explicit-noise frozen wrappers."""
import argparse,json,os,pathlib,subprocess,sys,hashlib,time
import numpy as np
from rl_vla.benchmark_contract import resolve


def restore_raw(data):
    raw={}
    for name,keys in json.loads(data['schema'].item()).items():
        cursor=raw
        for key in keys[:-1]:cursor=cursor.setdefault(key,{})
        x=data[name].copy();cursor[keys[-1]]=x.item() if x.ndim==0 else x
    return raw


def hashes(value,prefix=''):
    result={}
    for k,v in sorted(value.items()):
        name=prefix+'/'+k
        if isinstance(v,dict):result.update(hashes(v,name))
        else:
            x=np.asarray(v);result[name]={'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--wrapper',choices=['rldt','flashsac','dice']);a=p.parse_args();out=pathlib.Path(a.output)
    if not a.wrapper:
        out.mkdir(parents=True,exist_ok=False)
        os.environ.update(RL_VLA_ATTEMPT_DIR=str(out),RL_VLA_ATTEMPT_ID='parity_capture',RL_VLA_ATTEMPT_PHASE='diagnostic')
        from rl_vla.rldt_env import make_primitive_env
        from rl_vla.trajectory_recording import _flatten_observation
        _,c=resolve('rldt',0);env=make_primitive_env(c['env'])
        try:raw,info=env.reset(seed=0)
        finally:env.close()
        leaves=_flatten_observation(raw);schema={f'leaf_{i}':list(k) for i,k in enumerate(leaves)};arrays={name:leaves[tuple(keys)] for name,keys in schema.items()}
        noise=np.random.default_rng(9152026).standard_normal((1,10,32)).astype(np.float32)
        np.savez(out/'common_input.npz',**arrays,schema=np.asarray(json.dumps(schema)),noise=noise)
        (out/'reset.json').write_text(json.dumps(info,indent=2))
        for method in ['rldt','flashsac','dice']:
            with (out/f'{method}.log').open('x') as log:subprocess.run([sys.executable,__file__,'--output',str(out),'--wrapper',method],check=True,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        data={m:np.load(out/f'{m}.npz') for m in ['rldt','flashsac','dice']};meta={m:json.loads((out/f'{m}.json').read_text()) for m in data};comparisons={}
        for method in ['flashsac','dice']:
            comparisons[method]={key:float(np.abs(data[method][key]-data['rldt'][key]).max()) for key in ['normalized','decoded']}
            comparisons[method]['preprocessing_identical']=meta[method]['preprocessing']==meta['rldt']['preprocessing']
            assert comparisons[method]['preprocessing_identical']
            assert comparisons[method]['normalized']<=1e-5 and comparisons[method]['decoded']<=1e-5
        (out/'result.json').write_text(json.dumps({'comparisons':comparisons,'normalized_atol':1e-5,'decoded_atol':1e-5,'same_supplied_gaussian_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),'no_policy_noise_rng_assumption':True,'reset':info,'dice_batching_equivalence':meta['dice'].get('batching_equivalence')},indent=2)+'\n');return
    import jax
    with np.load(out/'common_input.npz',allow_pickle=False) as d:raw=restore_raw(d);noise=d['noise'].copy()
    _,c=resolve(a.wrapper,0);started=time.monotonic()
    if a.wrapper=='flashsac':
        from rl_vla.pro_residual.base import FrozenPi05Flow
        prior=FrozenPi05Flow(c['base']['checkpoint'],ode_steps=10);flow=prior.flow
        decoded,_,_=prior.infer(raw['openpi'],0,noise=noise)
        context=flow.prepare([raw]);normalized=np.asarray(prior._sample_explicit(flow.params,context,noise,flow.frozen))[0]
    elif a.wrapper=='dice':
        from rl_vla.dice_pi05 import FrozenPi05
        prior=FrozenPi05(c['policy'],c['dice']);flow=prior.flow
        _,_,_,full=prior.propose_explicit(raw,noise);normalized=full[0];decoded=flow.decode(full,[raw])[0]
    else:
        from rl_vla.rldt_pi05 import Pi05Flow
        flow=Pi05Flow.from_config(c['policy']);context=flow.prepare([raw]);sample=jax.jit(lambda params,frozen,ctx,z:flow.sample_with_noise(params,ctx,z,10,frozen=frozen)[0]);full=np.asarray(sample(flow.params,flow.frozen,context,noise));normalized=full[0];decoded=flow.decode(full,[raw])[0]
    result={'wrapper':a.wrapper,'preprocessing':hashes(flow._transform_observation(raw)),'policy_metadata':flow.metadata,'seconds':time.monotonic()-started}
    np.savez(out/f'{a.wrapper}.npz',normalized=normalized,decoded=decoded)
    (out/f'{a.wrapper}.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    if a.wrapper=='dice':
        key=jax.random.PRNGKey(771);prior.key=key;serial=[prior.propose(raw,16) for _ in range(16)];serialkey=jax.random.key_data(prior.key).copy();prior.key=key
        batched=prior.propose_many([raw]*16,16);deltas={}
        for i,name in enumerate(['features','noise','actions']):
            original=np.stack([x[i] for x in serial]);deltas[name]=float(np.abs(original-batched[i]).max())
        (out/'dice_batching_deltas.json').write_text(json.dumps({'max_abs':deltas,'required_features_atol':0,'required_noise_atol':0,'required_actions_atol':1e-4},indent=2)+'\n')
        print(json.dumps({'batching_max_abs':deltas}),flush=True)
        assert deltas['features']==0 and deltas['noise']==0 and deltas['actions']<=1e-4
        assert np.array_equal(serialkey,jax.random.key_data(prior.key))
        result['batching_equivalence']={'max_abs':deltas,'normalized_action_atol':1e-4,'serial_and_microbatch_final_key_equal':True,'fresh_latent_per_row':True,'rows':16,'candidates_per_row':16}
    np.savez(out/f'{a.wrapper}.npz',normalized=normalized,decoded=decoded)
    (out/f'{a.wrapper}.json').write_text(json.dumps(result,indent=2,default=str)+'\n')

if __name__=='__main__':main()
