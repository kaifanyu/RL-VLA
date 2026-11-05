"""One-shot evidence snapshot; never launches training/evaluation or changes source."""
import pathlib,json,hashlib,subprocess,datetime,os,csv
O=pathlib.Path('/workspace/smrithi/object0-launch-readiness-20261105');W=pathlib.Path('/workspace/upenn/worktrees/online-object0-ready-20261105');E=O/'evidence'
def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return None
def put(name,value):
 p=O/name;t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(value,indent=2,default=str)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
status={name:read(O/name/'result.json') for name in ['dice_original_production','dice_optimized_production','dice_optimized_diverse_rows16','dice_final_equivalence_restore','flash_runtime','rldt_runtime_v2','dice_runtime','common_real_parity']}
put('verification_status.json',{'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'production_launched':False,'completed_results':{k:v is not None for k,v in status.items()},'live_original_check_pids':{str(pid):pathlib.Path(f'/proc/{pid}').exists() for pid in [1262815,1270032]},'detached_finisher':read(E/'detached_finisher_status.json'),'batching_deltas':read(O/'common_real_parity/dice_batching_deltas.json'),'results':status})
# Persist exact seed config resolution, not just filenames; no model construction.
import sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['JAX_PLATFORMS']='cpu';sys.path.insert(0,str(W/'src'))
from rl_vla.benchmark_contract import resolve,canonical_hash
configs={}
for method in ['rldt','flashsac','dice']:
 configs[method]={}
 for seed in [0,1,2]:
  path,c=resolve(method,seed);configs[method][str(seed)]={'path':str(path),'raw_toml_sha256':sha(path),'resolved_config':c,'resolved_config_sha256':canonical_hash(c),'production_scope':seed==0}
put('resolved_seed_configs.json',configs)
files=[]
for root in [W/'src/rl_vla',W/'scripts',W/'configs/online_object0',W/'tests']:
 for p in sorted(root.rglob('*')):
  if p.is_file() and '__pycache__' not in str(p):files.append({'path':str(p.relative_to(W)),'sha256':sha(p),'bytes':p.stat().st_size})
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip();put('source_manifest.json',{'worktree':str(W),'branch':'online-object0-ready-20261105','commit':commit,'files':files,'git_status':subprocess.check_output(['git','status','--short'],cwd=W,text=True)})
# Primary supplied-noise equality can be assessed independently of batching parity.
parity={}
try:
 import numpy as np
 p=O/'common_real_parity';base=np.load(p/'rldt.npz');meta=read(p/'rldt.json')
 for method in ['flashsac','dice']:
  if not (p/f'{method}.npz').exists():continue
  d=np.load(p/f'{method}.npz');m=read(p/f'{method}.json')
  parity[method]={'normalized_max_abs':float(np.abs(d['normalized']-base['normalized']).max()),'decoded_max_abs':float(np.abs(d['decoded']-base['decoded']).max()),'preprocessing_equal':m['preprocessing']==meta['preprocessing']}
except (FileNotFoundError,TypeError):pass
put('primary_frozen_parity.json',parity)
rows=[
 {'gate':'Fresh online-only seed0 source/seed profiles','RLDT':'PASS','FlashSAC':'PASS','DICE':'PASS','evidence':'configs/online_object0; benchmark_contract.py; resolved_seed_configs.json'},
 {'gate':'Literal command guard and pending-chunk milestone source','RLDT':'PASS','FlashSAC':'PASS','DICE':'PASS','evidence':'rldt.py execute_chunk/train; pro_residual/rollout.py; dice.py'},
 {'gate':'Uninterrupted attempt IDs, intent/completion bounds, separate costs','RLDT':'PASS','FlashSAC':'PASS','DICE':'PASS','evidence':'attempt_accounting.py; launch_online_object0.py; flash_runtime/result.json'},
 {'gate':'Production-shape updates finite','RLDT':'PASS','FlashSAC':'PASS','DICE':'PASS','evidence':'rldt_runtime_v2/training/metrics.jsonl; flash_runtime/result.json; dice_optimized_diverse_rows16/result.json'},
 {'gate':'Normal280 horizon and restoration diagnostic completed','RLDT':'PASS' if status['rldt_runtime_v2'] else 'UNVERIFIED','FlashSAC':'PASS' if status['flash_runtime'] else 'UNVERIFIED','DICE':'PASS' if status['dice_runtime'] else 'UNVERIFIED','evidence':'verification_status.json (pending checks may finish after handoff)'},
 {'gate':'Frozen explicit-noise real-observation wrapper parity','RLDT':'PASS','FlashSAC':'PASS' if parity.get('flashsac',{}).get('normalized_max_abs',1)<=1e-5 else 'UNVERIFIED','DICE':'PASS' if parity.get('dice',{}).get('normalized_max_abs',1)<=1e-5 else 'UNVERIFIED','evidence':'primary_frozen_parity.json; common_real_parity/'},
 {'gate':'DICE serial/batched latent and proposal equivalence','RLDT':'PASS','FlashSAC':'PASS','DICE':'PASS' if read(O/'common_real_parity/dice.json') and read(O/'common_real_parity/dice.json').get('batching_equivalence') else 'FAIL','evidence':'Original assertion failed. Retry writes dice_batching_deltas.json; no tolerance silently relaxed.'},
 {'gate':'DICE full-shape CPU/GPU update and updated-state restore equivalence','RLDT':'PASS','FlashSAC':'PASS','DICE':'PASS' if status['dice_final_equivalence_restore'] else 'UNVERIFIED','evidence':'Detached benchmark runs only if serial/batch parity passes.'},
 {'gate':'Seed0-only three-job queue storage/RAM/CPU admission (source/dry-run)','RLDT':'PASS','FlashSAC':'PASS','DICE':'PASS','evidence':'seed0_schedule_dry_run.json; launch_online_object0_seed0.py (no --execute used)'}]
put('readiness_matrix.json',{'production_ready':False,'reason':'Handoff checkpoint: DICE optimization equivalence is an explicit blocker; assess current detached results before any readiness promotion. Final runtime/pytest checks may remain pending.','checks':rows})
with (O/'readiness_matrix.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
original=json.loads((E/'isolation.json').read_text());now=subprocess.check_output(['git','status','--short','--untracked-files=all'],cwd=original['original'],text=True)
put('preservation.json',{'original_status_unchanged':now==original['original_status'],'input_snapshot':read(E/'input_preservation.json'),'source_commit':commit,'no_production_campaign_launched':True})
put('artifact_manifest.json',[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.glob('*.json')) if p.name!='artifact_manifest.json'])
print(json.dumps({'source_commit':commit,'completed':{k:v is not None for k,v in status.items()},'original_status_unchanged':now==original['original_status']},indent=2))
