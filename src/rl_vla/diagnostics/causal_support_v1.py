"""Versioned strict-support causal consumers; structural NA is never an outcome."""
import csv
import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from rl_vla.diagnostics.metrics import paired_success

TASK='object_1'
POLICIES=('pi05_libero','rldt_000025','rldt_000050','rldt_000100')
CLOSED=('none','no_image','no_text','target:background_fill','distractor:background_fill','random:background_fill',
        'target:mosaic','distractor:mosaic','random:mosaic')
RAW='interventions_structural_na_v1'
ANALYSIS='causal_analysis_structural_na_v1'
APPROVAL='compiler_profile_integration/causal_structural_na_v1/root_acceptance.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def binding(path):
    return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def check_binding(row):
    if sha(row['path'])!=row['sha256']:raise ValueError(f'Changed support-aware source: {row["path"]}')
    return Path(row['path'])


def accepted_amendment(root):
    from rl_vla.diagnostics.causal_structural_validation_v1 import accepted_amendment as validate
    return validate(root)[2]


@lru_cache
def original_consumer():
    path=Path(__file__).resolve().parents[3]/'scripts/analyze_vla_trace_causal.py'
    spec=importlib.util.spec_from_file_location('_frozen_original_causal_consumer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def availability(episodes, expected_ids, *, evidence_validator):
    if len(episodes)!=10 or [r['reset']['initial_state_id'] for r in episodes]!=expected_ids:
        raise ValueError('Support-aware condition must resolve exactly all10 predetermined IDs')
    if [r['seed'] for r in episodes]!=list(range(100000,100010)):
        raise ValueError('Support-aware condition noise schedule differs')
    native=[];unavailable=[]
    for row in episodes:
        status=row.get('status','completed')
        if status=='structural_NA':
            if row.get('success','missing') is not None:raise ValueError('Structural NA cannot be a native Boolean outcome')
            if row.get('condition') not in ('random:background_fill','random:mosaic'):
                raise ValueError('Structural NA is restricted to original strict random-background controls')
            evidence_validator(row)
            unavailable.append(row)
        elif status=='completed':
            if type(row.get('success')) is not bool:raise ValueError('Completed native outcome must be Boolean')
            if not 0<row['primitive_steps']<=280:raise ValueError('Completed native controlled-step count is invalid')
            native.append(row)
        else:raise ValueError('Unresolved or unsupported case status cannot enter complete coverage')
    return {'attempted_cases':10,'completed_outcome_cases':len(native),'structural_na_cases':len(unavailable),
            'primary_outcome_available':not unavailable,'completed_native_successes':sum(r['success'] for r in native)}


def paired_primary(baseline, candidate, support):
    """Whole planned-pair estimand: never select the available subset."""
    if not support['primary_outcome_available']:
        fields=('baseline_successes','candidate_successes','baseline_rate','candidate_rate','difference','baseline_wilson95',
                'candidate_wilson95','paired_bootstrap95','bootstrap_seed','bootstrap_draws','recovered','regressed',
                'both_success','both_failure','mcnemar_exact_two_sided_p')
        return {**{key:None for key in fields},'episodes':10,'status':'structural_NA',**support,
                'caveat':'One or more exact original controls are undefined. Entire planned10-case primary success/paired effect unavailable; no imputation or complete-case statistic.'}
    if len(baseline)!=10 or len(candidate)!=10:raise ValueError('Available paired effect requires all10 cases')
    return {**paired_success(baseline,candidate),'status':'available',**support}


def write_csv(path, rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({key:json.dumps(value) if isinstance(value,(dict,list)) else value for key,value in row.items()})


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def open_loop(root,task,policy):
    """Unchanged original50-state/15-condition calculation; no closed-loop NA filtering."""
    root=Path(root).resolve(); original=original_consumer();OPEN=original.OPEN
    nominal_request_delta=original.nominal_request_delta
    protocol=read(root/"tasks"/task/"protocol.json")
    trace=root/"traces"/task/policy
    states=sorted(trace.glob("state_*"))
    if [int(state.name.split("_")[1]) for state in states]!=protocol["tasks"][task]["test_state_ids"]:
        raise ValueError("Open-loop intervention analysis needs all50 predetermined test states")
    open_rows=[]
    controller_path=root/"native_controller_contract.json"
    controller=json.loads(controller_path.read_text())
    for state in states:
        record=json.loads((state/"open_loop_summary.json").read_text())
        if not record.get("complete") or {row["condition"] for row in record["conditions"]}!=set(OPEN):
            raise ValueError("Expected all15 open-loop conditions")
        with np.load(state/"open_loop_actions.npz",allow_pickle=False) as archive:
            commands={name:archive[name] for name in archive.files if name.endswith("__decoded")}
        for row in record["conditions"]:
            decoded=row.get("decoded_controller_delta",row.get("physical_delta"))
            if row["condition"]=="none" and (row["normalized_delta"]["l2"]!=0 or decoded["l2"]!=0):
                raise ValueError("No-op intervention changed actions")
            open_rows.append({"task":task,"policy":policy,"state_id":int(state.name.split("_")[1]),"condition":row["condition"],
                              "normalized_l2":row["normalized_delta"]["l2"],"decoded_controller_l2":decoded["l2"],
                              "decoded_controller_xyz_l2":decoded["xyz_l2"],"severity":row["severity"],
                              "units":"LIBERO OSC/gripper controller command coordinates; not metres",
                              **nominal_request_delta(commands["none__decoded"],commands[row["condition"].replace(":","_")+"__decoded"],controller)})
    ratios=[]
    indexed={(row["state_id"],row["condition"]):row for row in open_rows}
    for state in protocol["tasks"][task]["test_state_ids"]:
        for mode in ("background_fill","mosaic"):
            a=indexed[state,f"target:{mode}"];b=indexed[state,f"distractor:{mode}"];control=indexed[state,f"random:{mode}"]
            ratios.append({"task":task,"policy":policy,"state_id":state,"replacement":mode,
                           "target_distractor_controller_sensitivity_ratio":(a["decoded_controller_l2"]+1e-12)/(b["decoded_controller_l2"]+1e-12),
                           "target_random_controller_sensitivity_ratio":(a["decoded_controller_l2"]+1e-12)/(control["decoded_controller_l2"]+1e-12),
                           "target_action_delta":a["decoded_controller_l2"],"distractor_action_delta":b["decoded_controller_l2"],
                           "matched_random_action_delta":control["decoded_controller_l2"],"epsilon":1e-12,
                           "target_pixel_severity":a["severity"],"distractor_pixel_severity":b["severity"],"random_pixel_severity":control["severity"],
                           "caveat":"Target/distractor areas and pixel severities differ; ratio alone is not object importance. Random control matches target mask area."})
    aggregates=[]
    for condition in OPEN:
        values=np.asarray([row["decoded_controller_l2"] for row in open_rows if row["condition"]==condition])
        rng=np.random.default_rng(20261031)
        boot=rng.choice(values,size=(20000,len(values))).mean(1)
        aggregates.append({"condition":condition,"states":len(values),"mean_decoded_controller_l2":float(values.mean()),
                           "paired_state_bootstrap95_mean":np.quantile(boot,[.025,.975]).tolist()})
    return open_rows,ratios,aggregates,binding(controller_path)


def analyze_policy(root, policy, certificate):
    """Versioned consumer: existing statistics for defined conditions; full-plan NA otherwise."""
    from rl_vla.diagnostics.causal_structural_validation_v1 import POLICIES as NATIVE_POLICIES
    from rl_vla.diagnostics.figure_artifacts import artifact
    root=Path(root).resolve()
    canonical=f"rldt_{int(policy.split('_')[1])}" if policy.startswith('rldt_') else policy
    if canonical not in NATIVE_POLICIES or certificate['coverage']['total_resolved']!=360:
        raise ValueError('Exact accepted Object1 execution certificate required')
    policy=f"rldt_{int(canonical.split('_')[1]):06d}" if canonical.startswith('rldt_') else canonical
    output=root/ANALYSIS/TASK/policy
    output.mkdir(parents=True,exist_ok=False)
    original=original_consumer()
    directory=root/RAW/TASK/canonical
    case_records=certificate.get('cases')
    if case_records is None:case_records=read(check_binding(certificate['case_records']))['cases']
    cases=[case for case in case_records if case['policy']==canonical]
    reports={};proxy_reports={};choices=[];supports={};availability_rows=[];events=[]
    for condition in CLOSED:
        selected=[case for case in cases if case['condition']==condition]
        if [case['episode'] for case in selected]!=list(range(10)):
            raise ValueError('Execution certificate omits an exact original condition identity')
        rows=[case['row'] for case in selected]
        witnesses={case['episode']:case for case in selected if case['status']=='structural_NA'}
        support=availability(rows,list(range(150,160)),evidence_validator=lambda row,witnesses=witnesses:check_binding(witnesses[row['episode']]['structural_evidence']))
        supports[condition]=support;reports[condition]=rows
        availability_rows.append({'task':TASK,'policy':policy,'condition':condition,**support,
                                  'primary_proxy_available':support['primary_outcome_available'],
                                  'completed_native_successes_scope':'Descriptive completed-case count only; not a primary rate when any case is unavailable'})
        proxies=[]
        for case in selected:
            row=case['row']
            identity={'task':TASK,'policy':policy,'condition':condition,'state_id':case['state_id'],'seed':case['seed']}
            if case['status']=='structural_NA':
                events.append({**identity,'episode':case['episode'],'structural_evidence':case['structural_evidence']})
                choices.append({**identity,'status':'structural_NA','native_success':None,'telemetry':None,
                                'first_grasp_candidate':None,'first_grasp_candidate_role':None,'first_candidate_step':None,
                                'all_sustained_grasp_candidates':None,'minimum_target_gripper_distance_m':None,
                                'minimum_distractor_gripper_distance_m':None,'reviewed':False,
                                'interpretation':'Undefined original control; partial trajectory excluded from all full-episode contact/outcome proxies'})
                proxies.append(dict(row,success=None))
            else:
                telemetry_path=check_binding(case['artifacts']['telemetry'])
                choice=original.contact_choice(read(telemetry_path),row['target'],row['distractor'])
                choices.append({**identity,'status':'completed','native_success':row['success'],'telemetry':str(telemetry_path),**choice})
                proxies.append(dict(row,success=choice['first_grasp_candidate_role']=='target'))
        proxy_reports[condition]=proxies
    if not supports['none']['primary_outcome_available']:
        raise ValueError('Unmodified native control cannot be structurally unavailable')
    paired=[dict(task=TASK,policy=policy,condition=condition,**paired_primary(reports['none'],reports[condition],supports[condition])) for condition in CLOSED]
    proxy_pairs=[dict(task=TASK,policy=policy,condition=condition,**paired_primary(proxy_reports['none'],proxy_reports[condition],supports[condition])) for condition in CLOSED]
    # Do not accidentally report native-success counts as proxy counts.
    for row,condition in zip(proxy_pairs,CLOSED,strict=True):
        row['completed_native_successes']=None
        row['completed_target_first_grasp_candidates']=sum(r['success'] is True for r in proxy_reports[condition])
        row['scope']='Unreviewed complete-episode contact/lift proxy; any structural NA makes whole-condition primary proxy rate/effect unavailable'
    open_rows,ratios,aggregates,controller=open_loop(root,TASK,policy)
    for name,rows in (('closed_loop_pairs.csv',paired),('open_loop_actions.csv',open_rows),
                     ('grasp_choice_proxies.csv',choices),('paired_target_grasp_proxies.csv',proxy_pairs),
                     ('causal_sensitivity_ratios.csv',ratios),('condition_availability.csv',availability_rows)):
        write_csv(output/name,rows)
    atomic_json(output/'structural_events.json',{'task':TASK,'policy':policy,'events':events,
                 'scope':'Evidence-backed undefined original inputs, not native success/failure outcomes'})
    inputs=[binding(root/'compiler_profile_integration/causal_structural_na_v1/execution_coverage.json'),
            certificate['queue_completion'],certificate['amendment'],certificate['root_acceptance'],controller,
            binding(directory/'summary.json'),binding(root/'methods/rldt/object_1/selection.json'),
            binding(Path(original.__file__)),binding(Path(__file__)),binding(Path(__file__).with_name('metrics.py'))]
    for state in sorted((root/'traces'/TASK/policy).glob('state_*')):
        inputs.extend(binding(state/name) for name in ('open_loop_summary.json','open_loop_actions.npz'))
    atomic_json(output/'input_manifest.json',{'inputs':inputs,'statistics':'Original fixed20,000-bootstrap/seed20261031 settings unchanged; no complete-case rates for unavailable conditions'})
    _plot(output,aggregates,paired)
    required=['closed_loop_pairs.csv','open_loop_actions.csv','grasp_choice_proxies.csv','paired_target_grasp_proxies.csv',
              'causal_sensitivity_ratios.csv','condition_availability.csv','structural_events.json','input_manifest.json',
              'open_loop_sensitivity.png','open_loop_sensitivity.pdf','closed_loop_availability.png','closed_loop_availability.pdf']
    artifacts=[artifact(output/name) for name in required]
    manifest={'schema':'support-aware-causal-artifacts-v1','complete':True,'passed':True,'task':TASK,'policy':policy,
              'artifacts':artifacts,'inputs':inputs,'validator_source':binding(__file__),
              'required_filenames':required,'statistics_recomputed_for_open_loop':'Exact original frozen calculation; all50states and15conditions retained'}
    atomic_json(output/'manifest.json',manifest)
    summary={'complete':True,'passed':True,'task':TASK,'policy':policy,'lineage':'strict_support_structural_na_v1',
             'closed_loop_conditions':9,'paired_episodes_per_condition':10,'open_loop_conditions':15,'open_loop_states':50,
             'attempted_cases':90,'completed_outcome_cases':sum(s['completed_outcome_cases'] for s in supports.values()),
             'structural_na_cases':sum(s['structural_na_cases'] for s in supports.values()),
             'paired_success':paired,'paired_target_first_grasp_proxy':proxy_pairs,'open_loop_aggregates':aggregates,
             'availability':availability_rows,'structural_events':events,'manifest':binding(output/'manifest.json'),
             'artifacts':artifacts,'execution_coverage':inputs[0],'amendment':certificate['amendment'],
             'root_acceptance':certificate['root_acceptance'],'selection_sha256':sha(root/'methods/rldt/object_1/selection.json'),
             'controller_contract_sha256':controller['sha256'],
             'units':'Decoded deltas are LIBERO OSC/gripper controller commands; nominal clipped/scaled request deltas are not achieved displacement',
             'limitations':['Any undefined original control makes its whole10-case primary success and paired/proxy effects unavailable; no imputation or complete-case estimate',
                            'All valid original256 completed episodes and23 whole conditions retained byte-identically',
                            'Single RL seed; first10 exploratory closed-loop cases; no checkpoint reselection',
                            'Contact/lift choices are unreviewed proxies; partial NA prefixes are never classified',
                            'Direct image/text-key knockouts can retain modality information in contextualized other prefix keys']}
    atomic_json(output/'summary.json',summary)
    return summary


def _plot(output,aggregates,paired):
    import matplotlib
    matplotlib.use('Agg')
    from textwrap import fill

    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(13,9))
    y=np.arange(len(aggregates))
    means=np.asarray([r['mean_decoded_controller_l2'] for r in aggregates])
    intervals=np.asarray([r['paired_state_bootstrap95_mean'] for r in aggregates])
    ax.errorbar(means,y,xerr=np.stack((means-intervals[:,0],intervals[:,1]-means)),fmt='o',capsize=3)
    ax.set_yticks(y,[r['condition'].replace(':',' / ') for r in aggregates]);ax.invert_yaxis()
    ax.set_xlabel('Decoded controller-command L2 change (50 matched states; bootstrap95% CI)')
    ax.set_title('Fixed-input open-loop sensitivity — unchanged original50-state analysis')
    ax.grid(axis='x',alpha=.25);fig.subplots_adjust(left=.30,right=.97,bottom=.12,top=.92)
    for extension in ('png','pdf'):fig.savefig(output/f'open_loop_sensitivity.{extension}',dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,7.5))
    for i,row in enumerate(paired):
        if row['primary_outcome_available']:
            lower,upper=row['paired_bootstrap95'];value=row['difference']
            ax.errorbar(value,i,xerr=[[value-lower],[upper-value]],fmt='o',capsize=4,color='#155a8a')
            label=f"{row['candidate_successes']}/10; Δ={value:+.2f} [{lower:+.2f}, {upper:+.2f}]"
        else:
            label=f"STRUCTURAL NA: {row['completed_outcome_cases']} native + {row['structural_na_cases']} undefined / 10 planned"
            ax.axhspan(i-.32,i+.32,facecolor='#eeeeee',hatch='///',edgecolor='#bbbbbb')
        ax.text(1.01,i,label,transform=ax.get_yaxis_transform(),va='center',fontsize=9)
    ax.axvline(0,color='black',linewidth=.7);ax.set_xlim(-1.05,1.05)
    ax.set_yticks(range(len(paired)),[r['condition'].replace(':',' / ') for r in paired]);ax.invert_yaxis()
    ax.set_xlabel('Paired success-rate change against none (original10 planned cases; bootstrap95% CI)')
    ax.set_title('Closed-loop primary effects and exact-control availability')
    fig.text(.02,.025,fill('Structural NA is neither success nor failure. If any exact original background control is undefined, the entire condition primary success and paired/proxy effects are unavailable. No complete-case mean is substituted.',150),fontsize=9)
    fig.subplots_adjust(left=.23,right=.65,bottom=.16,top=.91)
    for extension in ('png','pdf'):fig.savefig(output/f'closed_loop_availability.{extension}',dpi=160)
    plt.close(fig)


def verify_policy(root,policy,*,full_hash=False):
    canonical=f"rldt_{int(policy.split('_')[1])}" if policy.startswith('rldt_') else policy
    padded=f"rldt_{int(canonical.split('_')[1]):06d}" if canonical.startswith('rldt_') else canonical
    path=Path(root)/ANALYSIS/TASK/padded/'summary.json'
    if not path.exists():return {'task':TASK,'policy':canonical,'status':'pending','complete':False,'passed':None}
    try:
        summary=read(path)
        if summary.get('complete') is not True or summary.get('task')!=TASK or summary.get('policy')!=padded or summary.get('closed_loop_conditions')!=9 or summary.get('paired_episodes_per_condition')!=10 or summary.get('open_loop_conditions')!=15 or summary.get('open_loop_states')!=50 or summary.get('attempted_cases')!=90:
            raise ValueError('Support-aware causal cell does not cover the full fixed scope')
        manifest=read(check_binding(summary['manifest']))
        expected={'closed_loop_pairs.csv','open_loop_actions.csv','grasp_choice_proxies.csv','paired_target_grasp_proxies.csv',
                  'causal_sensitivity_ratios.csv','condition_availability.csv','structural_events.json','input_manifest.json',
                  'open_loop_sensitivity.png','open_loop_sensitivity.pdf','closed_loop_availability.png','closed_loop_availability.pdf'}
        if manifest.get('complete') is not True or manifest.get('passed') is not True or manifest['validator_source']!=binding(__file__) or set(manifest['required_filenames'])!=expected or {Path(a['path']).name for a in manifest['artifacts']}!=expected or len(manifest['artifacts'])!=len(expected):
            raise ValueError('Support-aware causal artifact manifest is incomplete or stale')
        if summary['artifacts']!=manifest['artifacts']:raise ValueError('Summary and manifest artifact lists differ')
        for item in manifest['inputs']:check_binding(item)
        for item in manifest['artifacts']:
            stat=Path(item['path']).stat()
            if item['identity']!=[stat.st_size,stat.st_mtime_ns,stat.st_ino] or full_hash and sha(item['path'])!=item['sha256']:
                raise ValueError('Bound support-aware figure/table changed')
        for field in ('paired_success','paired_target_first_grasp_proxy'):
            rows=summary[field]
            if [r['condition'] for r in rows]!=list(CLOSED):raise ValueError('Missing whole-condition primary output')
            for row in rows:
                if row['structural_na_cases'] and (row['primary_outcome_available'] is not False or any(row.get(k,'missing') is not None for k in ('candidate_rate','difference','paired_bootstrap95','candidate_successes'))):
                    raise ValueError('Unavailable condition was converted into a primary estimate')
        if sum(row['structural_na_cases'] for row in summary['availability'])!=summary['structural_na_cases'] or summary['completed_outcome_cases']+summary['structural_na_cases']!=90:
            raise ValueError('Support-aware causal case accounting differs')
        return {'task':TASK,'policy':canonical,'status':'complete','complete':True,'passed':True,'summary':binding(path),
                'manifest':summary['manifest'],'artifacts':manifest['artifacts'],
                'completed_outcome_cases':summary['completed_outcome_cases'],'structural_na_cases':summary['structural_na_cases']}
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {'task':TASK,'policy':canonical,'status':'failed','complete':False,'passed':False,'error':str(error)}
