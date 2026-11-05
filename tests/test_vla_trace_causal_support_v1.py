import copy

import pytest

from rl_vla.diagnostics.causal_support_v1 import availability, paired_primary
from rl_vla.diagnostics.metrics import paired_success


def episodes():
    return [{'episode':i,'seed':100000+i,'reset':{'initial_state_id':150+i},'condition':'random:background_fill',
             'primitive_steps':280,'success':i<7} for i in range(10)]


def test_available_ten_cases_reproduce_original_statistics_exactly():
    rows=episodes();support=availability(rows,list(range(150,160)),evidence_validator=lambda _:None)
    result=paired_primary(rows,rows,support)
    assert {k:result[k] for k in paired_success(rows,rows)}==paired_success(rows,rows)
    assert result['primary_outcome_available'] and result['completed_outcome_cases']==10


def test_one_na_makes_entire_condition_primary_unavailable_without_imputation():
    baseline=episodes();rows=copy.deepcopy(baseline);rows[8].update(status='structural_NA',success=None)
    checked=[];support=availability(rows,list(range(150,160)),evidence_validator=lambda row:checked.append(row['episode']))
    result=paired_primary(baseline,rows,support)
    assert checked==[8] and result['completed_outcome_cases']==9 and result['structural_na_cases']==1
    assert result['completed_native_successes']==7 and not result['primary_outcome_available']
    for key in ('candidate_successes','candidate_rate','difference','paired_bootstrap95','recovered','regressed','mcnemar_exact_two_sided_p'):
        assert result[key] is None
    assert rows[8]['success'] is None and len(rows)==10


def test_unverified_na_wrong_condition_or_nonboolean_native_outcome_rejected():
    rows=episodes();rows[8].update(status='structural_NA',success=None)
    def reject(_):raise ValueError('support witness missing')
    with pytest.raises(ValueError,match='witness'):availability(rows,list(range(150,160)),evidence_validator=reject)
    rows[8]['condition']='target:mosaic'
    with pytest.raises(ValueError,match='restricted'):availability(rows,list(range(150,160)),evidence_validator=lambda _:None)
    rows[8].pop('status');rows[8]['condition']='random:background_fill'
    with pytest.raises(ValueError,match='Boolean'):availability(rows,list(range(150,160)),evidence_validator=lambda _:None)


def test_na_cannot_be_dropped_or_shift_noise_identity():
    rows=episodes()
    with pytest.raises(ValueError,match='all10'):availability(rows[:-1],list(range(150,160)),evidence_validator=lambda _:None)
    rows[-1]['seed']=100000
    with pytest.raises(ValueError,match='noise'):availability(rows,list(range(150,160)),evidence_validator=lambda _:None)


def test_supported_figures_publish_last_and_keep_na_primary_empty(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from rl_vla.diagnostics import causal_support_v1 as s
    def write(relative,value):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return s.binding(path)
    acceptance=write('acceptance.json',{'passed':True});amendment=write('amendment.json',{'approved':True})
    queue=write('queue.json',{'complete':True});witness=write('witness.json',{'status':'structural_NA'})
    telemetry=write('telemetry.json',[{'primitive_steps':1}])
    write('interventions_structural_na_v1/object_1/pi05_libero/summary.json',{'complete':True})
    write('methods/rldt/object_1/selection.json',{'locked':True})
    controller=write('native_controller_contract.json',{'test':'unchanged openloop fixture'})
    cases=[]
    for condition in s.CLOSED:
        for i in range(10):
            row={'episode':i,'seed':100000+i,'reset':{'initial_state_id':150+i},'condition':condition,
                 'primitive_steps':5,'success':i<7,'target':'cream_cheese_1','distractor':'alphabet_soup_1'}
            case={'policy':'pi05_libero','condition':condition,'episode':i,'state_id':150+i,'seed':100000+i,
                  'row':row,'status':'completed','artifacts':{'telemetry':telemetry}}
            if condition=='random:background_fill' and i==8:
                row.update(status='structural_NA',success=None)
                case.update(status='structural_NA',structural_evidence=witness)
            cases.append(case)
    certificate={'coverage':{'total_resolved':360},'cases':cases,'queue_completion':queue,'amendment':amendment,'root_acceptance':acceptance}
    write('compiler_profile_integration/causal_structural_na_v1/execution_coverage.json',certificate)
    original=SimpleNamespace(__file__=str(Path(s.__file__).parents[3]/'scripts/analyze_vla_trace_causal.py'),
             contact_choice=lambda *_:{'first_grasp_candidate_role':'target','reviewed':False})
    monkeypatch.setattr(s,'original_consumer',lambda:original)
    names=['none','no_image','no_text']+[f'{r}:{m}' for r in ('target','distractor','random','robot','background','old_target') for m in ('background_fill','mosaic')]
    aggregates=[{'condition':c,'states':50,'mean_decoded_controller_l2':.1,'paired_state_bootstrap95_mean':[.05,.15]} for c in names]
    monkeypatch.setattr(s,'open_loop',lambda *_:([{'state_id':i,'condition':c,'normalized_l2':0.,'decoded_controller_l2':.1} for i in range(150,200) for c in names],
                    [{'state_id':i,'replacement':m} for i in range(150,200) for m in ('background_fill','mosaic')],aggregates,controller))
    result=s.analyze_policy(tmp_path,'pi05_libero',certificate)
    assert result['completed_outcome_cases']==89 and result['structural_na_cases']==1
    for field in ('paired_success','paired_target_first_grasp_proxy'):
        row=next(r for r in result[field] if r['condition']=='random:background_fill')
        assert row['candidate_successes'] is None and row['difference'] is None and row['candidate_rate'] is None
    assert s.verify_policy(tmp_path,'pi05_libero',full_hash=True)['complete']
    assert len(result['artifacts'])==12
    path=tmp_path/'causal_analysis_structural_na_v1/object_1/pi05_libero/summary.json'
    result['paired_success'][5]['candidate_rate']=.7
    path.write_text(json.dumps(result))
    assert s.verify_policy(tmp_path,'pi05_libero')['status']=='failed'
