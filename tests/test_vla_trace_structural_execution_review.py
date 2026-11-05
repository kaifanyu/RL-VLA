"""Independent mutations of the raw support-aware execution consumer."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from test_vla_trace_structural_execution import fixture_campaign, structural_case, write

from rl_vla.diagnostics import causal_structural_validation_v1 as validator


@pytest.mark.parametrize("mutation", ["mask_dtype", "mask_complement", "telemetry_order", "telemetry_tail", "severity_count", "prompt", "nonfinite_actions", "sampling_key", "key_after_split", "noise_shape", "artifact_outside"])
def test_rehashed_structural_prefix_mutations_are_rejected(tmp_path, mutation):
    row, amendment, evidence = structural_case(tmp_path)
    files = {Path(item['path']).name: Path(item['path']) for item in evidence['artifacts']}
    table = {'sampling_key': np.tile(np.array([1,2],np.uint32),(10,56,1)),
             'key_after_split': np.tile(np.array([3,4],np.uint32),(10,56,1))}
    def validate():
        return validator.validate_structural_case(row, policy='rldt_100', condition='random:background_fill', index=8,
            amendment=amendment, expected_prompt='pick cream cheese', key_table=table)
    validate()
    if mutation.startswith('mask_'):
        path=files['masks.npz']
        with np.load(path) as data: arrays={k:data[k].copy() for k in data.files}
        if mutation=='mask_dtype': arrays['image__cream_cheese_1']=arrays['image__cream_cheese_1'].astype(np.uint8)
        else: arrays['wrist_image__unowned_background'][200,200]=False
        np.savez(path,**arrays)
    elif mutation.startswith('telemetry_'):
        path=files['partial_telemetry.json']; data=json.loads(path.read_text())
        if mutation=='telemetry_order': data[1]['primitive_steps']=1
        else: data[-1]['simulator_state'][1]=2
        write(path,data)
    elif mutation=='severity_count':
        write(files['partial_intervention_severity.json'],[])
    elif mutation=='artifact_outside':
        source=files['masks.npz']; alternative=tmp_path/'other_masks'/source.name
        alternative.parent.mkdir();alternative.write_bytes(source.read_bytes());files['masks.npz']=alternative
    else:
        path=files['observation.npz']
        with np.load(path) as data: arrays={k:data[k].copy() for k in data.files}
        if mutation=='prompt': arrays['prompt']=np.asarray('pick soup')
        elif mutation=='nonfinite_actions': arrays['normalized_action_prefix'][0,0,0]=np.nan
        elif mutation=='noise_shape': arrays['initial_noise']=np.zeros((1,9,32))
        else: arrays[mutation][0]+=1
        np.savez(path,**arrays)
    evidence['artifacts']=[validator.binding(files[name]) for name in files]
    row['structural_evidence']=write(Path(row['structural_evidence']['path']),evidence)
    with pytest.raises(ValueError): validate()


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    # Native key regeneration is covered by the pinned real298-reference audit and
    # dedicated structural-case tests; this fixture exercises raw360-case accounting.
    monkeypatch.setattr(validator,'key_recurrence',lambda *_: ({},{}))
    raw=fixture_campaign(tmp_path)
    assert validator.validate_execution(tmp_path)['coverage']['total_resolved']==360
    return tmp_path,raw


@pytest.mark.parametrize('field,value', [('seed',100000),('initial_state_id',150),('checkpoint','wrong'),
    ('technical_smoke',True),('status','unsupported_error'),('success',None)])
def test_rehashed_new_native_episode_identity_and_outcome_rejected(campaign,field,value):
    run,raw=campaign
    cell=raw/'rldt_100/random_background_fill'
    path=cell/'summary.json'; report=json.loads(path.read_text())
    report['episodes'][8][field]=value
    write(path,report)
    (cell/'episodes.jsonl').write_text('\n'.join(json.dumps(row) for row in report['episodes'])+'\n')
    with pytest.raises(ValueError):validator.validate_execution(run)


@pytest.mark.parametrize('mutation',['action_shape','missing_video','telemetry_gap','wrong_reset_task','false_noop'])
def test_new_native_artifacts_and_reset_are_required(campaign,mutation):
    run,raw=campaign
    cell=raw/'rldt_100/random_background_fill'
    if mutation=='action_shape':np.savez(cell/'episode_008_actions.npz',normalized=np.zeros((1,9,32)))
    elif mutation=='missing_video':(cell/'episode_008.mp4').write_bytes(b'')
    elif mutation=='telemetry_gap':
        path=cell/'episode_008_telemetry.json'; rows=json.loads(path.read_text());rows.pop();write(path,rows)
    elif mutation=='false_noop':
        write(cell/'no_op.json',{'passed':True,'max_abs_difference':.01,'persisted_noise_recurrence_max_abs_difference':0.})
    else:
        path=cell/'summary.json';report=json.loads(path.read_text());report['episodes'][8]['reset']['task_id']=2
        write(path,report);(cell/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in report['episodes'])+'\n')
    with pytest.raises(ValueError):validator.validate_execution(run)


def test_case_publication_is_immutable_and_binds_all360(campaign):
    run,_=campaign
    certificate=validator.validate_execution(run)
    published=validator.publish_execution(run,certificate)
    assert len(json.loads(Path(published['case_records']['path']).read_text())['cases'])==360
    assert validator.publish_execution(run,certificate)==published
    changed=copy.deepcopy(certificate);changed['cases'][0]['seed']+=1
    with pytest.raises(ValueError,match='immutable'):
        validator.publish_execution(run,changed)


def test_full360_resolution_keeps_one_structural_case_outside_native_outcomes(campaign,monkeypatch):
    run,raw=campaign
    cell=raw/'rldt_100/random_background_fill'
    queue_path=raw/'queue_complete.json';queue=json.loads(queue_path.read_text())
    report_path=cell/'summary.json';report=json.loads(report_path.read_text())
    partial,_,witness=structural_case(cell)
    witness['amendment']=queue['amendment']
    row={**report['episodes'][8],**partial}
    witness['reset']=row['reset']
    row['structural_evidence']=write(Path(partial['structural_evidence']['path']),witness)
    report['episodes'][8]=row
    report.update(completed_outcome_cases=9,structural_na_cases=1,primary_outcome_available=False,
                  successes=None,primary_success_rate=None,primary_paired_comparison=None)
    write(report_path,report)
    (cell/'episodes.jsonl').write_text('\n'.join(json.dumps(r) for r in report['episodes'])+'\n')
    retained=Path(row['structural_evidence']['path']).parent
    (cell/'episode_008_telemetry.json').write_bytes((retained/'partial_telemetry.json').read_bytes())
    with np.load(retained/'observation.npz') as data:
        np.savez(cell/'episode_008_actions.npz',normalized=data['normalized_action_prefix'])
    write(run/'observation_bank/object_1/test/state_000158/metadata.json',{'prompt':'pick cream cheese'})
    top_path=cell.parent/'summary.json';top=json.loads(top_path.read_text())
    top['conditions'][5]={k:v for k,v in report.items() if k!='episodes'}
    top['primary_outcomes_all_available']=False
    top_binding=write(top_path,top)
    queue['policy_summaries'][3].update(summary=top_binding,native_completed=89,structural_NA=1)
    queue['coverage'].update(native_completed=359,structural_NA=1)
    write(queue_path,queue)
    keys={'sampling_key':np.tile(np.array([1,2],np.uint32),(10,56,1)),
          'key_after_split':np.tile(np.array([3,4],np.uint32),(10,56,1))}
    monkeypatch.setattr(validator,'key_recurrence',lambda *_:(keys,{}))
    certificate=validator.validate_execution(run)
    assert certificate['coverage']=={'planned_cases':360,'native_completed':359,'structural_NA':1,'total_resolved':360}
    unavailable=[c for c in certificate['cases'] if c['status']=='structural_NA']
    assert len(unavailable)==1 and unavailable[0]['row']['success'] is None
    assert 'artifacts' not in unavailable[0]
    assert len([c for c in certificate['cases'] if c['status']=='completed'])==359
    # A complete-case9-case success fraction cannot masquerade as the planned10 effect.
    report.update(successes=4,primary_success_rate=4/9)
    write(report_path,report)
    with pytest.raises(ValueError,match='primary estimate'):
        validator.validate_execution(run)
