"""Read-only selection of an explicitly accepted strict-support causal lineage."""
from pathlib import Path

from rl_vla.diagnostics import causal_support_v1 as support
from rl_vla.diagnostics import figure_artifacts as original_figures
from rl_vla.diagnostics.causal_structural_validation_v1 import APPROVAL, accepted_amendment


def adoption(root):
    path=Path(root)/APPROVAL
    if not path.exists():return {'active':False,'passed':None,'complete':False,'reason':'Strict-support amendment not yet accepted'}
    try:
        _,_,bindings=accepted_amendment(root)
        return {'active':True,'passed':True,'complete':True,**bindings}
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {'active':True,'passed':False,'complete':False,'reason':str(error)}


def execution(root,*,full_hash=False):
    adopted=adoption(root)
    path=Path(root)/'compiler_profile_integration/causal_structural_na_v1/execution_coverage.json'
    if not adopted['active']:return {**adopted,'status':'inactive'}
    if adopted['passed'] is False:return {**adopted,'status':'failed'}
    if not path.exists():return {**adopted,'complete':False,'passed':None,'status':'pending'}
    try:
        value=support.read(path)
        from rl_vla.diagnostics import causal_structural_validation_v1 as validator
        if value.get('schema')!='causal-structural-execution-coverage-v1' or value.get('complete') is not True or value.get('passed') is not True or value.get('task')!='object_1' or value['validator_source']!=support.binding(validator.__file__):
            raise ValueError('Strict-support execution certificate is incomplete or uses another validator')
        for name in ('queue_completion','root_acceptance','amendment','native_key_recurrence','case_records'):
            support.check_binding(value[name])
        queue=support.read(value['queue_completion']['path'])
        if value['policy_summaries']!=queue['policy_summaries'] or value['coverage']!=queue['coverage'] or value['coverage']['planned_cases']!=360 or value['coverage']['total_resolved']!=360 or value['coverage']['native_completed']+value['coverage']['structural_NA']!=360:
            raise ValueError('Strict-support certificate does not bind exact queue counts/policies')
        details=support.read(value['case_records']['path'])
        if len(details['cases'])!=360:raise ValueError('Strict-support certificate omits original cases')
        if full_hash:
            for item in details['native_artifacts']:support.check_binding(item)
            for witness in value['structural_cases']:
                support.check_binding(witness['evidence'])
                for item in witness['artifacts']:support.check_binding(item)
        return {'active':True,'status':'complete','complete':True,'passed':True,'certificate':support.binding(path),
                'coverage':value['coverage'],'value':value}
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {'active':True,'status':'failed','complete':False,'passed':False,'reason':str(error)}


def cell_ready(root,domain,task,policy):
    if domain=='causal_analysis' and task=='object_1' and adoption(root)['active']:
        return execution(root)['complete'] and support.verify_policy(root,policy)['complete']
    return original_figures.cell_ready(root,domain,task,policy)


def figure_coverage(root,*,full_hash=False):
    adopted=adoption(root)
    original_summary=Path(root)/'figure_artifacts/summary.json'
    global_error=None
    if original_summary.exists():
        declared=support.read(original_summary)
        if declared.get('passed') is False:global_error='Original figure validator reported failure'
        elif declared.get('complete') is True:
            try:original_figures.verify_coverage(root,full_hash=full_hash)
            except (OSError,ValueError,KeyError,TypeError) as error:global_error=str(error)
    execution_status=execution(root,full_hash=full_hash) if adopted['active'] else None
    rows=[]
    for domain,task,policy in original_figures.cells():
        if domain=='causal_analysis' and task=='object_1' and adopted['active']:
            row=support.verify_policy(root,policy,full_hash=full_hash)
            if not execution_status['complete']:
                row={**row,'status':execution_status['status'],'complete':False}
            rows.append({**row,'domain':domain});continue
        path=original_figures.manifest_path(root,domain,task,policy)
        if not path.exists():
            rows.append({'domain':domain,'task':task,'policy':policy,'status':'pending'});continue
        try:
            manifest=original_figures.verify_manifest(path,full_hash=full_hash)
            requirements=original_figures.requirements(root,domain,task,policy)
            if manifest.get('domain')!=domain or manifest.get('task')!=task or manifest.get('policy')!=policy or requirements is None or {str(p.resolve()) for p,_,_ in requirements}!={a['path'] for a in manifest['artifacts']}:
                raise ValueError('Original required figure artifact domain/coverage differs')
            rows.append({'domain':domain,'task':task,'policy':policy,'status':'complete','manifest':support.binding(path)})
        except (OSError,ValueError,KeyError,TypeError) as error:
            rows.append({'domain':domain,'task':task,'policy':policy,'status':'failed','error':str(error)})
    failed=sum(r['status']=='failed' for r in rows);pending=sum(r['status']=='pending' for r in rows)
    return {'complete':not failed and not pending and global_error is None,'passed':False if failed or global_error else None if pending else True,
            'global_error':global_error,
            'expected_cells':27,'completed_cells':sum(r['status']=='complete' for r in rows),
            'pending_cells':pending,'failed_cells':failed,'cells':rows,'adoption':adopted,
            'scope':'Original23 unaffected artifact cells plus4 accepted support-aware Object1 causal cells; no rewriting of original manifests'}


def readable_coverage(root,*,full_hash=False):
    from rl_vla.diagnostics.causal_figures import verify_coverage
    result=verify_coverage(root,full_hash=full_hash)
    if not adoption(root)['active']:return result
    status=execution(root,full_hash=full_hash)
    rows=[]
    for row in result['cells']:
        if row['task']=='object_1':
            row=support.verify_policy(root,row['policy'],full_hash=full_hash)
            if not status['complete']:row={**row,'status':status['status'],'complete':False,'passed':status['passed']}
        rows.append(row)
    failed=sum(r['status']=='failed' for r in rows);pending=sum(r['status']!='complete' and r['status']!='failed' for r in rows)
    return {'complete':not failed and not pending,'passed':False if failed else None if pending else True,
            'expected_cells':12,'completed_cells':sum(r['status']=='complete' for r in rows),
            'failed_cells':failed,'pending_cells':pending,'cells':rows,
            'scope':'Eight unchanged readable derivatives plus four accepted support-aware figures; original50-state open-loop statistics retained'}


def analysis_paths(root):
    root=Path(root)
    active=adoption(root)['active']
    paths=[p for p in sorted((root/'causal_analysis').glob('*/*/summary.json')) if not active or p.parent.parent.name!='object_1']
    if active and execution(root)['complete']:
        for policy in support.POLICIES:
            cell=support.verify_policy(root,policy)
            if cell['complete']:paths.append(Path(cell['summary']['path']))
    return paths


def operational_snapshot(root,*,process_reader=None):
    from rl_vla.diagnostics.operational_status import process, snapshot
    process_reader=process_reader or process
    original=snapshot(root,process_reader=process_reader)
    adopted=adoption(root)
    if not adopted['active']:return original
    root=Path(root);directory=root/'interventions_structural_na_v1/object_1'
    plan_path=directory/'queue_plan.json';status_path=directory/'queue_status.json'
    state='pending';reason='Accepted strict-support resumption has not published a queue plan';owner=None;alive=None
    if adopted['passed'] is False:state='failed';reason=adopted['reason']
    elif plan_path.exists():
        try:
            plan=support.read(plan_path);owner=plan['parent'];current=process_reader(owner['pid'])
            alive=current is not None and current['state']!='Z' and current['start_ticks']==owner.get('start_ticks',owner.get('proc_start_ticks'))
            status=support.read(status_path) if status_path.exists() else {}
            raw_complete=(directory/'queue_complete.json').exists()
            certified=execution(root)
            if status.get('state')=='failed' or certified['passed'] is False:
                state='failed';reason=status.get('error') or certified.get('reason','Structural resume failed')
            elif certified['complete']:state='complete';reason='Accepted360-case execution coverage; native outcomes and structural unavailability are distinct'
            elif not raw_complete and not alive:state='failed';reason='Accepted resume parent exited before complete raw coverage'
            else:reason='Accepted strict-support resumption or independent execution validation is pending'
        except (OSError,ValueError,KeyError,TypeError) as error:state='failed';reason=str(error)
    rows=[]
    for row in original['rows']:
        if row['domain']=='causal' and row['task']=='object_1':
            row={'domain':'causal','task':'object_1','state':state,'reason':reason,'parent':owner,'parent_alive':alive,
                 'status_path':str(status_path),'status_sha256':support.sha(status_path) if status_path.exists() else None,
                 'completion_path':str(directory/'queue_complete.json'),'completion_marker_present':(directory/'queue_complete.json').exists(),
                 'superseded_attempt':row,'adoption':adopted,'scope':'Original failed attempt preserved; accepted successor availability coverage selected explicitly'}
        rows.append(row)
    failures=[r for r in rows if r['state']=='failed']
    return {**original,'rows':rows,'failures':failures,'failed_queues':len(failures),'strict_support_adoption':adopted}
