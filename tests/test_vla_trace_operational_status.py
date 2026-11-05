import json

from rl_vla.diagnostics.operational_status import snapshot


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def launch(root):
    put(root/'diagnostic_policy_dispatch_launch/object_1_gpu6_v1.json',
        {'task': 'object_1', 'pid': 123, 'command': ['python', 'queue.py']})


def row(root, process=None):
    return next(v for v in snapshot(root, process_reader=lambda _: process)['rows'] if v['task']=='object_1' and v['domain']=='causal')


def test_explicit_failed_queue_is_failed_even_with_partial_conditions(tmp_path):
    launch(tmp_path)
    put(tmp_path/'interventions/object_1/queue_status.json', {'state':'failed','error':'Insufficient allowed pixels','completed_conditions':5})
    result=row(tmp_path)
    assert result['state']=='failed' and 'pixels' in result['reason']
    assert not result['completion_marker_present']


def test_parent_exit_without_completion_is_failure(tmp_path):
    launch(tmp_path)
    put(tmp_path/'interventions/object_1/queue_status.json', {'state':'running'})
    assert row(tmp_path)['state']=='failed'
    assert row(tmp_path, {'state':'Z','command':['python','queue.py'],'start_ticks':1})['state']=='failed'
    assert row(tmp_path, {'state':'S','command':['unrelated'],'start_ticks':1})['state']=='failed'


def test_live_parent_waits_and_completed_parent_may_exit(tmp_path):
    launch(tmp_path)
    assert row(tmp_path, {'state':'S','command':['python','queue.py'],'start_ticks':1})['state']=='pending'
    put(tmp_path/'interventions/object_1/queue_complete.json', {'complete':True,'passed':True})
    assert row(tmp_path)['state']=='complete'


def test_missing_launch_is_pending_and_failed_completion_never_passes(tmp_path):
    assert row(tmp_path)['state']=='pending'
    put(tmp_path/'interventions/object_1/queue_complete.json', {'complete':True,'passed':False})
    assert row(tmp_path)['state']=='failed'


def test_ood_process_identity_accounts_for_taskset_and_start_ticks(tmp_path):
    put(tmp_path/'ood_per_lane_release_review_v1/launch_manifest_v1.json', {'new_queues':{'object_1':{'parent':{
        'pid':123,'start_ticks':7,'command':['taskset','-c','77','python','ood.py']}}}})
    process={'state':'S','start_ticks':7,'command':['python','ood.py']}
    result=next(v for v in snapshot(tmp_path,process_reader=lambda _:process)['rows'] if v['domain']=='ood' and v['task']=='object_1')
    assert result['state']=='pending' and result['parent_alive']
    process['start_ticks']=8
    result=next(v for v in snapshot(tmp_path,process_reader=lambda _:process)['rows'] if v['domain']=='ood' and v['task']=='object_1')
    assert result['state']=='failed'
