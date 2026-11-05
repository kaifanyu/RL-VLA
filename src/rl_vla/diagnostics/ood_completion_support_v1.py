"""Read-only OOD acceptance with explicit repaired-causal resource provenance."""
from pathlib import Path

from rl_vla.diagnostics import ood_completion as original
from rl_vla.diagnostics.causal_structural_validation_v1 import binding, read, verify


def bridge_contract(root):
    root=Path(root).resolve()
    marker_path=root/'interventions/object_1/queue_complete.json'
    if not marker_path.exists():return None
    marker=read(marker_path)
    if marker.get('resource_release_only') is not True:
        if (root/'compiler_profile_integration/causal_structural_na_v1/root_acceptance.json').exists():
            raise ValueError('An accepted successor requires the explicit resource-only bridge, not an old native-completion claim')
        return None
    from rl_vla.diagnostics.causal_release_bridge_v1 import validate_inputs
    acceptance_path=verify(marker['bindings']['bridge_acceptance'])
    data=validate_inputs(root,acceptance_path)
    if data is None:raise ValueError('OOD resource marker precedes accepted structural execution coverage')
    expected={'version':'strict-support-resource-release-v1','complete':True,'passed':True,'technical_only':False,
              'resource_release_only':True,'task':'object_1','policies':list(original.POLICIES),
              'episodes_per_condition':10,'all_recorded_owners_exited':True,'all_owned_process_groups_gone':True,
              'gpu_compute_processes_at_release':[],'exclusive_gpu_lease_held_at_publication':True,
              'execution_coverage':data['coverage']['coverage'],'bindings':data['bindings'],
              'superseded_failed_parent':data['old_owner']['parent'],'resume_parent':data['completion']['parent'],
              'child_exits':data['completion']['child_exits'],'observed_descendants':data['completion']['observed_descendants'],
              'owned_process_groups':data['groups'],'conditions':read(data['completion']['queue_plan']['path'])['conditions']}
    if any(marker.get(name,'missing')!=value for name,value in expected.items()):
        raise ValueError('OOD resource-only bridge changed its accepted source/coverage/exit claim')
    # Validate the saved device identity, not current availability: OOD may now own it.
    from rl_vla.diagnostics.causal_release_bridge_v1 import GPU_UUID
    if marker.get('gpu_uuid')!=GPU_UUID:raise ValueError('OOD resource-only bridge refers to another physical GPU')
    return {'marker':binding(marker_path),'bindings':data['bindings'],'coverage':data['coverage']['coverage'],
            'scope':'Saved resource-release evidence only; original failed causal queue remains failed and structural NA is not a native outcome'}


def completion(root,task):
    result=original.completion(root,task)
    if task!='object_1':return result
    try:
        bridge=bridge_contract(root)
        if bridge is not None:
            result={**result,'causal_resource_bridge':bridge,
                    'sources':{**result['sources'],bridge['marker']['path']:bridge['marker']['sha256'],
                               **{item['path']:item['sha256'] for item in bridge['bindings'].values()}}}
        return result
    except (OSError,ValueError,KeyError,TypeError) as error:
        return {**result,'complete':False,'passed':False,'reason':f'Invalid repaired-causal resource bridge: {error}'}
