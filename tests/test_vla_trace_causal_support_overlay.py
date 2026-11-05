import json

from rl_vla.diagnostics import causal_support_overlay_v1 as overlay


def test_no_unapproved_lineage_adoption(tmp_path):
    assert not overlay.adoption(tmp_path)['active']
    path=tmp_path/overlay.APPROVAL;path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'approved':False,'passed':False}))
    assert overlay.adoption(tmp_path)['passed'] is False
    assert overlay.execution(tmp_path)['status']=='failed'


def test_active_resume_does_not_erase_original_failure(tmp_path,monkeypatch):
    original={'rows':[{'domain':'causal','task':'object_1','state':'failed','reason':'original exhausted support'}],
              'failures':[{}],'failed_queues':1}
    from rl_vla.diagnostics import operational_status
    monkeypatch.setattr(operational_status,'snapshot',lambda *_,**__:original)
    monkeypatch.setattr(overlay,'adoption',lambda _: {'active':True,'passed':True,'complete':True})
    monkeypatch.setattr(overlay,'execution',lambda _: {'active':True,'complete':False,'passed':None,'status':'pending'})
    directory=tmp_path/'interventions_structural_na_v1/object_1';directory.mkdir(parents=True)
    (directory/'queue_plan.json').write_text(json.dumps({'parent':{'pid':45,'start_ticks':123}}))
    (directory/'queue_status.json').write_text(json.dumps({'state':'running'}))
    result=overlay.operational_snapshot(tmp_path,process_reader=lambda _: {'state':'S','start_ticks':123})
    assert result['failed_queues']==0 and result['rows'][0]['state']=='pending'
    assert result['rows'][0]['superseded_attempt']['state']=='failed'
    stopped=overlay.operational_snapshot(tmp_path,process_reader=lambda _:None)
    assert stopped['rows'][0]['state']=='failed'


def test_figure_overlay_requires_unaffected23_cells(tmp_path,monkeypatch):
    monkeypatch.setattr(overlay,'adoption',lambda _: {'active':True,'passed':True,'complete':True})
    monkeypatch.setattr(overlay,'execution',lambda *_,**__: {'complete':True,'passed':True,'status':'complete'})
    monkeypatch.setattr(overlay.support,'verify_policy',lambda _,policy,**__: {'task':'object_1','policy':policy,'complete':True,'passed':True,'status':'complete'})
    result=overlay.figure_coverage(tmp_path)
    assert result['completed_cells']==4 and result['pending_cells']==23
    assert result['complete'] is False and result['passed'] is None
