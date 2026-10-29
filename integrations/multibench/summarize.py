"""Collect comparable success rates, interaction budgets, and time costs."""
from __future__ import annotations
import argparse
import csv
import io
import json
from pathlib import Path
from multibench.run_experiment import write


def summarize(campaign):
    campaign=Path(campaign).resolve()
    rows=[]
    for directory in sorted((campaign/'experiments').iterdir()):
        if not (directory/'experiment.json').is_file():
            continue
        spec=json.loads((directory/'experiment.json').read_text())
        status=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {}
        full_training_seconds=next((stage['elapsed_seconds'] for stage in status.get('stages', [])
                                    if stage['name']=='train'), None)
        validations={}
        path=directory/'checkpoint_validations.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                record=json.loads(line)
                validations[record['iteration']]=record
        selected=json.loads((directory/'selected_checkpoint.json').read_text()) if (directory/'selected_checkpoint.json').exists() else None
        reports=[('baseline_validation',0),('baseline_test',0)]
        reports += [(f'validation_{i:06d}',i) for i in (25,50,100)]
        reports += [('selected_test',selected['iteration'] if selected else None)]
        for name,iteration in reports:
            path=directory/f'{name}.json'
            if not path.exists():
                continue
            report=json.loads(path.read_text())
            counters=validations.get(iteration,{}).get('training_counters',{}) if iteration else {}
            rows.append({'benchmark':spec['benchmark'],'method':'baseline' if iteration==0 else spec['method'],
                         'backend':'jax' if spec['method']=='rldt' else 'torch',
                         'split':report['split'],'iteration':iteration,'successes':report['successes'],
                         'episodes':len(report['episodes']),'success_rate':report['success_rate'],
                         'train_controller_steps':counters.get('primitive_steps',0 if iteration==0 else None),
                         'train_policy_decisions':counters.get('decisions',0 if iteration==0 else None),
                         'train_seconds_excluding_validation':counters.get('elapsed_seconds',0 if iteration==0 else None),
                         'full_training_stage_wall_seconds':full_training_seconds if iteration else 0,
                         'evaluation_controller_steps':report['controller_steps'],
                         'evaluation_policy_decisions':report['policy_decisions'],
                         'evaluation_seconds':report['elapsed_seconds'],
                         'evaluation_model_load_seconds':report['model_load_seconds'],
                         'report':str(path)})
    result={'rows':rows,'note':'Compare success rates only within the same split. Each backend has its own raw-base baseline. train_seconds_excluding_validation is cumulative learner time through the checkpoint. full_training_stage_wall_seconds is the completed 100-iteration subprocess total, including startup, validation, and shutdown; it is not cumulative time through each earlier checkpoint.',
            'training_clock_scope':{'rldt':'Learner clock begins after model load and simulator process creation, before initial reset and optimizer initialization.',
                                    'flow_sde':'Learner clock begins at runner.run(), after Ray workers/model/optimizers initialize.'}}
    write(campaign/'results.json',result)
    if rows:
        buffer=io.StringIO()
        writer=csv.DictWriter(buffer,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
        temporary=campaign/'results.csv.tmp'
        temporary.write_text(buffer.getvalue());temporary.replace(campaign/'results.csv')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    args=parser.parse_args()
    result=summarize(args.campaign)
    print(json.dumps(result,indent=2))
