"""CPU-only independent check of the first published strict-support witness."""
import argparse
import json
import time
from pathlib import Path

from rl_vla.diagnostics import causal_structural_validation_v1 as validator


def main(run):
    run=run.resolve()
    destination=run/'compiler_profile_integration/causal_structural_na_v1/first_witness_independent_v1'
    destination.mkdir(exist_ok=False)
    source=validator.binding(__file__)
    while True:
        if validator.binding(__file__)!=source:raise ValueError('Independent witness watcher source changed')
        candidates=sorted((run/validator.RAW).glob('*/*/episode_*_structural_na/summary.json'),key=lambda p:(p.stat().st_mtime_ns,str(p)))
        for path in candidates:
            evidence=validator.read(path)
            condition_folder=path.parent.parent
            log=condition_folder/'episodes.jsonl'
            if not log.exists():continue
            rows=[]
            for line in log.read_text().splitlines(keepends=True):
                if line.endswith('\n'):rows.append(json.loads(line))
            matches=[r for r in rows if r['episode']==evidence['episode']]
            if not matches:continue
            if len(matches)!=1:raise ValueError('Duplicate completed structural evidence row')
            row=matches[0]
            acceptance,_,approved=validator.accepted_amendment(run)
            table,key_audit=validator.key_recurrence(run,acceptance)
            policy=evidence['policy'];index=evidence['episode'];condition=evidence['condition']
            checkpoint='pi05_libero' if policy=='pi05_libero' else str(run/'methods/rldt/object_1/training'/f"iteration_{int(policy.split('_')[1]):06d}")
            selection=run/'methods/rldt/object_1/selection.json'
            validator._identity(row,index,condition,checkpoint,validator.sha(selection))
            prompt=run/'observation_bank/object_1/test'/f'state_{150+index:06d}'/'metadata.json'
            witness=validator.validate_structural_case(row,policy=policy,condition=condition,index=index,
                amendment=approved['amendment'],expected_prompt=validator.read(prompt)['prompt'],key_table=table)
            result={'passed':True,'complete':True,'technical_validation_only':True,'task':'object_1',
                    'source':source,'validator_source':validator.binding(validator.__file__),**approved,
                    'native_key_recurrence':key_audit,'case_row':row,'witness':witness,
                    'prompt_metadata':validator.binding(prompt),'selection':validator.binding(selection),
                    'scope':'One actual published undefined-input witness. Exact original masks/support/keys/prefix checked; no full-episode outcome or population rate inferred.'}
            target=destination/'summary.json';temporary=destination/'summary.tmp'
            temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(target)
            print(json.dumps({'passed':True,'path':str(target),'sha256':validator.sha(target),'policy':policy,'condition':condition,'episode':index,'decision':evidence['decision'],'support':witness['support']}),flush=True)
            return
        time.sleep(15)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    try:main(args.run)
    except BaseException as error:
        path=args.run/'compiler_profile_integration/causal_structural_na_v1/first_witness_independent_v1/failure.json'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'passed':False,'error':repr(error)},indent=2)+'\n')
        raise
