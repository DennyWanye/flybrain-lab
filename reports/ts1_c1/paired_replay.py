"""Record paired evaluation on the same preselected validation case."""
from pathlib import Path
import json,hashlib
from flydrone.tellosim.training.campaign_v2 import evaluate_batch
from flydrone.tellosim.training.contracts import digest
from flydrone.tellosim.visual import atomic_json
root=Path('.').resolve();out=root/'reports/ts1_c1';case=json.loads((out/'cases.json').read_text())['validation'][0]
# Selection fixed here, before looking at any final model result.
selection={'case_id':case['case_id'],'case_hash':digest([case]),'selection':'first validation case; no success/failure-based selection'}
p=out/'comparison-selection.json'
if p.exists():assert json.loads(p.read_text())==selection
else:atomic_json(p,selection)
rows=[]
for phase,name in (('before','initial.pt'),('after','ppo/checkpoint.pt')):
    checkpoint=root/'runs/tellosim-sdk9/c1-s11'/name
    result=evaluate_batch(root,root/'data/male-v1.npz',checkpoint,[case],out/f'comparison-{phase}.json',f'c1-paired-{phase}',record_indices=(0,),batch=1)
    r=result['results'][0];rows.append({'label':'训练前' if phase=='before' else '训练后','run_id':r['run_id'],'case_id':case['case_id'],'success':r['success'],'distance_m':r['distance_m'],'return':r['return'],'checkpoint_sha256':result['checkpoint_sha256']})
atomic_json(out/'comparison.json',{'selection':selection,'runs':rows})
summary=json.loads((out/'summary.json').read_text());summary['comparison']=rows;atomic_json(out/'summary.json',summary)
print(json.dumps(rows,ensure_ascii=False))
