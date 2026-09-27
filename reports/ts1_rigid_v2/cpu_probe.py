import json,time
from pathlib import Path
from flydrone.tellosim.training.campaign_v2 import evaluate_batch
from flydrone.tellosim.visual import atomic_json
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2';cases=json.loads((OUT/'cases.json').read_text())['validation'][:2]
r=evaluate_batch(ROOT,ROOT/'data/male-v1.npz',ROOT/'runs/tellosim-sdk9/rigid-final-s11/checkpoint.pt',cases,OUT/'cpu-probe.json','cpu-probe',record_indices=(),device='cpu')
reference=json.loads((OUT/'final-s11-validation.progress.json').read_text())['results'];by_id={x['case_id']:x for x in reference}
comparison=[{'case_id':x['case_id'],'actions_exact':x['actions']==by_id[x['case_id']]['actions'],'distance_exact':x['distance_m']==by_id[x['case_id']]['distance_m'],'reward_exact':x['return']==by_id[x['case_id']]['return']} for x in r['results']]
atomic_json(OUT/'cpu-probe-comparison.json',{'comparison':comparison,'elapsed_s':r['elapsed_s']});print(r['elapsed_s'],comparison)
