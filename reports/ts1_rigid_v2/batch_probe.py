import json,time
from pathlib import Path
from flydrone.tellosim.training.campaign_v2 import evaluate_batch
from flydrone.tellosim.visual import atomic_json
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2';cases=json.loads((OUT/'cases.json').read_text())['validation'][:16]
r=evaluate_batch(ROOT,ROOT/'data/male-v1.npz',ROOT/'runs/tellosim-sdk9/rigid-final-s11/checkpoint.pt',cases,OUT/'batch16-probe.json','batch16-probe',record_indices=(),batch=16)
source=OUT/'final-s11-validation.progress.json';reference=json.loads(source.read_text())['results'];by_id={x['case_id']:x for x in reference}
checks=[{'case_id':x['case_id'],'actions_exact':x['actions']==by_id[x['case_id']]['actions'],'distance_exact':x['distance_m']==by_id[x['case_id']]['distance_m'],'reward_exact':x['return']==by_id[x['case_id']]['return']} for x in r['results']]
report={'checks':checks,'elapsed_s':r['elapsed_s'],'passed':all(c['actions_exact'] and c['distance_exact'] and c['reward_exact'] for c in checks)}
atomic_json(OUT/'batch16-probe-comparison.json',report);print(report);assert report['passed']
