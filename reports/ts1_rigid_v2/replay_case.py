from pathlib import Path
import argparse,json,datetime,hashlib
from flydrone.tellosim.training.campaign_v2 import evaluate_batch
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(11,22,33),required=True);p.add_argument('--case-id',required=True);a=p.parse_args()
root=Path('.').resolve();out=root/'reports/ts1_rigid_v2'
cases=json.loads((out/'cases.json').read_text())['sealed_test'];case=next(c for c in cases if c['case_id']==a.case_id)
checkpoint=root/f'runs/tellosim-sdk9/rigid-final-s{a.seed}/checkpoint.pt'
lock=json.loads((out/'frozen-checkpoints.json').read_text());assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() in lock['checkpoint_hashes']
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S%f');label=f'rigid-repro-s{a.seed}-{stamp}'
result=evaluate_batch(root,root/'data/male-v1.npz',checkpoint,[case],out/(label+'.json'),label,record_indices=(0,),batch=1)
print(json.dumps(result,ensure_ascii=False))
