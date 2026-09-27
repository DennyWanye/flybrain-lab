from pathlib import Path
import argparse,json,hashlib
from flydrone.tellosim.training.campaign_v2 import evaluate_batch
from flydrone.tellosim.training.contracts import digest
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2';checkpoint=ROOT/f'runs/tellosim-sdk9/rigid-final-s{a.seed}/checkpoint.pt'
cases=json.loads((OUT/'cases.json').read_text());lock=json.loads((OUT/'frozen-checkpoints.json').read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() in lock['checkpoint_hashes'] and digest(cases['sealed_test'])==lock['case_hash']
for split in ('validation','sealed_test'):
    target=OUT/f'final-s{a.seed}-{split}.json'
    if target.exists():raise FileExistsError(target)
    evaluate_batch(ROOT,ROOT/'data/male-v1.npz',checkpoint,cases[split],target,f'rigid-final-s{a.seed}-{split}')
