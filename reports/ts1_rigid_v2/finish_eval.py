"""Continue frozen evaluation at completed-case boundaries; never alter models."""
from pathlib import Path
import argparse,hashlib,json,time,shutil
from flydrone.tellosim.training.campaign_v2 import evaluate_batch
from flydrone.tellosim.training.c0_campaign import summarize
from flydrone.tellosim.training.contracts import digest
from flydrone.tellosim.visual import atomic_json
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2';checkpoint=ROOT/f'runs/tellosim-sdk9/rigid-final-s{a.seed}/checkpoint.pt'
sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest();splits=json.loads((OUT/'cases.json').read_text());lock=json.loads((OUT/'frozen-checkpoints.json').read_text())
assert sha in lock['checkpoint_hashes'] and digest(splits['sealed_test'])==lock['case_hash']
for split in ('validation','sealed_test'):
    cases=splits[split];target=OUT/f'final-s{a.seed}-{split}.json'
    if target.exists():
        previous=json.loads(target.read_text());assert previous['checkpoint_sha256']==sha and previous['case_hash']==digest(cases);continue
    prior=[];progress=target.with_suffix('.progress.json')
    if progress.exists():
        previous=json.loads(progress.read_text());assert previous['label']==f'rigid-final-s{a.seed}-{split}'
        prior=previous['results'];atomic_json(OUT/f'preserved-b4-s{a.seed}-{split}.json',{'checkpoint_sha256':sha,'case_hash':digest(cases),'results':prior})
    expected={c['case_id']:c for c in cases};done={r['case_id']:r for r in prior}
    assert len(done)==len(prior) and all(r['seed']==expected[k]['seed'] for k,r in done.items())
    remaining=[c for c in cases if c['case_id'] not in done]
    print(json.dumps({'seed':a.seed,'split':split,'preserved':len(prior),'remaining':len(remaining),'batch':64}),flush=True)
    partial=evaluate_batch(ROOT,ROOT/'data/male-v1.npz',checkpoint,remaining,OUT/f'remainder-s{a.seed}-{split}.json',f'rigid-accepted-s{a.seed}-{split}',record_indices=() if prior else (0,),batch=64)
    for row in partial['results']:
        assert row['case_id'] not in done;done[row['case_id']]=row
    ordered=[done[c['case_id']] for c in cases]
    result={**summarize(ordered,f'rigid-final-s{a.seed}-{split}'),'mode':'argmax','case_hash':digest(cases),'checkpoint_sha256':sha,
        'evaluation_device':'cuda','neural_backend':'csr_fp64_accum','batch_segments':[{'batch':4,'cases':len(prior)},{'batch':64,'cases':len(remaining)}],
        'elapsed_remaining_s':partial['elapsed_s'],'preserved_case_evidence':str(progress.name) if prior else None,
        'note':'Completed cases preserved before scheduling change; no case selection, training or weight updates during evaluation'}
    atomic_json(target,result);print(json.dumps({'seed':a.seed,'split':split,'successes':result['successes'],'episodes':result['episodes']}),flush=True)
