from pathlib import Path
from reports.ts1_rigid_v2.preflight import RuleClock
from flydrone.tellosim.training.c0_campaign import evaluate
from flydrone.tellosim.training.c1_campaign import prepare
root=Path('.').resolve();out=root/'reports/ts1_c1';cases=prepare(root)
r=evaluate(root,RuleClock(),cases['validation'],out/'rule-validation.json','c1-rule',mode='rule',record_indices=())
assert r['successes']>=99,r['successes']
r=evaluate(root,RuleClock(),cases['sealed_test'],out/'random-sealed.json','c1-uniform-random',mode='random',record_indices=())
print('BASELINES_COMPLETE',r['successes'])
