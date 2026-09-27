from pathlib import Path
import json
import torch
from reports.ts1_rigid_v2.full_graph_check import equal
from flydrone.tellosim.training.continuation import run
from flydrone.tellosim.visual import atomic_json
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2'
reference=torch.load(OUT/'perf-fp64-b4-continuous/resume.pt',weights_only=False)
results=[]
for i in range(3):
    target=OUT/f'fp64-repeat-{i}'
    report=run(ROOT,ROOT/'data/male-v1.npz',target,options=32,rollout=16,batch=4,seed=11,device='cuda',resume=OUT/'fp64-boundary.pt',wall_seconds=300)
    state=torch.load(target/'resume.pt',weights_only=False)
    checks={k:equal(reference[k],state[k]) for k in ('policy','optimizer','brain','lanes','rng')}
    checks['physics']=all(equal(a['session']['world'],b['session']['world']) for a,b in zip(reference['envs'],state['envs']))
    results.append({'repeat':i,'passed':all(checks.values()),'checks':checks});print(results[-1],flush=True)
atomic_json(OUT/'repeat-resume.json',{'results':results,'passed':all(r['passed'] for r in results),'scope':'Three restarts from the same saved PPO boundary, with concurrent GPU evaluation'})
assert all(r['passed'] for r in results)
