from pathlib import Path
import json
import numpy as np
import torch
from flydrone.tellosim.training.runtime import ReservoirAgent,load_checkpoint
from flydrone.tellosim.training.env import TrainingEnv
from flydrone.tellosim.training.contracts import SPEC
from flydrone.tellosim.visual import atomic_json
root=Path('.').resolve();out=root/'reports/sdk9_warmstart'
run=root/'runs/tellosim-sdk9/nearx-warmstart-s23-20260926'
summary=json.loads((run/'summary.json').read_text())
failed=next(r for r in summary['trained']['results'] if not r['success'])
case=next(c for c in json.loads((run/'cases.json').read_text())['validation'] if c['case_id']==failed['case_id'])
torch.set_num_threads(4);agent=ReservoirAgent(root/'data/male-v1.npz','cuda',23,profile='balanced_rate_v3')
for name,path in [('before',root/'runs/tellosim-sdk9/nearx-bootstrap-s23-20260926/checkpoint.pt'),('after',run/'checkpoint.pt')]:
    load_checkpoint(path,agent,root)
    env=TrainingEnv(root,case,agent,record=True,run_id=f'nearx-warmstart-s23-20260926-{name}-failure',policy_source='failure-diagnostic_sdk9')
    rows=[]
    try:
        while not env.terminated:
            action,_,_,policy=agent.decision(env.features,env.mask,True)
            observation=env.observation.copy()
            row={'time_s':env.session.tick/120,'measured_goal_error_m':(observation[:3]*[6,6,3]).tolist(),
                 'measured_velocity_mps':observation[3:6].tolist(),'command':SPEC['actions'][action],
                 'probabilities':policy['probabilities'],'value_estimate':policy['value_estimate']}
            result=env.step(action,policy);row.update(duration_s=result['k']/10,reward=result['reward'])
            rows.append(row)
        report={'checkpoint_sha256':agent.checkpoint_sha256,'case':case,'run_id':env.session.run_id,
                'reason':env.reason,'final_distance_m':float(np.linalg.norm(env.session.world.position-np.asarray(case['goal']))),
                'steps':rows,'scope':'Read-only reproduction of known validation failure; no learning from this case.'}
        atomic_json(out/f'failure-{name}.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('case','steps')}),flush=True)
    finally:env.close()
