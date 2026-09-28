from pathlib import Path
import json
import torch
from flydrone.tellosim.training.boundary_repair import validation_cases
from flydrone.tellosim.training.runtime import ReservoirAgent,load_checkpoint
from flydrone.tellosim.training.env import TrainingEnv
from flydrone.tellosim.training.contracts import SPEC
from flydrone.tellosim.visual import atomic_json
root=Path('.').resolve();torch.set_num_threads(4)
agent=ReservoirAgent(root/'data/male-v1.npz','cuda',23,profile='balanced_rate_v3')
load_checkpoint(root/'runs/tellosim-sdk9/boundary-repair-s23-20260926/checkpoint.pt',agent,root)
torch.manual_seed(960028);case=validation_cases()[28]
env=TrainingEnv(root,case,agent,record=True,run_id='boundary-s23-first-attempt-sampled-failure',policy_source='sampled-diagnostic_sdk9');rows=[]
try:
    while not env.terminated:
        action,_,_,policy=agent.decision(env.features,env.mask)
        rows.append({'t':env.session.tick/120,'error':float(env.observation[0]*6),
                     'action':SPEC['actions'][action],'probabilities':policy['probabilities'][:3]})
        env.step(action,policy)
    atomic_json(root/'reports/sdk9_boundary/sampled-failure.json',{'case':case,'reason':env.reason,'rows':rows})
    print(json.dumps({'reason':env.reason,'rows':rows},indent=2))
finally:env.close()
