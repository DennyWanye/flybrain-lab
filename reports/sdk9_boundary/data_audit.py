"""Read-only classification audit on explicitly marked training examples."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from flydrone.policy import ActorCritic
from flydrone.tellosim.training.runtime import distribution
from flydrone.tellosim.visual import atomic_json
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
data=np.load(a.run/'demonstrations.npz');x=torch.tensor(data['features']);m=data['masks'];y=data['teacher_actions']
meta=json.loads((a.run/'data-provenance.json').read_text())['rows']
boundary=np.array([.17<abs(r.get('measured_goal_error_m',99))<.30 for r in meta])
source=torch.load(a.run/'initial.pt',map_location='cpu',weights_only=False)
final=torch.load(a.run/'checkpoint.pt',map_location='cpu',weights_only=False)
assert source['contract']==final['contract']
results={}
torch.set_num_threads(4)
for label,state in [('before',source),('after',final)]:
    model=ActorCritic(x.shape[1],9);model.load_state_dict(state['policy'])
    with torch.no_grad():probs,_=distribution(model,x,m);pred=probs.probs.argmax(1).numpy()
    results[label]={'all_training_accuracy':float(np.mean(pred==y)),
        'boundary_training_examples':int(boundary.sum()),'boundary_training_accuracy':float(np.mean(pred[boundary]==y[boundary])),
        'wrong_direction_boundary':int(np.sum(boundary & (y!=0) & (pred!=0) & (pred!=y)))}
report={'scope':'Training-data diagnostic only, not held-out flight performance.',
    'contract_exactly_unchanged':True,'results':results,'examples':len(y)}
atomic_json(a.run/'feature-audit.json',report);print(json.dumps(report,indent=2))
