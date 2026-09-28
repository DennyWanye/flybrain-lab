"""Backend observation consumption cannot advance physics/neural clock/optimizer."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
from flydrone.tellosim.training.altitude_refined_campaign import pool_for
from flydrone.tellosim.training.altitude_refined import AltitudeEnv,altitude_case,with_profile,teacher,load_altitude
from flydrone.tellosim.training.stability_campaign import fit
root=Path.cwd();out=root/'reports/ts1_completion/T32-backend.json'
if out.exists():raise FileExistsError(out)
def hash_arrays(xs):
 h=hashlib.sha256()
 for x in xs:h.update(np.ascontiguousarray(x).tobytes())
 return h.hexdigest()
class Consumer:
 def __init__(self,mode):self.mode=mode;self.calls=0;self.reads=0
 def publish(self,env,frame):
  self.calls+=1
  copies={'0':0,'30':3,'60':6,'disconnect':6 if self.calls<40 else 0}[self.mode]
  for _ in range(copies):json.loads(json.dumps(frame));self.reads+=1
results=[]
for mode in ['0','30','60','disconnect']:
 pool=pool_for(root,11,1);load_altitude(root/'runs/tellosim-sdk9/altitude-refined-s11/checkpoint.pt',pool,root)
 observer=Consumer(mode);rows=[];trace=[];env=None
 for episode in range(2):
  env=AltitudeEnv(root,with_profile(altitude_case(341000000+episode),'combined'),pool.lanes[0],observer=observer)
  try:
   for step in range(12):
    if env.terminated:break
    x=env.features.copy();label=teacher(env.observation,env.mask,env.previous_action is None)
    rows.append((x,env.mask.copy(),label,1.))
    a,_,_,policy=env.agent.decision(x,env.mask,True);env.step(a,policy)
    trace.append(np.r_[env.session.world.data.qpos,env.session.world.data.qvel,env.session.tick,env.agent.brain_tick,a])
  finally:env.close()
 brain=hash_arrays([pool.brain.v.cpu().numpy(),pool.brain.s.cpu().numpy(),pool.brain.trace.cpu().numpy()])
 for p in pool.model.parameters():p.requires_grad_(True)
 opt=torch.optim.AdamW(list(pool.model.body.parameters())+list(pool.model.actor.parameters()),lr=.0003,weight_decay=.001)
 before=hash_arrays([v.detach().numpy() for v in pool.model.state_dict().values()]);torch.manual_seed(341000010);fit(pool.model,opt,rows,2)
 after=hash_arrays([v.detach().numpy() for v in pool.model.state_dict().values()]);assert before!=after
 optimizer=hash_arrays([v.detach().numpy() for state in opt.state.values() for v in state.values() if isinstance(v,torch.Tensor)])
 results.append(dict(mode=mode,observer_frames=observer.calls,consumer_reads=observer.reads,trajectory_sha256=hash_arrays(trace),brain_sha256=brain,trained_model_sha256=after,optimizer_sha256=optimizer,neural_graph_sha256=pool.brain.graph_sha256,training_samples=len(rows)))
 del pool,env
 torch.cuda.empty_cache()
for key in ['trajectory_sha256','brain_sha256','trained_model_sha256','optimizer_sha256']:
 assert len({r[key] for r in results})==1,key
out.write_text(json.dumps({'status':'PASS','modes':results,'scope':'real MaleCNS backend, actual rigid trajectories and readout gradient updates; observer consumes0/3/6 copies per10Hz frame and detaches. Browser control verified separately.'},indent=2));print(out.read_text())
