from pathlib import Path
import json,hashlib,gc
import numpy as np,torch
from flydrone.tellosim.training.spatial_continuous import SpatialPool,SpatialEnv,spatial_case,load_spatial
from flydrone.tellosim.training.altitude_refined import with_profile
from flydrone.tellosim.training.checkpoint import configure_exact_execution
root=Path.cwd();out=root/'reports/ts1_completion_c2w'
def hashed(xs):
 h=hashlib.sha256()
 for x in xs:h.update(np.ascontiguousarray(x).tobytes())
 return h.hexdigest()
class Consumer:
 def __init__(self,mode):self.mode=mode;self.calls=0;self.reads=0
 def publish(self,env,frame):
  self.calls+=1
  copies={'0':0,'30':3,'60':6,'disconnect':6 if self.calls<40 else 0}[self.mode]
  for _ in range(copies):json.loads(json.dumps(frame));self.reads+=1
results=[];torch.set_num_threads(4);configure_exact_execution('cuda')
for mode in ['0','30','60','disconnect']:
 pool=SpatialPool(root,11,batch=1);load_spatial(root/'runs/tellosim-sdk9/spatial-continuous-s11/checkpoint.pt',pool,root)
 observer=Consumer(mode);case=with_profile(spatial_case(558000001,'c2w-observer-development'),'combined')
 case.update(start=[0.,0.,1.],goal=[.4,.2,1.4],initial_yaw_rad=0.,target_yaw_rad=1.57)
 env=SpatialEnv(root,case,pool.lanes[0],observer=observer);trace=[]
 try:
  for _ in range(10):
   if env.terminated:break
   action,_,_,policy=env.agent.decision(env.features,env.mask,True);env.step(action,policy)
   trace.append(np.r_[env.session.world.data.qpos,env.session.world.data.qvel,env.session.world.target,env.session.tick,env.agent.brain_tick,action,env.hold])
  results.append({'mode':mode,'consumer_reads':observer.reads,'published_frames':observer.calls,'trajectory_sha256':hashed(trace),'brain_sha256':hashed([pool.brain.v.cpu().numpy(),pool.brain.s.cpu().numpy(),pool.brain.trace.cpu().numpy()]),'readout_sha256':hashed([value.detach().cpu().numpy() for model in pool.models.values() for value in model.state_dict().values()]),'physics_tick':env.session.tick,'brain_tick':env.agent.brain_tick,'graph_sha256':pool.brain.graph_sha256})
 finally:env.close()
 del env,pool;gc.collect();torch.cuda.empty_cache()
for key in ['trajectory_sha256','brain_sha256','readout_sha256','physics_tick','brain_tick']:assert len({r[key] for r in results})==1,key
(out/'T32-C2W.json').write_text(json.dumps({'status':'PASS','modes':results,'scope':'actual full MaleCNS C2W inference, anchored controller and120Hz hold clock;0/30/60FPS-equivalent consumption and disconnect identical; no optimization in frozen model'},indent=2))
print('C2W observer invariance PASS')
