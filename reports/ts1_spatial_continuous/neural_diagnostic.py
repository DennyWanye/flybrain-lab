from pathlib import Path
import json,os
import torch
from flydrone.tellosim.training.spatial_continuous import SpatialPool,SpatialEnv,load_spatial
from flydrone.tellosim.training.spatial_continuous_campaign import result_row
from flydrone.tellosim.training.checkpoint import configure_exact_execution
from reports.ts1_control_development_v.diagnose import snapshot
from reports.ts1_spatial_continuous.audit_replays import audit
ROOT=Path.cwd();OUT=ROOT/'reports/ts1_spatial_continuous'
def run():
 torch.set_num_threads(4);configure_exact_execution('cuda')
 pool=SpatialPool(ROOT,11,batch=1);load_spatial(ROOT/'runs/tellosim-sdk9/spatial-continuous-s11/checkpoint.pt',pool,ROOT)
 case=json.loads((ROOT/'reports/ts1_spatial_anchored/smoke-cases.json').read_text())[2]
 # Intentional regression of C2V's236/240-tick failure; not an unseen score.
 case['case_id']='c2w-regression-physical-hold'
 trace=[]
 class Observer:
  def publish(self,env,frame):
   if not hasattr(env,'device') or not hasattr(env,'features'):return
   trace.append({**snapshot(env),'neural_features':env.features.tolist(),'brain_tick':env.agent.brain_tick,'policy':env.last_policy})
 env=SpatialEnv(ROOT,case,pool.lanes[0],record=True,run_id='spatial-continuous-physical-hold-regression',policy_source='spatial_argmax_malecns',observer=Observer())
 actions=[];skills=[]
 try:
  while not env.terminated:
   action,_,_,policy=env.agent.decision(env.features,env.mask,True);actions.append(action);skills.append(env.phase);env.step(action,policy)
  result=result_row(env,actions,skills,env.session.run_id)
 finally:env.close()
 report={'purpose':'previously observed failure reproduction; no independent generalization claim','case':case,'result':result,'trace':trace}
 (OUT/'NEURAL_CONTROL_DIAGNOSTIC.json').write_text(json.dumps(report))
 evidence=audit('spatial-continuous-physical-hold-regression')
 assert result['success'] and evidence['final_valid_physics_ticks']==240
 (OUT/'PHYSICAL_HOLD_REGRESSION.json').write_text(json.dumps(evidence,indent=2))
if __name__=='__main__':run()
