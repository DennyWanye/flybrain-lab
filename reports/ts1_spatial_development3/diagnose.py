"""New half-command height-band development diagnosis, separate from every formal case."""
from pathlib import Path
import json,math,subprocess,sys
from flydrone.tellosim.training.spatial_precision import spatial_case
from flydrone.tellosim.training.altitude_refined import with_profile,PROFILES
from flydrone.tellosim.training.heading import wrap_angle
from flydrone.tellosim.training.spatial_precision_campaign import evaluate
root=Path.cwd();out=root/'reports/ts1_spatial_development3';out.mkdir(exist_ok=True)
if len(sys.argv)==1:
 cases=[]
 for i in range(24):
  c=spatial_case(410000000+i,f'c2-development3-{i}')
  import numpy as np
  c['goal'][2]=[.3,.5,1.5,1.7][i%4]+float(np.random.default_rng(410000000+i).uniform(-.025,.025))
  cases.append(with_profile(c,'clean'))
 (out/'cases.json').write_text(json.dumps(cases,indent=2))
 (out/'protocol.json').write_text(json.dumps({'purpose':'fresh development diagnosis after C2P aggregate boundary gate failure; no sealed case inspection or tuning','seeds':[410000000,410000023],'models':[11,22,33],'no_checkpoint_updates':True},indent=2))
 jobs=[]
 try:
  for seed in [11,22,33]:
   log=(out/f's{seed}.log').open('x');p=subprocess.Popen([sys.executable,'-u','-m','reports.ts1_spatial_development3.diagnose',str(seed)],stdout=log,stderr=subprocess.STDOUT);jobs.append((seed,p,log))
  (out/'processes.json').write_text(json.dumps([{'seed':s,'pid':p.pid} for s,p,_ in jobs]))
  for seed,p,log in jobs:p.wait(timeout=1200);assert p.returncode==0
 finally:
  for _,p,_ in jobs:
   if p.poll() is None:p.terminate()
  for _,p,log in jobs:
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   log.close()
  (out/'cleanup.json').write_text(json.dumps([{'seed':s,'pid':p.pid,'returncode':p.returncode,'absent':not Path(f'/proc/{p.pid}').exists()} for s,p,_ in jobs],indent=2))
else:
 seed=int(sys.argv[1]);cases=json.loads((out/'cases.json').read_text())
 from flydrone.tellosim.training.spatial_precision import SpatialEnv
 from flydrone.tellosim.training.spatial_precision_campaign import label_for
 original_step=SpatialEnv.step_iter;traces={}
 def traced(self,action,policy=None):
  sensor=self.device.latest_observation()
  before={'phase':self.phase,'action':action,'teacher':label_for(self),'tick':self.session.tick,'position':list(sensor.position_m) if sensor.position_m is not None else None,'velocity':list(sensor.velocity_mps) if sensor.velocity_mps is not None else None,'goal':self.case['goal'],'policy':policy,'features':self.features.tolist()}
  result=yield from original_step(self,action,policy)
  sensor=self.device.latest_observation();before.update(after_phase=self.phase,after_tick=self.session.tick,after_position=list(sensor.position_m) if sensor.position_m is not None else None,after_velocity=list(sensor.velocity_mps) if sensor.velocity_mps is not None else None,terminated=self.terminated)
  traces.setdefault(self.case['case_id'],[]).append(before)
  return result
 SpatialEnv.step_iter=traced

 evaluate(root,root/f'runs/tellosim-sdk9/spatial-precision-s{seed}/checkpoint.pt',cases,out/f's{seed}.json',f'spatial-development3-s{seed}',batch=24,record_indices=())

 if len(sys.argv)>1:(out/f's{seed}-traces.json').write_text(json.dumps(traces))
