from pathlib import Path
import json,math,sys,os,subprocess
import numpy as np
from flydrone.tellosim.training.spatial_orientation import SpatialEnv,spatial_case
from flydrone.tellosim.training.spatial_orientation_campaign import baseline
from flydrone.tellosim.training.altitude_refined import with_profile,PROFILES
root=Path.cwd();out=root/'reports/ts1_rule_development'
if len(sys.argv)==1:
 cases=[]
 for i in range(128):
  c=spatial_case(420000000+i,f'rule-development-{i}')
  if i<64:c['goal'][2]=[.3,.5,1.5,1.7][i%4]+float(np.random.default_rng(420000000+i).uniform(-.015,.015))
  cases.append(with_profile(c,tuple(PROFILES)[i//4%4]))
 (out/'cases.json').write_text(json.dumps(cases,indent=2));(out/'protocol.json').write_text(json.dumps({'scope':'rule-only development diagnosis after aggregate feasibility gate failure; no neural weight changes or heldout case reuse','seed_range':[420000000,420000127],'cases':128,'half_command_height_stratification':64,'profiles':'balanced clean/pose/force/combined'},indent=2))
 log=(out/'diagnosis.log').open('x');p=subprocess.Popen([sys.executable,'-u','-m','reports.ts1_rule_development.diagnose','child'],stdout=log,stderr=subprocess.STDOUT)
 (out/'process.json').write_text(json.dumps({'pid':p.pid}))
 try:p.wait(timeout=1200);assert p.returncode==0
 finally:
  if p.poll() is None:p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
  log.close();(out/'cleanup.json').write_text(json.dumps({'pid':p.pid,'returncode':p.returncode,'absent':not Path(f'/proc/{p.pid}').exists()}))
else:
 traces={};original=SpatialEnv.step_iter
 def traced(self,action,policy=None):
  sensor=self.device.latest_observation()
  before={'phase':self.phase,'action':action,'tick':self.session.tick,'observation':self.observation.tolist(),'goal':self.case['goal'],'position':sensor.position_m,'velocity':sensor.velocity_mps,'yaw_rad':sensor.yaw_rad}
  result=yield from original(self,action,policy)
  sensor=self.device.latest_observation();before.update(after_phase=self.phase,after_tick=self.session.tick,after_observation=self.observation.tolist(),after_position=sensor.position_m,after_velocity=sensor.velocity_mps,after_yaw_rad=sensor.yaw_rad)
  traces.setdefault(self.case['case_id'],[]).append(before);return result
 SpatialEnv.step_iter=traced
 baseline(root,json.loads((out/'cases.json').read_text()),'rule',out/'baseline.json')
 (out/'traces.json').write_text(json.dumps(traces))
 (out/'owned-memory-sample.json').write_text(json.dumps({'pid':os.getpid(),'private_kib':sum(int(r.split()[1]) for r in Path('/proc/self/smaps_rollup').read_text().splitlines() if r.startswith(('Private_Clean:','Private_Dirty:')))}))
