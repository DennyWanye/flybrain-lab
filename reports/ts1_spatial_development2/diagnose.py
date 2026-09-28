"""New development distribution, separate from all C2R formal cases."""
from pathlib import Path
import json,math,subprocess,sys
from flydrone.tellosim.training.spatial_refined import spatial_case
from flydrone.tellosim.training.altitude_refined import with_profile,PROFILES
from flydrone.tellosim.training.heading import wrap_angle
from flydrone.tellosim.training.spatial_refined_campaign import evaluate
root=Path.cwd();out=root/'reports/ts1_spatial_development2';out.mkdir(exist_ok=True)
if len(sys.argv)==1:
 cases=[]
 for i in range(24):
  c=spatial_case(350000000+i,f'c2-development2-{i}')
  c['goal'][2]=[.4,.6,1.4,1.6][i%4]
  c['target_yaw_rad']=wrap_angle(c['initial_yaw_rad']+[-1,1][i//4%2]*math.pi/2)
  cases.append(with_profile(c,tuple(PROFILES)[i//6]))
 (out/'cases.json').write_text(json.dumps(cases,indent=2))
 (out/'protocol.json').write_text(json.dumps({'purpose':'fresh development diagnosis after C2R aggregate instruction gate failure; no sealed case inspection or tuning','seeds':[350000000,350000023],'models':[11,22,33],'no_checkpoint_updates':True},indent=2))
 jobs=[]
 try:
  for seed in [11,22,33]:
   log=(out/f's{seed}.log').open('x');p=subprocess.Popen([sys.executable,'-u','-m','reports.ts1_spatial_development2.diagnose',str(seed)],stdout=log,stderr=subprocess.STDOUT);jobs.append((seed,p,log))
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
 evaluate(root,root/f'runs/tellosim-sdk9/spatial-refined-s{seed}/checkpoint.pt',cases,out/f's{seed}.json',f'spatial-development2-s{seed}',batch=24,record_indices=())
