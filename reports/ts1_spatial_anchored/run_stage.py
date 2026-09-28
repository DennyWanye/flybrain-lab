"""Scoped, recorded stage runner. No automatic acceptance or seed selection."""
from pathlib import Path
import sys,json,os,time,subprocess
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'reports/ts1_spatial_anchored'
def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2))
def child(stage):
 from flydrone.tellosim.training.spatial_anchored import spatial_case
 from flydrone.tellosim.training.spatial_anchored_campaign import baseline,evaluate,prepare,evaluate_seed,finalize
 from flydrone.tellosim.training.altitude_refined import with_profile,PROFILES
 if stage=='smoke':
  cases=[with_profile(spatial_case(531000000+i,f'c2v-smoke-{i}'),tuple(PROFILES)[i%4]) for i in range(8)]
  save('smoke-cases.json',cases)
  evaluate(ROOT,ROOT/'runs/tellosim-sdk9/spatial-anchored-s11/checkpoint.pt',cases,OUT/'smoke.json','spatial-anchored-smoke',batch=8,record_indices=(0,1,2,3,4,5,6,7))
 elif stage=='devrule':
  assert json.loads((OUT/'smoke.json').read_text())['successes']==8
  cases=[]
  for i in range(128):
   c=spatial_case(530000000+i,f'c2v-development-{i}')
   if i<64:c['goal'][2]=[.3,.5,1.5,1.7][i%4]+float(np.random.default_rng(530000000+i).uniform(-.015,.015))
   cases.append(with_profile(c,tuple(PROFILES)[i//4%4]))
  save('development-cases.json',cases)
  save('DEVELOPMENT_PROTOCOL.json',{'cases':128,'required_rule_successes':127,'new_seed_range':[530000000,530000127],'half_command_height_stratification':64,'formal_not_started':True,'neural_development_required':{'cases':32,'successes':30},'selection':'fixed transferred C2Q weights; no selection'})
  baseline(ROOT,cases,'rule',OUT/'development-rule.json')
 elif stage=='devneural':
  assert json.loads((OUT/'development-rule.json').read_text())['successes']>=127
  cases=json.loads((OUT/'development-cases.json').read_text())[::4]
  # All disturbance profiles are selected evenly using a fixed stratified index.
  source=json.loads((OUT/'development-cases.json').read_text())
  cases=[source[(i//4)*16+(i%4)*4+(i//16)] for i in range(32)]
  save('neural-development-cases.json',cases)
  evaluate(ROOT,ROOT/'runs/tellosim-sdk9/spatial-anchored-s11/checkpoint.pt',cases,OUT/'development-neural.json','spatial-anchored-development',batch=32)
 elif stage=='formal-rule':
  assert json.loads((OUT/'development-rule.json').read_text())['successes']>=127
  assert json.loads((OUT/'development-neural.json').read_text())['successes']>=30
  baseline(ROOT,prepare(ROOT)['validation'],'rule',OUT/'rule-validation.json')
 elif stage.startswith('formal-s'):
  assert json.loads((OUT/'rule-validation.json').read_text())['successes']>=99
  evaluate_seed(ROOT,int(stage.removeprefix('formal-s')))
 elif stage=='formal-random':
  assert json.loads((OUT/'rule-validation.json').read_text())['successes']>=99
  baseline(ROOT,prepare(ROOT)['sealed_test'],'random',OUT/'random-sealed.json')
 elif stage=='finalize':finalize(ROOT)
 else:raise ValueError(stage)
if __name__=='__main__':
 stage=sys.argv[1]
 if len(sys.argv)>2:
  try:child(stage)
  finally:
   save(stage+'-memory.json',{'pid':os.getpid(),'private_kib':sum(int(x.split()[1]) for x in Path('/proc/self/smaps_rollup').read_text().splitlines() if x.startswith(('Private_Clean:','Private_Dirty:')))})
 else:
  cmd=[sys.executable,'-u','-m','reports.ts1_spatial_anchored.run_stage',stage,'child']
  log=(OUT/(stage+'.log')).open('x');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  save(stage+'-process.json',{'pid':p.pid,'command':cmd,'cwd':str(ROOT),'started':time.time()})
  print(json.dumps({'stage':stage,'pid':p.pid}),flush=True)
  try:
   code=p.wait(timeout=10800)
  finally:
   if p.poll() is None:p.terminate()
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   log.close();save(stage+'-cleanup.json',{'pid':p.pid,'returncode':p.returncode,'absent':not Path(f'/proc/{p.pid}').exists()})
  print(json.dumps({'stage':stage,'returncode':p.returncode}),flush=True)
  sys.exit(code)
