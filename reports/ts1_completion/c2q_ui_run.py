from pathlib import Path
import json,os,sys,subprocess,time
root=Path.cwd();out=root/'reports/ts1_completion'
if len(sys.argv)==1:
 log=(out/'c2q-ui-run.log').open('x');p=subprocess.Popen([sys.executable,'-u','-m','reports.ts1_completion.c2q_ui_run','child'],stdout=log,stderr=subprocess.STDOUT)
 (out/'c2q-ui-process.json').write_text(json.dumps({'pid':p.pid}))
 try:p.wait(timeout=600);assert p.returncode==0
 finally:
  if p.poll() is None:p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
  log.close();(out/'c2q-ui-cleanup.json').write_text(json.dumps({'pid':p.pid,'returncode':p.returncode,'proc_absent':not Path(f'/proc/{p.pid}').exists()}))
else:
 from flydrone.tellosim.training.spatial_orientation import SpatialPool,spatial_case
 from flydrone.tellosim.training.spatial_orientation_campaign import evaluate
 from flydrone.tellosim.training.altitude_refined import with_profile
 original=SpatialPool.observe_lanes
 def paced(self,pending):
  value=original(self,pending);time.sleep(.15);return value
 SpatialPool.observe_lanes=paced
 case=with_profile(spatial_case(510000000,'c2q-ui-development-0'),'combined')
 (out/'c2q-ui-case.json').write_text(json.dumps({'case':case,'scope':'new UI diagnostic only, not formal score; wall-clock pacing outside simulator and neural update','wall_pacing_s_per_sample':.15},indent=2))
 evaluate(root,root/'runs/tellosim-sdk9/spatial-orientation-s11/checkpoint.pt',[case],out/'c2q-ui-result.json','spatial-orientation-ui-s11',batch=1,record_indices=(0,))
 (out/'c2q-ui-memory-sample.json').write_text(json.dumps({'pid':os.getpid(),'private_kib':sum(int(r.split()[1]) for r in Path('/proc/self/smaps_rollup').read_text().splitlines() if r.startswith(('Private_Clean:','Private_Dirty:')))}))
