"""Run from the extracted package root; never imports the original checkout."""
from pathlib import Path
import json,hashlib,os,gc,threading,urllib.request
import numpy as np,torch
root=Path.cwd()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def verify_files():
 p=root/'PACKAGE_MANIFEST.json'
 if not p.exists():return 0
 files=json.loads(p.read_text())
 for name,digest in files.items():assert sha(root/name)==digest,name
 return len(files)
initial_files=verify_files()
from flyview.tellosim_api import Observatory
import flyview.tellosim_api as module
assert Path(module.__file__).resolve().is_relative_to(root)
from flydrone.tellosim.training.spatial_continuous import SpatialPool,load_spatial,SKILLS
from flydrone.tellosim.training.checkpoint import configure_exact_execution
configure_exact_execution('cuda');torch.set_num_threads(4)
summary=json.loads((root/'reports/ts1_spatial_continuous/summary.json').read_text());assert summary['acceptance']['passed'] and summary['new_training_actions']==0
assert json.loads((root/'reports/ts1_completion_c2w/TRACKED_EXPECTATIONS.json').read_text())['regression_tests']==211
assert sha(root/'data/male-v1.npz')=='badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9'
peak=0;checks=0;replays=0
for seed in [11,22,33]:
 pool=SpatialPool(root,seed,batch=1);state=load_spatial(root/f'runs/tellosim-sdk9/spatial-continuous-s{seed}/checkpoint.pt',pool,root)
 assert state['seed']==seed
 vector=np.zeros(26,np.float32);vector[:3]=[.1,.2,.3];vector[7]=1;vector[8]=1/3;vector[9]=1;vector[12:15]=1
 for sample,skill in enumerate(SKILLS,1):
  pool.lanes[0].switch_skill(skill);features=pool.observe_lanes({0:(vector,sample)})
  assert np.isfinite(features).all() and features.shape==(1,128)
  action,_,_,info=pool.lanes[0].decision(features[0],np.ones(9,bool),True)
  assert 0<=action<9 and info['input_source']=='reservoir_v_trace';checks+=1
 peak=max(peak,sum(int(x.split()[1]) for x in Path('/proc/self/smaps_rollup').read_text().splitlines() if x.startswith(('Private_Clean:','Private_Dirty:'))))
 del pool;gc.collect();torch.cuda.empty_cache()
 for kind in ['spatial-continuous','altitude-refined']:
  folder=root/f'reports/vis/tellosim/{kind}-s{seed}-sealed_test-3';m=json.loads((folder/'manifest.json').read_text());assert m['complete'] and not m['partial']
  for parts in m['streams'].values():
   for part in parts:assert sha(folder/part['file'])==part['sha256']
  replays+=1
assert (root/'flyview/static/vendor/three/three.module.js').stat().st_size>10000
from flyview.server import Viewer,Handler,ThreadingHTTPServer
server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.viewer=Viewer(root,root/'reports/vis/views');server.tellosim=Observatory(root)
worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start();routes=0
try:
 for route in ['/tellosim','/static/tellosim.js','/api/tellosim/runs']:
  with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}'+route,timeout=15) as response:
   body=response.read();assert response.status==200 and len(body)>100;routes+=1
   if route.endswith('runs'):
    catalog=json.loads(body);assert catalog['spatial']['run_prefix']=='spatial-continuous' and catalog['spatial']['JOINT_3D_TASK_VERIFIED']
    assert catalog['completion']['regression_tests']==211 and not catalog['completion']['REAL_FLIGHT_READY']
    assert all(any(r['run_id']==f'spatial-continuous-s{s}-sealed_test-3' for r in catalog['runs']) for s in [11,22,33])
finally:server.shutdown();worker.join(timeout=5);server.tellosim.close();server.server_close()
assert not worker.is_alive();after_files=verify_files();assert initial_files==after_files
print(json.dumps({'status':'PASS','pid':os.getpid(),'sampled_private_kib':peak,'actual_extracted_source':str(module.__file__),'strict_full_graph_inference_checks':checks,'replays_checked':replays,'http_routes_checked':routes,'file_hashes_before_and_after':after_files,'server_thread_joined':True,'inference_probe_is_not_task_score':True}))
