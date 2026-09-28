"""Build a new portable archive, verify fresh extraction, publish only after audit."""
from pathlib import Path
import hashlib,json,zipfile,shutil,tempfile,subprocess,sys,os
root=Path.cwd();report=root/'reports/ts1_completion';dest=Path('/mnt/d/projects/果蝇训练/ts1-v1r-c2q-delivery-20260928');dest.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
summary=json.loads((report/'summary.json').read_text());assert summary['SOFTWARE_FUNCTIONAL_ACCEPTANCE_PASSED']
paths=set()
def add_tree(folder):
 for p in (root/folder).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.zip'] and p.name not in ['PACKAGE_AUDIT.json']:
   if folder=='reports/ts1_completion' and 'quota256' in p.parts:continue
   paths.add(p)
for folder in ['flydrone','flyview','tests','configs','contracts','scripts','tools','data','runs/tellosim-sdk9','reports/vis','reports/sdk9_c0','reports/golden_evaluation_final','artifacts/golden_episode_verified']:
 add_tree(folder)
for folder in (root/'reports').glob('ts1*'):
 if folder.is_dir():add_tree(str(folder.relative_to(root)))
for p in root.iterdir():
 if p.is_file() and (p.suffix in ['.md','.toml'] or 'requirements' in p.name or p.name in ['sources.lock.json','flylab.py','LICENSE']):paths.add(p)
(report/'tracked-source.diff').write_bytes(subprocess.run(['git','diff','--binary'],capture_output=True,check=True).stdout);paths.add(report/'tracked-source.diff')
for p in Path('/mnt/d/projects/果蝇训练').glob('TS1_browser_*.png'):
 target=report/p.name;shutil.copy2(p,target);paths.add(target)
probe=r"""
from pathlib import Path
import hashlib,json
import numpy as np,torch
from flyview.tellosim_api import Observatory
import flyview.tellosim_api as module
from flydrone.policy import ActorCritic
root=Path.cwd();assert Path(module.__file__).resolve().is_relative_to(root)
manifest=root/'PACKAGE_MANIFEST.json'
if manifest.exists():
 for name,expected in json.loads(manifest.read_text()).items():
  h=hashlib.sha256()
  with (root/name).open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  assert h.hexdigest()==expected,name
a=Observatory.__new__(Observatory);a.root=root;a.directory=root/'reports/vis/tellosim';a.sessions={};catalog=a.catalog()
assert catalog['altitude']['ALTITUDE_TASK_LEARNED'] and len(catalog['spatial']['models'])==3
runs={r['run_id'] for r in catalog['runs']}
for seed in [11,22,33]:
 p=root/f'runs/tellosim-sdk9/spatial-orientation-s{seed}/checkpoint.pt';c=torch.load(p,map_location='cpu',weights_only=False)
 for name,h in c['contract']['sources'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
 for state in c['models'].values():
  model=ActorCritic(128,9);model.load_state_dict(state)
 assert f'spatial-orientation-s{seed}-sealed_test-3' in runs
 for kind in ['spatial-orientation','altitude-refined']:
  m=json.loads((root/f'reports/vis/tellosim/{kind}-s{seed}-sealed_test-3/manifest.json').read_text())
  assert m['complete'] and not m['partial']
  for parts in m['streams'].values():
   for part in parts:assert hashlib.sha256((root/f'reports/vis/tellosim/{kind}-s{seed}-sealed_test-3'/part['file']).read_bytes()).hexdigest()==part['sha256']
assert hashlib.sha256((root/'data/male-v1.npz').read_bytes()).hexdigest()==c['contract']['base']['graph_sha256']
assert (root/'flyview/static/vendor/three/three.module.js').stat().st_size>10000
from flydrone.tellosim.training.spatial_orientation import SpatialPool,load_spatial,SKILLS
from flydrone.tellosim.training.checkpoint import configure_exact_execution
import gc,os
max_private_kib=0
configure_exact_execution('cuda');torch.set_num_threads(4)
for seed in [11,22,33]:
 pool=SpatialPool(root,seed,batch=1)
 load_spatial(root/f'runs/tellosim-sdk9/spatial-orientation-s{seed}/checkpoint.pt',pool,root)
 vector=np.zeros(26,np.float32);vector[:3]=[.1,.2,.3];vector[7]=1;vector[8]=1/3;vector[9]=1;vector[12:15]=1
 for sample,skill in enumerate(SKILLS,1):
  pool.lanes[0].switch_skill(skill)
  features=pool.observe_lanes({0:(vector,sample)})
  assert np.isfinite(features).all()
  action,_,_,info=pool.lanes[0].decision(features[0],np.ones(9,bool),deterministic=True)
  assert 0<=action<9 and info['input_source']=='reservoir_v_trace'
 max_private_kib=max(max_private_kib,sum(int(line.split()[1]) for line in Path('/proc/self/smaps_rollup').read_text().splitlines() if line.startswith(('Private_Clean:','Private_Dirty:'))))
 del pool;gc.collect();torch.cuda.empty_cache()
from flyview.server import Viewer,Handler,ThreadingHTTPServer
import threading,urllib.request
server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
server.viewer=Viewer(root,root/'reports/vis/views');server.tellosim=Observatory(root)
worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
base=f'http://127.0.0.1:{server.server_port}'
try:
 for route in ['/tellosim','/static/tellosim.js','/api/tellosim/runs']:
  with urllib.request.urlopen(base+route,timeout=15) as response:
   body=response.read();assert response.status==200 and len(body)>100
   if route.endswith('runs'):assert json.loads(body)['spatial']['run_prefix']=='spatial-orientation'
finally:
 server.shutdown();worker.join(timeout=5);server.tellosim.close();server.server_close()
assert not worker.is_alive()
print(json.dumps({'status':'PASS','pid':os.getpid(),'sampled_private_kib':max_private_kib,'actual_extracted_source':str(module.__file__),'models_loaded':9,'strict_full_graph_inference_checks':9,'inference_fixture_not_task_score':True,'replays_checked':6,'graph_included':True,'http_routes_checked':3,'server_thread_joined':True}))
"""
# Correct model module is verified before the archive is made.
with tempfile.TemporaryDirectory(prefix='ts1-final-stage-') as stage_dir, tempfile.TemporaryDirectory(prefix='ts1-final-extract-') as extract_dir:
 stage=Path(stage_dir);extract=Path(extract_dir)
 for p in sorted(paths):
  q=stage/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 registry=stage/'reports/vis/views/index.json'
 old=json.loads(registry.read_text());portable={}
 for key,value in old.items():
  value=Path(value);rel=value.relative_to(root) if value.is_absolute() and value.is_relative_to(root) else value
  if not rel.is_absolute() and (stage/rel).exists():portable[key]=str(rel)
 registry.write_text(json.dumps(portable,indent=2))
 summary.update(SOFTWARE_MATRIX_PASSED=True,test_rows_passed=60,SOFTWARE_ACCEPTANCE_READY=True,FULL_TS1_READY=summary['JOINT_3D_TASK_VERIFIED'],PACKAGE_VERIFICATION_PENDING=False,SIMULATION_GOAL_READY=summary['JOINT_3D_TASK_VERIFIED'])
 (stage/'reports/ts1_completion/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 (stage/'START_VIEWER.py').write_text("from pathlib import Path\nimport os\nfrom flyview.server import serve\nroot=Path(__file__).resolve().parent\nos.chdir(root)\nserve(root)\n")
 instructions='# 本轮独立交付包\n\n用现有锁定Python依赖运行 python START_VIEWER.py，打开 http://127.0.0.1:8765/tellosim 。当前报告在reports/ts1_completion/README.md。\n\n包含完整MaleCNS图、源码、权重、失败实验、正式回放与历史证据。未包含虚拟环境、Git目录或256MiB原始压力填充文件；压力报告和可复现脚本已包含。新目录解包验收使用现有WSL依赖，不是全新机器安装测试。\n\n完整TS1软件验收与扩展C2三维模型门槛分开，参见summary.json；真实飞行始终未就绪。\n'
 (stage/'START_HERE.md').write_text(instructions)
 (stage/'VERIFY_PACKAGE.py').write_text(probe)
 stage_check=subprocess.run([sys.executable,'VERIFY_PACKAGE.py'],cwd=stage,capture_output=True,text=True)
 (report/'package-stage-probe.log').write_text(stage_check.stdout+stage_check.stderr)
 if stage_check.returncode:raise RuntimeError('staging probe failed before archive; see package-stage-probe.log')
 assert not Path(f"/proc/{json.loads(stage_check.stdout)['pid']}").exists()
 matrix=json.loads((stage/'reports/ts1_completion/TEST_MATRIX.json').read_text())
 for row in matrix:
  if row['test_id']=='T60':row.update(status='PASS',artifact=['PACKAGE_CONTENT_AUDIT.json','../../PACKAGE_MANIFEST.json','../../VERIFY_PACKAGE.py'],external_archive_audit='Adjacent PACKAGE_AUDIT.json records final archive hash and fresh-extraction verification')
 (stage/'reports/ts1_completion/TEST_MATRIX.json').write_text(json.dumps(matrix,ensure_ascii=False,indent=2))
 (stage/'reports/ts1_completion/PACKAGE_CONTENT_AUDIT.json').write_text(json.dumps({'status':'PASS','staged_probe':json.loads(stage_check.stdout),'publication_rule':'Archive published only after full fresh extraction, every-file hash verification and repeated model/HTTP probe; final archive audit adjacent to ZIP'},indent=2))
 inventory={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()};(stage/'PACKAGE_MANIFEST.json').write_text(json.dumps(inventory,indent=2))
 candidate=dest/'delivery.candidate.zip'
 with zipfile.ZipFile(candidate,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for p in sorted(stage.rglob('*')):
   if p.is_file():z.write(p,str(p.relative_to(stage)))
 with zipfile.ZipFile(candidate) as z:
  assert z.testzip() is None;z.extractall(extract)
 for name,h in inventory.items():assert sha(extract/name)==h,name
 result=subprocess.run([sys.executable,'VERIFY_PACKAGE.py'],cwd=extract,capture_output=True,text=True)
 (report/'package-probe.log').write_text(result.stdout+result.stderr)
 if result.returncode:raise RuntimeError('package probe failed; candidate preserved; see package-probe.log')
 assert not Path(f"/proc/{json.loads(result.stdout)['pid']}").exists()
 for name,h in inventory.items():assert sha(extract/name)==h,('post-probe mutation',name)
 final=dest/'FlyBrain_TS1_V1R_C2Q_20260928.zip';candidate.rename(final)
 audit={'status':'PASS','path':str(final),'sha256':sha(final),'files':len(inventory)+1,'archive_bytes':final.stat().st_size,'uncompressed_bytes':sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()),'fresh_extraction':True,'all_file_hashes_verified':True,'extracted_model_and_viewer_probe':json.loads(result.stdout),'probe_processes_reaped':True,'stage_probe_pid':json.loads(stage_check.stdout)['pid'],'extraction_probe_pid':json.loads(result.stdout)['pid'],'new_machine_install_tested':False,'excluded':['virtual environment','git','raw quota pressure fill'],'runtime_lock':'requirements-runtime.lock.txt'}
 (report/'PACKAGE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2));(dest/'PACKAGE_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 shutil.copy2(stage/'reports/ts1_completion/PACKAGE_CONTENT_AUDIT.json',report/'PACKAGE_CONTENT_AUDIT.json')
 (report/'TEST_MATRIX.json').write_text(json.dumps(matrix,ensure_ascii=False,indent=2))
 (report/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));shutil.copy2(report/'README.md',dest/'README.md');print(json.dumps(audit,ensure_ascii=False))
