"""Publish only after staging and fresh-extraction verification; preserve old packages."""
from pathlib import Path
import hashlib,json,zipfile,shutil,tempfile,subprocess,sys,os
root=Path.cwd();report=root/'reports/ts1_completion_c2w';dest=Path('/mnt/d/projects/果蝇训练/ts1-v1r-c2w-delivery-20260928-r1')
summary=json.loads((report/'summary.json').read_text())
assert summary['SOFTWARE_FUNCTIONAL_ACCEPTANCE_PASSED'] and summary['JOINT_3D_TASK_VERIFIED'] and summary['regression_tests']==211
for name,key in [('BROWSER_C2W.json','status'),('T32-C2W.json','status'),('HISTORY_AUDIT.json','all_original_bytes_preserved')]:
 value=json.loads((report/name).read_text())[key];assert value in [True,'PASS']
assert json.loads((root/'reports/ts1_spatial_continuous/summary.json').read_text())['acceptance']['passed']
assert json.loads((root/'reports/ts1_spatial_continuous/TRAINING_AND_EVALUATION_AUDIT.json').read_text())['passed']
assert json.loads((root/'reports/ts1_spatial_continuous/FORMAL_REPLAY_AUDIT.json').read_text())['passed']
dest.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2))
paths=set()
def add_tree(folder):
 for p in (root/folder).rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.zip']:continue
  if 'quota256' in p.parts:continue
  if report in p.parents and (p.name.startswith('package-') or p.name in ['package.log','PACKAGE_AUDIT.json']):continue
  paths.add(p)
for folder in ['flydrone','flyview','tests','configs','contracts','scripts','tools','data','runs/tellosim-sdk9','reports/vis','reports/sdk9_c0','reports/golden_evaluation_final','artifacts/golden_episode_verified']:add_tree(folder)
for folder in (root/'reports').glob('ts1*'):
 if folder.is_dir():add_tree(str(folder.relative_to(root)))
for p in root.iterdir():
 if p.is_file() and (p.suffix in ['.md','.toml'] or 'requirements' in p.name or p.name in ['sources.lock.json','flylab.py','LICENSE']):paths.add(p)
probe=(report/'verify_package_template.py').read_text();processes={}
def run_probe(folder,label):
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.Popen([sys.executable,'-B','VERIFY_PACKAGE.py'],cwd=folder,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 processes[label]={'pid':p.pid,'running':True};write(report/'package-probe-processes.json',processes)
 try:stdout,stderr=p.communicate(timeout=600)
 finally:
  if p.poll() is None:
   p.terminate()
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  processes[label]={'pid':p.pid,'returncode':p.returncode,'absent':not Path(f'/proc/{p.pid}').exists()};write(report/'package-probe-processes.json',processes)
 (report/f'package-{label}.log').write_text(stdout+stderr)
 assert p.returncode==0,(label,stderr[-1500:])
 result=json.loads(stdout.strip().splitlines()[-1]);assert result['status']=='PASS' and processes[label]['absent'];return result
with tempfile.TemporaryDirectory(prefix='ts1-c2w-stage-') as stage_dir,tempfile.TemporaryDirectory(prefix='ts1-c2w-extract-') as extract_dir:
 stage=Path(stage_dir);extract=Path(extract_dir)
 for p in sorted(paths):
  q=stage/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 registry=stage/'reports/vis/views/index.json';old=json.loads(registry.read_text());portable={}
 for key,value in old.items():
  value=Path(value);rel=value.relative_to(root) if value.is_absolute() and value.is_relative_to(root) else value
  if not rel.is_absolute() and (stage/rel).exists():portable[key]=str(rel)
 write(registry,portable)
 write(stage/'reports/ts1_completion_c2w/TRACKED_EXPECTATIONS.json',{'regression_tests':211,'R2_audit':'../ts1_spatial_continuous/TRAINING_AND_EVALUATION_AUDIT.json','publication_rule':'not published unless all fresh extraction hashes, full graph inference, replay and HTTP checks pass'})
 (stage/'START_VIEWER.py').write_text(chr(10).join(['from pathlib import Path','import os','from flyview.server import serve','root=Path(__file__).resolve().parent','os.chdir(root)','serve(root)','']))
 (stage/'VERIFY_PACKAGE.py').write_text(probe)
 (stage/'START_HERE.md').write_text('# V1R + C2W 模拟阶段交付\n\n使用现有锁定Python依赖运行 `python START_VIEWER.py`，打开 http://127.0.0.1:8765/tellosim 。运行 `python -B VERIFY_PACKAGE.py` 可复核清单、包内完整图与三个模型、回放、HTTP。\n\nC2W三组正式门槛全部通过，权重来自C2Q，本轮新增训练动作0。完整报告：reports/ts1_completion_c2w/README.md；最终状态：reports/ts1_completion_c2w/summary.json。所有历史失败版本和证据保留。\n\n新目录解包核验使用现有WSL依赖，不是全新机器安装测试。未包含venv/Git/256MiB原始压力填充文件。真实无人机飞行未授权、未执行、未就绪。\n\n交付只在新目录全部文件哈希、包内完整图推理、回放/API及运行后再次哈希均通过后发布。最终ZIP摘要在包旁PACKAGE_AUDIT.json，避免自身摘要循环。\n')
 stage_probe=run_probe(stage,'stage');print('staging probe PASS',flush=True)
 final_summary={**summary,'SOFTWARE_ACCEPTANCE_READY':True,'SOFTWARE_MATRIX_PASSED':True,'FULL_TS1_READY':True,'SIMULATION_GOAL_READY':True,'PACKAGE_VERIFICATION_PENDING':False,'test_rows_passed':60,'scope':'R2 original formal gates and R8 software/browser/package acceptance; simulation only; real flight disabled'}
 write(stage/'reports/ts1_completion_c2w/summary.json',final_summary)
 matrix=json.loads((report/'TEST_MATRIX.json').read_text())
 for row in matrix:
  if row['test_id']=='T60':row.update(status='PASS',artifact=['PACKAGE_CONTENT_AUDIT.json','../../PACKAGE_MANIFEST.json','../../VERIFY_PACKAGE.py'],external_archive_audit='Adjacent PACKAGE_AUDIT.json after fresh extraction and post-inference hash verification')
 write(stage/'reports/ts1_completion_c2w/TEST_MATRIX.json',matrix)
 content_audit={'status':'PASS','staging_probe':stage_probe,'publication_rule':'Final ZIP is published only after fresh extraction and unchanged hashes after actual probes; ZIP hash is in adjacent audit'}
 write(stage/'reports/ts1_completion_c2w/PACKAGE_CONTENT_AUDIT.json',content_audit)
 assert not list(stage.rglob('*.pyc'))
 manifest={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()};write(stage/'PACKAGE_MANIFEST.json',manifest)
 candidate=dest/'delivery.candidate.zip'
 with zipfile.ZipFile(candidate,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for p in sorted(stage.rglob('*')):
   if p.is_file():z.write(p,str(p.relative_to(stage)))
 print('archive created; fresh extraction begins',flush=True)
 with zipfile.ZipFile(candidate) as z:
  assert z.testzip() is None;z.extractall(extract)
 extracted_probe=run_probe(extract,'extracted')
 assert extracted_probe['file_hashes_before_and_after']==len(manifest)
 # Independently recheck after the subprocess has exited.
 assert all(sha(extract/name)==expected for name,expected in manifest.items())
 final=dest/'FlyBrain_TS1_V1R_C2W_20260928.zip';candidate.replace(final)
 audit={'status':'PASS','path':str(final),'sha256':sha(final),'files':len(manifest)+1,'archive_bytes':final.stat().st_size,'uncompressed_bytes':sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()),'fresh_extraction':True,'every_file_hash_before_and_after_actual_inference':True,'staging_probe':stage_probe,'extracted_probe':extracted_probe,'probe_processes':processes,'new_machine_install_tested':False,'REAL_FLIGHT_READY':False,'excluded':['venv','Git','pycache','raw256MiB quota fill'],'runtime_lock':'requirements-runtime.lock.txt'}
 write(dest/'PACKAGE_AUDIT.json',audit);write(report/'PACKAGE_AUDIT.json',audit)
 write(report/'summary.json',final_summary);write(report/'TEST_MATRIX.json',matrix);write(report/'PACKAGE_CONTENT_AUDIT.json',content_audit)
 evidence=dest/'evidence';evidence.mkdir()
 for name in ['summary.json','TEST_MATRIX.json','README.md','CHANGES.md','PACKAGE_AUDIT.json','PACKAGE_CONTENT_AUDIT.json','BROWSER_C2W.json','T32-C2W.json','HISTORY_AUDIT.json','IMPACT_AND_RETENTION.json','PRE_PACKAGE_PROCESS_AUDIT.json','regression.xml','TS1_C2W_replay_20260928.png','TS1_C2W_pan_20260928.png']:shutil.copy2(report/name,evidence/name)
 print(json.dumps(audit),flush=True)
