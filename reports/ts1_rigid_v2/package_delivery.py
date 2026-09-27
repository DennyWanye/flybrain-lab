"""Build portable evidence; run extracted source; hash every file."""
from pathlib import Path
import hashlib,json,zipfile,shutil,tempfile,subprocess,sys,datetime
ROOT=Path('.').resolve();REPORT=ROOT/'reports/ts1_rigid_v2'
DEST=Path('/mnt/d/projects/果蝇训练/ts1-rigid-v2-delivery')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    summary=json.loads((REPORT/'summary.json').read_text());assert all('sealed' in x for x in summary['models'])
    DEST.mkdir(parents=True,exist_ok=True);paths=set();excluded=[]
    def add_tree(name):
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.zip','.pyc'):paths.add(p)
    for name in ('flydrone','flyview','configs','contracts','tests','scripts','tools','reports/sdk9_c0','reports/vis/views','reports/vis/live','reports/golden_evaluation_final','artifacts/golden_episode_verified'):add_tree(name)
    valid_dirs={'perf-fp64-b1','perf-fp64-b4-continuous','perf-fp64-b4-resumed','fp64-repeat-0','fp64-repeat-1','fp64-repeat-2'}
    for p in REPORT.iterdir():
        if p.is_file() and p.suffix not in ('.zip','.pyc'):
            if p.name in ('release-boundary.pt','fork-boundary.pt'):excluded.append(str(p.relative_to(ROOT)))
            else:paths.add(p)
        elif p.name in valid_dirs:add_tree(p.relative_to(ROOT))
        elif p.is_dir() and p.name!='__pycache__':excluded.append(str(p.relative_to(ROOT)))
    for name in ('flylab.py','README.md','README_OPEN_SOURCE.md','LICENSE','NOTICE.md','sources.lock.json','pyproject.toml'):
        if (ROOT/name).exists():paths.add(ROOT/name)
    paths.update(p for p in ROOT.glob('*requirements*') if p.is_file())
    for p in (ROOT/'runs/tellosim-sdk9').iterdir():
        if p.is_dir() and (p.name.startswith(('rigid-final-','rigid-adapt-')) or p.name in ('c0-s11-20260927','c0-s22-20260927','c0-s33-20260927')):add_tree(p.relative_to(ROOT))
    for p in (ROOT/'reports/vis/tellosim').iterdir():
        if p.is_file():paths.add(p)
        elif p.name.startswith(('rigid-accepted-','rigid-final-','rigid-v2-demo-final','c0-','sdk9_c0-','golden-episode','ts1_rigid_v2-')):add_tree(p.relative_to(ROOT))
    diff=REPORT/'tracked-source.diff';diff.write_bytes(subprocess.run(['git','diff','--binary'],cwd=ROOT,check=True,capture_output=True).stdout);paths.add(diff)
    probe=r'''
from pathlib import Path
import json,threading,urllib.request
from http.server import ThreadingHTTPServer
import flyview.server as server
from flyview.tellosim_api import Observatory
root=Path('.').resolve()
assert Path(server.__file__).resolve().is_relative_to(root)
httpd=ThreadingHTTPServer(('127.0.0.1',0),server.Handler);httpd.viewer=server.Viewer(root,root/'reports/vis/views');httpd.tellosim=Observatory(root)
thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
base=f'http://127.0.0.1:{httpd.server_address[1]}'
def get(path):
    with urllib.request.urlopen(base+path,timeout=10) as r:return r.read()
try:
    assert b'neural-status' in get('/tellosim')
    assert b'rigidModels' in get('/static/tellosim.js')
    assert len(get('/static/vendor/three/three.module.js'))>10000
    catalog=json.loads(get('/api/tellosim/runs'));assert len(catalog['rigid']['models'])==3
    known={x['run_id'] for x in catalog['runs']}
    for row in catalog['rigid']['models']:
        assert row['run_id'] in known
        m=json.loads(get('/api/tellosim/runs/'+row['run_id']+'/manifest'));assert m['complete'] and m['policy_checkpoint_sha256']
        for chunks in m['streams'].values():
            for c in chunks:assert (root/'reports/vis/tellosim'/row['run_id']/c['file']).is_file()
    views=json.loads(get('/api/views'))['views'];assert all(v.get('status')!='ERROR' for v in views)
    assert json.loads(get('/api/views/golden-episode/manifest'))['schema_version']=='golden_view/1.0'
    print('PACKAGE_SMOKE_OK')
finally:
    httpd.shutdown();thread.join(timeout=5);httpd.tellosim.close();httpd.server_close()
'''
    with tempfile.TemporaryDirectory(prefix='flybrain-rigid-stage-') as directory:
        stage=Path(directory)
        for p in sorted(paths):
            out=stage/p.relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
        registry=stage/'reports/vis/views/index.json';old=json.loads(registry.read_text());portable={};omitted=[]
        for key,value in old.items():
            rel=Path(value).relative_to(ROOT) if Path(value).is_absolute() else Path(value)
            if (stage/rel).is_dir():portable[key]=str(rel)
            else:omitted.append(key)
        registry.write_text(json.dumps(portable,indent=2))
        launcher="from pathlib import Path\nimport os\nfrom flyview.server import serve\nroot=Path(__file__).resolve().parent\nos.chdir(root)\nserve(root)\n"
        (stage/'START_VIEWER.py').write_text(launcher)
        instructions='# 打开交付包\n\n先 cd 到解压根目录，使用已安装锁定依赖的 Python 执行 `python START_VIEWER.py`，再打开 http://127.0.0.1:8765/tellosim 。如 8765 已被现有服务占用，先使用现有页面。\n\n本轮报告：reports/ts1_rigid_v2/README.md。纯回放无需大图；训练需要外部 data/male-v1.npz，原位置 /home/denny/projects/flybrain_lab_4spark_v0_1/data/male-v1.npz，哈希见 DELIVERY_MANIFEST.json。依赖锁在 reports/sdk9_c0/requirements.lock.txt。虚拟环境不在包内；新目录验证使用现有 WSL 依赖，不等于新机器自动安装验证。\n\n最终可恢复文件只取 runs/tellosim-sdk9/rigid-final-sXX/resume.pt。rigid-adapt 中旧后端恢复文件仅作来源审计，不可按最终合同继续。完整 TS1 和真实飞行仍未就绪。\n'
        (stage/'START_HERE.md').write_text(instructions)
        result=subprocess.run([sys.executable,'-c',probe],cwd=stage,capture_output=True,text=True,timeout=60)
        (REPORT/'package-smoke.log').write_text(result.stdout+'\n'+result.stderr)
        assert result.returncode==0,result.stderr
        assert 'PACKAGE_SMOKE_OK' in result.stdout
        records=json.loads((REPORT/'TEST_MATRIX.json').read_text())
        for row in records:
            if row['test_id']=='T60':row.update(status='PASS',actual='新临时目录加载解包源码、API/静态依赖/三个模型回放及 Golden 注册均通过；使用现有 WSL 依赖，未测全新机器',elapsed_s=None)
        (REPORT/'TEST_MATRIX.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
        md=REPORT/'TEST_MATRIX.md';lines=md.read_text().splitlines();row=next(x for x in records if x['test_id']=='T60')
        lines=[f"| T60 | PASS | {row['input']} | {row['actual']} | PACKAGE_AUDIT.json / package-smoke.log |" if x.startswith('| T60 |') else x for x in lines]
        md.write_text('\n'.join(lines)+'\n')
        for name in ('TEST_MATRIX.json','TEST_MATRIX.md','package-smoke.log'):shutil.copy2(REPORT/name,stage/'reports/ts1_rigid_v2'/name)
        files=[p for p in stage.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
        manifest={str(p.relative_to(stage)):sha(p) for p in sorted(files)}
        metadata={'format':'flybrain.rigid.delivery/1.0','files':manifest,'external_graph':{'path':'data/male-v1.npz','original_path':str(ROOT/'data/male-v1.npz'),'sha256':'badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9'},'runtime':'Ubuntu-24.04 WSL; existing locked dependencies; venv and full graph omitted','excluded_diagnostic_checkpoints':excluded,'registry_transform':'Original absolute paths converted to package-relative paths; START_VIEWER.py sets cwd to root','omitted_registry_entries':omitted,'MODEL_READY_FOR_NEXT_STAGE':summary['MODEL_READY_FOR_NEXT_STAGE'],'FULL_TS1_READY':False,'REAL_FLIGHT_READY':False}
        package=DEST/'FlyBrain_TS1_Rigid_v2_20260927.zip'
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(files):z.write(p,p.relative_to(stage))
            z.writestr('DELIVERY_MANIFEST.json',json.dumps(metadata,ensure_ascii=False,indent=2))
        with tempfile.TemporaryDirectory(prefix='flybrain-rigid-extract-') as unpacked:
            with zipfile.ZipFile(package) as z:
                assert z.testzip() is None;z.extractall(unpacked)
            for name,expected in manifest.items():assert sha(Path(unpacked)/name)==expected,name
            result2=subprocess.run([sys.executable,'-c',probe],cwd=unpacked,capture_output=True,text=True,timeout=60)
            assert result2.returncode==0 and 'PACKAGE_SMOKE_OK' in result2.stdout,result2.stderr
    audit={'zip':str(package),'sha256':sha(package),'bytes':package.stat().st_size,'verified_extracted_files':len(manifest),'all_hashes_match':True,'fresh_directory_api_smoke':True,'fresh_machine':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (REPORT/'PACKAGE_AUDIT.json').write_text(json.dumps(audit,indent=2))
    for p in REPORT.iterdir():
        if p.is_file() and p.suffix in ('.md','.json','.png','.xml','.log'):shutil.copy2(p,DEST/p.name)
    models=DEST/'models';models.mkdir(exist_ok=True)
    for p in (ROOT/'runs/tellosim-sdk9').glob('rigid-final-s*'):shutil.copytree(p,models/p.name,dirs_exist_ok=True)
    print(json.dumps(audit))
if __name__=='__main__':main()
