"""Build portable evidence; run extracted source; hash every file."""
from pathlib import Path
import hashlib,json,zipfile,shutil,tempfile,subprocess,sys,datetime
ROOT=Path('.').resolve();REPORT=ROOT/'reports/ts1_heading'
DEST=Path('/mnt/d/projects/果蝇训练/ts1-heading-delivery')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    summary=json.loads((REPORT/'summary.json').read_text());assert summary.get('status')=='evaluated' and len(summary.get('models',[]))==3 and all('sealed' in x for x in summary['models'])
    DEST.mkdir(parents=True,exist_ok=True);paths=set();excluded=[]
    def add_tree(name):
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.zip','.pyc'):paths.add(p)
    for name in ('flydrone','flyview','configs','contracts','tests','scripts','tools','reports/sdk9_c0','reports/vis/views','reports/vis/live','reports/golden_evaluation_final','artifacts/golden_episode_verified'):add_tree(name)
    for p in (ROOT/'reports/ts1_c1').iterdir():
        if p.is_file() and p.suffix in ('.json','.md','.xml','.py'):paths.add(p)
    valid_dirs={'verified-fp64-b1','verified-fp64-b4-continuous','verified-fp64-b4-resumed','stop-sigint','stop-sigterm','stop-wall'}
    for p in REPORT.iterdir():
        if p.is_file() and p.suffix not in ('.zip','.pyc'):
            if p.name in ('smoke-only.pt','PACKAGE_AUDIT.json'):excluded.append(str(p.relative_to(ROOT)))
            else:paths.add(p)
        elif p.name in valid_dirs:add_tree(p.relative_to(ROOT))
        elif p.is_dir() and p.name!='__pycache__':excluded.append(str(p.relative_to(ROOT)))
    for name in ('flylab.py','README.md','README_OPEN_SOURCE.md','LICENSE','NOTICE.md','sources.lock.json','pyproject.toml'):
        if (ROOT/name).exists():paths.add(ROOT/name)
    paths.update(p for p in ROOT.glob('*requirements*') if p.is_file())
    for p in (ROOT/'runs/tellosim-sdk9').iterdir():
        if p.is_dir() and (p.name.startswith(('rigid-final-','rigid-adapt-','c1-s','heading-s')) or p.name in ('c0-s11-20260927','c0-s22-20260927','c0-s33-20260927')):add_tree(p.relative_to(ROOT))
    for p in (ROOT/'reports/vis/tellosim').iterdir():
        if p.is_file():paths.add(p)
        elif p.name.startswith(('rigid-accepted-','rigid-final-','rigid-v2-demo-final','c0-','sdk9_c0-','golden-episode','ts1_rigid_v2-','c1-','heading-','manual-deca69a07a27')) and not p.name.startswith('c1-smoke-'):add_tree(p.relative_to(ROOT))
    for p in (ROOT/'reports/ts1_rigid_v2').iterdir():
        if p.is_file() and p.suffix in ('.md','.json','.xml'):paths.add(p)
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
    catalog=json.loads(get('/api/tellosim/runs'));assert len(catalog['heading']['models'])==3
    known={x['run_id'] for x in catalog['runs']}
    for row in catalog['heading']['models']:
        assert row['run_id'] in known
        m=json.loads(get('/api/tellosim/runs/'+row['run_id']+'/manifest'));assert m['complete'] and m['policy_checkpoint_sha256']
        assert m['observation_schema']=='tellosim.heading_observation26/1.0'
        assert m['task_spec']['allowed_actions']==[0,7,8]
        first=m['streams']['transition'][0]['file'];frame=json.loads((root/'reports/vis/tellosim'/row['run_id']/first).read_text().splitlines()[0]);assert 'heading' in frame
        for chunks in m['streams'].values():
            for c in chunks:assert (root/'reports/vis/tellosim'/row['run_id']/c['file']).is_file()
    views=json.loads(get('/api/views'))['views'];assert all(v.get('status')!='ERROR' for v in views)
    assert json.loads(get('/api/views/golden-episode/manifest'))['schema_version']=='golden_view/1.0'
    print('PACKAGE_SMOKE_OK')
finally:
    httpd.shutdown();thread.join(timeout=5);httpd.tellosim.close();httpd.server_close()
'''
    with tempfile.TemporaryDirectory(prefix='flybrain-heading-stage-') as directory:
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
        instructions='# 打开 H1 主动转向交付包\n\n先 cd 到解压根目录，使用锁定依赖环境执行 `python START_VIEWER.py`，打开 http://127.0.0.1:8765/tellosim 。报告在 reports/ts1_heading/README.md。\n\n纯回放无需外部大图。训练需要 data/male-v1.npz，原图哈希见 DELIVERY_MANIFEST.json；虚拟环境不在包内，锁定依赖在 reports/sdk9_c0/requirements.lock.txt。新目录验证使用现有 WSL 依赖，不是新机器安装验证。\n\n本轮权重 runs/tellosim-sdk9/heading-sXX/checkpoint.pt 是独立主动转向读出，用 heading.load_heading 加载；C1 导航权重保留在 c1-sXX/ppo/checkpoint.pt，用原导航 loader 加载。H1 未支持精确中途训练恢复；完整 TS1 和真机未就绪。\n'
        (stage/'START_HERE.md').write_text(instructions)
        result=subprocess.run([sys.executable,'-c',probe],cwd=stage,capture_output=True,text=True,timeout=60)
        (REPORT/'package-smoke.log').write_text(result.stdout+'\n'+result.stderr)
        assert result.returncode==0,result.stderr
        assert 'PACKAGE_SMOKE_OK' in result.stdout
        shutil.copy2(REPORT/'package-smoke.log',stage/'reports/ts1_heading/package-smoke.log')
        files=[p for p in stage.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
        manifest={str(p.relative_to(stage)):sha(p) for p in sorted(files)}
        metadata={'format':'flybrain.heading.delivery/1.0','files':manifest,'external_graph':{'path':'data/male-v1.npz','original_path':str(ROOT/'data/male-v1.npz'),'sha256':'badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9'},'runtime':'Ubuntu-24.04 WSL; existing locked dependencies; venv and full graph omitted','excluded_diagnostic_checkpoints':excluded,'registry_transform':'Original absolute paths converted to package-relative paths; START_VIEWER.py sets cwd to root','omitted_registry_entries':omitted,'MODEL_READY_FOR_NEXT_STAGE':summary['MODEL_READY_FOR_NEXT_STAGE'],'HEADING_TASK_LEARNED':summary['HEADING_TASK_LEARNED'],'JOINT_NAVIGATION_AND_HEADING_LEARNED':False,'FULL_TS1_READY':False,'REAL_FLIGHT_READY':False}
        package=DEST/'FlyBrain_TS1_Heading_20260927.zip'
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(files):z.write(p,p.relative_to(stage))
            z.writestr('DELIVERY_MANIFEST.json',json.dumps(metadata,ensure_ascii=False,indent=2))
        with tempfile.TemporaryDirectory(prefix='flybrain-heading-extract-') as unpacked:
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
    for p in (ROOT/'runs/tellosim-sdk9').glob('heading-s*'):shutil.copytree(p,models/p.name,dirs_exist_ok=True)
    print(json.dumps(audit))
if __name__=='__main__':main()
