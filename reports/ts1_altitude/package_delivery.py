"""Build portable evidence; run extracted source; hash every file."""
from pathlib import Path
import hashlib,json,zipfile,shutil,tempfile,subprocess,sys,datetime
ROOT=Path('.').resolve();REPORT=ROOT/'reports/ts1_altitude'
DEST=Path('/mnt/d/projects/果蝇训练/ts1-altitude-delivery')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    summary=json.loads((REPORT/'summary.json').read_text());assert summary.get('status')=='evaluated' and len(summary.get('models',[]))==3 and all('sealed' in x for x in summary['models'])
    DEST.mkdir(parents=True,exist_ok=True);paths=set();excluded=[]
    def add_tree(name):
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.zip','.pyc'):paths.add(p)
    for name in ('flydrone','flyview','configs','contracts','tests','scripts','tools','reports/sdk9_c0','reports/vis/views','reports/vis/live','reports/golden_evaluation_final','artifacts/golden_episode_verified'):add_tree(name)
    for p in list((ROOT/'reports/ts1_c1').iterdir())+list((ROOT/'reports/ts1_heading').iterdir())+list((ROOT/'reports/ts1_joint').iterdir())+list((ROOT/'reports/ts1_joint_refined').iterdir())+list((ROOT/'reports/ts1_robust').iterdir())+list((ROOT/'reports/ts1_stability').iterdir()):
        if p.is_file() and p.suffix in ('.json','.md','.xml','.py'):paths.add(p)
    valid_dirs={'verified-fp64-b1','verified-fp64-b4-continuous','verified-fp64-b4-resumed','stop-sigint','stop-sigterm','stop-wall'}
    for p in REPORT.iterdir():
        if p.is_file() and p.suffix not in ('.zip','.pyc'):
            if p.name in ('smoke-only.pt','smoke-bundle.json','smoke-rule.json','smoke-learned.json','PACKAGE_AUDIT.json','NEURAL_SMOKE.json','PHYSICS_SMOKE.json'):excluded.append(str(p.relative_to(ROOT)))
            else:paths.add(p)
        elif p.name in valid_dirs:add_tree(p.relative_to(ROOT))
        elif p.is_dir() and p.name!='__pycache__':excluded.append(str(p.relative_to(ROOT)))
    for name in ('flylab.py','README.md','README_OPEN_SOURCE.md','LICENSE','NOTICE.md','sources.lock.json','pyproject.toml'):
        if (ROOT/name).exists():paths.add(ROOT/name)
    paths.update(p for p in ROOT.glob('*requirements*') if p.is_file())
    for p in (ROOT/'runs/tellosim-sdk9').iterdir():
        if p.name.startswith(('joint-refined-smoke','robust-smoke','stability-smoke','altitude-smoke')):excluded.append(str(p.relative_to(ROOT)));continue
        if p.is_dir() and (p.name.startswith(('rigid-final-','rigid-adapt-','c1-s','heading-s','joint-s','joint-refined-s','robust-s','stability-s','altitude-s')) or p.name in ('c0-s11-20260927','c0-s22-20260927','c0-s33-20260927')):add_tree(p.relative_to(ROOT))
    for p in (ROOT/'reports/vis/tellosim').iterdir():
        if p.is_file():paths.add(p)
        elif p.name.startswith(('rigid-accepted-','rigid-final-','rigid-v2-demo-final','c0-','sdk9_c0-','golden-episode','ts1_rigid_v2-','c1-','heading-','joint-','robust-','stability-','altitude-','manual-deca69a07a27')) and not p.name.startswith(('c1-smoke-','altitude-smoke-','altitude-joint-retention-')):add_tree(p.relative_to(ROOT))
    for p in (ROOT/'reports/ts1_rigid_v2').iterdir():
        if p.is_file() and p.suffix in ('.md','.json','.xml'):paths.add(p)
    diff=REPORT/'tracked-source.diff';diff.write_bytes(subprocess.run(['git','diff','--binary'],cwd=ROOT,check=True,capture_output=True).stdout);paths.add(diff)
    probe=r'''
from pathlib import Path
import json,hashlib
from flyview.tellosim_api import Observatory
import flyview.tellosim_api as api_module
root=Path('.').resolve()
assert Path(api_module.__file__).resolve().is_relative_to(root)
assert 'ts1_altitude/summary.json' in (root/'flyview/tellosim_api.py').read_text()
summary=json.loads((root/'reports/ts1_altitude/summary.json').read_text())
assert summary['stage']=='V1' and len(summary['models'])==3
api=Observatory.__new__(Observatory);api.root=root;api.directory=root/'reports/vis/tellosim';api.sessions={}
catalog=api.catalog();assert catalog['altitude']==summary
known={r['run_id'] for r in catalog['runs']}
assert all(r['run_id'] in known and r['down_run_id'] in known for r in summary['models'])
for row in summary['models']:
    for run in (row['run_id'],row['down_run_id']):
        folder=root/'reports/vis/tellosim'/run
        m=json.loads((folder/'manifest.json').read_text())
        assert m['complete'] and m['disturbance_evidence']['profile']=='combined'
        assert m['disturbance_evidence']['absolute_impulse_ns']>0
        for chunks in m['streams'].values():
            for c in chunks:assert hashlib.sha256((folder/c['file']).read_bytes()).hexdigest()==c['sha256']
        for source in [m['altitude_checkpoint']]:
            assert hashlib.sha256((root/source['path']).read_bytes()).hexdigest()==source['sha256']
assert len((root/'flyview/static/vendor/three/three.module.js').read_bytes())>10000
print('PACKAGE_OFFLINE_OK')
'''
    with tempfile.TemporaryDirectory(prefix='flybrain-joint-stage-') as directory:
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
        instructions='# 打开V1独立高度控制训练交付包\n\n解压后cd至根目录，使用既有锁定依赖运行 `python START_VIEWER.py`，打开 http://127.0.0.1:8765/tellosim 。报告在reports/ts1_altitude/README.md。\n\n独立高度权重位于runs/tellosim-sdk9/altitude-sXX/checkpoint.pt。每种子新增2048个上下高度训练动作，并学习连续神经状态标签。三维位置与朝向联合任务尚未验证。页面点击V1高度上升或下降入口。纯回放无需外部大图；重新评估需要data/male-v1.npz及已有WSL依赖环境，完整大图和虚拟环境未打包。这是新目录验证，不是新机器安装验证。完整TS1和真机未就绪。\n'
        (stage/'START_HERE.md').write_text(instructions)
        result=subprocess.run([sys.executable,'-c',probe],cwd=stage,capture_output=True,text=True,timeout=60)
        (REPORT/'package-smoke.log').write_text(result.stdout+'\n'+result.stderr)
        assert result.returncode==0,result.stderr
        assert 'PACKAGE_OFFLINE_OK' in result.stdout
        shutil.copy2(REPORT/'package-smoke.log',stage/'reports/ts1_altitude/package-smoke.log')
        files=[p for p in stage.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
        manifest={str(p.relative_to(stage)):sha(p) for p in sorted(files)}
        metadata={'format':'flybrain.altitude.delivery/1.0','files':manifest,'external_graph':{'path':'data/male-v1.npz','original_path':str(ROOT/'data/male-v1.npz'),'sha256':'badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9'},'runtime':'Ubuntu-24.04 WSL; existing locked dependencies; venv and full graph omitted','excluded_diagnostic_checkpoints':excluded,'registry_transform':'Original absolute paths converted to package-relative paths; START_VIEWER.py sets cwd to root','omitted_registry_entries':omitted,'MODEL_READY_FOR_NEXT_STAGE':summary['MODEL_READY_FOR_NEXT_STAGE'],'ALTITUDE_TASK_LEARNED':summary['ALTITUDE_TASK_LEARNED'],'JOINT_3D_TASK_VERIFIED':False,'SINGLE_POLICY_JOINT_TRAINED':False,'FULL_TS1_READY':False,'REAL_FLIGHT_READY':False}
        package=DEST/'FlyBrain_TS1_Altitude_20260927.zip'
        with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(files):z.write(p,p.relative_to(stage))
            z.writestr('DELIVERY_MANIFEST.json',json.dumps(metadata,ensure_ascii=False,indent=2))
        with tempfile.TemporaryDirectory(prefix='flybrain-joint-extract-') as unpacked:
            with zipfile.ZipFile(package) as z:
                assert z.testzip() is None;z.extractall(unpacked)
            for name,expected in manifest.items():assert sha(Path(unpacked)/name)==expected,name
            result2=subprocess.run([sys.executable,'-c',probe],cwd=unpacked,capture_output=True,text=True,timeout=60)
            assert result2.returncode==0 and 'PACKAGE_OFFLINE_OK' in result2.stdout,result2.stderr
    audit={'zip':str(package),'sha256':sha(package),'bytes':package.stat().st_size,'verified_extracted_files':len(manifest),'all_hashes_match':True,'fresh_directory_offline_smoke':True,'browser_interaction_verified':False,'fresh_machine':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (REPORT/'PACKAGE_AUDIT.json').write_text(json.dumps(audit,indent=2))
    for p in REPORT.iterdir():
        if p.is_file() and p.suffix in ('.md','.json','.png','.xml','.log'):shutil.copy2(p,DEST/p.name)
    models=DEST/'models';models.mkdir(exist_ok=True)
    for p in (ROOT/'runs/tellosim-sdk9').glob('altitude-s*'):
        if 'smoke' not in p.name:shutil.copytree(p,models/p.name,dirs_exist_ok=True)
    print(json.dumps(audit))
if __name__=='__main__':main()
