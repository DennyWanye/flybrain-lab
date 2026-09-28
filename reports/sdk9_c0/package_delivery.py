"""Package local source, model evidence and replays; verify every archived byte."""
from pathlib import Path
import hashlib,json,zipfile,shutil,tempfile,subprocess
ROOT=Path('.').resolve();REPORT=ROOT/'reports/sdk9_c0';DEST=Path('/mnt/d/projects/果蝇训练/sdk9-c0-delivery')

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    paths=set()
    for name in ('flydrone','flyview','configs','contracts','tests','scripts','tools','reports/sdk9_c0','reports/vis/views'):
        for p in (ROOT/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.zip','.pyc'):paths.add(p)
    for name in ('flylab.py','README.md','README_OPEN_SOURCE.md','LICENSE','NOTICE.md','sources.lock.json'):
        if (ROOT/name).exists():paths.add(ROOT/name)
    for p in ROOT.glob('*requirements*'):
        if p.is_file():paths.add(p)
    for directory in (ROOT/'runs/tellosim-sdk9').glob('c0-*20260927'):
        paths.update(p for p in directory.rglob('*') if p.is_file())
    for directory in (ROOT/'reports/vis/tellosim').iterdir():
        if directory.is_dir() and (directory.name.startswith(('c0-','sdk9_c0-')) or directory.name=='golden-episode'):
            paths.update(p for p in directory.rglob('*') if p.is_file())
    for name in ('training-summary.json','learning-summary.json'):
        p=ROOT/'reports/vis/tellosim'/name
        if p.exists():paths.add(p)
    diff=subprocess.run(['git','diff','--binary'],cwd=ROOT,check=True,capture_output=True).stdout
    (REPORT/'tracked-source.diff').write_bytes(diff);paths.add(REPORT/'tracked-source.diff')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    metadata={'format':'flybrain.c0.delivery/1.0','files':manifest,
        'external_graph':{'path':'data/male-v1.npz','sha256':'badc33a247894fe12c4d68d3ff791393857cffa39ff2ab81869608a11fa1e0c9'},
        'runtime':'Existing Ubuntu-24.04 WSL environment; virtual environment and large graph omitted',
        'readiness_source':'reports/sdk9_c0/summary.json'}
    package=DEST/'FlyBrain_C0_20260927.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(paths):archive.write(p,p.relative_to(ROOT))
        archive.writestr('DELIVERY_MANIFEST.json',json.dumps(metadata,ensure_ascii=False,indent=2))
    with tempfile.TemporaryDirectory(prefix='flybrain-c0-verify-') as directory:
        with zipfile.ZipFile(package) as archive:
            assert archive.testzip() is None
            archive.extractall(directory)
        for name,expected in manifest.items():assert hashlib.sha256((Path(directory)/name).read_bytes()).hexdigest()==expected,name
    for p in REPORT.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    models=DEST/'models';models.mkdir(exist_ok=True)
    for path in (ROOT/'runs/tellosim-sdk9').glob('c0-*20260927'):
        shutil.copytree(path,models/path.name,dirs_exist_ok=True)
    audit={'zip':str(package),'sha256':hashlib.sha256(package.read_bytes()).hexdigest(),'bytes':package.stat().st_size,
        'verified_extracted_files':len(manifest),'all_hashes_match':True}
    (DEST/'PACKAGE_AUDIT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
if __name__=='__main__':main()
