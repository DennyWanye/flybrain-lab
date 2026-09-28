"""Snapshot all training evidence without mutating source, weights or old packages."""
import hashlib,json,os,re,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT=Path('/mnt/d/projects/果蝇训练/github-ts1-c2w-shadow-v1-20260928')
OUT.mkdir(exist_ok=False)
REPORT=ROOT/'reports/github_release_20260928'
paths=set()
excluded=[]
for base in ('runs','reports','artifacts','logs'):
    for p in Path(base).rglob('*'):
        if not p.is_file():continue
        if '__pycache__' in p.parts or p.suffix=='.pyc':continue
        if any(str(p).startswith(prefix) for prefix in ('reports/ts1_completion/quota256/','reports/vis/live/','reports/github_release_20260928/')):
            excluded.append({'path':str(p),'bytes':p.stat().st_size});continue
        paths.add(p)
for raw in subprocess.check_output(['git','ls-files','-z']).decode().split('\0'):
    if raw and Path(raw).is_file() and not raw.startswith(('reports/github_release_20260928/','reports/vis/live/')):paths.add(Path(raw))
for base in ('flydrone','flyview','tests','docs','configs','contracts'):
    for p in Path(base).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'node_modules' not in p.parts:paths.add(p)
paths.update([Path('data/male-v1.npz'),Path('data/male-v1.npz.json')])
# File names and token patterns only; no secret values are printed or saved.
secret=re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
manifest={};total=0;last=time.monotonic()
for i,p in enumerate(sorted(paths)):
    if p.is_symlink():raise RuntimeError('unexpected symlink '+str(p))
    h=hashlib.sha256();tail=b''
    scan=p.suffix.lower() not in ('.pt','.pth','.npz','.png','.jpg','.gif','.zip','.pdf')
    with p.open('rb') as stream:
        while block:=stream.read(1024*1024):
            h.update(block)
            if scan and secret.search(tail+block):raise RuntimeError('credential-like content in '+str(p))
            tail=block[-256:]
    size=p.stat().st_size;manifest[str(p)]={'bytes':size,'sha256':h.hexdigest()};total+=size
    if time.monotonic()-last>25:print('HASH',i+1,len(paths),flush=True);last=time.monotonic()
metadata={'schema':'flybrain.release-artifacts/1','release_tag':'ts1-c2w-shadow-v1-20260928','files':manifest,'payload_files':len(manifest),'payload_bytes':total,'excluded_local_files':excluded,'exclusions':'caches, live Viewer state, raw upstream downloads, quota stress filler, environment and Git; originals retained locally','REAL_FLIGHT_READY':False}
manifest_bytes=(json.dumps(metadata,indent=2)+'\n').encode()
(REPORT/'ARTIFACT_MANIFEST.json').write_bytes(manifest_bytes)
(OUT/'ARTIFACT_MANIFEST.json').write_bytes(manifest_bytes)
archive=OUT/'FlyBrain_C2W_ShadowV1_Training_Records_20260928.zip'
last=time.monotonic()
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
    for i,name in enumerate(manifest):
        z.write(name,name)
        if time.monotonic()-last>25:print('ZIP',i+1,len(manifest),flush=True);last=time.monotonic()
    z.writestr('reports/github_release_20260928/ARTIFACT_MANIFEST.json',manifest_bytes)
print('VERIFY_ZIP',archive.stat().st_size,flush=True)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(manifest)+1
    for name,expected in manifest.items():
        h=hashlib.sha256()
        with z.open(name) as stream:
            while block:=stream.read(1024*1024):h.update(block)
        assert h.hexdigest()==expected['sha256'],name
        assert z.getinfo(name).file_size==expected['bytes'],name
h=hashlib.sha256()
with archive.open('rb') as stream:
    while block:=stream.read(1024*1024):h.update(block)
assets={'tag':metadata['release_tag'],'files':[{'name':archive.name,'bytes':archive.stat().st_size,'sha256':h.hexdigest()},{'name':'ARTIFACT_MANIFEST.json','bytes':len(manifest_bytes),'sha256':hashlib.sha256(manifest_bytes).hexdigest()}],'archive_member_hashes_verified':len(manifest),'credential_pattern_findings':0,'new_training_actions':0,'REAL_FLIGHT_READY':False}
(REPORT/'RELEASE_ASSETS.json').write_text(json.dumps(assets,indent=2)+'\n')
(OUT/'SHA256SUMS').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in assets['files']))
(OUT/'RELEASE_ASSETS.json').write_text(json.dumps(assets,indent=2)+'\n')
print(json.dumps(assets),flush=True)
