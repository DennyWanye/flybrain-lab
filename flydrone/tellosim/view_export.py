"""Map single-session recorder identities to the enclosing training episode.

The physical recorder owns one local environment (0); TrainingEnv supplies the
outer lane/case identity. This explicit export step changes identity metadata
only, never observations, decisions, rewards or trajectories. Legacy runs are
not implicitly rewritten.
"""
import hashlib,json
from pathlib import Path
from .visual import packed,atomic_json

IDENTITY=('run_id','epoch','env_id','episode_id')
def sha(body):return hashlib.sha256(body).hexdigest()
def normalize_episode_identity(directory):
    directory=Path(directory);path=directory/'manifest.json';manifest=json.loads(path.read_bytes())
    if manifest.get('identity_export'):
        return manifest['identity_export']
    if manifest.get('schema_version')!='tellosim.view/2.0' or not manifest.get('complete') or not manifest.get('case'):
        raise ValueError('identity export requires a complete training episode recording')
    expected={key:manifest[key] for key in IDENTITY}
    plans=[];audit=[]
    for kind,chunks in manifest['streams'].items():
        for chunk in chunks:
            source=directory/chunk['file']
            if source.parent!=directory or not source.name.endswith('.jsonl'):raise ValueError('invalid recording chunk path')
            body=source.read_bytes()
            if sha(body)!=chunk['sha256']:raise ValueError('source chunk hash mismatch')
            rows=[json.loads(line) for line in body.splitlines() if line];before=[];after=[];changed=0;local=set()
            for row in rows:
                if row.get('run_id')!=expected['run_id'] or row.get('epoch')!=expected['epoch']:raise ValueError('foreign run or epoch in recording')
                if row.get('env_id') not in (0,expected['env_id']) or row.get('episode_id') not in ('episode-0',expected['episode_id']):raise ValueError('foreign lane or episode in recording')
                before.append({k:v for k,v in row.items() if k not in IDENTITY})
                local.add((row.get('env_id'),row.get('episode_id')))
                changed+=int(any(row.get(k)!=v for k,v in expected.items()));row.update(expected)
                after.append({k:v for k,v in row.items() if k not in IDENTITY})
            payload_hash=sha(packed(before));assert payload_hash==sha(packed(after))
            new_body=b'\n'.join(packed(row) for row in rows)+b'\n'
            audit.append({'file':chunk['file'],'source_sha256':sha(body),'export_sha256':sha(new_body),'changed_rows':changed,'source_local_identities':[list(x) for x in sorted(local)],'nonidentity_payload_sha256':payload_hash})
            plans.append((source,new_body,chunk))
    # Every source is validated before replacing any export chunk.
    for source,body,chunk in plans:
        temp=source.with_suffix('.identity.tmp');temp.write_bytes(body);temp.replace(source)
        chunk.update(bytes=len(body),sha256=sha(body))
    evidence={'format':'tellosim.identity_export/1','identity':expected,'changes':'identity metadata only; numeric payload unchanged','chunks':audit}
    manifest['identity_export']=evidence;manifest['bytes']=sum(c['bytes'] for cs in manifest['streams'].values() for c in cs)
    atomic_json(path,manifest);return evidence
