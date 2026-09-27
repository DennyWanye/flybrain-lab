import hashlib,json
import pytest
from flydrone.tellosim.visual import Recording
from flydrone.tellosim.view_export import normalize_episode_identity

def test_export_preserves_numeric_payload_and_maps_local_identity(tmp_path):
    rec=Recording(tmp_path,{'run_id':'actual','epoch':'epoch-a','episode_id':'episode-0'})
    rec.add('trajectory',0,{'x_m':1.25,'quaternion_wxyz':[1,0,0,0]})
    rec.manifest.update(env_id=2,episode_id='case-a',case={'case_id':'case-a'})
    rec.add('neural',12,{'brain':[.25,.5]});rec.close()
    result=normalize_episode_identity(tmp_path)
    assert result['identity']['env_id']==2
    for chunk in result['chunks']:
        row=json.loads((tmp_path/chunk['file']).read_text())
        assert row['env_id']==2 and row['episode_id']=='case-a'
        assert chunk['changed_rows']==1
        payload={k:v for k,v in row.items() if k not in ('run_id','epoch','env_id','episode_id')}
        body=json.dumps([payload],ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        assert hashlib.sha256(body).hexdigest()==chunk['nonidentity_payload_sha256']
    assert normalize_episode_identity(tmp_path)==result

def test_export_rejects_foreign_lane_without_rewriting(tmp_path):
    rec=Recording(tmp_path,{'run_id':'actual','epoch':'epoch-a','episode_id':'case-a','env_id':2,'case':{'case_id':'case-a'}})
    rec.add('trajectory',0,{'env_id':3,'x_m':1.25});rec.close()
    chunk=tmp_path/'trajectory-00000.jsonl';before=chunk.read_bytes()
    with pytest.raises(ValueError,match='foreign lane'):normalize_episode_identity(tmp_path)
    assert chunk.read_bytes()==before
