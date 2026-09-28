import json
from flydrone.tellosim.recording_v3 import RecordingV3,AuditedVisualSession
from tests.flydrone.test_rigid_training import project

def test_quota_closes_all_stream_buffers_and_records_missing_ranges(tmp_path):
    r=RecordingV3(tmp_path,{'run_id':'q','epoch':'e','episode_id':'q'},quota=10000)
    for i in range(240):r.add('physics',i,{'x':i})
    for i in range(240,250):r.add('neural',i,{'status':'not_recorded'})
    r.close('fixture')
    m=json.loads((tmp_path/'manifest.json').read_text())
    assert m['partial'] and m['bytes']<=10000 and not r.buffers
    assert {x['stream'] for x in m['missing_intervals']}=={'physics','neural'}
    assert next(x for x in m['missing_intervals'] if x['stream']=='physics')['first_tick']==0

def test_new_session_uses_versioned_recorder(project):
    s=AuditedVisualSession(project,run_id='recorder3-fixture')
    try:assert s.recording.manifest['recording_contract']=='tellosim.recording/3'
    finally:s.close()
