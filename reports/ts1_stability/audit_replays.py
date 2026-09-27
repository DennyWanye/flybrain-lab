"""Audit formal recordings, including explicit physical perturbation evidence."""
from pathlib import Path
import json,hashlib
root=Path.cwd();out=root/'reports/ts1_stability';expected={}
for seed in (11,22,33):
    for suffix in ('sealed_test','before-sealed'):
        for row in json.loads((out/f's{seed}-{suffix}.json').read_text())['results']:
            if row['run_id']:expected[row['run_id']]=row
runs=[]
for name,result in sorted(expected.items()):
    folder=root/'reports/vis/tellosim'/name;m=json.loads((folder/'manifest.json').read_text())
    assert m['complete'] and not m.get('partial') and m['outcome']==result['reason']
    assert m['disturbance_evidence']==result['disturbance']
    assert m['case']['disturbance_profile']==result['disturbance']['profile']
    if result['disturbance']['profile'] in ('force','combined'):assert result['disturbance']['absolute_impulse_ns']>0
    chunks=0;transitions=[];force_records=0
    for stream,parts in m['streams'].items():
        for part in parts:
            p=folder/part['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==part['sha256'];chunks+=1
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            assert all(r.get('episode_id')==m['episode_id'] and r.get('env_id')==m['env_id'] for r in rows)
            if stream=='transition':transitions.extend(rows)
            force_records+=sum(any(abs(v)>0 for v in r.get('control',{}).get('external_force_world_n',[])) for r in rows)
    if result['disturbance']['profile'] in ('force','combined'):assert force_records>0
    assert transitions and transitions[-1]['finished'] and transitions[-1]['termination_reason']==m['outcome']
    for frame in transitions:
        if frame.get('policy'):
            policy=frame['policy'];assert policy['skill']==frame['task_phase']
            assert policy['input_source']=='reservoir_v_trace' and policy['checkpoint_sha256']==m['skill_bundle']['sources'][policy['skill']]['sha256']
        assert len(frame['observation_names'])==26
    runs.append({'run_id':name,'outcome':m['outcome'],'profile':result['disturbance']['profile'],'chunks_verified':chunks,
                 'disturbance_evidence_verified':True,'identity_and_policy_sources_verified':True,'force_rows_found':force_records})
assert len(runs)==15
(out/'REPLAY_AUDIT.json').write_text(json.dumps({'all_passed':True,'runs':runs},indent=2));print('15 formal replay recordings verified')
