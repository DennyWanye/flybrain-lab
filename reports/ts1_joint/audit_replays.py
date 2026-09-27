"""Read-only integrity and identity audit for all formal joint replay chunks."""
from pathlib import Path
import json,hashlib
ROOT=Path.cwd();OUT=ROOT/'reports/ts1_joint';runs=[]
expected={r['run_id'] for seed in (11,22,33) for split in ('sealed_test','instruction_pairs') for r in json.loads((OUT/f's{seed}-{split}.json').read_text())['results'] if r['run_id']}
for directory in sorted(ROOT/'reports/vis/tellosim'/run for run in expected):
    m=json.loads((directory/'manifest.json').read_text());assert m['complete'] and not m.get('partial')
    assert m['observation_schema']=='tellosim.joint_observation26/1.0'
    assert m['skill_bundle']['new_training_actions']==0
    transitions=[];count=0
    for stream,chunks in m['streams'].items():
        for c in chunks:
            p=directory/c['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==c['sha256'];count+=1
            rows=[json.loads(line) for line in p.read_text().splitlines()]
            assert all(r.get('episode_id')==m['episode_id'] and r.get('env_id')==m['env_id'] for r in rows)
            if stream=='transition':transitions.extend(rows)
    assert transitions
    for frame in transitions:
        phase=frame['task_phase'];policy=frame.get('policy')
        if policy:
            assert policy['skill']==phase
            assert policy['checkpoint_sha256']==m['skill_bundle']['sources'][phase]['sha256']
            assert policy['input_source']=='reservoir_v_trace'
        assert len(frame['observation_names'])==26
    last=transitions[-1]
    assert last['finished'] and last['termination_reason']==m['outcome']
    assert all(e['physics_state_unchanged'] for e in last['phase_events'])
    runs.append({'run_id':m['run_id'],'chunks_verified':count,'phase_changes':len(last['phase_events']),'outcome':m['outcome'],'identity_verified':True,'phase_policy_source_verified':True})
assert len(runs)==12,len(runs)
(OUT/'REPLAY_AUDIT.json').write_text(json.dumps({'runs':runs,'all_passed':True},indent=2));print('12 formal replay recordings verified')
