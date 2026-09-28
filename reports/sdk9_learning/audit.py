from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
root=Path(__file__).resolve().parents[2];reports=root/'reports/sdk9_learning';runs=[];recordings=[]
for directory in sorted((root/'runs/tellosim-sdk9').glob('nearx-*')):
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['status']=='completed'
    checkpoint=directory/'checkpoint.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==summary['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert not set(state['mapping']['input_indices'])&set(state['mapping']['readout_indices'])
    if (directory/'transitions.jsonl').exists():
        rows=[json.loads(line) for line in (directory/'transitions.jsonl').read_text().splitlines()]
        assert len(rows)==summary['options']
        assert sum(r['k'] for r in rows)==summary['base_ticks']
        for row in rows:
            assert row['after_tick']-row['before_tick']==row['k']*12
            assert row['brain_after']-row['brain_before']==row['k']*4
            assert row['mask'][row['action']]
            assert math.isclose(row['Gamma'],.995**row['k'],abs_tol=1e-12)
            if 'reward_base_terms' in row:
                assert math.isclose(row['reward'],sum(.995**i*sum(t.values()) for i,t in enumerate(row['reward_base_terms'])),abs_tol=1e-9)
            if summary['profile']=='balanced_rate_v3' and row['action']==0 and not row['terminal']:assert row['k']>=20
    else:
        data=np.load(directory/'demonstrations.npz',allow_pickle=False)
        assert data['features'].shape==(summary['options'],128) and np.isfinite(data['features']).all()
        assert data['masks'][np.arange(len(data['teacher_actions'])),data['teacher_actions']].all()
        assert hashlib.sha256((directory/'demonstrations.npz').read_bytes()).hexdigest()==state['extra']['demonstrations_sha256']
        assert state['value_trained'] is False and summary['ppo_updates']==0
        cases=json.loads((directory/'cases.json').read_text())
        assert not set(cases['training_seeds'])&{x['seed'] for x in cases['validation']}
    runs.append({'run':directory.name,'checkpoint_sha256':summary['checkpoint_sha256'],'audit':'passed'})
for directory in sorted((root/'reports/vis/tellosim').glob('nearx-*')):
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['complete'] and not manifest['partial']
    counts={}
    for kind,chunks in manifest['streams'].items():
        ticks=[];count=0
        for chunk in chunks:
            path=directory/chunk['file'];assert path.resolve().is_relative_to(directory.resolve())
            raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==chunk['sha256']
            rows=[json.loads(x) for x in raw.splitlines()];assert len(rows)==chunk['count']
            count+=len(rows);ticks.extend(r['sim_tick'] for r in rows)
            if kind=='transition':assert all(len(r['observation'])==26 for r in rows)
            if kind=='neural':assert all(len(r['brain']['neurons'])==64 for r in rows)
        if kind=='trajectory':assert ticks==list(range(len(ticks)))
        counts[kind]=count
    recordings.append({'run_id':directory.name,'counts':counts,'audit':'passed'})
golden=hashlib.sha256((root/'artifacts/golden_episode_verified/replay.jsonl').read_bytes()).hexdigest()
assert golden=='08bebe5164ae67a8b66577d17c88552a5d6aed39e5ca386977d5d05ea2e04a7e'
report={'runs':runs,'recordings':recordings,'golden_sha256_unchanged':golden,'browser':'NOT_VERIFIED_POLICY_BLOCKED'}
(reports/'artifact-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'audited_models':len(runs),'audited_recordings':len(recordings),'golden_unchanged':True}))
