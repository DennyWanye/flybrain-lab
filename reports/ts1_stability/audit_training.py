"""Audit J2R data origin, real readout updates, unchanged graph/input contract."""
from pathlib import Path
import json,sys
import numpy as np
import torch
root=Path.cwd();sys.path.insert(0,str(root))
from flydrone.tellosim.training.stability_campaign import SEEDS,SKILLS,STAGES,source_hashes,old_bundle,bundle_spec
from flydrone.tellosim.training.joint import sha
from flydrone.tellosim.training.contracts import digest
out=root/'reports/ts1_stability';heldout=set()
for name in ['ts1_robust','ts1_stability','ts1_joint_refined','ts1_joint']:
    case_document=json.loads((root/f'reports/{name}/cases.json').read_text())
    for split in ['validation','sealed_test','instruction_pairs']:
        heldout.update(r['seed'] for r in case_document.get(split,[]))
report=[]
for seed in SEEDS:
    bundle=bundle_spec(root,seed);old=old_bundle(root,seed);folder=root/f'runs/tellosim-sdk9/stability-s{seed}'
    t=json.loads((folder/'training.json').read_text());provenance=json.loads((folder/'provenance.json').read_text())
    assert t['status']=='completed' and not t['smoke_only'] and t['options']==sum(STAGES)
    assert sum(r['kind']=='decision' for r in provenance)==sum(STAGES)
    assert t['source_hashes']==source_hashes(root) and t['original_bundle_hash']==digest(old)
    assert not heldout.intersection(r['seed'] for r in provenance)
    assert all(0<=r['seed']-(150000000+seed*100000+r['stage']*10000)<16*500 for r in provenance)
    for skill in SKILLS:
        info=t['skills'][skill];records=[r for r in provenance if r['skill']==skill]
        assert info['new_examples']==len(records) and info['options']==sum(r['kind']=='decision' for r in records)
        assert any(r['kind']=='intermediate' for r in records) and set(r['profile'] for r in records)=={'clean','pose','force','combined'}
        data=np.load(folder/skill/'demonstrations.npz');assert len(data['labels'])==info['examples']
        assert np.array_equal(data['labels'][info['rehearsal_examples']:],[r['label'] for r in records])
        rehearsal=np.load(root/f'runs/tellosim-sdk9/robust-s{seed}/{skill}/demonstrations.npz')
        assert all(np.array_equal(data[key][:info['rehearsal_examples']],rehearsal[key]) for key in ['features','masks','labels','weights'])
        initial=torch.load(root/old['sources'][skill]['path'],map_location='cpu',weights_only=False)
        final=torch.load(folder/skill/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert initial['contract']==final['contract'] and final['extra']['sources']==source_hashes(root)
        delta=float(torch.sqrt(sum((initial['policy'][k]-v).square().sum() for k,v in final['policy'].items())))
        assert delta>0 and abs(delta-info['parameter_delta_l2'])<1e-6
        assert sha(root/old['sources'][skill]['path'])==old['sources'][skill]['sha256']
        assert sha(folder/skill/'checkpoint.pt')==info['checkpoint_sha256'] and info['roundtrip_exact']
        report.append(dict(seed=seed,skill=skill,real_training_actions=info['options'],new_neural_examples=len(records),
            intermediate_examples=sum(r['kind']=='intermediate' for r in records),parameter_delta_l2=delta,
            heldout_overlap=0,old_checkpoint_unchanged=True,base_contract_unchanged=True,roundtrip_exact=True))
assert sum(r['real_training_actions'] for r in report)==len(SEEDS)*sum(STAGES)
(out/'TRAINING_AUDIT.json').write_text(json.dumps(report,indent=2));print('six J2R readouts and dense data origin audited')
