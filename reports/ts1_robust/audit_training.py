"""Verify real updates, data separation and frozen source/weight lineage."""
from pathlib import Path
import sys,json
root=Path.cwd();sys.path.insert(0,str(root))
import torch
from flydrone.tellosim.training.robust_campaign import SEEDS,SKILLS,source_hashes,old_bundle,bundle_spec
from flydrone.tellosim.training.joint import sha
out=root/'reports/ts1_robust';cases=json.loads((out/'cases.json').read_text());heldout={r['seed'] for values in cases.values() for r in values};report=[]
for seed in SEEDS:
    bundle=bundle_spec(root,seed);old=old_bundle(root,seed)
    for skill in SKILLS:
        folder=root/f'runs/tellosim-sdk9/robust-s{seed}/{skill}';t=json.loads((folder/'training.json').read_text());provenance=json.loads((folder/'provenance.json').read_text())
        assert t['status']=='completed' and not t['smoke_only'] and t['options']==1280 and len(provenance)==1280
        assert not heldout.intersection(r['seed'] for r in provenance)
        assert t['source_hashes']==source_hashes(root) and t['source']==old['sources'][skill]
        initial=torch.load(root/t['source']['path'],map_location='cpu',weights_only=False)
        final=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert initial['contract']==final['contract']
        delta=float(torch.sqrt(sum((initial['policy'][k]-v).square().sum() for k,v in final['policy'].items())))
        assert delta>0 and abs(delta-t['parameter_delta_l2'])<1e-6
        assert sha(root/t['source']['path'])==t['source']['sha256']
        assert sha(folder/'checkpoint.pt')==t['checkpoint_sha256'] and t['roundtrip_exact']
        assert set(r['profile'] for r in provenance)=={'clean','pose','force','combined'}
        report.append({'seed':seed,'skill':skill,'real_training_actions':1280,'parameter_delta_l2':delta,'heldout_overlap':0,'old_checkpoint_unchanged':True,'base_contract_unchanged':True,'roundtrip_exact':True})
(out/'TRAINING_AUDIT.json').write_text(json.dumps(report,indent=2));print('six trained skill readouts audited')
