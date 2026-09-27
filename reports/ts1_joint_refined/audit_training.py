"""Check real parameter updates, old-head immutability and held-out split isolation."""
from pathlib import Path
import json,torch,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from flydrone.tellosim.training.joint import bundle_spec
ROOT=Path.cwd();OUT=ROOT/'reports/ts1_joint_refined'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cases=json.loads((OUT/'cases.json').read_text());heldout={c['seed'] for rows in cases.values() for c in rows};protocol=json.loads((OUT/'protocol.json').read_text());audit=[]
for seed in (11,22,33):
    run=ROOT/f'runs/tellosim-sdk9/joint-refined-s{seed}';t=json.loads((run/'training.json').read_text());provenance=json.loads((run/'provenance.json').read_text())
    assert t['options']==t['new_examples']==len(provenance)==2816 and t['roundtrip_exact'] and not t['smoke_only']
    assert not heldout & {r['seed'] for r in provenance}
    old_bundle=bundle_spec(ROOT,seed);assert old_bundle==protocol['original_bundles'][str(seed)]
    old=torch.load(ROOT/old_bundle['sources']['navigation']['path'],map_location='cpu',weights_only=False)
    new=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert old['contract']==new['contract'] and new['value_trained'] is False
    delta=float(torch.sqrt(sum((old['policy'][k]-v).square().sum() for k,v in new['policy'].items())))
    assert delta>0 and abs(delta-t['parameter_delta_l2'])<1e-6
    assert sha(run/'checkpoint.pt')==t['checkpoint_sha256']
    assert t['training_source_sha256']==sha(ROOT/'flydrone/tellosim/training/joint_refine.py')
    audit.append({'seed':seed,'new_training_actions':len(provenance),'parameter_delta_l2':delta,'heldout_overlap':0,'original_navigation_and_heading_unchanged':True,'same_graph_mapping_physics_contract':True,'roundtrip_exact':True})
(OUT/'TRAINING_AUDIT.json').write_text(json.dumps(audit,indent=2));print('Three training lineages and split isolation verified')
