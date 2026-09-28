"""Post-freeze acceptance audit, no model selection or tuning."""
from pathlib import Path
import json,hashlib
import torch
from flydrone.tellosim.training.spatial_continuous import SKILLS,SPATIAL_SPEC,spatial_contract,sha
from flydrone.tellosim.training.spatial_continuous_campaign import source_hashes
from flydrone.tellosim.training.contracts import digest
ROOT=Path.cwd();OUT=ROOT/'reports/ts1_spatial_continuous'
def read(p):return json.loads(p.read_text())
def main():
 summary=read(OUT/'summary.json');cases=read(OUT/'cases.json');protocol=read(OUT/'protocol.json');locked=read(OUT/'frozen-checkpoints.json')
 assert protocol['split_hashes']=={k:digest(v) for k,v in cases.items()}
 heldout={c['seed'] for group in cases.values() for c in group}
 reports=[]
 for entry in locked['models']:
  seed=entry['seed'];checkpoint=ROOT/entry['path'];origin=read(checkpoint.with_name('MODEL_ORIGIN.json'))
  assert sha(checkpoint)==entry['sha256']==origin['checkpoint_sha256']
  assert origin['source_hashes']==source_hashes(ROOT) and origin['new_training_actions']==0 and origin['parameter_identical']
  source=ROOT/origin['source_checkpoint'];assert sha(source)==origin['source_sha256']
  before=torch.load(source,map_location='cpu',weights_only=False);after=torch.load(checkpoint,map_location='cpu',weights_only=False)
  assert all(torch.equal(value,after['models'][skill][key]) for skill in SKILLS for key,value in before['models'][skill].items())
  provenance=read(source.with_name('provenance.json'));assert not heldout.intersection(x['seed'] for x in provenance)
  rows=[]
  for split,split_cases in cases.items():
   evidence=read(OUT/f's{seed}-{split}.json')
   assert evidence['checkpoint_sha256']==entry['sha256'] and evidence['case_hash']==digest(split_cases)
   assert [r['case_id'] for r in evidence['results']]==[c['case_id'] for c in split_cases]
   for row in evidence['results']:
    assert row['brain_tick']==4*(1+row['physics_ticks']//12)
    assert all(event['physics_state_unchanged'] for event in row['phase_events'])
    assert all(action in SPATIAL_SPEC['allowed_actions'][phase] for action,phase in zip(row['actions'],row['action_skills']))
    assert row['duration_s']<=180 and row['physics_ticks']%12==0
    if row['success']:
     assert row['duration_s']>=8 and row['stable_hold_s']>=2-1e-8
     assert row['height_error_m']<=.1 and row['horizontal_distance_m']<=.2
     assert row['horizontal_speed_mps']<=.08 and row['vertical_speed_mps']<=.08
     assert row['heading_error_deg']<=16 and row['angular_speed_rad_s']<=.08 and row['final_phase']=='heading'
   rows.append({'split':split,'episodes':evidence['episodes'],'successes':evidence['successes'],'case_hash':evidence['case_hash']})
  reports.append({'seed':seed,**origin,'heldout_training_overlap':0,'splits':rows,'all_physics_clock_mask_checks':True})
 (OUT/'TRAINING_AND_EVALUATION_AUDIT.json').write_text(json.dumps({'passed':True,'models':reports,'new_training_actions':0,'prior_training_actions':16896,'physical_hold_implementation':'per120Hz tick, no10Hz approximation','original_acceptance':summary['acceptance']},indent=2))
 print(json.dumps({'audit_passed':True,'model_gate_passed':summary['acceptance']['passed']}))
if __name__=='__main__':main()
