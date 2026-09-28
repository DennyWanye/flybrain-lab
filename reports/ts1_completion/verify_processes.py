"""Verify only the processes explicitly owned by this delivery session."""
from pathlib import Path
import json
root=Path.cwd();report=root/'reports/ts1_completion'
folders=['ts1_completion','ts1_altitude_refined','ts1_spatial','ts1_spatial_refined','ts1_spatial_precision','ts1_spatial_orientation','ts1_mlp_pilot','ts1_spatial_development2','ts1_spatial_development3','ts1_spatial_settled','ts1_spatial_handoff','ts1_spatial_stationary','ts1_rule_development','ts1_phase_development','ts1_phase_development2','ts1_phase_development3']
owned={};memory={};records=[]
def walk(value):
 if isinstance(value,dict):
  if isinstance(value.get('pid'),int):yield value
  for v in value.values():yield from walk(v)
 elif isinstance(value,list):
  for v in value:yield from walk(v)
for folder in folders:
 p=root/'reports'/folder
 for f in p.glob('*cleanup*.json'):
  records.append(str(f.relative_to(root)))
  for r in walk(json.loads(f.read_text())):owned[r['pid']]={'source':str(f.relative_to(root)),'returncode':r.get('returncode')}
 for f in p.glob('*memory-sample.json'):
  for r in walk(json.loads(f.read_text())):
   if 'private_kib'in r:memory[r['pid']]=max(memory.get(r['pid'],0),r['private_kib'])
for pid,record in owned.items():record['proc_absent']=not Path(f'/proc/{pid}').exists()
assert owned and all(r['proc_absent'] for r in owned.values()),owned
remaining=[]
modules=['training.altitude_refined_campaign','training.spatial_campaign','training.spatial_refined_campaign','training.spatial_precision_campaign','training.spatial_orientation_campaign','training.direct_mlp','reports.ts1_spatial_development2.diagnose','reports.ts1_spatial_development3.diagnose','reports.ts1_spatial_precision.smoke_fit','reports.ts1_spatial_orientation.start_after_smoke','training.spatial_settled_campaign','training.spatial_handoff_campaign','training.spatial_stationary_campaign','reports.ts1_spatial_settled.run_campaign','reports.ts1_spatial_settled.development_check','reports.ts1_spatial_stationary.run_campaign','reports.ts1_spatial_stationary.development_check','reports.ts1_rule_development.diagnose','reports.ts1_phase_development.diagnose','reports.ts1_phase_development2.diagnose','reports.ts1_phase_development3.diagnose','reports.ts1_completion.c2q_ui_run']
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except (FileNotFoundError,PermissionError):continue
 if any(' -m '+m+' ' in cmd for m in modules):remaining.append({'pid':int(p.name),'command':cmd})
assert not remaining,remaining
viewer_path=report/'viewer-c2q-process.json'
if not viewer_path.exists():viewer_path=report/'viewer-c2p-process.json'
viewer=json.loads(viewer_path.read_text());v=Path('/proc')/str(viewer['pid']);assert v.exists() and '-m flyview serve' in (v/'cmdline').read_bytes().replace(b'\0',b' ').decode()
result={'status':'PASS','owned_children_verified_absent':len(owned),'processes':owned,'cleanup_records':records,'private_memory_samples_kib':memory,'sum_released_sampled_private_kib':sum(memory.values()),'memory_note':'Sum across different completed phases, not simultaneous RAM reduction or machine memory measurement','retained_viewer_pid':viewer['pid'],'remaining_owned_training_processes':remaining}
(report/'FINAL_PROCESS_AUDIT.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ['processes','cleanup_records','private_memory_samples_kib']})
