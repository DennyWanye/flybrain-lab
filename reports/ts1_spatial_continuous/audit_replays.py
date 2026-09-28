"""Audit recorded physics and neural evidence, independently of task success flags."""
from pathlib import Path
import json,hashlib,math,sys
import numpy as np
ROOT=Path.cwd();OUT=ROOT/'reports/ts1_spatial_continuous'
def audit(run_id):
 folder=ROOT/'reports/vis/tellosim'/run_id;m=json.loads((folder/'manifest.json').read_text());data={};bootstrap=[]
 assert m['complete'] and not m['partial']
 for stream,parts in m['streams'].items():
  rows=[]
  for part in parts:
   p=folder/part['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==part['sha256']
   chunk=[json.loads(x) for x in p.read_text().splitlines()]
   for r in chunk:
    assert r['run_id']==run_id and r['epoch']==m['epoch']
    if r['episode_id']!=m['episode_id']:
     assert stream=='trajectory' and r['sim_tick']==0 and r['episode_id']=='episode-0'
     bootstrap.append({'stream':stream,'tick':0,'episode_id':'episode-0'})
   rows.extend(chunk)
  data[stream]=rows
 goal=np.array(m['case']['goal']);yaw=m['case']['target_yaw_rad'];frames=data['transition'];end=frames[-1]['sim_tick']
 assert frames[-1]['finished'] and frames[-1]['termination_reason']==m['outcome']
 tail=[r for r in data['trajectory'] if end-240<r['sim_tick']<=end]
 assert len(tail)==240 and [r['sim_tick'] for r in tail]==list(range(end-239,end+1))
 def valid(r):
  return np.linalg.norm(np.array([r['x_m'],r['y_m']])-goal[:2])<=.2 and abs(r['z_m']-goal[2])<=.1 and np.linalg.norm(r['velocity_mps'][:2])<=.08 and abs(r['velocity_mps'][2])<=.08 and abs((yaw-r['yaw_rad']+math.pi)%(2*math.pi)-math.pi)<=math.radians(16) and abs(r['angular_velocity_body_rad_s'][2])<=.08 and r['airborne'] and not r['collision']
 good=sum(valid(r) for r in tail)
 if m['outcome']=='success':assert good==240,(run_id,good)
 decisions=[r for r in frames if r.get('policy')]
 checkpoint=m['skill_bundle']['checkpoint_sha256']
 assert decisions and all(r['policy']['input_source']=='reservoir_v_trace' and r['policy']['feature_dim']==128 and r['policy']['checkpoint_sha256']==checkpoint for r in decisions)
 clocks={}
 for r in frames:
  sample=r['sensor_contract']['sample_id'];clock=r['brain_tick']
  if sample in clocks:assert clocks[sample]==clock
  clocks[sample]=clock
 pairs=sorted(clocks.items())
 assert all(b[1]-a[1]==4*(b[0]-a[0]) for a,b in zip(pairs,pairs[1:]))
 assert data['neural'] and all(r['brain']['status']=='recorded' and r['brain']['sample_phase']=='after_substep_3' for r in data['neural'])
 return {'run_id':run_id,'outcome':m['outcome'],'complete':True,'graph_sha256':m['graph_sha256'],'checkpoint_sha256':checkpoint,'neural_frames':len(data['neural']),'policy_frames':len(decisions),'final_valid_physics_ticks':good,'required_physics_ticks':240,'neural_clock_verified':True,'constructor_bootstrap_rows':bootstrap}
if __name__=='__main__':
 mode=sys.argv[1]
 runs=[f'spatial-continuous-smoke-{i}' for i in range(8)] if mode=='smoke' else [f'spatial-continuous-s{s}-sealed_test-{i}' for s in [11,22,33] for i in range(4)]
 rows=[audit(run) for run in runs]
 (OUT/(mode.upper()+'_REPLAY_AUDIT.json')).write_text(json.dumps({'passed':True,'runs':rows},indent=2))
 print(json.dumps({'passed':True,'runs':len(rows)}))
