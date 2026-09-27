from pathlib import Path
import json,math,time
from types import SimpleNamespace
import numpy as np
from flydrone.tellosim.visual import VisualSession,VisualWorld,scene_config,atomic_json
from flydrone.tellosim.physics.rigid import PROFILE,rotation
from flydrone.tellosim.training.env import curriculum_case
from flydrone.tellosim.training.c0_campaign import evaluate
from flydrone.tellosim.training.contracts import digest
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2'
class RuleClock:
    physics_profile=PROFILE;stop_dwell_s=2.;feature_source='reservoir';training_method='RULE BASELINE; no neural model'
    def __init__(self):self.brain=SimpleNamespace(graph_sha256='NOT_USED_RULE_BASELINE',mapping_sha256='NOT_USED');self.reset()
    def reset(self):self.brain_tick=0;self.sample_id=-1
    def observe(self,obs,sample_id):
        assert sample_id>self.sample_id;self.sample_id=sample_id;self.brain_tick+=4;return np.zeros(1,np.float32)

def main():
    started=time.monotonic();cases={'validation':[curriculum_case(5100000+i,f'rigid-validation-{i:03d}') for i in range(100)],
        'sealed_test':[curriculum_case(5200000+i,f'rigid-sealed-{i:03d}') for i in range(300)]}
    atomic_json(OUT/'cases.json',cases);atomic_json(OUT/'case-hashes.json',{k:digest(v) for k,v in cases.items()})
    hover=[]
    for i in range(5):
        rng=np.random.default_rng(770+i);w=VisualWorld(scene_config(ROOT));pos=np.array([0,0,1.])+rng.uniform(-.1,.1,3)
        w.reset(tuple(pos));axis=rng.normal(size=3);axis/=np.linalg.norm(axis);angle=rng.uniform(-5,5)*math.pi/180
        w.data.qpos[3:7]=np.r_[math.cos(angle/2),axis*math.sin(angle/2)]
        import mujoco
        mujoco.mj_forward(w.model,w.data)
        w.powered=True;w.set_target((0,0,1));w.step(1200)
        error=np.abs(w.position-[0,0,1.]);tilt=math.degrees(math.acos(np.clip(rotation(w.data.qpos[3:7])[2,2],-1,1)))
        hover.append({'case':i,'error_m':error.tolist(),'tilt_deg':tilt,'passed':bool(max(error)<=.1 and tilt<15)})
    free=VisualWorld(scene_config(ROOT));free.reset((0,0,2));free.step(24)
    fall={'z_m':float(free.position[2]),'expected_z_m':2-.5*9.81*.2**2,'passed':bool(abs(free.position[2]-(2-.5*9.81*.2**2))<.02)}
    s=VisualSession(ROOT,mode='script',run_id='rigid-v2-demo-final')
    while not s.finished:s.advance(12)
    demo={'run_id':s.run_id,'outcome':s.recording.manifest['outcome'],'operations':list(s.history),'collision':s.collision_latched}
    saved=OUT/'rule-validation.json'
    result=json.loads(saved.read_text()) if saved.exists() else evaluate(ROOT,RuleClock(),cases['validation'],saved,'rigid-rule-validation','rule',record_indices=())
    assert result['case_hash']==digest(cases['validation']) and result['episodes']==100
    report={'physics':PROFILE,'hover':hover,'gravity':fall,'demo':demo,'rule_successes':result['successes'],'rule_episodes':result['episodes'],
        'passed':all(r['passed'] for r in hover) and fall['passed'] and demo['outcome']=='script_completed' and result['success_rate']>=.99,
        'elapsed_s':time.monotonic()-started}
    atomic_json(OUT/'preflight.json',report);print(json.dumps(report),flush=True)
    summary=json.loads((OUT/'summary.json').read_text());summary.update(demo_run=s.run_id,preflight=report['passed']);atomic_json(OUT/'summary.json',summary)
if __name__=='__main__':main()
