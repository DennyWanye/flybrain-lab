"""Read-only legacy command/phase diagnosis; run from the existing WSL checkout."""
from pathlib import Path
from dataclasses import asdict
import json, os, hashlib, time
import numpy as np
from flydrone.tellosim.training.spatial_orientation import SpatialEnv, SpatialDiagnostic, spatial_case
from flydrone.tellosim.training.spatial_orientation_campaign import label_for, result_row
from flydrone.tellosim.training.altitude_refined import with_profile, PROFILES

ROOT = Path.cwd()
OUT = ROOT / 'reports/ts1_control_development_v'

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2))

def snapshot(env):
    w = env.session.world
    return dict(tick=env.session.tick, phase=env.phase, truth=w.position.tolist(),
                velocity=w.velocity.tolist(), yaw=w.yaw_rad, yaw_rate=float(w.data.qvel[5]),
                target=w.target.tolist(), target_yaw=w.target_yaw,
                sensor=asdict(env.device.latest_observation()), observation=env.observation.tolist(),
                hold=env.hold, altitude_hold=env.altitude_hold, navigation_hold=env.navigation_hold,
                height_drift_hold=env.height_drift_hold, drift_hold=env.drift_hold,
                operation=dict(env.session.operation), phase_events=list(env.phase_events))

def episode(case, scripted=False):
    env = SpatialEnv(ROOT, case, SpatialDiagnostic())
    rows=[]; actions=[]; skills=[]
    original=env.session.command
    commands=[]
    def command(wire, request_id, **kwargs):
        before=snapshot(env); result=original(wire,request_id,**kwargs)
        commands.append(dict(wire=wire,before=before,after=snapshot(env)))
        return result
    env.session.command=command
    try:
        while not env.terminated:
            if scripted:
                if len(actions)>=18:break
                # SDK commands are exercised directly, independent of phase masks.
                wire=['stop','cw 30','forward 20'][len(actions)%3]
                op=env.session.command(wire,f'diagnostic-{len(actions)}')
                start=env.session.tick
                while env.session.tick-start<240 or env.session.operation['client']=='sent':
                    env.device.advance(12);env.sensors.sample(env.session);env.prepare_observation()
                    rows.append(snapshot(env))
                    if env.session.finished:break
                actions.append(wire)
            else:
                action=label_for(env);actions.append(action);skills.append(env.phase)
                it=env.step_iter(action)
                for _ in it:
                    env.features=env.agent.observe(env.observation,env.device.latest_observation().sample_id).copy()
                    rows.append(snapshot(env))
        result=dict(case=case,commands=commands,trace=rows)
        if not scripted:result['result']=result_row(env,actions,skills,None)
        write(case['case_id']+'.json', result)
        print(json.dumps(dict(case=case['case_id'],commands=len(commands),result=result.get('result',{}).get('reason'))),flush=True)
    finally:env.close()

def main():
    OUT.mkdir(exist_ok=False)
    inventory={}
    paths=set()
    for pattern in ['flydrone/**/*.py','flyview/**/*.py','flyview/static/*','runs/tellosim-sdk9/**/*.pt','reports/ts1*/**/*.json','reports/ts1*/**/*.py','reports/ts1*/**/*.md']:
        paths.update(ROOT.glob(pattern))
    for p in sorted(paths):
        if p.is_file() and OUT not in p.parents:
            inventory[str(p.relative_to(ROOT))]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    write('historical-inventory.json',inventory)
    write('process.json',dict(pid=os.getpid(),cwd=str(ROOT),started=time.time(),scope='legacy diagnostic only'))
    # Existing development cases are deliberate reproductions, not new evaluation.
    cases=json.loads((ROOT/'reports/ts1_rule_development/cases.json').read_text())
    for index in [13,29]:episode(cases[index])
    for i,profile in enumerate(PROFILES):
        case=with_profile(spatial_case(520000000+i,f'command-{profile}'),profile)
        case['start']=[0.,0.,1.];case['goal']=[0.,0.,1.5]
        episode(case,True)
    write('memory.json',dict(pid=os.getpid(),private_kib=sum(int(x.split()[1]) for x in Path('/proc/self/smaps_rollup').read_text().splitlines() if x.startswith(('Private_Clean:','Private_Dirty:')))))

if __name__=='__main__':main()
