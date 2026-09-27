from pathlib import Path
import math,time
import numpy as np
from flydrone.tellosim.visual import VisualSession,scene_config,atomic_json
from flydrone.tellosim.physics.rigid import rotation
ROOT=Path('.').resolve();OUT=ROOT/'reports/ts1_rigid_v2'
def execute(s,wire,key):
    s.command(wire,key);metrics={'max_tilt_deg':0.,'max_yaw_rate_deg_s':0.,'max_step_m':0.,'max_thrust_n':0.,'max_torque_nm':0.};last=s.world.position
    while s.operation['client']=='sent' and not s.finished:
        s.advance(1);w=s.world;R=rotation(w.data.qpos[3:7]);c=w.last_control
        metrics['max_tilt_deg']=max(metrics['max_tilt_deg'],math.degrees(math.acos(np.clip(R[2,2],-1,1))))
        metrics['max_yaw_rate_deg_s']=max(metrics['max_yaw_rate_deg_s'],abs(math.degrees(w.data.qvel[5])))
        metrics['max_step_m']=max(metrics['max_step_m'],float(np.linalg.norm(w.position-last)));last=w.position
        metrics['max_thrust_n']=max(metrics['max_thrust_n'],c.get('thrust_n',0.))
        metrics['max_torque_nm']=max(metrics['max_torque_nm'],max(map(abs,c.get('torque_body_nm',[0.]))))
    return {'wire':wire,'client':s.operation['client'],'position_m':s.world.position.tolist(),'target_m':s.world.target.tolist(),
        'error_m':float(np.linalg.norm(s.world.position-s.world.target)),'duration_s':(s.tick-s.operation['sent_tick'])/120,**metrics}
started=time.monotonic();rows=[]
for i,wire in enumerate(('forward 500','back 500','left 500','right 500','up 500','down 20','cw 360','ccw 360','go 21 21 21 50')):
    scene=scene_config(ROOT);scene['room_size_m']=[14,14,8]
    s=VisualSession(ROOT,scene=scene,recording_enabled=False);execute(s,'command','sdk');execute(s,'takeoff','takeoff');execute(s,'speed 100','speed')
    r=execute(s,wire,'motion');r['passed']=r['client']=='ack_ok' and r['max_tilt_deg']<=15 and r['max_yaw_rate_deg_s']<=45.5 and r['max_thrust_n']<=1.962+1e-9 and r['max_torque_nm']<=.01
    rows.append(r);s.close()
scene=scene_config(ROOT);scene['obstacles']=[{'center_m':[.8,0,1],'half_extents_m':[.01,1,1]}]
s=VisualSession(ROOT,scene=scene,recording_enabled=False);execute(s,'command','sdk');execute(s,'takeoff','up');execute(s,'speed 100','speed');contact=execute(s,'forward 200','wall');contact['passed']=bool(s.collision_latched and contact['client']=='ack_error' and s.world.position[0]<.8);s.close()
result={'cases':rows,'thin_obstacle':contact,'passed':all(r['passed'] for r in rows) and contact['passed'],'elapsed_s':time.monotonic()-started}
atomic_json(OUT/'physics-edges.json',result);print(result);assert result['passed']
