"""T25 controller bounds and separately integrated external wrench matrix."""
from pathlib import Path
from dataclasses import replace,asdict
import json,math
import numpy as np
from flydrone.tellosim.physics.world import SimWorld,WorldConfig
from flydrone.tellosim.physics.rigid import wrench,rotation
from flydrone.tellosim.visual import atomic_json
root=Path.cwd();out=root/'reports/ts1_completion/T25-dynamics.json'
if out.exists():raise FileExistsError(out)
base=WorldConfig(controller_profile='rigid_body_thrust_v2');rows=[]
for limit in [.2,.6]:
    for signs in [(1,1,1),(-1,1,-1),(1,-1,-1),(-1,-1,1)]:
        w=SimWorld(replace(base,max_accel_mps2=limit));w.set_target(tuple(w.position+np.array(signs)*100));w._apply_controller()
        a=np.array(w.last_control['desired_acceleration_mps2'])
        assert abs(np.linalg.norm(a[:2])-limit)<1e-12 and abs(abs(a[2])-limit)<1e-12
        rows.append({'check':'outer_acceleration_saturation','limit':limit,'signs':signs,'commanded_mps2':a.tolist()})
for ratio in [.8,1.,2.]:
    w=SimWorld(replace(base,max_thrust_weight_ratio=ratio));force,torque,log=wrench(w,[0,0,100])
    assert abs(log['thrust_n']-ratio*base.mass_kg*9.81)<1e-12
    assert np.linalg.norm(np.cross(force,rotation(w.data.qpos[3:7])[:,2]))<1e-12
    rows.append({'check':'thrust_saturation_body_axis','ratio':ratio,**log})
for limit in [5.,15.]:
    w=SimWorld(replace(base,max_accel_mps2=50.,max_tilt_deg=limit));w.set_target((100,0,1))
    angles=[];commanded=[]
    for _ in range(240):
        w.step();a=np.array(w.last_control['desired_acceleration_mps2']);commanded.append(math.degrees(math.atan2(np.linalg.norm(a[:2]),9.81+a[2])))
        angles.append(math.degrees(math.acos(np.clip(rotation(w.data.qpos[3:7])[2,2],-1,1))))
    assert abs(angles[-1]-limit)<1
    rows.append({'check':'tilt_parameter_effect','target_limit_deg':limit,'pre_inner_tilt_cap_acceleration_angle_max_deg':max(commanded),'integrated_peak_deg':max(angles),'integrated_final_deg':angles[-1]})
for rate in [math.radians(15),math.radians(45)]:
    for accel in [.25,1.]:
        w=SimWorld(replace(base,max_yaw_rate_rad_s=rate,max_yaw_accel_rad_s2=accel));w.set_yaw(100.)
        speeds=[0.]
        for _ in range(720):w.step();speeds.append(float(w.data.qvel[5]))
        observed_accel=max(abs(np.diff(speeds)/base.dt));assert max(speeds)<=rate+1e-6 and observed_accel<=accel+1e-6 and speeds[-1]>.95*rate
        rows.append({'check':'yaw_rate_and_acceleration','rate_limit':rate,'accel_limit':accel,'peak_rate':max(speeds),'peak_accel':observed_accel})
for cap in [.0001,.01]:
    w=SimWorld(replace(base,max_torque_nm=cap));w.data.qvel[3:6]=[100,-100,100] # isolated actuator unit fixture, never a task trajectory
    _,_,log=wrench(w,[0,0,0]);assert max(abs(x) for x in log['torque_body_nm'])<=cap+1e-12 and max(abs(x) for x in log['torque_body_nm'])>=cap-1e-12
    rows.append({'check':'torque_saturation_fixture','cap_nm':cap,'torque_nm':log['torque_body_nm']})
for axis in range(4):
    for sign in [-1,1]:
        clean=SimWorld(base);wind=SimWorld(base);original=wind._apply_controller;applied=[]
        external=np.zeros(6);external[axis if axis<3 else 5]=sign*(.006 if axis<3 else .00003)
        def disturbed():
            original();motor=wind.data.xfrc_applied[wind.body_id].copy();wind.data.xfrc_applied[wind.body_id]+=external
            actual=wind.data.xfrc_applied[wind.body_id].copy();np.testing.assert_allclose(actual-motor,external,atol=1e-15)
            wind.last_control.update(external_force_world_n=external[:3].tolist(),external_torque_world_nm=external[3:].tolist());applied.append(actual-motor)
        wind._apply_controller=disturbed;target=wind.target.copy()
        for _ in range(120):clean.step();wind.step()
        np.testing.assert_array_equal(wind.target,target)
        displacement=float(np.linalg.norm(wind.position-clean.position));yaw=abs(wind.yaw_rad-clean.yaw_rad)
        assert displacement>1e-5 if axis<3 else yaw>1e-5
        np.testing.assert_allclose(np.sum(applied,axis=0)*base.dt,external,atol=1e-12)
        rows.append({'check':'external_wrench_separation_no_accumulation','axis':axis,'sign':sign,'external':external.tolist(),'integrated_position_difference_m':displacement,'integrated_yaw_difference_rad':yaw})
atomic_json(out,{'status':'PASS','checks':rows,'default_config':asdict(base),'scope':'controller requests/actuator bounds; actual integrated state responds dynamically and is never clamped or rewritten; parameter fixtures are not model results','tilt_note':'target tilt is capped; transient integrated angle is reported separately, not falsely called a hard state bound'})
print(json.dumps({'status':'PASS','checks':len(rows),'tilt_cases':[x for x in rows if x['check']=='tilt_parameter_effect']}))
