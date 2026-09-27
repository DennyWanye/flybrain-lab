"""Engineering SDK3 telemetry subset; world position is deliberately absent."""
import math

UNITS={'pitch':'deg','roll':'deg','yaw':'deg','vgx':'dm/s','vgy':'dm/s','vgz':'dm/s',
       'tof':'cm','h':'cm','bat':'percent','baro':'m','time':'s'}

def parse_state(text):
    if not isinstance(text,str) or len(text)>2048:raise ValueError('invalid state packet')
    raw={}
    for item in text.strip().strip(';').split(';'):
        if not item or ':' not in item:raise ValueError('invalid state field')
        key,value=item.split(':',1)
        if key in raw:raise ValueError('duplicate state field')
        try:number=float(value)
        except ValueError:raise ValueError('nonnumeric state field') from None
        if not math.isfinite(number):raise ValueError('nonfinite state field')
        raw[key]=number
    fields={key:raw.get(key) for key in UNITS}
    height=raw.get('h');battery=raw.get('bat')
    if height is not None and height<0:raise ValueError('negative height')
    if battery is not None and not 0<=battery<=100:raise ValueError('battery outside 0..100')
    return {'raw':raw,'fields':fields,'units':UNITS,'height_m':None if height is None else height/100,
        'velocity_body_mps':[None if raw.get(k) is None else raw[k]/10 for k in ('vgx','vgy','vgz')],
        'barometer_m':raw.get('baro'),'battery_fraction':None if battery is None else battery/100,
        'position_world_m':None,'position_source':None,'mission_pad_valid':False,
        'profile':'sdk3_subset_engineering_v1','velocity_unit_note':'dm/s per chosen SDK3 profile; hardware calibration unverified'}


def serialize_state(session):
    yaw=session.world.yaw_rad;c,s=math.cos(yaw),math.sin(yaw)
    vx,vy,vz=session.world.velocity;z=float(session.world.position[2])
    # Legacy packets retain their original fields for reproducible old runs.
    values={'yaw':math.degrees(yaw),'vgx':(c*vx+s*vy)*10,'vgy':(-s*vx+c*vy)*10,
        'vgz':vz*10,'tof':max(0,z*100),'h':max(0,z*100),
        'bat':max(0,100-session.tick/120/12),'baro':z,'time':session.tick/120}
    if session.world.config.controller_profile=='rigid_body_thrust_v2':
        from ..physics.rigid import rotation
        R=rotation(session.world.data.qpos[3:7]);body=R.T@session.world.velocity
        values.update(pitch=math.degrees(math.asin(max(-1.,min(1.,-R[2,0])))),
            roll=math.degrees(math.atan2(R[2,1],R[2,2])),vgx=body[0]*10,vgy=body[1]*10,vgz=body[2]*10)
    return ''.join(f'{key}:{value:.6f};' for key,value in values.items())
