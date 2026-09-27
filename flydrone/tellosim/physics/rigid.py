"""Engineering rigid-body controller, world-frame wrench / body-frame omega.

Conventions follow MuJoCo: free-joint angular qvel is local; xfrc_applied
is a Cartesian wrench at the center of mass expressed in world coordinates.
This is not an aircraft calibration or per-motor aerodynamic model.
"""
import math
import numpy as np

PROFILE = 'rigid_body_thrust_v2'
LEGACY = 'bounded_level_body_surrogate'

def rotation(quaternion):
    w,x,y,z=np.asarray(quaternion,dtype=float)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def heading(quaternion):
    R=rotation(quaternion)
    return math.atan2(R[1,0],R[0,0])

def wrench(world, acceleration):
    c=world.config;R=rotation(world.data.qpos[3:7])
    desired=np.asarray(acceleration,dtype=float)+np.array([0.,0.,9.81])
    desired[2]=max(.1,desired[2])
    xy=np.linalg.norm(desired[:2]);limit=desired[2]*math.tan(math.radians(c.max_tilt_deg))
    if xy>limit:desired[:2]*=limit/xy
    z_des=desired/np.linalg.norm(desired)
    # Roll/pitch stabilization uses measured heading; yaw is a separately
    # bounded desired angular rate, integrated by the SAME rigid body.
    yaw=world.yaw_rad;xc=np.array([math.cos(yaw),math.sin(yaw),0.])
    yd=np.cross(z_des,xc);yd/=np.linalg.norm(yd);xd=np.cross(yd,z_des)
    Rd=np.column_stack((xd,yd,z_des))
    skew=.5*(Rd.T@R-R.T@Rd)
    e=np.array([skew[2,1],skew[0,2],skew[1,0]])
    vertical=R.T@np.array([0.,0.,1.])
    e-=vertical*np.dot(e,vertical)
    yaw_rate=float(np.clip(c.yaw_kp*(world.target_yaw-yaw),-c.max_yaw_rate_rad_s,c.max_yaw_rate_rad_s))
    desired_rate=-8.*e+vertical*yaw_rate
    omega=world.data.qvel[3:6].copy()
    alpha=12.*(desired_rate-omega)
    yaw_alpha=float(np.dot(alpha,vertical));alpha-=vertical*yaw_alpha
    alpha=np.clip(alpha,-8.,8.)+vertical*np.clip(yaw_alpha,-c.max_yaw_accel_rad_s2,c.max_yaw_accel_rad_s2)
    inertia=world.model.body_inertia[world.body_id]
    torque_body=np.clip(inertia*alpha+np.cross(omega,inertia*omega),-c.max_torque_nm,c.max_torque_nm)
    thrust=float(np.clip(c.mass_kg*np.dot(desired,R[:,2]),0.,c.max_thrust_weight_ratio*c.mass_kg*9.81))
    force=thrust*R[:,2]
    return force,R@torque_body,{'thrust_n':thrust,'torque_body_nm':torque_body.tolist(),
        'motor_force_world_n':force.tolist(),'desired_acceleration_mps2':np.asarray(acceleration).tolist()}


def translation_reference(start, goal, speed, acceleration, elapsed):
    """Rest-to-rest triangular/trapezoidal reference; never edits plant state."""
    start=np.asarray(start);delta=np.asarray(goal)-start;distance=float(np.linalg.norm(delta))
    if distance<1e-12:return np.asarray(goal).copy(),np.zeros(3),np.zeros(3)
    direction=delta/distance;peak=min(speed,math.sqrt(distance*acceleration))
    ramp=peak/acceleration;cruise=max(0.,distance/peak-ramp)
    if elapsed<ramp:
        s=.5*acceleration*elapsed**2;v=acceleration*elapsed;a=acceleration
    elif elapsed<ramp+cruise:
        s=.5*peak*ramp+peak*(elapsed-ramp);v=peak;a=0.
    elif elapsed<2*ramp+cruise:
        t=elapsed-ramp-cruise;s=.5*peak*ramp+peak*cruise+peak*t-.5*acceleration*t*t;v=peak-acceleration*t;a=-acceleration
    else:s=distance;v=0.;a=0.
    return start+direction*s,direction*v,direction*a
