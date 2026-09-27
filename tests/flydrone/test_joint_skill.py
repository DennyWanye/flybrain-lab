"""Joint task invariants; rule fixtures do not prove learned performance."""
import math,json
from dataclasses import replace
import numpy as np
import pytest
from tests.flydrone.test_rigid_training import project
from flydrone.tellosim.training.joint import JointEnv,joint_case,wrap_angle
from flydrone.tellosim.training.joint_campaign import JointDiagnostic
from flydrone.tellosim.training.c0_campaign import expert
from flydrone.tellosim.training.heading import teacher

def case():
    c=joint_case(19);c.update(start=[0.,0.,1.],goal=[1.,0.,1.],initial_yaw_rad=0.,target_yaw_rad=math.pi/2);return c

@pytest.mark.parametrize('at_position',[False,True])
def test_only_one_goal_cannot_succeed(project,at_position):
    c=case()
    if at_position:c['goal']=c['start'][:]
    else:c['target_yaw_rad']=0.
    env=JointEnv(project,c,JointDiagnostic())
    try:
        while not env.terminated:env.step(0)
        assert env.reason in ('phase_deadline','task_deadline') and env.hold==0
    finally:env.close()

def test_measured_phase_state_anchor_and_continuous_physics(project):
    env=JointEnv(project,case(),JointDiagnostic())
    try:
        sensor=env.sensors.latest
        env.sensors.latest=replace(sensor,position_m=(1.,0.,1.),velocity_mps=(0.,0.,0.),yaw_rad=.4)
        env.prepare_observation();assert env.measured_position_state()==(True,False)
        # Truth remains far away. The manager reads only the substituted measurement.
        assert np.linalg.norm(env.session.world.position[:2]-[1,0])>.8
        before=env.session.world.export_state();tick=env.agent.brain_tick;sample=env.agent.sample_id
        env.switch_phase('heading');env.prepare_observation()
        np.testing.assert_array_equal(before['integration'],env.session.world.export_state()['integration'])
        assert env.agent.brain_tick==tick and env.agent.sample_id==sample
        assert env.heading_anchor==[1.,0.,1.] and np.flatnonzero(env.mask).tolist()==[0,7,8]
        assert math.atan2(env.observation[6],env.observation[7])==pytest.approx(math.pi/2-.4)
        env.sensors.latest=replace(sensor,position_m=(1.3,0.,1.),velocity_mps=(0.,0.,0.))
        env.prepare_observation();assert env.measured_position_state()==(False,True)
        env.switch_phase('navigation');env.prepare_observation()
        assert env.case['goal']==[1.,0.,1.] and env.observation[0]<0
        env.sensors.latest=replace(sensor,position_m=None,velocity_mps=None,yaw_rad=None)
        env.prepare_observation();assert env.measured_position_state()==(False,False)
        assert np.flatnonzero(env.mask).tolist()==[0]
    finally:env.close()

@pytest.mark.parametrize('sign',[-1,1])
def test_continuous_task_clocks_and_joint_hold(project,sign):
    c=case();c['target_yaw_rad']*=sign;env=JointEnv(project,c,JointDiagnostic());actions=[]
    try:
        start=env.start_tick
        while not env.terminated:
            phase=env.phase
            a=expert(env.observation,env.mask,env.steps==0) if phase=='navigation' else teacher(env.observation,env.mask)
            actions.append((phase,a));result=env.step(a)
            assert result['brain_after']-result['brain_before']==4*result['k']
        assert env.reason=='success' and env.hold>=2-1e-8
        assert env.agent.brain_tick==4*(1+(env.session.tick-start)//12)
        assert len(env.phase_events)==1 and env.phase_events[0]['physics_state_unchanged']
        assert ('heading',8 if sign>0 else 7) in actions
        assert np.linalg.norm(env.session.world.position[:2]-np.asarray(c['goal'])[:2])<=.2
        assert abs(wrap_angle(env.target_yaw-env.session.world.yaw_rad))<=math.radians(16)
        assert sum(o['wire']=='takeoff' for o in env.session.requests.values())==1
    finally:env.close()
