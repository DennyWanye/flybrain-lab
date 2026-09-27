"""Altitude task semantics; synthetic fixtures never count as model success."""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
from tests.flydrone.test_rigid_training import project,pool,ROOT
from flydrone.tellosim.training.altitude import AltitudeEnv,AltitudeBase,altitude_case,teacher,with_profile,save_altitude,load_altitude
from flydrone.tellosim.training.heading_campaign import RuleDiagnostic
from flydrone.tellosim.training.runtime import save_checkpoint

def test_measured_height_goal_mask_and_invalid_pose(project):
    e=AltitudeEnv(project,with_profile(altitude_case(1,direction=1),'clean'),RuleDiagnostic())
    try:
        assert teacher(e.observation,e.mask)==5
        original=e.mask.copy();e.case['goal'][2]=.5;e.prepare_observation()
        assert teacher(e.observation,e.mask)==6
        np.testing.assert_array_equal(original,e.mask)
        e.sensors.latest=replace(e.sensors.latest,position_m=(0,0,.3));e.prepare_observation();assert not e.mask[6] and e.mask[5]
        e.sensors.latest=replace(e.sensors.latest,position_m=None,yaw_rad=None);e.prepare_observation();assert np.flatnonzero(e.mask).tolist()==[0]
    finally:e.close()

def test_crossing_target_fast_does_not_count_as_stable():
    world=SimpleNamespace(position=np.array([0.,0.,1.5]),velocity=np.array([0.,0.,.2]),yaw_rad=0.,data=SimpleNamespace(qvel=np.zeros(6)))
    e=SimpleNamespace(session=SimpleNamespace(world=world),case={'goal':[0,0,1.5]},target_yaw=0.)
    assert not AltitudeBase.stable_state(e,{'client':'ack_ok'},0)
    world.velocity[2]=.079;assert AltitudeBase.stable_state(e,{'client':'ack_ok'},0)
    world.position[2]=1.601;assert not AltitudeBase.stable_state(e,{'client':'ack_ok'},0)

@pytest.mark.parametrize('sign',[-1,1])
def test_real_up_down_options_minimum_exposure_and_stable_hold(project,sign):
    c=with_profile(altitude_case(9,direction=sign),'combined');c['goal'][2]=1+sign*.4
    e=AltitudeEnv(project,c,RuleDiagnostic());actions=[]
    try:
        while not e.terminated:
            a=teacher(e.observation,e.mask,e.steps==0);actions.append(a);e.step(a)
        assert e.reason=='success' and e.hold>=2-1e-8
        assert (5 if sign>0 else 6) in actions
        assert (e.session.tick-e.start_tick)/120>=8-1e-8
        assert abs(e.session.world.position[2]-c['goal'][2])<=.1 and abs(e.session.world.velocity[2])<=.08
        assert e.agent.brain_tick==4*(1+(e.session.tick-e.start_tick)//12)
        assert e.disturbance_evidence()['absolute_vertical_impulse_ns']>0
    finally:e.close()

def test_vertical_force_changes_integrated_height_without_target_rewrite(project):
    c=altitude_case(19);clean=AltitudeEnv(project,with_profile(c,'clean'),RuleDiagnostic());force=AltitudeEnv(project,with_profile(c,'force'),RuleDiagnostic())
    try:
        target=force.session.world.target.copy()
        for _ in range(60):clean.device.advance(12);force.device.advance(12)
        np.testing.assert_array_equal(target,force.session.world.target)
        assert abs(clean.session.world.position[2]-force.session.world.position[2])>.001
        assert force.disturbance_evidence()['absolute_vertical_impulse_ns']>0
    finally:clean.close();force.close()

def test_stationary_wrong_height_cannot_succeed(project):
    c=altitude_case(23);c['goal'][2]=1.6;e=AltitudeEnv(project,with_profile(c,'clean'),RuleDiagnostic())
    try:
        while not e.terminated:e.step(0)
        assert e.reason=='task_deadline' and e.hold==0
    finally:e.close()

def test_altitude_checkpoint_typed_and_replay_fields(project):
    p=pool(project,1);path=project/'altitude.pt';save_altitude(path,p,None,ROOT,0);load_altitude(path,p,ROOT)
    old=project/'old.pt';save_checkpoint(old,p,None,ROOT,0,0)
    with pytest.raises(ValueError,match='altitude checkpoint'):load_altitude(old,p,ROOT)
    e=AltitudeEnv(project,with_profile(altitude_case(6),'clean'),p.lanes[0],record=True,run_id='altitude-fixture')
    e.step(5);e.close('fixture_end')
    folder=project/'reports/vis/tellosim/altitude-fixture';m=json.loads((folder/'manifest.json').read_text())
    rows=[json.loads(line) for f in folder.glob('transition-*.jsonl') for line in f.read_text().splitlines()]
    assert m['scene']['task_kind']=='altitude_hold' and m['task_spec']['allowed_actions']==[0,5,6]
    assert rows and all('altitude' in r and 'heading' not in r for r in rows)
    assert all(r['episode_id']==m['episode_id'] for r in rows)
