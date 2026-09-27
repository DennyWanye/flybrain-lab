"""Heading task acceptance; synthetic graph fixtures are not trained-model evidence."""
import json,math
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from tests.flydrone.test_rigid_training import project,pool,ROOT
from flydrone.tellosim.training.heading import HeadingEnv,heading_case,teacher,wrap_angle,save_heading,load_heading,HEADING_SPEC
from flydrone.tellosim.training.heading_campaign import RuleDiagnostic
from flydrone.tellosim.training.runtime import save_checkpoint


def test_measured_target_error_wrap_and_sensor_failure(project):
    case=heading_case(1);case.update(initial_yaw_rad=math.radians(170),target_yaw_rad=math.radians(-170))
    env=HeadingEnv(project,case,RuleDiagnostic())
    try:
        assert math.atan2(env.observation[6],env.observation[7])==pytest.approx(math.radians(20),abs=.001)
        assert teacher(env.observation,env.mask)==8
        assert np.flatnonzero(env.mask).tolist()==[0,7,8]
        env.sensors.latest=replace(env.sensors.latest,yaw_rad=math.radians(-160))
        env.prepare_observation()
        assert math.atan2(env.observation[6],env.observation[7])==pytest.approx(math.radians(-10),abs=.001)
        assert teacher(env.observation,env.mask)==0
        env.sensors.latest=replace(env.sensors.latest,position_m=None,yaw_rad=None,velocity_mps=None)
        env.prepare_observation()
        assert env.observation[12]==0 and not env.observation[6:8].any()
        assert np.flatnonzero(env.mask).tolist()==[0]
    finally:env.close()


def test_stationary_wrong_heading_cannot_succeed(project):
    env=HeadingEnv(project,heading_case(8),RuleDiagnostic())
    try:
        while not env.terminated:env.step(0)
        assert env.reason=='task_deadline' and env.hold==0
        assert abs(wrap_angle(env.target_yaw-env.session.world.yaw_rad))>HEADING_SPEC['heading_tolerance_rad']
    finally:env.close()


@pytest.mark.parametrize('sign',[-1,1])
def test_physical_turn_sign_and_two_second_hold(project,sign):
    case=heading_case(3);case.update(initial_yaw_rad=math.radians(170),target_yaw_rad=wrap_angle(math.radians(170)+sign*math.pi/2))
    env=HeadingEnv(project,case,RuleDiagnostic());actions=[]
    try:
        while not env.terminated:
            a=teacher(env.observation,env.mask,env.steps==0);actions.append(a);env.step(a)
        assert env.reason=='success' and env.hold>=2-1e-8
        assert (8 if sign>0 else 7) in actions
        assert abs(wrap_angle(env.target_yaw-env.session.world.yaw_rad))<=math.radians(16)
        assert abs(env.session.world.data.qvel[5])<=.08
    finally:env.close()


def test_heading_readout_cannot_load_navigation_checkpoint(project):
    p=pool(project,1);path=project/'heading.pt';save_heading(path,p,None,ROOT,0)
    load_heading(path,p,ROOT)
    old=project/'navigation.pt';save_checkpoint(old,p,None,ROOT,0,0,{})
    with pytest.raises(ValueError,match='contract'):load_heading(old,p,ROOT)


def test_recorded_target_and_episode_identity(project):
    p=pool(project,1);case=heading_case(6);env=HeadingEnv(project,case,p.lanes[0],record=True,run_id='heading-fixture')
    env.step(8);env.close('fixture_end')
    directory=project/'reports/vis/tellosim/heading-fixture';m=json.loads((directory/'manifest.json').read_text())
    assert m['observation_schema']==HEADING_SPEC['observation']
    assert m['scene']['target_yaw_rad']==case['target_yaw_rad']
    assert m['task_spec']['allowed_actions']==[0,7,8]
    for stream in ('trajectory','transition'):
        rows=[json.loads(line) for path in directory.glob(stream+'-*.jsonl') for line in path.read_text().splitlines()]
        assert rows and all(row['episode_id']==case['case_id'] for row in rows)
    assert any('heading' in row for row in rows)
