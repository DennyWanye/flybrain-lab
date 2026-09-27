"""Disturbance mechanics and contracts; rule fixtures are not model evidence."""
import math
import numpy as np
import pytest
from tests.flydrone.test_rigid_training import project
from flydrone.tellosim.training.robust_env import with_profile, RobustJointEnv, PROFILES
from flydrone.tellosim.training.robust_campaign import training_case
from flydrone.tellosim.training.joint import JointEnv, joint_case
from flydrone.tellosim.training.joint_campaign import JointDiagnostic


def test_clean_wrapper_preserves_original_physics_and_clock(project):
    case=with_profile(joint_case(17),'clean')
    original=JointEnv(project,case,JointDiagnostic());wrapped=RobustJointEnv(project,case,JointDiagnostic())
    try:
        for action in [0,1,0,8,0]:
            original.step(action);wrapped.step(action)
            np.testing.assert_array_equal(original.session.world.data.qpos,wrapped.session.world.data.qpos)
            np.testing.assert_array_equal(original.session.world.data.qvel,wrapped.session.world.data.qvel)
            np.testing.assert_array_equal(original.features,wrapped.features)
            assert original.agent.brain_tick==wrapped.agent.brain_tick
        assert wrapped.disturbance_ticks==0 and wrapped.disturbance_abs_impulse_ns==0
    finally: original.close();wrapped.close()


def test_force_enters_integration_without_changing_command_target(project):
    case=joint_case(27)
    clean=RobustJointEnv(project,with_profile(case,'clean'),JointDiagnostic())
    gust=RobustJointEnv(project,with_profile(case,'force'),JointDiagnostic())
    try:
        initial=gust.session.world.position.copy();target=gust.session.world.target.copy()
        for _ in range(60):
            clean.device.advance(12);gust.device.advance(12)
        assert gust.disturbance_ticks>0 and gust.disturbance_abs_impulse_ns>0
        assert np.linalg.norm(gust.session.world.position-clean.session.world.position)>.001
        np.testing.assert_array_equal(gust.session.world.target,target)
        assert np.linalg.norm(gust.session.world.position-initial)<.2
        assert 'external_force_world_n' in gust.session.world.last_control
    finally: clean.close();gust.close()


def test_delayed_observation_is_sampled_earlier_than_current_truth(project):
    case=with_profile(joint_case(37),'pose');env=RobustJointEnv(project,case,JointDiagnostic())
    try:
        env.step(0);sensor=env.device.latest_observation()
        assert sensor.pose_age_s==pytest.approx(.1)
        assert sensor.sample_tick==env.session.tick-12
        assert sensor.position_m is not None
        assert env.agent.brain_tick==4*(1+(env.session.tick-env.start_tick)//12)
        assert env.session.recording.manifest['disturbance_evidence']['profile']=='pose'
    finally:env.close()


def test_training_distribution_balances_profiles_and_preserves_skill_contract():
    for skill in ('navigation','heading'):
        cases=[training_case(121100000+i*500,i,skill) for i in range(16)]
        for profile in PROFILES:assert sum(c['disturbance_profile']==profile for c in cases)==4
        assert all(c['goal'][2]==c['start'][2]==1. for c in cases)
        if skill=='heading':assert all(c['goal']==c['start'] for c in cases)
        for c in cases:
            assert max(abs(x) for x in c['goal'][:2])<3
            assert c['delay'] in (0,1) and c['noise'] in (0.,.002)
