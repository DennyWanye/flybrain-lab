from pathlib import Path
"""Focused C2 semantics, never substitute fixture graph results for training."""
import numpy as np
import pytest
from tests.flydrone.test_rigid_training import project,ROOT
from flydrone.tellosim.training.spatial_settled import SpatialEnv,SpatialPool,SpatialDiagnostic,spatial_case,save_spatial,load_spatial
from flydrone.tellosim.training.spatial_settled_campaign import label_for
from flydrone.tellosim.training.altitude_refined import with_profile

@pytest.mark.parametrize('height',[.45,1.55])
def test_continuous_full3d_rule_and_non1m_navigation(project,height):
    c=with_profile(spatial_case(248000000+round(height*100)),'combined')
    c.update(start=[0,0,1],goal=[.6,.4,height],initial_yaw_rad=0.,target_yaw_rad=1.57)
    e=SpatialEnv(project,c,SpatialDiagnostic())
    try:
        while not e.terminated:e.step(label_for(e))
        assert e.reason=='success' and e.hold>=2-1e-8
        assert set(x['to'] for x in e.phase_events)>={'navigation','heading'}
        assert all(x['physics_state_unchanged'] for x in e.phase_events)
        assert e.navigation_heights and abs(np.mean(e.navigation_heights)-1)>.3
        assert e.agent.brain_tick==4*(1+(e.session.tick-e.start_tick)//12)
        assert abs(e.session.world.velocity[2])<=.08
        assert e.disturbance_evidence()['absolute_vertical_impulse_ns']>0
    finally:e.close()

def test_handoff_preserves_physical_state_and_clock(project):
    e=SpatialEnv(project,with_profile(spatial_case(248000011),'clean'),SpatialDiagnostic())
    try:
        before=e.session.world.export_state();tick=e.agent.brain_tick
        e.switch_phase('navigation');e.switch_phase('heading');e.switch_phase('altitude')
        after=e.session.world.export_state()
        np.testing.assert_array_equal(before['integration'],after['integration'])
        np.testing.assert_array_equal(before['target'],after['target'])
        assert e.agent.brain_tick==tick
    finally:e.close()

def test_spatial_checkpoint_and_duplicate_sample(project):
    p=SpatialPool(ROOT,11,batch=2,graph=project/'graph.npz',device='cpu',initialize=False)
    path=project/'spatial.pt';save_spatial(path,p,None,ROOT,0);load_spatial(path,p,ROOT)
    v=np.zeros(26,np.float32);v[12:15]=1
    p.lanes[0].observe(v,1)
    with pytest.raises(ValueError,match='stale'):p.lanes[0].observe(v,1)
    tick=p.lanes[0].brain_tick;p.lanes[0].switch_skill('heading')
    assert p.lanes[0].brain_tick==tick
    with pytest.raises(ValueError):p.lanes[0].switch_skill('unknown')


def test_absolute_simulator_deadline_propagates_once(tmp_path):
    from flydrone.tellosim.training.spatial_settled import SpatialEnv, SpatialDiagnostic, spatial_case
    env=SpatialEnv(Path.cwd(),spatial_case(470000063),SpatialDiagnostic())
    try:
        # Exercise actual simulator closure at its tick boundary without a180s rollout.
        env.session.world.data.time=179.9
        env.session.world.tick=21588
        env.session.tick=21588
        env.start_tick=21588
        env.phase_start_tick=21588
        env.step(0)
        assert env.terminated and env.reason=='sandbox_time_limit_180s'
        assert env.session.finished
        env.close();env.close()
    finally:env.close()


def test_navigation_encoder_body_invariance_and_distance_signal():
    from flydrone.tellosim.training.spatial_settled import NavigationEncoder34
    import numpy as np
    x=np.zeros((2,26),np.float32);x[:,12:15]=1;x[:,0]=.17/6
    x[0,6:9]=[0,1,.5/3];x[1,6:9]=[1,0,1.5/3]
    y=NavigationEncoder34()(x)
    np.testing.assert_array_equal(y[0],y[1])
    x[1,0]=.19/6;y=NavigationEncoder34()(x)
    assert np.linalg.norm(y[0,4:6]-y[1,4:6])>.04
    assert y.shape==(2,34)

def test_closed_simulator_terminal_frame_is_flushed(project):
    from flydrone.tellosim.training.spatial_settled import SpatialEnv,SpatialDiagnostic,spatial_case
    import json
    env=SpatialEnv(project,spatial_case(470000062),SpatialDiagnostic(),record=True,run_id='c2s-clock-fixture')
    try:
        env.session.world.data.time=179.9;env.session.world.tick=21588;env.session.tick=21588
        env.start_tick=21588;env.phase_start_tick=21588
        env.step(0);env.close();env.close()
        m=json.loads((env.session.recording.directory/'manifest.json').read_text());chunks=m['streams']['transition']
        row=json.loads((env.session.recording.directory/chunks[-1]['file']).read_text().splitlines()[-1])
        assert row['finished'] and row['termination_reason']=='sandbox_time_limit_180s'
        assert m['task_result']=='sandbox_time_limit_180s'
    finally:env.close()


def test_heading_encoder_removes_position_and_height_nuisance():
    from flydrone.tellosim.training.spatial_settled import HeadingEncoder34
    x=np.zeros((2,26),np.float32);x[:,12:15]=1;x[:,6]=np.sin(.14);x[:,7]=np.cos(.14)
    x[0,:3]=[.01,-.02,.025];x[1,:3]=[-.03,.02,-.025];x[:,8]=[.5/3,1.6/3]
    original=x.copy();y=HeadingEncoder34()(x)
    np.testing.assert_array_equal(x,original);np.testing.assert_array_equal(y[0],y[1])
    x[1,3]=.08
    assert not np.array_equal(HeadingEncoder34()(x)[0],HeadingEncoder34()(x)[1])


def test_heading_encoder_continuous_signed_error_and_invalid_input():
    from flydrone.tellosim.training.spatial_settled import HeadingEncoder34
    e=HeadingEncoder34();x=np.zeros((3,26),np.float32);angles=np.deg2rad([8,21,-8]);x[:,6]=np.sin(angles);x[:,7]=np.cos(angles)
    y=e(x);assert y.shape==(3,34) and np.isfinite(y).all()
    assert np.linalg.norm(y[0,:6]-y[1,:6])>.1
    np.testing.assert_allclose(y[0,:6].reshape(3,2)[:,::-1],y[2,:6].reshape(3,2))
    with pytest.raises(ValueError):e(np.zeros(26))
    with pytest.raises(ValueError):e(np.full((1,26),np.nan))


def test_phase_estimator_filters_measurement_noise_without_mutating_observation(project):
    from types import SimpleNamespace
    from flydrone.tellosim.training.spatial_settled import SpatialEnv,SpatialDiagnostic,spatial_case
    e=SpatialEnv(project,spatial_case(470000060),SpatialDiagnostic())
    try:
        e._phase_window.clear();e.observation[12:15]=1;original=e.observation.copy()
        for i in range(20):
            sensor=SimpleNamespace(sample_id=900+i,position_m=(0,0,1.001+(-1)**i*.002),velocity_mps=(0,0,(-1)**i*.04))
            e.update_phase_estimate(sensor)
        pos,vel=e.phase_estimate();assert abs(pos[2]-1.001)<1e-10 and abs(vel[2])<1e-10
        np.testing.assert_array_equal(e.observation,original)
        before=list(e._phase_window);e.update_phase_estimate(sensor)
        np.testing.assert_array_equal(e._phase_window,before)
        e.observation[12]=0;e.update_phase_estimate(sensor);assert e.phase_estimate() is None and not e._phase_window
    finally:e.close()


def test_good_phase_estimate_cannot_pass_bad_physical_height(project):
    from flydrone.tellosim.training.spatial_settled import SpatialEnv,SpatialDiagnostic,spatial_case
    c=spatial_case(470000059);c.update(start=[0,0,1],goal=[0,0,1.4],initial_yaw_rad=0.,target_yaw_rad=0.)
    e=SpatialEnv(project,c,SpatialDiagnostic())
    try:
        e.switch_phase('heading');e.phase_estimate=lambda:(np.array(c['goal']),np.zeros(3))
        e.step(0)
        assert abs(e.session.world.position[2]-1.4)>.1
        assert e.hold==0 and e.reason!='success'
    finally:e.close()


def test_physical_acceptance_thresholds_remain_identical():
    from flydrone.tellosim.training.spatial_settled import SPATIAL_SPEC as new
    from flydrone.tellosim.training.spatial_orientation import SPATIAL_SPEC as old
    for key in ['task_deadline_s','absolute_simulator_deadline_s','phase_deadline_s','height_tolerance_m','heading_tolerance_rad','yaw_rate_tolerance_rad_s','success_hold_s','success_speed_mps','target_radius_m','allowed_actions','altitude_handoff_hold_s','navigation_handoff_hold_s']:
        assert new[key]==old[key]
