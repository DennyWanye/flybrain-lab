"""Focused C2 semantics, never substitute fixture graph results for training."""
import numpy as np
import pytest
from tests.flydrone.test_rigid_training import project,ROOT
from flydrone.tellosim.training.spatial import SpatialEnv,SpatialPool,SpatialDiagnostic,spatial_case,save_spatial,load_spatial
from flydrone.tellosim.training.spatial_campaign import label_for
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
