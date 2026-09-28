"""V1R focused checks; fixture graph is never model-readiness evidence."""
import numpy as np
import pytest
from tests.flydrone.test_rigid_training import project,pool,ROOT
from flydrone.tellosim.training.altitude_refined import AltitudeEncoder34,AltitudeEnv,altitude_case,with_profile,teacher,save_altitude,load_altitude
from flydrone.tellosim.training.altitude import load_altitude as load_v1
from flydrone.tellosim.training.heading_campaign import RuleDiagnostic

def test_smooth_multiscale_encoding_and_invalid_input():
    x=np.zeros((5,26),np.float32);x[:,2]=np.array([-.11,-.09,0,.09,.11])/3
    z=AltitudeEncoder34()(x)
    assert z.shape==(5,34) and np.isfinite(z).all()
    assert np.all(z[4,[0,2,4]]>z[3,[0,2,4]])
    np.testing.assert_allclose(z[0,:6].reshape(3,2)[:,::-1],z[4,:6].reshape(3,2))
    with pytest.raises(ValueError):AltitudeEncoder34()(np.full((1,26),np.nan))

@pytest.mark.parametrize('height',[.291,.509,1.491,1.709])
def test_new_development_grid_physics_hold(project,height):
    c=with_profile(altitude_case(218000000+round(height*1000)),'combined');c['goal'][2]=height
    env=AltitudeEnv(project,c,RuleDiagnostic())
    try:
        while not env.terminated:env.step(teacher(env.observation,env.mask,env.steps==0))
        assert env.reason=='success' and env.hold>=2-1e-8
        assert abs(env.session.world.position[2]-height)<=.1
        assert env.stop_dwell_s==2.
        assert env.disturbance_evidence()['absolute_vertical_impulse_ns']>0
    finally:env.close()

def test_checkpoint_encoder_and_version_fail_closed(project):
    p=pool(project,1);p.brain.encoder=AltitudeEncoder34();path=project/'v1r.pt'
    save_altitude(path,p,None,ROOT,0);load_altitude(path,p,ROOT)
    with pytest.raises(ValueError,match='contract'):load_v1(path,p,ROOT)
    p.brain.encoder=object()
    with pytest.raises(ValueError,match='encoder'):load_altitude(path,p,ROOT)
