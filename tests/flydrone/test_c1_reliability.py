"""Acceptance checks for isolated stochastic lanes and read-only C1 observation."""
import copy,json,math
from dataclasses import replace
import numpy as np
import torch
import pytest
from tests.flydrone.test_rigid_training import project,pool,equal_nested,complete
from flydrone.tellosim.training.env import TrainingEnv,curriculum_case
from flydrone.tellosim.training.contracts import action_mask
from flydrone.tellosim.training.observer import TrainingObserver
from flydrone.tellosim.training.parallel import collect_rollout
from flydrone.tellosim.training.checkpoint import save_training,load_training
from flydrone.tellosim.visual import VisualSession
from flydrone.tellosim.sdk.channels import CommandChannel,FaultProfile

def test_lane_action_rng_is_independent_of_other_lane_order_and_global_rng(project):
    p=pool(project);x=np.zeros(p.brain.feature_dim,np.float32);mask=np.ones(9,bool)
    state=p.lanes[1].action_rng.get_state();expected=[p.lanes[1].decision(x,mask)[0] for _ in range(32)]
    p.lanes[1].action_rng.set_state(state)
    actual=[]
    for k in range(32):
        for _ in range(k%5):p.lanes[0].decision(x,mask)
        torch.rand(7);p.lanes[2].reset();actual.append(p.lanes[1].decision(x,mask)[0])
    assert actual==expected

def test_c1_reset_yaw_mask_and_sensor_frame(project):
    p=pool(project,1);case=curriculum_case(21000001,curriculum='C1');case['initial_yaw_rad']=math.pi/2
    env=TrainingEnv(project,case,p.lanes[0],record=True,run_id='yaw-proof')
    assert env.session.world.yaw_rad==pytest.approx(math.pi/2,abs=.01)
    assert np.flatnonzero(env.mask).tolist()==[0,1,2,3,4,7,8]
    assert env.observation[6]==pytest.approx(1,abs=.001)
    with pytest.raises(ValueError,match='frame mismatch'):replace(env.sensors.latest,frame_id='wrong').vector(case['goal'],None,0,1)
    before=env.session.world.position;env.step(1)
    assert env.session.world.position[1]-before[1]>.15
    env.close()
    rows=[json.loads(line) for f in (project/'reports/vis/tellosim/yaw-proof').glob('trajectory-*.jsonl') for line in f.read_text().splitlines()]
    assert rows[0]['yaw_rad']==pytest.approx(math.pi/2)

def test_observer_does_not_change_policy_physics_or_training_snapshot(project):
    p=pool(project,1);case=curriculum_case(21000017,curriculum='C1')
    env=TrainingEnv(project,case,p.lanes[0]);opt=torch.optim.Adam(p.model.parameters());saved=project/'boundary.pt'
    save_training(saved,p,opt,[env],{},project)
    expected=[]
    for _ in range(3):
        action,_,_,policy=p.lanes[0].decision(env.features,env.mask);expected.append((action,env.step(action,policy),env.session.world.export_state()))
    observer=TrainingObserver(project,'probe',1,'test fixture, not trained-model evidence')
    q=pool(project,1);qopt=torch.optim.Adam(q.model.parameters());envs,_=load_training(saved,q,qopt,project);other=envs[0];other.observer=observer
    for action,result,world in expected:
        actual,_,_,policy=q.lanes[0].decision(other.features,other.mask);assert actual==action
        equal_nested(result,other.step(actual,policy));equal_nested(world,other.session.world.export_state())
    # Observation objects must not be pickled into exact training checkpoints.
    save_training(project/'observed.pt',q,qopt,[other],{},project)
    observer.close('completed');env.close();other.close()
    path=next((project/'reports/vis/tellosim').glob('probe-*/live.latest.json'));frame=json.loads(path.read_text())
    assert frame['env_id']==0 and frame['episode_id']==case['case_id'] and frame['finished']
    assert 'observer' not in torch.load(project/'observed.pt',weights_only=False)['envs'][0]['fields']

def test_observer_episode_identity_and_bounded_queue(project):
    observer=TrainingObserver(project,'four-lanes',4,'fixture');p=pool(project)
    envs=[TrainingEnv(project,curriculum_case(21000400+i,curriculum='C1'),p.lanes[i],observer=observer) for i in range(4)]
    for env in envs:env.step(0);env.close()
    prior_seq=observer.sequence[2];env=TrainingEnv(project,curriculum_case(21000500,curriculum='C1'),p.lanes[2],observer=observer)
    assert observer.sequence[2]>prior_seq and observer.latest[2]['episode_id']==env.case['case_id']
    assert observer.latest[1]['episode_id']!=env.case['case_id']
    assert all(isinstance(w._latest,(dict,type(None))) for w in observer.writers.values())
    env.close();observer.close('completed')

def test_policy_input_cannot_mutate_world_and_contains_no_truth_fields(project):
    p=pool(project,1);env=TrainingEnv(project,curriculum_case(21000300,curriculum='C1'),p.lanes[0]);before=env.session.world.position.copy()
    assert env.features.shape==(128,) and env.mask.shape==(9,) and env.features.dtype==np.float32
    x=env.features.copy();x[:]=0;p.lanes[0].decision(x,env.mask)
    np.testing.assert_array_equal(before,env.session.world.position)
    assert not hasattr(x,'world') and not hasattr(x,'reward');env.close()

def test_protective_stop_does_not_consume_late_old_reply(project):
    s=VisualSession(project,recording_enabled=False);complete(s,'command','sdk');complete(s,'takeoff','up')
    channel=CommandChannel(s,FaultProfile(reply_delay_ticks=120))
    first=channel.submit('forward 20','move');channel.advance(360)
    with pytest.raises(ValueError,match='busy'):channel.submit('back 20','too-early')
    channel.submit('stop','safe-stop');channel.advance(240)
    assert channel.poll('move')['client']=='cancelled_local'
    assert channel.poll('safe-stop')['client']=='ack_ok'
    channel.submit('back 20','new-move');channel.advance(1)
    assert channel.poll('new-move')['client']=='sent';s.close()
