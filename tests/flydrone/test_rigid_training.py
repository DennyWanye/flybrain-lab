"""Physical invariants and interrupted-vs-continuous training evidence."""
from pathlib import Path
import copy, math, shutil
import numpy as np
import pytest
import scipy.sparse as sp
import torch
from flydrone.tellosim.visual import VisualSession, VisualWorld, scene_config
from flydrone.tellosim.physics.rigid import PROFILE, rotation, translation_reference
from flydrone.tellosim.sdk.state_codec import serialize_state,parse_state
from flydrone.tellosim.training.parallel import ReservoirPool,collect_rollout
from flydrone.tellosim.training.env import TrainingEnv,curriculum_case
from flydrone.tellosim.training.checkpoint import save_training,load_training
from flydrone.tellosim.training.runtime import update
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture
def project(tmp_path):
    shutil.copytree(ROOT/'configs',tmp_path/'configs')
    n=2048;rng=np.random.default_rng(19)
    matrix=sp.random(n,n,density=.006,random_state=rng,format='csr',dtype=np.float32);matrix.data[:]=.05
    np.savez(tmp_path/'graph.npz',n=n,indptr=matrix.indptr,indices=matrix.indices,data=matrix.data,ids=np.arange(n).astype(str))
    torch.set_num_threads(2)
    return tmp_path

def complete(session,wire,key):
    session.command(wire,key)
    rows=[]
    for _ in range(6000):
        if session.operation['client']!='sent':break
        session.advance(1)
        rows.append((session.world.position,session.world.data.qpos[3:7].copy(),session.world.last_control.copy()))
    assert session.operation['client']=='ack_ok',session.operation
    return rows

def test_reference_has_bounded_velocity_acceleration_and_exact_endpoint():
    for distance in (.02,2.):
        samples=[translation_reference([0,0,0],[distance,0,0],.2,.3,t) for t in np.arange(0,14,.001)]
        pos,vel,acc=map(np.asarray,zip(*samples))
        assert np.max(np.linalg.norm(vel,axis=1))<=.2+1e-12
        assert np.max(np.linalg.norm(np.diff(vel,axis=0)/.001,axis=1))<=.3+1e-9
        assert np.all(np.diff(pos[:,0])>=-1e-12)
        np.testing.assert_array_equal(pos[-1],[distance,0,0]);assert not vel[-1].any()

def test_rigid_body_route_rotates_thrust_and_moves_in_new_heading(project):
    s=VisualSession(project,recording_enabled=False)
    complete(s,'command','sdk');complete(s,'takeoff','up')
    rows=complete(s,'forward 40','f1');before=s.world.position.copy()
    assert before[0]==pytest.approx(.4,abs=.035)
    assert max(abs(rotation(q)[2,0]) for _,q,_ in rows)>.01
    for _,q,c in rows:
        assert 0<=c['thrust_n']<=1.962+1e-9
        assert max(abs(x) for x in c['torque_body_nm'])<=.01
    complete(s,'cw 90','yaw');complete(s,'forward 40','f2')
    assert s.world.position[1]==pytest.approx(-.4,abs=.05)
    assert abs(s.world.position[0]-before[0])<.035
    yaw=s.world.yaw_rad;complete(s,'ccw 360','turn360')
    assert s.world.yaw_rad-yaw==pytest.approx(2*math.pi,abs=math.radians(2))
    assert np.linalg.norm(s.world.data.qvel[3:6])<math.radians(5)
    packet=parse_state(serialize_state(s));assert packet['fields']['pitch'] is not None
    complete(s,'stop','stop');complete(s,'land','land')
    assert not s.airborne and not s.collision_latched
    assert s.world.position[2]==pytest.approx(.045,abs=.003)

def test_wrench_follows_actual_body_axis_not_desired_horizontal_force(project):
    w=VisualWorld(scene_config(project));w.reset((0,0,1));w.powered=True;w.set_target((1,0,1))
    tilted=False
    for _ in range(180):
        R=rotation(w.data.qpos[3:7]);w.data.xfrc_applied[:]=0;w._apply_controller()
        force=w.data.xfrc_applied[w.body_id,:3].copy()
        np.testing.assert_allclose(np.cross(force,R[:,2]),0,atol=1e-12)
        tilted |= abs(R[2,0])>.01
        # Advance integration only here; avoid calling the controller twice.
        import mujoco
        mujoco.mj_step(w.model,w.data)
    assert tilted and w.position[0]>.05

def test_mid_motion_state_restore_is_bit_exact_and_keeps_request_aliases(project):
    s=VisualSession(project,recording_enabled=False)
    complete(s,'command','sdk');complete(s,'takeoff','up');s.command('forward 100','moving');s.advance(173)
    saved=s.export_state();s.advance(221);expected=s.export_state()
    other=VisualSession(project,recording_enabled=False);other.restore_state(saved);other.advance(221)
    np.testing.assert_array_equal(expected['world']['integration'],other.export_state()['world']['integration'])
    assert other.operation is other.requests['moving'] and other.operation is other.history[-1]
    assert other.world.trajectory==s.world.trajectory
    assert other.rng.bit_generator.state==s.rng.bit_generator.state

def pool(project,batch=4):
    return ReservoirPool(project/'graph.npz',seed=71,profile='balanced_rate_v3',batch=batch,physics_profile=PROFILE)

def test_masked_batch_keeps_inactive_lanes_unchanged(project):
    p=pool(project);v=p.brain.v.clone();tr=p.brain.trace.clone()
    x=np.ones(26,np.float32)*.1;p.observe_lanes({1:(x,1),3:(-x,1)})
    assert torch.equal(v[:,[0,2]],p.brain.v[:,[0,2]]) and torch.equal(tr[:,[0,2]],p.brain.trace[:,[0,2]])
    assert [a.brain_tick for a in p.lanes]==[0,4,0,4]
    before=p.brain.v[:,3].clone();p.lanes[1].reset();assert torch.equal(before,p.brain.v[:,3])

def equal_nested(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_nested(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_nested(x,y)
    else:assert a==b

def test_ppo_boundary_resume_matches_continuation_on_four_lanes(project):
    p=pool(project);opt=torch.optim.Adam(p.model.parameters(),lr=1e-5);envs=[None]*4
    def make(i):return TrainingEnv(project,curriculum_case(6100000+i,noise=.005,delay=1,channel_profile={'request_delay_ticks':3,'reply_delay_ticks':3}),p.lanes[i])
    rows,_,_=collect_rollout(p,envs,[2]*4,make);update(p,opt,rows,epochs=1)
    assert all(rows[i]['truncated'] for i in (1,3,5,7))
    saved=project/'resume.pt';save_training(saved,p,opt,envs,{'options':8},project)
    expected_rows,_,_=collect_rollout(p,envs,[2]*4,make);update(p,opt,expected_rows,epochs=1)
    expected_policy=copy.deepcopy(p.model.state_dict());expected_brain=copy.deepcopy(p.brain.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
    q=pool(project);qopt=torch.optim.Adam(q.model.parameters(),lr=1e-5)
    restored,counters=load_training(saved,q,qopt,project)
    def make_q(i):return TrainingEnv(project,curriculum_case(6100000+i,noise=.005,delay=1,channel_profile={'request_delay_ticks':3,'reply_delay_ticks':3}),q.lanes[i])
    actual_rows,_,_=collect_rollout(q,restored,[2]*4,make_q);equal_nested(expected_rows,actual_rows)
    update(q,qopt,actual_rows,epochs=1)
    equal_nested(expected_policy,q.model.state_dict());equal_nested(expected_brain,q.brain.state_dict());equal_nested(expected_opt,qopt.state_dict())
    assert counters=={'options':8}
    for env in envs+restored:
        if env:env.close()
    wrong=pool(project,1);wrongopt=torch.optim.Adam(wrong.model.parameters(),lr=1e-5)
    with pytest.raises(ValueError,match='contract changed'):load_training(saved,wrong,wrongopt,project)


def test_delayed_inflight_channel_and_sensor_snapshot(project):
    p=pool(project,1);case=curriculum_case(6111111,noise=.01,delay=2,channel_profile={'request_delay_ticks':30,'reply_delay_ticks':24})
    env=TrainingEnv(project,case,p.lanes[0]);env.device.channel.submit('forward 20','inflight');env.device.advance(12)
    saved=env.export_state();assert env.device.channel.operations['inflight']['device_id'] is None
    for _ in range(80):env.device.advance(12);env.sensors.sample(env.session)
    q=pool(project,1);other=TrainingEnv(project,case,q.lanes[0]);other.restore_state(saved)
    for _ in range(80):other.device.advance(12);other.sensors.sample(other.session)
    equal_nested(env.session.world.export_state(),other.session.world.export_state())
    equal_nested(env.device.channel.operations,other.device.channel.operations)
    equal_nested(env.sensors.latest,other.sensors.latest)
    env.close();other.close()


def test_batched_evaluation_matches_single_lane_with_mixed_terminations(project):
    from flydrone.tellosim.training.runtime import save_checkpoint
    from flydrone.tellosim.training.campaign_v2 import evaluate_batch
    p=pool(project)
    with torch.no_grad():
        p.model.actor.weight.zero_();p.model.actor.bias.zero_();p.model.actor.bias[0]=10
    saved=project/'eval.pt';save_checkpoint(saved,p,None,project,0,0)
    cases=[dict(case_id=f'eval-{i}',seed=5100200+i,start=[0,0,1],goal=[.05,0,1],**extra)
        for i,extra in enumerate(({}, {'dropout':1.},{'noise':.005},{'reply_loss':1.}))]
    one=evaluate_batch(project,project/'graph.npz',saved,cases,project/'one.json','one',record_indices=(),batch=1,device='cpu')
    four=evaluate_batch(project,project/'graph.npz',saved,cases,project/'four.json','four',record_indices=(),batch=4,device='cpu')
    equal_nested(one['results'],four['results'])
    assert {r['reason'] for r in four['results']}=={'success','localization_lost','command_unknown'}


def test_accumulation_backend_preserves_graph_and_is_versioned(project):
    from flydrone.tellosim.training.runtime import ReservoirAgent,save_checkpoint,load_checkpoint
    legacy=ReservoirAgent(project/'graph.npz',profile='balanced_rate_v3',physics_profile=PROFILE)
    accurate=ReservoirAgent(project/'graph.npz',profile='balanced_rate_v3',physics_profile=PROFILE,neural_backend='csr_fp64_accum')
    assert accurate.brain.w.dtype==torch.float64
    assert torch.equal(accurate.brain.w.values(),legacy.brain.w.values().double())
    assert accurate.brain.graph_sha256==legacy.brain.graph_sha256 and accurate.brain.mapping_sha256==legacy.brain.mapping_sha256
    save_checkpoint(project/'fp64.pt',accurate,None,project,0,0)
    with pytest.raises(ValueError,match='mismatch'):load_checkpoint(project/'fp64.pt',legacy,project)
    load_checkpoint(project/'fp64.pt',accurate,project)
