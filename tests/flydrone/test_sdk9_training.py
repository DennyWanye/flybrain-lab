from pathlib import Path
import json
import shutil
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from flydrone.tellosim.training.contracts import Encoder34,SensorObservation,action_mask,RealDeviceDisabled
from flydrone.tellosim.training.env import TrainingEnv
from flydrone.tellosim.training.runtime import ReservoirAgent,distribution,smdp_gae,save_checkpoint,load_checkpoint,update
from flydrone.policy import ActorCritic

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def project(tmp_path):
    (tmp_path/'configs').mkdir();shutil.copytree(ROOT/'configs/tellosim',tmp_path/'configs/tellosim')
    return tmp_path


def sensor(**kwargs):
    args=dict(sample_id=1,sample_tick=12,received_tick=12,frame_id='room_map',position_m=(1.,1.,1.),
              yaw_rad=np.pi/2,velocity_mps=(1.,0.,0.),height_m=1.,battery_fraction=.9,pose_age_s=0.,state_age_s=0.)
    args.update(kwargs);return SensorObservation(**args)


def test_observation_rotation_signed_encoding_and_validity():
    observation=sensor().vector((1,2,1),2,1.2,.8)
    assert len(observation)==26
    assert observation[0]==pytest.approx(1/6) and observation[1]==pytest.approx(0,abs=1e-7)
    assert observation[4]==pytest.approx(-1)
    assert observation[12:15].tolist()==[1,1,1] and observation[19]==1
    encoded=Encoder34()(observation[None])[0]
    assert encoded.shape==(34,) and np.all(encoded>=0)
    np.testing.assert_allclose(encoded[:16:2]-encoded[1:16:2],observation[:8])
    np.testing.assert_array_equal(encoded[16:],observation[8:])
    invalid=sensor(position_m=None,yaw_rad=None,velocity_mps=None,height_m=None,battery_fraction=None).vector((5,5,5),None,0,1)
    assert not invalid[:10].any() and not invalid[12:15].any()
    assert action_mask(invalid).tolist()==[True]+[False]*8
    with pytest.raises(ValueError,match='frame'):sensor(frame_id='different_map').vector((0,0,1),None,0,1)
    with pytest.raises(ValueError):Encoder34()(np.full((1,26),np.nan))


def test_smdp_discount_terminal_and_truncation():
    # Last terminal MUST ignore V(next)=999; preceding option lasts 3 base ticks.
    a,r=smdp_gae([1,2],[.5,.7],[.7,999],[.9**3,.9**2],[False,True],[False,False])
    assert a[1]==pytest.approx(1.3)
    assert a[0]==pytest.approx(1+.9**3*.7-.5+.9**3*.95*1.3)
    # An external truncation bootstraps but cannot propagate into another episode.
    a,_=smdp_gae([1,100],[.5,0],[2,0],[.8,.8],[False,True],[True,False])
    assert a[0]==pytest.approx(1+.8*2-.5)


class ClockAgent:
    def __init__(self):
        self.brain=SimpleNamespace(graph_sha256='fixture',mapping_sha256='fixture')
        self.reset()
    def reset(self):self.brain_tick=0;self.last=-1
    def observe(self,obs,sample_id):
        assert sample_id>self.last
        self.last=sample_id;self.brain_tick+=4
        return np.zeros(8,np.float32)


def test_shared_executor_clock_success_hold_and_rewards(project):
    case={'case_id':'clock','seed':7,'start':[0,0,1],'goal':[.6,0,1]}
    env=TrainingEnv(project,case,ClockAgent())
    try:
        assert env.session.world.scene['solid_walls']
        assert env.session.world.position[2]>.95  # actual takeoff completed
        while not env.terminated:
            relative=env.observation[:2]*6
            action=0 if np.linalg.norm(relative)<.17 else 1
            before=env.agent.brain_tick
            transition=env.step(action)
            assert env.agent.brain_tick-before==4*transition['k']
            assert transition['after_tick']-transition['before_tick']==12*transition['k']
            micro=np.asarray([sum(x.values()) for x in transition['reward_base_terms']])
            assert transition['reward']==pytest.approx(sum(micro*.995**np.arange(len(micro))))
        assert env.reason=='success' and env.hold>=2-1e-8
        assert transition['terminal']
        assert sum(t['success_bonus'] for t in transition['reward_base_terms'])==5
        with pytest.raises(RuntimeError):env.step(0)
    finally:env.close()


@pytest.mark.parametrize('fault,reason',[({'reply_loss':1.},'command_unknown'),({'dropout':1.},'localization_lost')])
def test_faults_terminate_without_inventing_success(project,fault,reason):
    env=TrainingEnv(project,{'seed':8,'start':[0,0,1],'goal':[1,0,1],**fault},ClockAgent())
    try:
        while not env.terminated:env.step(0 if not env.observation[12] else 1)
        assert env.reason==reason
        assert env.last_reward['reward_components']['failure_penalty']==-5
    finally:env.close()


def test_policy_distribution_honors_stored_mask():
    model=ActorCritic(4,9)
    masks=np.zeros((2,9),bool);masks[:,0]=True
    dist,_=distribution(model,torch.zeros(2,4),masks)
    assert dist.probs[:,0].tolist()==[1,1]
    assert not dist.probs[:,1:].any()
    with pytest.raises(ValueError):distribution(model,torch.zeros(2,4),np.zeros((2,9),bool))


def test_checkpoint_contract_and_readonly_brain(project):
    n=2048;rng=np.random.default_rng(1)
    matrix=sp.random(n,n,density=.005,random_state=rng,format='csr',dtype=np.float32)
    matrix.data[:]=.01
    path=project/'graph.npz';np.savez(path,n=n,indptr=matrix.indptr,indices=matrix.indices,data=matrix.data,ids=np.arange(n).astype(str))
    torch.set_num_threads(2)
    agent=ReservoirAgent(path,readout=16)
    x=agent.observe(np.ones(26,np.float32)*.5,1)
    before=agent.brain_tick;agent.snapshot();agent.brain.current_features();assert agent.brain_tick==before
    with pytest.raises(ValueError,match='duplicate'):agent.observe(np.zeros(26,np.float32),1)
    optimizer=torch.optim.Adam(agent.model.parameters(),lr=1e-3)
    checkpoint=project/'model.pt';save_checkpoint(checkpoint,agent,optimizer,project,0,0)
    expected=agent.decision(x,np.ones(9,bool),True)
    for p in agent.model.parameters():p.data.zero_()
    load_checkpoint(checkpoint,agent,project)
    assert agent.decision(x,np.ones(9,bool),True)==expected
    legacy=project/'legacy.pt';torch.save({'policy_state_dict':agent.model.state_dict()},legacy)
    with pytest.raises(ValueError,match='format'):load_checkpoint(legacy,agent,project)
    world=project/'configs/tellosim/world_room6.json';value=json.loads(world.read_text());value['room_size_m']=[8,8,3];world.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='mismatch'):load_checkpoint(checkpoint,agent,project)
    with pytest.raises(RuntimeError,match='Real flight'):RealDeviceDisabled()


def test_near_curriculum_is_seeded_and_mask_is_not_goal_dependent():
    from flydrone.tellosim.training.env import curriculum_case
    for seed in range(20):
        case=curriculum_case(seed,curriculum='C0-near-x')
        assert case==curriculum_case(seed,curriculum='C0-near-x')
        delta=np.asarray(case['goal'])-case['start']
        assert .3<=abs(delta[0])<=.65 and delta[1]==delta[2]==0
        obs=sensor(position_m=(0,0,1),yaw_rad=0.).vector(case['goal'],None,0,1)
        mask=action_mask(obs,'C0-near-x')
        assert mask.tolist()==[True,True,True]+[False]*6
        obs[:3]*=-1
        np.testing.assert_array_equal(mask,action_mask(obs,'C0-near-x'))


def test_balanced_profile_controls_and_checkpoint_boundaries(project):
    n=2048;rng=np.random.default_rng(19)
    matrix=sp.random(n,n,density=.006,random_state=rng,format='csr',dtype=np.float32)
    matrix.data[:]=.05
    path=project/'balanced.npz'
    np.savez(path,n=n,indptr=matrix.indptr,indices=matrix.indices,data=matrix.data,ids=np.arange(n).astype(str))
    torch.set_num_threads(2)
    agent=ReservoirAgent(path,profile='balanced_rate_v2')
    assert not set(agent.brain.input_indices)&set(agent.brain.readout_indices)
    for channel in range(34):
        neuron=agent.brain.readout_indices[channel]
        sources=agent.brain.input_indices[agent.brain.input_channels==channel]
        assert abs(matrix[neuron,sources]).sum()>0
    obs=sensor(position_m=(0,0,1),yaw_rad=0.,velocity_mps=(0,0,0)).vector((.15,0,1),None,0,1)
    forward=[];back=[]
    for i in range(20):forward.append(agent.observe(obs,i))
    agent.reset();obs[0]*=-1
    for i in range(20):back.append(agent.observe(obs,i))
    assert np.linalg.norm(np.asarray(forward)-back)>0
    features=back[-1];mask=action_mask(obs,'C0-near-x')
    action,logp,value,_=agent.decision(features,mask)
    optimizer=torch.optim.Adam(agent.model.parameters(),lr=1e-3)
    old=[x.detach().clone() for x in agent.model.parameters()]
    row={'features':features,'mask':mask,'action':action,'logp':logp,'value':value,
         'next_value':value,'reward':1.,'Gamma':.9,'terminal':False,'truncated':True}
    before=agent.brain.advance_count
    update(agent,optimizer,[row,row],epochs=1)
    assert agent.brain.advance_count==before
    assert any(not torch.equal(a,b) for a,b in zip(old,agent.model.parameters()))
    saved=project/'balanced.pt';save_checkpoint(saved,agent,optimizer,project,1,2)
    expected=agent.decision(features,mask,True)
    load_checkpoint(saved,agent,project)
    assert agent.decision(features,mask,True)==expected
    legacy=ReservoirAgent(path)
    with pytest.raises(ValueError,match='mismatch'):load_checkpoint(saved,legacy,project)
    zero=ReservoirAgent(path,profile='balanced_rate_v2',feature_source='zero_brain_control')
    assert not zero.observe(obs,1).any()
    with pytest.raises(ValueError,match='mismatch'):load_checkpoint(saved,zero,project)
    raw=ReservoirAgent(path,profile='balanced_rate_v2',feature_source='raw_observation_control')
    raw_features=raw.observe(obs,1)
    np.testing.assert_allclose(raw_features[:3],obs[:3]*[6,6,3])
    assert not raw_features[26:].any()
    assert raw.decision(raw_features,mask)[3]['input_source']=='raw_observation_control'


def test_stop_dwell_changes_decision_interval_without_changing_hold(project):
    agent=ClockAgent();agent.stop_dwell_s=2.
    env=TrainingEnv(project,{'case_id':'dwell','seed':13,'start':[0,0,1],'goal':[.6,0,1]},agent)
    try:
        result=env.step(0)
        assert not result['terminal'] and result['k']==20
        assert result['Gamma']==pytest.approx(.995**20)
        assert env.session.world.scene['stable_hold_required_s']==2.
        assert env.session.world.scene['target_radius_m']==.2
        assert result['brain_after']-result['brain_before']==80
    finally:env.close()


def test_supervised_readout_learns_without_advancing_brain():
    from flydrone.tellosim.training.bootstrap import fit
    torch.manual_seed(31);torch.set_num_threads(2)
    agent=SimpleNamespace(model=ActorCritic(3,9),brain=SimpleNamespace(advance_count=19))
    rng=np.random.default_rng(22);centers=np.eye(3,dtype=np.float32)*2
    mask=np.array([True,True,True]+[False]*6)
    rows=[(centers[i]+rng.normal(0,.05,3).astype(np.float32),mask.copy(),i) for i in range(3) for _ in range(16)]
    optimizer=torch.optim.Adam(agent.model.parameters(),lr=.005)
    metrics=fit(agent,optimizer,rows,epochs=30)
    assert metrics['training_label_accuracy']>.95 and agent.brain.advance_count==19
    heldout=np.stack([centers[i]+rng.normal(0,.05,3).astype(np.float32) for i in range(3)])
    with torch.no_grad():policy,_=distribution(agent.model,torch.tensor(heldout),np.stack([mask]*3))
    assert policy.probs.argmax(1).tolist()==[0,1,2]
    assert not policy.probs[:,3:].any()


def test_critic_prefit_preserves_actor_and_uses_option_discount():
    from flydrone.tellosim.training.critic_warmup import fit_value_head,discounted_returns,actor_hash
    np.testing.assert_allclose(discounted_returns([1,2,3],[.5,.25,.9]),[2.375,2.75,3])
    torch.manual_seed(31)
    agent=SimpleNamespace(model=ActorCritic(4,9),brain=SimpleNamespace(advance_count=17),value_trained=False)
    x=np.random.default_rng(31).normal(size=(64,4)).astype(np.float32)
    with torch.no_grad():
        probabilities=agent.model(torch.tensor(x))[0].probs.clone()
        targets=(agent.model.body(torch.tensor(x))[:,0]*.5+1).numpy()
    before=actor_hash(agent.model)
    metrics=fit_value_head(agent,x,targets,epochs=50)
    assert metrics['training_mse_after']<metrics['training_mse_before']
    assert before==actor_hash(agent.model) and agent.value_trained
    assert torch.equal(probabilities,agent.model(torch.tensor(x))[0].probs)


def test_frozen_body_ppo_updates_heads_without_changing_body():
    torch.manual_seed(19)
    agent=SimpleNamespace(model=ActorCritic(4,9),value_trained=False)
    for p in agent.model.body.parameters():p.requires_grad_(False)
    original={k:v.clone() for k,v in agent.model.state_dict().items()}
    features=torch.randn(8,4);masks=np.ones((8,9),bool)
    with torch.no_grad():policy,values=distribution(agent.model,features,masks)
    actions=policy.sample();logps=policy.log_prob(actions)
    rows=[dict(features=features[i].numpy(),mask=masks[i],action=int(actions[i]),logp=float(logps[i]),
               value=float(values[i]),next_value=0.,reward=float(i%3-1),Gamma=.9,terminal=True,truncated=False)
          for i in range(8)]
    optimizer=torch.optim.Adam([p for p in agent.model.parameters() if p.requires_grad],lr=1e-4)
    update(agent,optimizer,rows)
    now=agent.model.state_dict()
    assert all(torch.equal(v,now[k]) for k,v in original.items() if k.startswith('body.'))
    assert any(not torch.equal(v,now[k]) for k,v in original.items() if k.startswith('actor.'))
    assert any(not torch.equal(v,now[k]) for k,v in original.items() if k.startswith('critic.'))


def test_boundary_repair_cases_are_separate_and_teacher_uses_measurement():
    from flydrone.tellosim.training.boundary_repair import boundary_case,validation_cases
    from flydrone.tellosim.training.bootstrap import teacher
    validation=validation_cases()
    assert len(validation)==32 and len({c['seed'] for c in validation})==32
    recovery=validation_cases(True)
    assert len(recovery)==40 and len({c['seed'] for c in recovery})==40
    assert all(c['cohort']=='recovery_holdout' for c in recovery[-8:])
    for seed in range(1223000,1223032):
        case=boundary_case(seed)
        assert .205<=abs(case['goal'][0]-case['start'][0])<=.265
        assert case==boundary_case(seed) and case['seed'] not in {c['seed'] for c in validation}
    env=SimpleNamespace(steps=3,observation=np.zeros(26),mask=np.array([True]*3+[False]*6))
    env.observation[0]=.217/6
    assert teacher(env)==1
    env.observation[0]=-.217/6
    assert teacher(env)==2
    env.observation[0]=.1/6
    assert teacher(env)==0


def test_boundary_repair_gate_detects_swapped_successes():
    from flydrone.tellosim.training.boundary_repair import compare
    left={'all_modes':[dict(case_id='a',cohort='previous16',mode='argmax',success=True),
                       dict(case_id='b',cohort='previous16',mode='argmax',success=False)]}
    right={'all_modes':[dict(case_id='a',cohort='previous16',mode='argmax',success=False),
                       dict(case_id='b',cohort='previous16',mode='argmax',success=True)]}
    row=compare(left,right)[0]
    assert row['before']==row['after']==1 and row['regressed_cases']==['a']


def test_recovery_exploration_keeps_expert_labels(monkeypatch):
    from flydrone.tellosim.training import boundary_repair as repair
    executed=[]
    class StubEnv:
        def __init__(self,*args,**kwargs):
            self.steps=0;self.terminated=False;self.reason=None
            self.features=np.zeros(128,np.float32);self.mask=np.array([True]*3+[False]*6)
            self.observation=np.zeros(26,np.float32);self.observation[0]=.24/6
        def step(self,action,policy):
            executed.append(action);self.steps+=1
            if self.steps==4:self.terminated=True;self.reason='fixture_end'
        def close(self):pass
    monkeypatch.setattr(repair,'TrainingEnv',StubEnv)
    rows=[];metadata=[]
    repair.collect(None,None,[dict(seed=123,cohort='boundary')],rows,metadata,True,True)
    assert executed==[0,0,1,1]
    assert [r[2] for r in rows]==[0,1,1,1]  # Do not label forced pauses as expert STOP.


@pytest.mark.parametrize('profile,executed', [({'request_drop':1.},False),({'reply_drop':1.},True)])
def test_training_channel_preserves_clock_and_observer_truth(project,profile,executed):
    env=TrainingEnv(project,{'seed':9,'start':[0,0,1],'goal':[1,0,1],'channel_profile':profile},ClockAgent())
    try:
        result=env.step(0)
        assert env.reason=='command_unknown' and result['terminal']
        assert result['brain_after']-result['brain_before']==4*result['k']
        observer=env.device.observed_operation()
        assert observer['client']=='unknown_execution' and observer['response'] is None
        assert (observer['device']=='completed') is executed
        assert 'device' not in env.device.poll('policy-0')
        assert ('channel-policy-0' in env.session.requests) is executed
    finally:env.close()
