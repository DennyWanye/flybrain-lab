"""Continuous navigation -> heading task using frozen learned skills.

The measured-pose phase manager selects a readout, never a movement direction.
This module does not claim end-to-end joint-policy training.
"""
from pathlib import Path
from dataclasses import asdict
import copy,hashlib,json,math
import numpy as np
import torch
from .env import TrainingEnv,sample_case
from .heading import HeadingEnv,HEADING_SPEC,wrap_angle,load_heading
from .parallel import ReservoirPool,AgentLane
from .runtime import load_checkpoint,distribution
from .contracts import SPEC,ACTIONS,action_mask,digest
from ..sdk.codec import TelloCommand
from ..visual import atomic_json

JOINT_SPEC={**SPEC,'observation':'tellosim.joint_observation26/1.0','curriculum':'J1',
    'task':'navigate_then_heading_and_hold','phase_deadline_s':60,'task_deadline_s':120,
    'heading_tolerance_rad':math.radians(16),'yaw_rate_tolerance_rad_s':.08,
    'navigation_handoff_hold_s':2.,'recovery_distance_m':.2,'recovery_hold_s':.5,
    'phase_observations':{'navigation':SPEC['observation'],'heading':HEADING_SPEC['observation']},
    'allowed_actions':{'navigation':[0,1,2,3,4,7,8],'heading':[0,7,8]},
    'phase_manager':'measured pose only; switch at completed SDK option; no direction rule',
    'heading_anchor':'measured arrival XY and instructed altitude; final success and recovery always use original global position target',
    'neural_handoff':'reset lane neural state; preserve physical state and neural tick counter; one 10Hz physical/sensor warmup tick',
    'trainable':'none; frozen previously trained C1 and H1 readouts',
    'SINGLE_POLICY_JOINT_TRAINED':False}
METHOD='Composition of frozen learned C1 navigation and H1 heading skills; measured-pose phase manager; no new training'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def joint_case(seed,case_id=None):
    case=sample_case(seed,case_id or f'joint-{seed}');rng=np.random.default_rng(seed+1701)
    yaw=float(rng.uniform(-math.pi,math.pi));delta=float(rng.choice([-1,1])*rng.uniform(math.radians(35),math.pi))
    return {**case,'curriculum':'J1','initial_yaw_rad':yaw,'target_yaw_rad':wrap_angle(yaw+delta)}

def bundle_spec(root,seed):
    root=Path(root);files={'navigation':f'runs/tellosim-sdk9/c1-s{seed}/ppo/checkpoint.pt','heading':f'runs/tellosim-sdk9/heading-s{seed}/checkpoint.pt'}
    return {'format':'tellosim.joint_skill_bundle/1','seed':seed,'spec':JOINT_SPEC,'method':METHOD,
        'sources':{skill:{'path':path,'sha256':sha(root/path)} for skill,path in files.items()},
        'source_hashes':{name:sha(root/name) for name in ('flydrone/tellosim/training/joint.py','flydrone/tellosim/training/joint_campaign.py')},
        'new_training_actions':0,'SINGLE_POLICY_JOINT_TRAINED':False}

class JointLane(AgentLane):
    def __init__(self,pool,index):super().__init__(pool,index);self.skill='navigation'
    def reset(self):super().reset();self.skill='navigation'
    def switch_skill(self,skill):
        if skill not in ('navigation','heading'):raise ValueError('unknown skill')
        self.skill=skill;mask=np.zeros(self.brain.batch,bool);mask[self.index]=True;self.brain.reset(mask)
        # Preserve sample_id and brain_tick: the next observation must be a NEW
        # physical sample. A phase change does not teleport or advance the brain.
    @torch.no_grad()
    def decision(self,features,mask,deterministic=False):
        policy,value=distribution(self.pool.models[self.skill],torch.as_tensor(features[None],dtype=torch.float32),[mask])
        action=policy.probs.argmax(-1) if deterministic else torch.multinomial(policy.probs,1,generator=self.action_rng).squeeze(-1)
        return int(action.item()),float(policy.log_prob(action).item()),float(value.item()),{
            'selected_action':int(action.item()),'probabilities':policy.probs[0].tolist(),
            'value_estimate':float(value.item()),'value_trained':self.pool.value_flags[self.skill],
            'mask':np.asarray(mask,bool).tolist(),'input_source':'reservoir_v_trace','feature_dim':self.brain.feature_dim,
            'skill':self.skill,'checkpoint_sha256':self.pool.bundle['sources'][self.skill]['sha256']}

class JointPool(ReservoirPool):
    def __init__(self,root,bundle,device='cuda',batch=64):
        root=Path(root);self.bundle=json.loads(Path(bundle).read_text()) if isinstance(bundle,(str,Path)) else bundle
        if self.bundle!=bundle_spec(root,self.bundle['seed']):raise ValueError('joint bundle or source contract mismatch')
        super().__init__(root/'data/male-v1.npz',device,seed=self.bundle['seed'],profile='balanced_rate_v3',batch=batch,physics_profile='rigid_body_thrust_v2',neural_backend='csr_fp64_accum')
        load_checkpoint(root/self.bundle['sources']['navigation']['path'],self,root);nav=copy.deepcopy(self.model);nav_value=self.value_trained
        load_heading(root/self.bundle['sources']['heading']['path'],self,root);heading=copy.deepcopy(self.model)
        self.models={'navigation':nav.eval(),'heading':heading.eval()};self.value_flags={'navigation':nav_value,'heading':False}
        for model in self.models.values():
            for param in model.parameters():param.requires_grad_(False)
        self.lanes=[JointLane(self,i) for i in range(batch)];self.training_method=METHOD;self.checkpoint_sha256=digest(self.bundle)

class JointEnv(HeadingEnv):
    def __init__(self,root,case,agent,**kwargs):
        if case.get('curriculum')!='J1':raise ValueError('joint task requires J1 cases')
        self.target_yaw=float(case['target_yaw_rad'])
        if not math.isfinite(self.target_yaw):raise ValueError('invalid target yaw')
        self.heading_anchor=None;self.phase='navigation';self.phase_start_tick=None;self.navigation_hold=0.;self.drift_hold=0.;self.phase_events=[]
        TrainingEnv.__init__(self,root,case,agent,**kwargs)
        self.phase_start_tick=self.start_tick;self.session.recording.save()
    def potential(self):
        position=min(float(np.linalg.norm(self.session.world.position-np.asarray(self.case['goal']))),6.)/6
        heading=abs(wrap_angle(self.target_yaw-self.session.world.yaw_rad))/math.pi
        return -position-.25*heading
    def prepare_observation(self):
        sensor=self.device.latest_observation();start=self.phase_start_tick if self.phase_start_tick is not None else self.start_tick
        self.observation=sensor.vector(self.heading_anchor if self.phase=='heading' else self.case['goal'],self.previous_action,self.previous_duration,1-(self.session.tick-start)/120/60)
        if self.phase=='navigation':self.mask=action_mask(self.observation,'C1')
        else:
            if self.observation[12]:
                error=wrap_angle(self.target_yaw-sensor.yaw_rad);self.observation[6:8]=[math.sin(error),math.cos(error)]
            self.mask=np.zeros(9,bool);self.mask[0]=True
            if self.observation[12] and self.observation[14]:self.mask[7:9]=True
    def measured_position_state(self):
        s=self.device.latest_observation()
        if not self.observation[12] or not self.observation[13] or not self.observation[14]:return False,False
        pos=np.asarray(s.position_m);goal=np.asarray(self.case['goal']);distance=np.linalg.norm(pos[:2]-goal[:2])
        stable=distance<=.2 and np.linalg.norm(np.asarray(s.velocity_mps)[:2])<=.08 and abs(pos[2]-goal[2])<=.1
        drift=distance>JOINT_SPEC['recovery_distance_m'] or abs(pos[2]-goal[2])>.15
        return bool(stable),bool(drift)
    def switch_phase(self,phase):
        before=self.session.world.export_state();old=self.phase;clock=self.agent.brain_tick
        if phase=='heading':
            pose=self.device.latest_observation().position_m
            if pose is None:raise ValueError('cannot hand off without measured pose')
            self.heading_anchor=[pose[0],pose[1],self.case['goal'][2]]
        self.agent.switch_skill(phase);self.phase=phase;self.phase_start_tick=self.session.tick
        self.navigation_hold=0.;self.drift_hold=0.;self.hold=0.;self.previous_action=None;self.previous_duration=0.;self.last_policy=None
        after=self.session.world.export_state()
        # Physical state belongs to the simulator; only neural memory is reset.
        np.testing.assert_array_equal(before['integration'],after['integration'])
        np.testing.assert_array_equal(before['target'],after['target']);assert before['target_yaw']==after['target_yaw']
        assert self.agent.brain_tick==clock
        self.phase_events.append({'tick':self.session.tick,'from':old,'to':phase,'reason':'position_settled' if phase=='heading' else 'position_drift',
            'source':'external_pose_mock','physics_state_unchanged':True,'neural_tick_preserved':clock})
    def step_iter(self,action,policy=None):
        if self.terminated:raise RuntimeError('step after terminal')
        if type(action) is not int or not 0<=action<9 or not self.mask[action]:raise ValueError('invalid or masked action')
        self.last_policy=policy
        if policy:self.publish()
        before=self.session.tick;brain_before=self.agent.brain_tick
        op=self.device.submit(TelloCommand(*ACTIONS[action]),f'policy-{self.steps}')
        terms=[];discounted=0.;raw=0.;discount=1.;warming=False
        while True:
            self.device.advance(12);self.sensors.sample(self.session)
            if not warming:self.previous_action=action;self.previous_duration=(self.session.tick-before)/120
            self.prepare_observation();yield self
            status=self.device.poll(op['operation_id']);elapsed=(self.session.tick-self.start_tick)/120
            phase_elapsed=(self.session.tick-(self.phase_start_tick or self.start_tick))/120
            position_ok,drift=self.measured_position_state()
            ack=status['client']=='ack_ok' or (action==0 and status['client']=='sent')
            self.navigation_hold=self.navigation_hold+.1 if self.phase=='navigation' and position_ok and ack else 0.
            self.drift_hold=self.drift_hold+.1 if self.phase=='heading' and drift else 0.
            pos=self.session.world.position;goal=np.asarray(self.case['goal'])
            stable=(self.phase=='heading' and np.linalg.norm(pos[:2]-goal[:2])<=.2 and np.linalg.norm(self.session.world.velocity[:2])<=.08
                and abs(pos[2]-goal[2])<=.1 and abs(wrap_angle(self.target_yaw-self.session.world.yaw_rad))<=JOINT_SPEC['heading_tolerance_rad']
                and abs(float(self.session.world.data.qvel[5]))<=.08 and ack)
            self.hold=self.hold+.1 if stable else 0.;self.invalid_age=self.invalid_age+.1 if not self.observation[12] else 0.
            self.reason=('collision' if self.session.collision_latched else 'command_unknown' if status['client']=='unknown_execution' else
                'command_error' if status['client']=='ack_error' else 'localization_lost' if self.invalid_age>=.5-1e-8 else
                'success' if self.hold>=2-1e-8 else 'task_deadline' if elapsed>=120-1e-8 else 'phase_deadline' if phase_elapsed>=60-1e-8 else None)
            self.terminated=self.reason is not None
            phi=0. if self.terminated else self.potential();components={'potential':SPEC['gamma_base']*phi-self.previous_phi,'step_cost':-.001,
                'success_bonus':5. if self.reason=='success' else 0.,'failure_penalty':-5. if self.terminated and self.reason!='success' else 0.}
            self.previous_phi=phi;reward=sum(components.values());raw+=reward;discounted+=discount*reward;discount*=SPEC['gamma_base'];terms.append(components)
            done=status['client']!='sent' and (action!=0 or self.previous_duration>=self.stop_dwell_s-1e-8)
            if self.terminated or warming:break
            if done:
                switch='heading' if self.phase=='navigation' and self.navigation_hold>=2-1e-8 else 'navigation' if self.phase=='heading' and self.drift_hold>=.5-1e-8 else None
                if switch:self.switch_phase(switch);warming=True;continue
                break
        self.steps+=1;self.raw_return+=raw;k=(self.session.tick-before)//12
        if self.agent.brain_tick-brain_before!=4*k:raise AssertionError('joint neural/physics clock mismatch')
        self.last_reward={'reward_total':raw,'reward_components':{key:sum(t[key] for t in terms) for key in terms[0]},'discounted_option_return':discounted,'duration_base_ticks':k,'Gamma':discount}
        self.publish(result_only=True)
        return {'reward':discounted,'raw_reward':raw,'Gamma':discount,'k':k,'terminal':self.terminated,'truncated':False,
            'next_features':self.features.copy(),'next_observation':self.observation.copy(),'next_mask':self.mask.copy(),
            'reason':self.reason,'before_tick':before,'after_tick':self.session.tick,'brain_before':brain_before,'brain_after':self.agent.brain_tick}

    def publish(self,result_only=False):
        s=self.session
        s.scene.update(target_yaw_rad=self.target_yaw,heading_tolerance_rad=JOINT_SPEC['heading_tolerance_rad'],task_kind='navigation_heading_hold')
        s.recording.manifest.update(observation_schema=JOINT_SPEC['observation'],observation_names=JOINT_SPEC['names'],task_spec=JOINT_SPEC,scene=s.scene,curriculum='J1',training_method=getattr(self.agent,'training_method','heading diagnostic'),
            observation_note='Phase-specific channels 6/7: absolute yaw in navigation; measured target error in heading. Two frozen learned heads; no raw-input bypass or joint-training claim')
        s.recording.manifest.update(skill_bundle=getattr(self.agent,'bundle',None),phase_observation_names={'navigation':SPEC['names'],'heading':HEADING_SPEC['names']})
        if self.live is None and self.observer is None:return
        self.seq+=1
        s=self.session;sensor=self.device.latest_observation()
        brain=self.agent.snapshot()
        values=self.observation.tolist()
        valid=[bool(values[12])]*3+[bool(values[13])]*3+[bool(values[12])]*2+[bool(values[14])]*2+[True]*16
        frame={'run_id':s.run_id,'epoch':s.epoch,'episode_id':self.episode_id,'env_id':self.env_id,'seq':self.seq,
            'sim_tick':s.tick,'time_s':s.tick/120,'truth':{'position_m':s.world.position.tolist(),
            'velocity_mps':s.world.velocity.tolist(),'yaw_rad':s.world.yaw_rad,
            'quaternion_wxyz':s.world.data.qpos[3:7].copy().tolist(),'angular_velocity_body_rad_s':s.world.data.qvel[3:6].copy().tolist(),'collision':s.collision_latched},
            'sensor':{'position_m':sensor.position_m,'valid':sensor.position_m is not None,
                'age_s':sensor.pose_age_s,'sample_tick':sensor.sample_tick,'source':'external_pose_mock'},
            'sensor_contract':asdict(sensor),'observation':values,'brain_observation':values,
            'observation_valid':valid,'observation_schema':JOINT_SPEC['observation'],'policy':self.last_policy,
            'brain':brain,'neural_tick':s.tick,'neural_substep':3,'brain_tick':self.agent.brain_tick,
            'reward':self.last_reward,'stable_hold_s':self.hold,'operation':self.device.observed_operation(),
            'finished':self.terminated,'trail':list(s.trail),'history':list(s.history)[-20:]}
        frame['heading']={'target_yaw_rad':self.target_yaw,'measured_yaw_rad':sensor.yaw_rad,'error_rad':wrap_angle(self.target_yaw-sensor.yaw_rad) if sensor.yaw_rad is not None else None,'tolerance_rad':JOINT_SPEC['heading_tolerance_rad']}
        frame.update(task_phase=self.phase,heading_anchor_m=self.heading_anchor,phase_events=list(self.phase_events),navigation_hold_s=self.navigation_hold,active_observation_schema=SPEC['observation'] if self.phase=='navigation' else HEADING_SPEC['observation'],observation_names=SPEC['names'] if self.phase=='navigation' else HEADING_SPEC['names'])
        frame.update(result_only=result_only,scene=s.scene,episode_return=self.raw_return,episode_steps=self.steps,termination_reason=self.reason)
        if self.observer is not None:self.observer.publish(self,frame)
        if self.live is None:return
        self.session.recording.add('transition',s.tick,{k:v for k,v in frame.items() if k not in {'trail','history','seq'}})
        self.session.recording.add('neural',s.tick,{'brain':brain,'neural_substep':3,'brain_tick':self.agent.brain_tick})
        # File consumers see finished only after the recording has been flushed.
        if not self.terminated:self.live.publish(frame)
        self.final_frame=frame

