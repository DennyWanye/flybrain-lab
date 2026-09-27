"""Isolated learned heading skill. Legacy navigation code and weights stay intact."""
from pathlib import Path
from dataclasses import asdict
import hashlib,math,json,time
import numpy as np
import torch
from .env import TrainingEnv
from .contracts import SPEC,ACTIONS,digest
from .runtime import checkpoint_contract
from ..sdk.codec import TelloCommand
from ..visual import atomic_json
from ..view_export import normalize_episode_identity

HEADING_SPEC={**SPEC,'observation':'tellosim.heading_observation26/1.0',
    'names':SPEC['names'][:6]+['target_yaw_error_sin','target_yaw_error_cos']+SPEC['names'][8:],
    'curriculum':'H1','task':'in_place_target_heading_and_hold',
    'heading_tolerance_rad':math.radians(16),'yaw_rate_tolerance_rad_s':.08,
    'allowed_actions':[0,7,8],'goal_source':'external task instruction; yaw from external measured pose',
    'precision_note':'SDK actions rotate 30 degrees; target acceptance is within 16 degrees'}

def wrap_angle(angle):return math.atan2(math.sin(angle),math.cos(angle))

def heading_case(seed,case_id=None):
    rng=np.random.default_rng(seed);xy=rng.uniform(-1.5,1.5,2);yaw=float(rng.uniform(-math.pi,math.pi))
    delta=float(rng.choice([-1,1])*rng.uniform(math.radians(35),math.pi))
    return {'case_id':case_id or f'heading-train-{seed}','seed':seed,'curriculum':'H1',
        'start':[float(xy[0]),float(xy[1]),1.],'goal':[float(xy[0]),float(xy[1]),1.],
        'initial_yaw_rad':yaw,'target_yaw_rad':wrap_angle(yaw+delta),
        'noise':0.,'delay':0,'dropout':0.,'reply_loss':0.}

class HeadingEnv(TrainingEnv):
    def __init__(self,root,case,agent,**kwargs):
        if case.get('curriculum')!='H1':raise ValueError('heading skill requires H1 cases')
        self.target_yaw=float(case['target_yaw_rad'])
        if not math.isfinite(self.target_yaw):raise ValueError('nonfinite heading target')
        if not np.allclose(case['start'],case['goal'],rtol=0,atol=0):raise ValueError('H1 is an in-place task, not joint navigation')
        super().__init__(root,case,agent,**kwargs)
        self.session.recording.save()
    def potential(self):
        angle=abs(wrap_angle(self.target_yaw-self.session.world.yaw_rad))/math.pi
        drift=min(float(np.linalg.norm(self.session.world.position-np.asarray(self.case['goal']))),6.)/6
        return -angle-.2*drift
    def prepare_observation(self):
        sensor=self.device.latest_observation();elapsed=(self.session.tick-self.start_tick)/120
        self.observation=sensor.vector(self.case['goal'],self.previous_action,self.previous_duration,1-elapsed/60)
        if self.observation[12]:
            error=wrap_angle(self.target_yaw-sensor.yaw_rad);self.observation[6:8]=[math.sin(error),math.cos(error)]
        self.mask=np.zeros(9,bool);self.mask[0]=True
        if self.observation[12] and self.observation[14]:self.mask[7:9]=True
    def publish(self,result_only=False):
        s=self.session
        s.scene.update(target_yaw_rad=self.target_yaw,heading_tolerance_rad=HEADING_SPEC['heading_tolerance_rad'],task_kind='heading_hold')
        s.recording.manifest.update(observation_schema=HEADING_SPEC['observation'],observation_names=HEADING_SPEC['names'],task_spec=HEADING_SPEC,scene=s.scene,curriculum='H1',training_method=getattr(self.agent,'training_method','heading diagnostic'),
            observation_note='Channels 6/7 are sin/cos of measured target-heading error; versioned heading-only readout; no raw-input bypass')
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
            'observation_valid':valid,'observation_schema':HEADING_SPEC['observation'],'policy':self.last_policy,
            'brain':brain,'neural_tick':s.tick,'neural_substep':3,'brain_tick':self.agent.brain_tick,
            'reward':self.last_reward,'stable_hold_s':self.hold,'operation':self.device.observed_operation(),
            'finished':self.terminated,'trail':list(s.trail),'history':list(s.history)[-20:]}
        frame['heading']={'target_yaw_rad':self.target_yaw,'measured_yaw_rad':sensor.yaw_rad,'error_rad':wrap_angle(self.target_yaw-sensor.yaw_rad) if sensor.yaw_rad is not None else None,'tolerance_rad':HEADING_SPEC['heading_tolerance_rad']}
        frame.update(result_only=result_only,scene=s.scene,episode_return=self.raw_return,episode_steps=self.steps,termination_reason=self.reason)
        if self.observer is not None:self.observer.publish(self,frame)
        if self.live is None:return
        self.session.recording.add('transition',s.tick,{k:v for k,v in frame.items() if k not in {'trail','history','seq'}})
        self.session.recording.add('neural',s.tick,{'brain':brain,'neural_substep':3,'brain_tick':self.agent.brain_tick})
        # File consumers see finished only after the recording has been flushed.
        if not self.terminated:self.live.publish(frame)
        self.final_frame=frame

    def step_iter(self,action,policy=None):
        if self.terminated:raise RuntimeError('step after terminal')
        if type(action) is not int or not 0<=action<9 or not self.mask[action]:
            raise ValueError('invalid or masked action')
        self.last_policy=policy
        if policy:self.publish()
        before=self.session.tick;brain_before=self.agent.brain_tick
        op=self.device.submit(TelloCommand(*ACTIONS[action]),f'policy-{self.steps}')
        terms=[];discounted=0.;raw=0.;discount=1.
        while True:
            self.device.advance(12)
            self.sensors.sample(self.session)
            self.previous_action=action
            self.previous_duration=(self.session.tick-before)/120
            self.prepare_observation()
            yield self
            status=self.device.poll(op['operation_id'])
            elapsed=(self.session.tick-self.start_tick)/120
            pos=self.session.world.position
            distance=float(np.linalg.norm(pos[:2]-np.asarray(self.case['goal'])[:2]))
            stable=(distance<=.2 and np.linalg.norm(self.session.world.velocity[:2])<=.08
                    and abs(pos[2]-self.case['goal'][2])<=.1
                    and abs(wrap_angle(self.target_yaw-self.session.world.yaw_rad))<=HEADING_SPEC['heading_tolerance_rad']
                    and abs(float(self.session.world.data.qvel[5]))<=HEADING_SPEC['yaw_rate_tolerance_rad_s']
                    and (status['client']=='ack_ok' or (action==0 and status['client']=='sent')))
            self.hold=self.hold+.1 if stable else 0.
            self.invalid_age=self.invalid_age+.1 if not self.observation[12] else 0.
            self.reason=('collision' if self.session.collision_latched else
                         'command_unknown' if status['client']=='unknown_execution' else
                         'command_error' if status['client']=='ack_error' else
                         'localization_lost' if self.invalid_age>=.5-1e-8 else
                         'success' if self.hold>=2-1e-8 else
                         'task_deadline' if elapsed>=60-1e-8 else None)
            self.terminated=self.reason is not None
            phi=0. if self.terminated else self.potential()
            components={'potential':SPEC['gamma_base']*phi-self.previous_phi,'step_cost':-.001,
                        'success_bonus':5. if self.reason=='success' else 0.,
                        'failure_penalty':-5. if self.terminated and self.reason!='success' else 0.}
            self.previous_phi=phi
            reward=sum(components.values());raw+=reward;discounted+=discount*reward;discount*=SPEC['gamma_base']
            terms.append(components)
            # STOP dwell is an explicit checkpoint-bound option duration.
            # It never navigates toward the goal or changes the success hold.
            if self.terminated or (status['client']!='sent' and (action!=0 or self.previous_duration>=self.stop_dwell_s-1e-8)):
                break
        self.steps+=1;self.raw_return+=raw
        k=(self.session.tick-before)//12
        if self.agent.brain_tick-brain_before!=4*k:raise AssertionError('neural/simulator clock mismatch')
        self.last_reward={'reward_total':raw,'reward_components':{key:sum(t[key] for t in terms) for key in terms[0]},
                          'discounted_option_return':discounted,'duration_base_ticks':k,'Gamma':discount}
        self.publish(result_only=True)
        return {'reward':discounted,'raw_reward':raw,'Gamma':discount,'k':k,'terminal':self.terminated,
                'truncated':False,'next_features':self.features.copy(),'next_observation':self.observation.copy(),
                'next_mask':self.mask.copy(),'reason':self.reason,'reward_base_terms':terms,
                'before_tick':before,'after_tick':self.session.tick,'brain_before':brain_before,'brain_after':self.agent.brain_tick}


    def close(self,outcome=None):
        recorded=hasattr(self,'session') and hasattr(self.session.recording,'directory')
        super().close(outcome)
        if recorded and self.session.recording.manifest.get('complete'):
            normalize_episode_identity(self.session.recording.directory)

def heading_contract(pool,root):
    root=Path(root)
    return {'base':checkpoint_contract(pool,root),'task_spec':HEADING_SPEC,'task_spec_hash':digest(HEADING_SPEC),
        'task_sources':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('flydrone/tellosim/training/heading.py','flydrone/tellosim/training/heading_campaign.py')}}

def save_heading(path,pool,optimizer,root,options,extra=None):
    state={'format':'tellosim.heading_readout/1','contract':heading_contract(pool,root),
        'policy':pool.model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,'options':options,
        'value_trained':False,'extra':extra or {},'restore_mode':'inference or explicit stage warm start; not exact mid-rollout resume',
        'method':'measured-pose teacher demonstrations and DAgger; frozen real MaleCNS; no PPO claim'}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def load_heading(path,pool,root):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format')!='tellosim.heading_readout/1' or state['contract']!=heading_contract(pool,root):raise ValueError('heading checkpoint contract mismatch')
    pool.model.load_state_dict(state['policy']);pool.value_trained=False
    pool.training_method=state['method'];pool.checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest();pool.reset();return state

def teacher(observation,mask,first=False):
    # Training/baseline only. Evaluation calls the saved learned readout.
    if first or not observation[12]:return 0
    error=math.atan2(float(observation[6]),float(observation[7]))
    if abs(error)<=math.radians(15):return 0
    action=8 if error>0 else 7
    return action if mask[action] else 0
