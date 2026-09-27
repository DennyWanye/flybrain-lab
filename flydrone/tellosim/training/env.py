from __future__ import annotations
from collections import deque
from dataclasses import asdict
import math
import os
import uuid
import numpy as np

from ..visual import VisualSession, scene_config
from ..sdk.codec import TelloCommand, encode_command
from ...vis.live import LatestWriter
from .contracts import ACTIONS, SPEC, SensorObservation, action_mask


class ExternalPoseMock:
    """Extra sensor, not an undocumented Tello world-coordinate API."""
    def __init__(self,seed,noise=0.,delay=0,dropout=0.):
        if not np.isfinite(noise) or noise<0 or type(delay) is not int or not 0<=delay<=4 or not np.isfinite(dropout) or not 0<=dropout<=1:
            raise ValueError('invalid sensor profile')
        self.rng=np.random.default_rng(seed)
        self.noise,self.delay,self.dropout=noise,delay,dropout
        self.history=deque(maxlen=delay+1)
        self.previous=None
        self.latest=None

    def sample(self,session):
        tick=session.tick
        # Only this simulated sensor has access to the world; its public output
        # carries noisy/delayed measurements, never a reference to the world.
        measured=session.world.position+self.rng.normal(0,self.noise,3)
        yaw=session.world.yaw_rad+float(self.rng.normal(0,self.noise*.2))
        self.history.append((tick,measured,yaw))
        sample_tick,position,yaw=self.history[0]
        valid=self.rng.random()>=self.dropout
        velocity=None
        if valid and self.previous is not None and sample_tick>self.previous[0]:
            velocity=tuple((position-self.previous[1])/((sample_tick-self.previous[0])/120))
        self.previous=(sample_tick,position.copy()) if valid else None
        height=float(max(0,session.world.position[2]+self.rng.normal(0,self.noise)))
        # Explicit engineering battery model, not a measured battery curve.
        battery=max(0.,1-tick/120/1200)
        self.latest=SensorObservation(tick//12,sample_tick,tick,'room_map',tuple(position) if valid else None,
            yaw if valid else None,velocity if valid else None,height,battery,(tick-sample_tick)/120,0.)
        return self.latest


class SimulatedDevice:
    def __init__(self,session,sensors,reply_loss=0.,seed=0,channel_profile=None):
        self._session=session;self._sensors=sensors
        if not np.isfinite(reply_loss) or not 0<=reply_loss<=1:raise ValueError('invalid reply loss')
        self.rng=np.random.default_rng(seed);self.reply_loss=reply_loss
        self.channel=None
        if channel_profile is not None:
            from ..sdk.channels import CommandChannel,FaultProfile
            self.channel=CommandChannel(session,FaultProfile(**channel_profile),seed)

    def submit(self,command,request_id):
        if self.channel:return self.channel.submit(encode_command(command),request_id)
        return self._session.command(encode_command(command),request_id,lose_reply=bool(self.rng.random()<self.reply_loss))

    def advance(self,ticks):
        if self.channel:self.channel.advance(ticks)
        else:self._session.advance(ticks)

    def poll(self,operation_id):
        if self.channel:return self.channel.poll(operation_id)
        op=next((o for o in self._session.requests.values() if o['operation_id']==operation_id),None)
        if op is None:raise KeyError(operation_id)
        # Client code cannot inspect simulated device completion truth.
        return {k:op[k] for k in ('operation_id','request_id','wire','client','response')}

    def observed_operation(self):
        # Observer-only evidence; never returned through the policy poll contract.
        if not self.channel or self.channel.active is None:
            return self._session.operation.copy() if self._session.operation else None
        op=self.channel.operations[self.channel.active]
        device=self._session.requests.get(op['device_id'],{})
        return {**self.channel.poll(op['operation_id']),'sent_tick':op['sent_tick'],
            'verb':op['verb'],'device':device.get('device','not_observed'),
            'device_operation_id':device.get('operation_id'),'source':'fault_channel',
            'completed_tick':device.get('completed_tick')}

    def latest_observation(self):
        return self._sensors.latest

    def protective_stop(self,request_id):
        return self._session.command('stop',request_id)


class TrainingEnv:
    """C0 option environment using the exact visual sandbox executor."""
    def __init__(self,root,case,agent,record=False,run_id=None,policy_source='malecns_ppo_sdk9',observer=None):
        self.case=dict(case);self.agent=agent;self.previous_action=None;self.previous_duration=0.
        self.terminated=False;self.reason=None;self.hold=0.;self.raw_return=0.;self.steps=0
        self.previous_phi=None;self.seq=0;self.live=None;self.observer=observer
        self.env_id=getattr(agent,"index",0);self.episode_id=case.get("case_id",f"episode-seed-{case['seed']}")
        self.stop_dwell_s=getattr(agent,'stop_dwell_s',.5)
        self.last_policy=None;self.last_reward=None;self.invalid_age=0.
        scene=scene_config(root,bool(case.get('obstacles',False)))
        scene.update(controller=getattr(agent,'physics_profile','bounded_level_body_surrogate'),target_xyz_m=case['goal'],target_radius_m=.2,stable_hold_required_s=2.)
        self.session=VisualSession(root,mode='training',seed=case['seed'],scene=scene,
            start_position=(case['start'][0],case['start'][1],.047),recording_enabled=record,
            run_id=run_id or f'train-{uuid.uuid4().hex[:12]}',initial_yaw_rad=float(case.get('initial_yaw_rad',0.)))
        self.sensors=ExternalPoseMock(case['seed'],case.get('noise',0.),case.get('delay',0),case.get('dropout',0.))
        self.device=SimulatedDevice(self.session,self.sensors,case.get('reply_loss',0.),case['seed']+1,case.get('channel_profile'))
        manifest=self.session.recording.manifest
        manifest.update(training_method=getattr(agent,'training_method','PPO or diagnostic baseline'),stop_min_s=self.stop_dwell_s,agent_profile=getattr(agent,'profile','legacy'),feature_source=getattr(agent,'feature_source','reservoir'),
            policy_source=policy_source if getattr(agent,'feature_source','reservoir')=='reservoir' else policy_source+'_'+agent.feature_source,mode='TRAINING',observation_schema=SPEC['observation'],
            policy_checkpoint_sha256=getattr(agent,'checkpoint_sha256',None) if policy_source!='malecns_ppo_sdk9' else None,
            observation_names=SPEC['names'],curriculum=case.get('curriculum','C0'),source_epoch=self.session.epoch,owner_pid=os.getpid(),
            observation_note='26 measured channels; encoder 34; actor/critic receive only reservoir v/trace' if getattr(agent,'feature_source','reservoir')=='reservoir' else 'DIAGNOSTIC CONTROL: '+agent.feature_source,
            case=case,env_id=self.env_id,episode_id=self.episode_id,mission_manager_actions=['command','takeoff'],graph_sha256=agent.brain.graph_sha256,mapping_sha256=agent.brain.mapping_sha256,
            limitations=['Frozen connectome; only actor/critic readout is trained',
                         'External localization sensor required for future transfer',
                         'Engineering battery/physics models, not calibrated to aircraft',
                         'Replay records the final neural substep of each 10Hz sample'])
        self.session.recording.save()
        if record:
            self.live=LatestWriter(self.session.recording.directory,manifest)
        try:
            self.session.command('command','bootstrap-sdk')
            self.session.command('takeoff','bootstrap-takeoff')
            for _ in range(400):
                self.session.advance(12)
                if self.session.operation['client']!='sent':break
            if self.session.operation['client']!='ack_ok':raise RuntimeError('bootstrap takeoff failed')
            self.start_tick=self.session.tick
            self.agent.reset()
            self.sensors.sample(self.session)
            self.observe()
            self.previous_phi=self.potential()
        except BaseException:
            self.close('bootstrap_error');raise

    def potential(self):
        return -min(float(np.linalg.norm(self.session.world.position-np.asarray(self.case['goal']))),6.)/6

    def prepare_observation(self):
        sensor=self.device.latest_observation()
        elapsed=(self.session.tick-self.start_tick)/120
        self.observation=sensor.vector(self.case['goal'],self.previous_action,self.previous_duration,1-elapsed/60)
        self.mask=action_mask(self.observation,self.case.get('curriculum','C0'))

    def observe(self):
        self.prepare_observation()
        self.features=self.agent.observe(self.observation,self.device.latest_observation().sample_id).copy()
        self.publish()

    def publish(self,result_only=False):
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
            'observation_valid':valid,'observation_schema':SPEC['observation'],'policy':self.last_policy,
            'brain':brain,'neural_tick':s.tick,'neural_substep':3,'brain_tick':self.agent.brain_tick,
            'reward':self.last_reward,'stable_hold_s':self.hold,'operation':self.device.observed_operation(),
            'finished':self.terminated,'trail':list(s.trail),'history':list(s.history)[-20:]}
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

    def step(self,action,policy=None):
        iterator=self.step_iter(action,policy)
        while True:
            try:next(iterator)
            except StopIteration as done:return done.value
            self.features=self.agent.observe(self.observation,self.device.latest_observation().sample_id).copy()
            self.publish()

    def export_state(self):
        import copy
        if self.live is not None:raise RuntimeError('exact training snapshots use unrecorded rollouts')
        excluded={'agent','session','sensors','device','live','observer','final_frame'}
        sensor={k:v for k,v in self.sensors.__dict__.items() if k!='rng'}
        channel=self.device.channel
        return {'fields':copy.deepcopy({k:v for k,v in self.__dict__.items() if k not in excluded}),
            'session':self.session.export_state(),'sensor':copy.deepcopy(sensor),
            'sensor_rng':copy.deepcopy(self.sensors.rng.bit_generator.state),
            'device_rng':copy.deepcopy(self.device.rng.bit_generator.state),
            'channel':None if channel is None else copy.deepcopy({'operations':channel.operations,'active':channel.active,'rng':channel.rng.bit_generator.state})}

    def restore_state(self,state):
        import copy
        if self.case!=state['fields']['case']:raise ValueError('case mismatch')
        self.session.restore_state(state['session'])
        self.__dict__.update(copy.deepcopy(state['fields']))
        self.sensors.__dict__.update(copy.deepcopy(state['sensor']))
        self.sensors.rng.bit_generator.state=copy.deepcopy(state['sensor_rng'])
        self.device.rng.bit_generator.state=copy.deepcopy(state['device_rng'])
        channel=state['channel']
        if (channel is None)!=(self.device.channel is None):raise ValueError('channel contract mismatch')
        if channel is not None:
            self.device.channel.operations=copy.deepcopy(channel['operations']);self.device.channel.active=channel['active']
            self.device.channel.rng.bit_generator.state=copy.deepcopy(channel['rng'])

    def close(self,outcome=None):
        if not hasattr(self,'session'):return
        outcome=outcome or self.reason or 'external_truncation'
        # Local episode shutdown requests stop, but does not rewrite a lost ack.
        if self.session.operation and self.session.operation['client'] in {'sent','unknown_execution'}:
            self.device.protective_stop('episode-protective-stop')
            for _ in range(100):
                self.session.advance(12)
                if self.session.operation['client']!='sent':break
        self.session.recording.manifest.update(policy_return=self.raw_return,task_result=self.reason,
            policy_steps=self.steps,success=self.reason=='success')
        self.session.close(outcome)
        if self.live:
            if hasattr(self,'final_frame'):
                final={**self.final_frame,'finished':True};self.live.publish(final)
            self.live.close();self.live=None


def sample_case(seed,case_id=None,**overrides):
    rng=np.random.default_rng(seed)
    for _ in range(10000):
        start=rng.uniform(-2,2,2);goal=rng.uniform(-2,2,2)
        if .6<=np.linalg.norm(goal-start)<=2.5:break
    return {'case_id':case_id or f'train-{seed}','seed':seed,'start':[float(start[0]),float(start[1]),1.],
            'goal':[float(goal[0]),float(goal[1]),1.],'noise':0.,'delay':0,'dropout':0.,'reply_loss':0.,**overrides}


def curriculum_case(seed,case_id=None,curriculum='C0',**overrides):
    if curriculum=='C0':return sample_case(seed,case_id,**overrides)
    if curriculum=='C1':
        yaw=float(np.random.default_rng(seed+7919).uniform(-math.pi,math.pi))
        return sample_case(seed,case_id,curriculum='C1',initial_yaw_rad=yaw,**overrides)
    if curriculum not in ('C0-near-x','C0-near-xy'):raise ValueError('unknown curriculum')
    rng=np.random.default_rng(seed)
    start=rng.uniform(-.6,.6,2);distance=float(rng.uniform(.3,.65))
    axis=0 if curriculum=='C0-near-x' else int(rng.integers(2))
    direction=int(rng.choice([-1,1]));goal=start.copy();goal[axis]+=direction*distance
    return {'case_id':case_id or f'train-{seed}','seed':seed,'start':[float(start[0]),float(start[1]),1.],
            'goal':[float(goal[0]),float(goal[1]),1.],'noise':0.,'delay':0,'dropout':0.,'reply_loss':0.,
            'curriculum':curriculum,**overrides}
