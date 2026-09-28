"""C2 continuous XYZ + instructed yaw + hold; independent of frozen V1/J2R/V1R."""
from pathlib import Path
from dataclasses import asdict
from collections import deque
import copy,json,math
import numpy as np
import torch
from .env import TrainingEnv
from .heading import HeadingEnv,HEADING_SPEC,wrap_angle
from .parallel import ReservoirPool,AgentLane
from .contracts import SPEC,ACTIONS,action_mask,digest,Encoder34
from .runtime import checkpoint_contract,distribution
from .joint import joint_case,sha
from .joint_campaign import JointDiagnostic
from .altitude_refined import AltitudeDisturbanceMixin,AltitudeEncoder34,altitude_mask,ALTITUDE_SPEC,ENCODER_SPEC,DISTURBANCE_SPEC,load_altitude
from .stability_campaign import bundle_spec as legacy_bundle
from .robust_campaign import load_skill
from ..sdk.codec import TelloCommand
from ..visual import atomic_json

SKILLS=('altitude','navigation','heading')
SPATIAL_SPEC={**SPEC,'curriculum':'C2','observation':'tellosim.spatial_observation26/1.0',
 'task':'continuous_xyz_then_instructed_yaw_and_hold','task_deadline_s':180.,'absolute_simulator_deadline_s':180.,'phase_deadline_s':60.,
 'target_height_range_m':[.25,1.75],'height_tolerance_m':.1,'vertical_speed_tolerance_mps':.08,
 'heading_tolerance_rad':math.radians(16),'yaw_rate_tolerance_rad_s':.08,'minimum_episode_s':8.,
 'stop_min_s':2.,'navigation_handoff_hold_s':2.,'altitude_handoff_hold_s':2.,'recovery_distance_m':.2,'recovery_hold_s':.5,
 'allowed_actions':{'altitude':[0,5,6],'navigation':[0,1,2,3,4,7,8],'heading':[0,7,8]},
 'phase_observations':{'altitude':ALTITUDE_SPEC['observation'],'navigation':SPEC['observation'],'heading':HEADING_SPEC['observation']},
 'phase_manager':'measured pose/velocity only, completed SDK option; no movement direction selection',
 'neural_handoff':'reset neural lane, preserve physical integration state, target, clock; one new10Hz observation',
 'trainable':'all C2Q readouts and MaleCNS frozen; measured-phase estimator repair only',
 'phase_estimator':{'kind':'trailing_measured_mean/1.0','window_samples':20,'min_samples':10,'sample_hz':10,'fields':'measured world XYZ and measured velocity; cleared on invalid data','policy_observation':'unchanged raw26','final_success':'unchanged true physical state and original tolerances'},
 'SINGLE_POLICY_JOINT_TRAINED':False}
SOURCE_FILES=('flydrone/tellosim/training/spatial_orientation.py','flydrone/tellosim/training/spatial_orientation_campaign.py','flydrone/tellosim/training/spatial_precision.py','flydrone/tellosim/training/spatial_precision_campaign.py','flydrone/tellosim/training/spatial_refined.py','flydrone/tellosim/training/spatial_refined_campaign.py','flydrone/tellosim/training/spatial_settled.py','flydrone/tellosim/training/spatial_settled_campaign.py',
 'flydrone/tellosim/training/altitude_refined.py','flydrone/tellosim/training/altitude_refined_campaign.py',
 'flydrone/tellosim/training/joint.py','flydrone/tellosim/training/stability_campaign.py',
 'flydrone/tellosim/training/heading.py','flydrone/tellosim/training/parallel.py','flydrone/tellosim/training/contracts.py')

def spatial_case(seed,case_id=None):
    c=joint_case(seed,case_id or f'c2-{seed}');rng=np.random.default_rng(seed+4401)
    c['goal'][2]=float(1+rng.choice([-1,1])*rng.uniform(.25,.75));c['curriculum']='C2'
    return c

NAVIGATION_ENCODER_SPEC={'kind':'navigation_body_distance34/1.0','direction_scale_m':.6,'distance_scales_m':[.12,.36],
 'channels_0_to_3':'signed body XY tanh populations','channels_4_to_5':'continuous radial XY distance tanh populations',
 'nuisance_invariance':'body-relative navigation ignores absolute yaw and absolute height; validity and measured velocities preserved',
 'policy_input':'128 actual frozen MaleCNS downstream features; no raw-input or action-rule bypass'}

class NavigationEncoder34(Encoder34):
    def __init__(self):super().__init__('balanced_rate_v2')
    def __call__(self,observations):
        obs=np.array(observations,dtype=np.float32,copy=True)
        # Goal and velocity are already in the measured body frame.
        obs[:,6]=0;obs[:,7]=obs[:,12];obs[:,8]=obs[:,14]/3
        encoded=super().__call__(obs);xy=obs[:,:2]*6
        for axis in range(2):
            signed=np.tanh(xy[:,axis]/.6)
            encoded[:,axis*2]=.12+.88*np.maximum(signed,0)
            encoded[:,axis*2+1]=.12+.88*np.maximum(-signed,0)
        radius=np.linalg.norm(xy,axis=1)
        for i,scale in enumerate([.12,.36]):encoded[:,4+i]=.12+.88*np.tanh(radius/scale)
        return encoded

HEADING_ENCODER_SPEC={'kind':'heading_relative_angle34/1.0','angle_scales_rad':[.15,.45,1.35],
 'channels_0_to_5':'signed continuous yaw error at three scales; no angle threshold or action choice',
 'nuisance_invariance':'relative-yaw control ignores residual XY/Z goal displacement and absolute height; measured velocities and validity preserved',
 'policy_input':'128 actual frozen MaleCNS features; no raw-angle bypass'}

class HeadingEncoder34(Encoder34):
    def __init__(self):super().__init__('balanced_rate_v2')
    def __call__(self,observations):
        obs=np.array(observations,dtype=np.float32,copy=True)
        if obs.ndim!=2 or obs.shape[1]!=26 or not np.isfinite(obs).all():raise ValueError('expected finite [batch,26] observations')
        angle=np.arctan2(obs[:,6],obs[:,7])
        obs[:,:3]=0;obs[:,8]=obs[:,14]/3
        encoded=super().__call__(obs)
        for i,scale in enumerate(HEADING_ENCODER_SPEC['angle_scales_rad']):
            signed=np.tanh(angle/scale)
            encoded[:,2*i]=.12+.88*np.maximum(signed,0)
            encoded[:,2*i+1]=.12+.88*np.maximum(-signed,0)
        return encoded

class SpatialLane(AgentLane):
    def __init__(self,pool,index):super().__init__(pool,index);self.skill='altitude'
    def reset(self):super().reset();self.skill='altitude'
    def switch_skill(self,skill):
        if skill not in SKILLS:raise ValueError('invalid C2 skill')
        self.skill=skill;mask=np.zeros(self.brain.batch,bool);mask[self.index]=True;self.brain.reset(mask)
    @torch.no_grad()
    def decision(self,features,mask,deterministic=False):
        pi,value=distribution(self.pool.models[self.skill],torch.as_tensor(features[None],dtype=torch.float32),[mask])
        action=pi.probs.argmax(-1) if deterministic else torch.multinomial(pi.probs,1,generator=self.action_rng).squeeze(-1)
        return int(action.item()),float(pi.log_prob(action).item()),float(value.item()),dict(selected_action=int(action.item()),
            probabilities=pi.probs[0].tolist(),value_estimate=float(value.item()),value_trained=False,mask=np.asarray(mask,bool).tolist(),
            input_source='reservoir_v_trace',feature_dim=128,skill=self.skill,checkpoint_sha256=getattr(self.pool,'checkpoint_sha256',None))

from .spatial_orientation import SpatialPool as TrainedSpatialPool

class SpatialPool(TrainedSpatialPool):
    def __init__(self,root,seed,batch=16,graph=None,device='cuda',initialize=True):
        super().__init__(root,seed,batch,graph,device,initialize)
        if initialize:
            from .spatial_orientation import load_spatial as load_trained
            source=Path(root)/f'runs/tellosim-sdk9/spatial-orientation-s{seed}/checkpoint.pt'
            load_trained(source,self,root)
            self.original_sources={skill:{'path':str(source.relative_to(root)),'sha256':sha(source),'head':skill,'used_for_initial_weights':True} for skill in SKILLS}
        self.training_method='C2S: frozen formally trained C2Q readouts; measured phase-state smoothing; no optimization'
        for model in self.models.values():
            for parameter in model.parameters():parameter.requires_grad_(False)

class SpatialDiagnostic(JointDiagnostic):
    def reset(self):super().reset();self.skill='altitude'
    def snapshot(self):return {'status':'not_applicable','reason':'rule diagnostic; no neural model','neurons':[]}


def spatial_contract(pool,root):
    return {'base':checkpoint_contract(pool,root),'task':SPATIAL_SPEC,'disturbance':DISTURBANCE_SPEC,
        'phase_encoders':{'altitude':ENCODER_SPEC,'navigation':NAVIGATION_ENCODER_SPEC,'heading':HEADING_ENCODER_SPEC},
        'sources':{name:sha(Path(root)/name) for name in SOURCE_FILES},'initial_sources':pool.original_sources}

def save_spatial(path,pool,opts,root,options,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'format':'tellosim.spatial_settled_readouts/1','contract':spatial_contract(pool,root),
        'models':{skill:model.state_dict() for skill,model in pool.models.items()},
        'optimizers':{skill:opt.state_dict() for skill,opt in opts.items()} if opts else None,
        'seed':pool.seed,'options':options,'extra':extra or {},'value_trained':False,'restore_mode':'inference or explicit stage warm start'}
    temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)

def load_spatial(path,pool,root):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format')!='tellosim.spatial_settled_readouts/1' or state['contract']!=spatial_contract(pool,root):raise ValueError('spatial contract mismatch')
    for skill in SKILLS:pool.models[skill].load_state_dict(state['models'][skill])
    pool.reset();pool.checkpoint_sha256=sha(path);pool.bundle={'checkpoint_path':str(path),'checkpoint_sha256':sha(path),'spec':SPATIAL_SPEC,'initial_sources':pool.original_sources}
    return state

class SpatialBase(HeadingEnv):
    def __init__(self,root,case,agent,**kwargs):
        self._phase_window=deque(maxlen=20);self._phase_sample=-1;self._phase_valid=False
        if case.get('curriculum')!='C2':raise ValueError('spatial task requires C2 cases')
        start=np.asarray(case['start'],float);goal=np.asarray(case['goal'],float)
        if start.shape!=(3,) or goal.shape!=(3,) or not np.isfinite(np.r_[start,goal]).all() or start[2]!=1 or not .25<=goal[2]<=1.75:raise ValueError('invalid spatial case')
        self.target_yaw=float(case['target_yaw_rad'])
        if not math.isfinite(self.target_yaw):raise ValueError('invalid target yaw')
        self.altitude_anchor=[*case['start'][:2],case['goal'][2]];self.heading_anchor=None;self.phase='altitude';self.phase_start_tick=None;self.navigation_hold=0.;self.altitude_hold=0.;self.height_drift_hold=0.;self.drift_hold=0.;self.phase_events=[];self.navigation_heights=[]
        TrainingEnv.__init__(self,root,case,agent,**kwargs)
        self.phase_start_tick=self.start_tick;self.session.recording.save()
    def potential(self):
        position=min(float(np.linalg.norm(self.session.world.position-np.asarray(self.case['goal']))),6.)/6
        heading=abs(wrap_angle(self.target_yaw-self.session.world.yaw_rad))/math.pi
        return -position-.25*heading
    def prepare_observation(self):
        sensor=self.device.latest_observation();start=self.phase_start_tick if self.phase_start_tick is not None else self.start_tick
        self.observation=sensor.vector(self.heading_anchor if self.phase=='heading' else self.altitude_anchor if self.phase=='altitude' else self.case['goal'],self.previous_action,self.previous_duration,1-(self.session.tick-start)/120/60)
        if self.phase=='altitude':self.mask=altitude_mask(self.observation,sensor)
        elif self.phase=='navigation':
            self.mask=action_mask(self.observation,'C1')
            if sensor.position_m is not None:self.navigation_heights.append(float(sensor.position_m[2]))
        else:
            if self.observation[12]:
                error=wrap_angle(self.target_yaw-sensor.yaw_rad);self.observation[6:8]=[math.sin(error),math.cos(error)]
            self.mask=np.zeros(9,bool);self.mask[0]=True
            if self.observation[12] and self.observation[14]:self.mask[7:9]=True
        self.update_phase_estimate(sensor)
    def update_phase_estimate(self,sensor):
        self._phase_valid=bool(all(self.observation[i] for i in (12,13,14)) and sensor.position_m is not None and sensor.velocity_mps is not None)
        if not self._phase_valid:self._phase_window.clear();return
        if sensor.sample_id==self._phase_sample:return
        self._phase_sample=sensor.sample_id
        self._phase_window.append(np.asarray([*sensor.position_m,*sensor.velocity_mps],float))
    def phase_estimate(self):
        if not self._phase_valid or len(self._phase_window)<10:return None
        estimate=np.mean(self._phase_window,axis=0)
        return estimate[:3],estimate[3:]
    def measured_position_state(self):
        s=self.device.latest_observation()
        if not self.observation[12] or not self.observation[13] or not self.observation[14]:return False,False
        estimate=self.phase_estimate()
        if estimate is None:return False,False
        pos,velocity=estimate;goal=np.asarray(self.case['goal']);distance=np.linalg.norm(pos[:2]-goal[:2])
        stable=distance<=.2 and np.linalg.norm(velocity[:2])<=.08 and abs(pos[2]-goal[2])<=.1 and abs(velocity[2])<=.08
        drift=distance>SPATIAL_SPEC['recovery_distance_m']
        return bool(stable),bool(drift)
    def switch_phase(self,phase):
        before=self.session.world.export_state();old=self.phase;clock=self.agent.brain_tick
        if phase=='altitude':
            pose=self.device.latest_observation().position_m
            if pose is None:raise ValueError('cannot recover altitude without measured pose')
            self.altitude_anchor=[pose[0],pose[1],self.case['goal'][2]]
        if phase=='heading':
            pose=self.device.latest_observation().position_m
            if pose is None:raise ValueError('cannot hand off without measured pose')
            self.heading_anchor=[pose[0],pose[1],self.case['goal'][2]]
        self.agent.switch_skill(phase);self.phase=phase;self.phase_start_tick=self.session.tick
        self.navigation_hold=0.;self.altitude_hold=0.;self.height_drift_hold=0.;self.drift_hold=0.;self.hold=0.;self.previous_action=None;self.previous_duration=0.;self.last_policy=None
        after=self.session.world.export_state()
        # Physical state belongs to the simulator; only neural memory is reset.
        np.testing.assert_array_equal(before['integration'],after['integration'])
        np.testing.assert_array_equal(before['target'],after['target']);assert before['target_yaw']==after['target_yaw']
        assert self.agent.brain_tick==clock
        self.phase_events.append({'tick':self.session.tick,'from':old,'to':phase,'reason':f'{old}_to_{phase}','height_m':float(self.session.world.position[2]),
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
            sensor=self.device.latest_observation()
            estimate=self.phase_estimate()
            height_error=abs(estimate[0][2]-self.case['goal'][2]) if estimate is not None else float('inf')
            height_ok=(height_error<=.1 and estimate is not None and abs(estimate[1][2])<=.08 and np.linalg.norm(estimate[1][:2])<=.08)
            self.altitude_hold=self.altitude_hold+.1 if self.phase=='altitude' and height_ok and ack else 0.
            self.height_drift_hold=self.height_drift_hold+.1 if self.phase!='altitude' and height_error>.1 else 0.
            self.navigation_hold=self.navigation_hold+.1 if self.phase=='navigation' and position_ok and ack else 0.
            self.drift_hold=self.drift_hold+.1 if self.phase=='heading' and drift else 0.
            pos=self.session.world.position;goal=np.asarray(self.case['goal'])
            stable=(self.phase=='heading' and np.linalg.norm(pos[:2]-goal[:2])<=.2 and np.linalg.norm(self.session.world.velocity[:2])<=.08
                and abs(pos[2]-goal[2])<=.1 and abs(self.session.world.velocity[2])<=.08 and abs(wrap_angle(self.target_yaw-self.session.world.yaw_rad))<=SPATIAL_SPEC['heading_tolerance_rad']
                and abs(float(self.session.world.data.qvel[5]))<=.08 and ack)
            self.hold=self.hold+.1 if stable else 0.;self.invalid_age=self.invalid_age+.1 if not self.observation[12] else 0.
            self.reason=('collision' if self.session.collision_latched else 'command_unknown' if status['client']=='unknown_execution' else
                'command_error' if status['client']=='ack_error' else 'localization_lost' if self.invalid_age>=.5-1e-8 else
                'sandbox_time_limit_180s' if self.session.finished else 'success' if self.hold>=2-1e-8 and elapsed>=8-1e-8 else 'task_deadline' if elapsed>=180-1e-8 else 'phase_deadline' if phase_elapsed>=60-1e-8 else None)
            self.terminated=self.reason is not None
            phi=0. if self.terminated else self.potential();components={'potential':SPEC['gamma_base']*phi-self.previous_phi,'step_cost':-.001,
                'success_bonus':5. if self.reason=='success' else 0.,'failure_penalty':-5. if self.terminated and self.reason!='success' else 0.}
            self.previous_phi=phi;reward=sum(components.values());raw+=reward;discounted+=discount*reward;discount*=SPEC['gamma_base'];terms.append(components)
            done=status['client']!='sent' and (action!=0 or self.previous_duration>=self.stop_dwell_s-1e-8)
            if self.terminated or warming:break
            if done:
                switch=('navigation' if self.phase=='altitude' and self.altitude_hold>=2-1e-8 else
                    'altitude' if self.phase!='altitude' and self.height_drift_hold>=.5-1e-8 else
                    'heading' if self.phase=='navigation' and self.navigation_hold>=2-1e-8 else
                    'navigation' if self.phase=='heading' and self.drift_hold>=.5-1e-8 else None)
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
        s.scene.update(target_yaw_rad=self.target_yaw,heading_tolerance_rad=SPATIAL_SPEC['heading_tolerance_rad'],task_kind='spatial_heading_hold')
        s.recording.manifest.update(observation_schema=SPATIAL_SPEC['observation'],observation_names=SPATIAL_SPEC['names'],task_spec=SPATIAL_SPEC,scene=s.scene,curriculum='C2',training_method=getattr(self.agent,'training_method','heading diagnostic'),
            observation_note='Phase-specific channels 6/7: absolute yaw in navigation; measured target error in heading. Three readouts trained on continuous 3D task; no raw-input bypass; frozen connectome')
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
            'observation_valid':valid,'observation_schema':SPATIAL_SPEC['observation'],'policy':self.last_policy,
            'brain':brain,'neural_tick':s.tick,'neural_substep':3,'brain_tick':self.agent.brain_tick,
            'reward':self.last_reward,'stable_hold_s':self.hold,'operation':self.device.observed_operation(),
            'finished':self.terminated,'trail':list(s.trail),'history':list(s.history)[-20:]}
        frame['heading']={'target_yaw_rad':self.target_yaw,'measured_yaw_rad':sensor.yaw_rad,'error_rad':wrap_angle(self.target_yaw-sensor.yaw_rad) if sensor.yaw_rad is not None else None,'tolerance_rad':SPATIAL_SPEC['heading_tolerance_rad']}
        frame['altitude']={'target_z_m':self.case['goal'][2],'measured_z_m':sensor.position_m[2] if sensor.position_m else None,'error_m':self.case['goal'][2]-sensor.position_m[2] if sensor.position_m else None,'vertical_speed_mps':sensor.velocity_mps[2] if sensor.velocity_mps else None,'tolerance_m':.1}
        frame.update(task_phase=self.phase,heading_anchor_m=self.heading_anchor,phase_events=list(self.phase_events),navigation_hold_s=self.navigation_hold,active_observation_schema=SPATIAL_SPEC['phase_observations'][self.phase],observation_names=SPEC['names'] if self.phase=='navigation' else HEADING_SPEC['names'])
        frame.update(result_only=result_only,scene=s.scene,episode_return=self.raw_return,episode_steps=self.steps,termination_reason=self.reason)
        if self.observer is not None:self.observer.publish(self,frame)
        if self.live is None:return
        self.session.recording.add('transition',s.tick,{k:v for k,v in frame.items() if k not in {'trail','history','seq'}})
        self.session.recording.add('neural',s.tick,{'brain':brain,'neural_substep':3,'brain_tick':self.agent.brain_tick})
        # File consumers see finished only after the recording has been flushed.
        if not self.terminated:self.live.publish(frame)
        self.final_frame=frame


class SpatialEnv(AltitudeDisturbanceMixin,SpatialBase):
    def close(self,outcome=None):
        # The simulator may independently close at its absolute safety deadline.
        if hasattr(self,'session') and self.session.finished:
            self.session.recording.manifest.update(task_result=self.reason,policy_return=self.raw_return,policy_steps=self.steps,success=self.reason=='success')
            if hasattr(self.session.recording,'flush'):self.session.recording.flush()
            self.session.recording.save()
            if getattr(self,'_original_controller',None) is not None:
                self.session.world._apply_controller=self._original_controller
                self._original_controller=None
            if self.live:
                if hasattr(self,'final_frame'):self.live.publish({**self.final_frame,'finished':True})
                self.live.close();self.live=None
            return
        super().close(outcome)
