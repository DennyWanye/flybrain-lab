"""V1R independent altitude skill; original physics and legacy skills stay frozen."""
from pathlib import Path
from dataclasses import asdict
import math,json,hashlib
import numpy as np
import torch
from .env import TrainingEnv
from .heading import HeadingEnv,wrap_angle
from .contracts import SPEC,ACTIONS,digest,Encoder34
from .runtime import checkpoint_contract
from .robust_env import PROFILES as LEGACY_PROFILES,with_profile,disturbance_wrench as horizontal_wrench
from ..sdk.codec import TelloCommand
from ..visual import atomic_json

ALTITUDE_SPEC={**SPEC,'observation':'tellosim.altitude_refined_observation26/1.0','curriculum':'V1R',
    'task':'in_place_target_altitude_and_hold','allowed_actions':[0,5,6],
    'height_tolerance_m':.1,'vertical_speed_tolerance_mps':.08,'horizontal_speed_tolerance_mps':.08,
    'horizontal_tolerance_m':.2,'yaw_drift_tolerance_rad':math.radians(16),'yaw_rate_tolerance_rad_s':.08,
    'minimum_episode_s':8.,'task_deadline_s':60.,'success_hold_s':2.,
    'target_height_range_m':[.25,1.75],'bootstrap_height_m':1.,
    'safety_mask':'measured pose: down only above .35m; up only below 2.1m; independent of target',
    'precision_note':'20cm SDK increments; true height error <=10cm AND vertical speed <=.08m/s held for2s; observe at least8s',
    'encoder':'altitude_multiscale34/1.0','stop_min_s':2.,
    'trainable':'independent altitude actor readout; original MaleCNS and J2R navigation/heading frozen'}
PROFILES={k:{**v,'vertical_force_n':v['force_n']} for k,v in LEGACY_PROFILES.items()}
DISTURBANCE_SPEC={'format':'tellosim.altitude_disturbance/1','profiles':PROFILES,'pulse_start_s':5.,'pulse_period_s':12.,'pulse_duration_s':2.,
    'force_frame':'world XYZ; horizontal .006N plus alternating vertical +/-.006N; yaw torque .00003Nm',
    'pose_delay_unit_s':.1,'noise':'Gaussian position sigma .002m, yaw sigma .0004rad in pose/combined',
    'scope':'synthetic engineering disturbances, not aircraft-calibrated wind'}

ENCODER_SPEC = {
    'kind': 'altitude_multiscale34/1.0',
    'altitude_scales_m': [.08, .24, .72],
    'formula': 'signed tanh(error_m / scale), positive/negative pair per scale',
    'channels_0_to_5': 'altitude error population (replaces XY and Z goal pairs for in-place altitude only)',
    'other_channels': 'unchanged rate_offset34/2.0',
    'offset': .12, 'amplitude': .88,
    'policy_input': '128 frozen downstream MaleCNS v/trace features only',
}

class AltitudeEncoder34(Encoder34):
    """Smooth multi-scale sensory encoding; no action or tolerance decision."""
    def __init__(self):
        super().__init__('balanced_rate_v2')

    def __call__(self, observations):
        obs = np.asarray(observations, dtype=np.float32)
        encoded = super().__call__(obs)
        error = obs[:, 2] * 3.
        for i, scale in enumerate(ENCODER_SPEC['altitude_scales_m']):
            signed = np.tanh(error / scale)
            encoded[:, 2*i] = .12 + .88 * np.maximum(signed, 0)
            encoded[:, 2*i+1] = .12 + .88 * np.maximum(-signed, 0)
        return encoded


def disturbance_wrench(seed,elapsed_s,profile):
    force,torque=horizontal_wrench(seed,elapsed_s,profile)
    phase=elapsed_s-5.
    if phase>=0 and phase%12.<2.:
        force[2]=PROFILES[profile]['vertical_force_n']*(1 if (seed+int(phase//12))%2 else -1)
    return force,torque

def altitude_case(seed,case_id=None,direction=None,near=False):
    rng=np.random.default_rng(seed);xy=rng.uniform(-1.5,1.5,2);yaw=float(rng.uniform(-math.pi,math.pi))
    sign=int(rng.choice([-1,1])) if direction is None else direction
    if sign not in (-1,1):raise ValueError('altitude direction must be signed')
    delta=float(rng.uniform(.04,.24) if near else rng.uniform(.25,.75))
    return dict(case_id=case_id or f'v1r-train-{seed}',seed=seed,curriculum='V1R',
        start=[float(xy[0]),float(xy[1]),1.],goal=[float(xy[0]),float(xy[1]),1.+sign*delta],
        initial_yaw_rad=yaw,noise=0.,delay=0,dropout=0.,reply_loss=0.)

def altitude_mask(observation,sensor):
    mask=np.zeros(9,bool);mask[0]=True
    if observation[12] and observation[14] and sensor.position_m is not None:
        z=sensor.position_m[2];mask[5]=z<2.1;mask[6]=z>.35
    return mask

def teacher(observation,mask,first=False):
    if first or not observation[12]:return 0
    error=float(observation[2])*3
    if abs(error)<=.1:return 0
    action=5 if error>0 else 6
    return action if mask[action] else 0

class AltitudeBase(HeadingEnv):
    def __init__(self,root,case,agent,**kwargs):
        if case.get('curriculum')!='V1R':raise ValueError('altitude task requires V1R case')
        start=np.asarray(case['start'],float);goal=np.asarray(case['goal'],float)
        if start.shape!=(3,) or goal.shape!=(3,) or not np.isfinite(np.r_[start,goal]).all():raise ValueError('invalid altitude case')
        if not np.array_equal(start[:2],goal[:2]) or start[2]!=1. or not .25<=goal[2]<=1.75:raise ValueError('invalid V1R height or horizontal goal')
        self.target_yaw=float(case['initial_yaw_rad'])
        if not math.isfinite(self.target_yaw):raise ValueError('nonfinite yaw')
        TrainingEnv.__init__(self,root,case,agent,**kwargs)
        self.session.recording.save()
    def potential(self):
        return -abs(float(self.session.world.position[2])-self.case['goal'][2])-.2*float(np.linalg.norm(self.session.world.position[:2]-self.case['goal'][:2]))
    def prepare_observation(self):
        sensor=self.device.latest_observation();elapsed=(self.session.tick-self.start_tick)/120
        self.observation=sensor.vector(self.case['goal'],self.previous_action,self.previous_duration,1-elapsed/60)
        self.mask=altitude_mask(self.observation,sensor)
    def stable_state(self,status,action):
        world=self.session.world;goal=np.asarray(self.case['goal'])
        return bool(np.linalg.norm(world.position[:2]-goal[:2])<=.2 and abs(world.position[2]-goal[2])<=.1
            and np.linalg.norm(world.velocity[:2])<=.08 and abs(world.velocity[2])<=.08
            and abs(wrap_angle(self.target_yaw-world.yaw_rad))<=ALTITUDE_SPEC['yaw_drift_tolerance_rad']
            and abs(float(world.data.qvel[5]))<=.08
            and (status['client']=='ack_ok' or (action==0 and status['client']=='sent')))
    def publish(self,result_only=False):
        s=self.session
        s.scene.update(task_kind='altitude_hold',height_tolerance_m=.1,vertical_speed_tolerance_mps=.08)
        s.recording.manifest.update(observation_schema=ALTITUDE_SPEC['observation'],observation_names=ALTITUDE_SPEC['names'],task_spec=ALTITUDE_SPEC,scene=s.scene,curriculum='V1R',training_method=getattr(self.agent,'training_method','altitude diagnostic'),
            observation_note='Measured 26-channel pose/state input, altitude error channel2; policy sees frozen MaleCNS features only')
        s.recording.manifest['altitude_checkpoint']={'path':getattr(self.agent,'checkpoint_path',None),'sha256':getattr(self.agent,'checkpoint_sha256',None)}
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
            'observation_valid':valid,'observation_schema':ALTITUDE_SPEC['observation'],'policy':self.last_policy,
            'brain':brain,'neural_tick':s.tick,'neural_substep':3,'brain_tick':self.agent.brain_tick,
            'reward':self.last_reward,'stable_hold_s':self.hold,'operation':self.device.observed_operation(),
            'finished':self.terminated,'trail':list(s.trail),'history':list(s.history)[-20:]}
        frame['altitude']={'target_z_m':self.case['goal'][2],'measured_z_m':sensor.position_m[2] if sensor.position_m else None,'error_m':self.case['goal'][2]-sensor.position_m[2] if sensor.position_m else None,'vertical_speed_mps':sensor.velocity_mps[2] if sensor.velocity_mps else None,'tolerance_m':.1,'speed_tolerance_mps':.08}
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
            stable=self.stable_state(status,action)
            self.hold=self.hold+.1 if stable else 0.
            self.invalid_age=self.invalid_age+.1 if not self.observation[12] else 0.
            self.reason=('collision' if self.session.collision_latched else
                         'command_unknown' if status['client']=='unknown_execution' else
                         'command_error' if status['client']=='ack_error' else
                         'localization_lost' if self.invalid_age>=.5-1e-8 else
                         'success' if self.hold>=2-1e-8 and elapsed>=ALTITUDE_SPEC['minimum_episode_s']-1e-8 else
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

class AltitudeDisturbanceMixin:
    def __init__(self, root, case, agent, **kwargs):
        self.disturbance_profile = case.get('disturbance_profile', 'clean')
        if self.disturbance_profile not in PROFILES: raise ValueError('unknown disturbance profile')
        expected = with_profile(case, self.disturbance_profile)
        if any(case.get(k, 0) != expected[k] for k in ('noise', 'delay', 'dropout')):
            raise ValueError('case does not match fixed disturbance profile')
        self.disturbance_vertical_abs_impulse_ns = 0.; self.disturbance_ticks = 0; self.disturbance_impulse_ns = np.zeros(3)
        self.disturbance_angular_impulse_nms = np.zeros(3)
        self.disturbance_abs_impulse_ns = 0.; self._original_controller = None
        super().__init__(root, case, agent, **kwargs)
        world = self.session.world
        self._original_controller = world._apply_controller
        origin = float(world.data.time)
        def controller_with_disturbance():
            self._original_controller()
            force, torque = disturbance_wrench(case['seed'], float(world.data.time) - origin, self.disturbance_profile)
            world.data.xfrc_applied[world.body_id, :3] += force
            world.data.xfrc_applied[world.body_id, 3:] += torque
            world.last_control.update(external_force_world_n=force.tolist(), external_torque_world_nm=torque.tolist())
            if np.any(force) or np.any(torque):
                self.disturbance_ticks += 1
                self.disturbance_vertical_abs_impulse_ns += abs(float(force[2])) * world.config.dt
                self.disturbance_impulse_ns += force * world.config.dt
                self.disturbance_angular_impulse_nms += torque * world.config.dt
                self.disturbance_abs_impulse_ns += float(np.linalg.norm(force)) * world.config.dt
        world._apply_controller = controller_with_disturbance
        self.publish()

    def disturbance_evidence(self):
        return {'profile': self.disturbance_profile, 'applied_physics_ticks': self.disturbance_ticks,
                'impulse_world_ns': self.disturbance_impulse_ns.tolist(),
                'angular_impulse_world_nms': self.disturbance_angular_impulse_nms.tolist(),
                'absolute_impulse_ns': self.disturbance_abs_impulse_ns,'absolute_vertical_impulse_ns': self.disturbance_vertical_abs_impulse_ns}

    def publish(self, result_only=False):
        self.session.scene.update(disturbance_profile=self.disturbance_profile,
                                  disturbance_spec=DISTURBANCE_SPEC)
        self.session.recording.manifest.update(disturbance_spec=DISTURBANCE_SPEC,
                                               disturbance_evidence=self.disturbance_evidence())
        super().publish(result_only)

    def close(self, outcome=None):
        try: super().close(outcome)
        finally:
            if self._original_controller is not None:
                self.session.world._apply_controller = self._original_controller
                self._original_controller = None


class AltitudeEnv(AltitudeDisturbanceMixin,AltitudeBase):pass

def altitude_contract(pool,root):
    root=Path(root)
    base=checkpoint_contract(pool,root)
    base['encoder']=ENCODER_SPEC
    if not isinstance(pool.brain.encoder,AltitudeEncoder34):raise ValueError('V1R requires its versioned encoder')
    return dict(base=base,task_spec=ALTITUDE_SPEC,disturbance_spec=DISTURBANCE_SPEC,
        task_sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['flydrone/tellosim/training/altitude_refined.py','flydrone/tellosim/training/altitude_refined_campaign.py','flydrone/tellosim/training/stability_campaign.py','flydrone/tellosim/training/robust_env.py','flydrone/tellosim/training/heading.py','flydrone/tellosim/training/parallel.py','flydrone/tellosim/training/contracts.py']})

def save_altitude(path,pool,optimizer,root,options,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(format='tellosim.altitude_refined_readout/1',contract=altitude_contract(pool,root),policy=pool.model.state_dict(),
        optimizer=optimizer.state_dict() if optimizer else None,options=options,value_trained=False,extra=extra or {},
        restore_mode='inference or explicit stage warm start; not mid-rollout resume')
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def load_altitude(path,pool,root):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format')!='tellosim.altitude_refined_readout/1' or state['contract']!=altitude_contract(pool,root):raise ValueError('altitude checkpoint contract mismatch')
    pool.model.load_state_dict(state['policy']);pool.value_trained=False;pool.reset()
    pool.training_method=state['extra'].get('method','untrained altitude diagnostic')
    pool.checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest();pool.checkpoint_path=str(Path(path).resolve().relative_to(Path(root).resolve())) if Path(path).resolve().is_relative_to(Path(root).resolve()) else str(Path(path).resolve());return state
