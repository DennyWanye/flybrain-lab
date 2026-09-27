from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import hashlib
import json
import math
import numpy as np

from ..sdk.codec import TelloCommand, encode_command

NAMES = ['goal_forward','goal_left','goal_up','velocity_forward','velocity_left','velocity_up',
         'yaw_sin','yaw_cos','height','battery','pose_age','state_age','pose_valid','velocity_valid',
         'state_valid','previous_duration','remaining_time']+[f'previous_action_{i}' for i in range(9)]
ACTIONS = [('stop',()),('forward',(20,)),('back',(20,)),('left',(20,)),('right',(20,)),
           ('up',(20,)),('down',(20,)),('cw',(30,)),('ccw',(30,))]
SPEC = {'observation':'tellosim.observation26/2.0','names':NAMES,'encoder':'signed8_pairs_plus18/1.0',
        'action_schema':'sdk9-v1','actions':[encode_command(TelloCommand(v,a)) for v,a in ACTIONS],
        'physics_hz':120,'base_ticks':12,'neural_substeps':4,'lif_decay_dt_s':.020,
        'gamma_base':.995,'lambda_option':.95,'task_deadline_s':60,'success_hold_s':2,
        'target_radius_m':.2,'success_speed_mps':.08,'success_height_error_m':.1,
        'input_source':'external_pose_mock_and_simulated_sdk_state','frame_id':'room_map',
        'controller':'VisualSession/1.0','trainable':'actor_critic_readout_only','curriculum':'C0',
        'pose_loss_terminal_age_s':.5,'stop_min_s':.5}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class Encoder34:
    def __init__(self, profile='legacy'):
        if profile not in ('legacy','balanced_rate_v2'):raise ValueError('unknown encoder profile')
        self.profile=profile

    def __call__(self, observations):
        obs=np.asarray(observations,dtype=np.float32)
        if obs.ndim!=2 or obs.shape[1]!=26 or not np.isfinite(obs).all():
            raise ValueError('expected finite [batch,26] observations')
        pairs=np.stack((np.maximum(obs[:,:8],0),np.maximum(-obs[:,:8],0)),axis=-1).reshape(len(obs),16)
        encoded=np.concatenate((pairs,obs[:,8:]),axis=1).astype(np.float32)
        if self.profile=='balanced_rate_v2':
            # Baseline firing avoids losing small signed signals below the LIF
            # spike threshold. Goal channels use a 1 m rather than 6 m scale.
            gains=np.ones(34,np.float32);gains[:6]=[6,6,6,6,3,3]
            encoded=.12+.88*np.clip(encoded*gains,0,1)
        return encoded.astype(np.float32)


@dataclass(frozen=True)
class SensorObservation:
    sample_id: int
    sample_tick: int
    received_tick: int
    frame_id: str
    position_m: tuple | None
    yaw_rad: float | None
    velocity_mps: tuple | None
    height_m: float | None
    battery_fraction: float | None
    pose_age_s: float
    state_age_s: float

    def vector(self, goal, previous_action, previous_duration, remaining):
        if self.frame_id != 'room_map':
            raise ValueError('pose frame mismatch')
        x=np.zeros(26,np.float32)
        pose_valid=self.position_m is not None and self.yaw_rad is not None and self.pose_age_s<=.5
        velocity_valid=pose_valid and self.velocity_mps is not None
        state_valid=self.height_m is not None and self.battery_fraction is not None and self.state_age_s<=.5
        if pose_valid:
            yaw=self.yaw_rad;c,s=math.cos(yaw),math.sin(yaw)
            rotation=np.asarray([[c,s,0],[-s,c,0],[0,0,1]])
            x[:3]=np.clip(rotation@(np.asarray(goal)-self.position_m)/[6,6,3],-1,1)
            x[6:8]=[s,c]
            if velocity_valid:x[3:6]=np.clip(rotation@self.velocity_mps,-1,1)
        if state_valid:x[8:10]=[np.clip(self.height_m/3,0,1),np.clip(self.battery_fraction,0,1)]
        x[10:17]=[min(self.pose_age_s,1),min(self.state_age_s,1),pose_valid,velocity_valid,state_valid,
                  min(previous_duration/10,1),np.clip(remaining,0,1)]
        if previous_action is not None:
            if not 0<=previous_action<9:raise ValueError('invalid previous action')
            x[17+previous_action]=1
        if not np.isfinite(x).all():raise ValueError('nonfinite observation')
        return x


def action_mask(observation, curriculum='C0'):
    if curriculum not in ('C0','C0-near-x','C0-near-xy','C1'):raise ValueError('unknown curriculum')
    mask=np.zeros(9,bool);mask[0]=True
    if observation[12] and observation[14]:mask[1:3 if curriculum=='C0-near-x' else 5]=True
    if curriculum=='C1' and observation[12] and observation[14]:mask[7:9]=True
    return mask


class Device(Protocol):
    """Transfer boundary. Simulation tick advancement/reset is NOT this API."""
    def submit(self, command:TelloCommand, request_id:str)->dict: ...
    def poll(self, operation_id:str)->dict: ...
    def latest_observation(self)->SensorObservation: ...
    def protective_stop(self, request_id:str)->dict: ...


class RealDeviceDisabled:
    def __init__(self,*args,**kwargs):
        raise RuntimeError('Real flight is disabled. A validated SDK transport and external pose provider are required.')
