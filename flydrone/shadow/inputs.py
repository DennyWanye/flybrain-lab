"""Strict SI, z-up, common monotonic-clock observation boundary.

The upstream acquisition adapter must synchronize timestamps and resample at
10 Hz. This module neither estimates missing pose nor substitutes simulator truth.
"""
from dataclasses import dataclass
import math
import numpy as np
from flydrone.tellosim.training.contracts import SensorObservation, action_mask

SCHEMA = "flybrain.shadow.sample/1"
SKILLS = ("altitude", "navigation", "heading")


def number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: finite number required")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name}: out of range")
    return value


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: nonnegative integer required")
    return value


def vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name}: three SI coordinates required")
    return np.asarray([number(v, name) for v in value], dtype=float)


def fields(value, required, name):
    if not isinstance(value, dict) or set(value) != set(required.split()):
        raise ValueError(f"{name}: unexpected or missing fields")


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name}: boolean required")
    return value


@dataclass(frozen=True)
class Profile:
    source_id: str
    provenance: str
    source_frame: str
    yaw_to_room_rad: float = 0.
    translation_m: tuple = (0., 0., 0.)
    max_age_s: float = .2
    clock_uncertainty_s: float = .02

    def __post_init__(self):
        for name in ('source_id', 'source_frame'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name}: nonempty string required')
        if self.provenance not in ('synthetic_fixture', 'simulated_observation', 'real_recording'):
            raise ValueError('explicit provenance required')
        number(self.yaw_to_room_rad, 'yaw_to_room_rad', -math.pi, math.pi)
        vector(self.translation_m, 'translation_m')
        number(self.max_age_s, 'max_age_s', .001, .5)
        number(self.clock_uncertainty_s, 'clock_uncertainty_s', 0., .02)

    def rotation(self):
        c, s = math.cos(self.yaw_to_room_rad), math.sin(self.yaw_to_room_rad)
        return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


@dataclass
class Prepared:
    sample_id: int
    time_ns: int
    phase: str
    sensor: SensorObservation
    observation: np.ndarray
    mask: np.ndarray
    blocked_reasons: list


class ObservationBridge:
    def __init__(self, profile):
        self.profile = profile
        self.last_seq = None
        self.last_time = None
        self.last_capture = {}

    def prepare(self, packet):
        fields(packet, 'schema seq time_ns pose state context', 'sample')
        if packet['schema'] != SCHEMA:
            raise ValueError('unsupported sample schema')
        seq = integer(packet['seq'], 'seq')
        now = integer(packet['time_ns'], 'time_ns')
        if self.last_seq is not None:
            if seq != self.last_seq + 1:
                raise ValueError('duplicate, reordered or missing sample; start a new session')
            if now - self.last_time != 100_000_000:
                raise ValueError('10Hz clock discontinuity; start a new session')
        reasons = []
        ages = {}
        captures = {}
        for name in ('pose', 'state'):
            item = packet[name]
            extra = ('frame_id position_m velocity_mps yaw_rad' if name == 'pose'
                     else 'height_m battery_fraction airborne')
            fields(item, 'captured_ns received_ns valid ' + extra, name)
            valid = boolean(item['valid'], name + '.valid')
            captured = integer(item['captured_ns'], name + '.captured_ns')
            received = integer(item['received_ns'], name + '.received_ns')
            if not captured <= received <= now:
                raise ValueError(name + ': future or reversed timestamps')
            if captured < self.last_capture.get(name, -1):
                raise ValueError(name + ': reordered source capture')
            captures[name] = captured
            ages[name] = (now - captured) / 1e9
            if not valid:
                reasons.append(name + '_invalid')
            if ages[name] + self.profile.clock_uncertainty_s > self.profile.max_age_s + 1e-12:
                reasons.append(name + '_stale')
        p, s, ctx = packet['pose'], packet['state'], packet['context']
        if p['frame_id'] != self.profile.source_frame:
            raise ValueError('pose frame mismatch')
        pos = None if p['position_m'] is None else vector(p['position_m'], 'position_m')
        vel = None if p['velocity_mps'] is None else vector(p['velocity_mps'], 'velocity_mps')
        yaw = None if p['yaw_rad'] is None else number(p['yaw_rad'], 'yaw_rad', -math.pi, math.pi)
        if p['valid'] and (pos is None or vel is None or yaw is None):
            raise ValueError('valid pose must include position, velocity and yaw')
        height = None if s['height_m'] is None else number(s['height_m'], 'height_m', 0., 100.)
        battery = None if s['battery_fraction'] is None else number(s['battery_fraction'], 'battery_fraction', 0., 1.)
        airborne = boolean(s['airborne'], 'airborne')
        if s['valid'] and (height is None or battery is None):
            raise ValueError('valid state requires measured height and battery')
        if not airborne:
            reasons.append('not_airborne')
        fields(ctx, 'phase goal_m target_yaw_rad previous_action previous_duration_s remaining_s operation_status', 'context')
        phase = ctx['phase']
        if phase not in SKILLS:
            raise ValueError('unknown phase')
        goal = vector(ctx['goal_m'], 'goal_m')
        number(float(goal[2]), 'goal_z', .25, 1.75)
        target_yaw = number(ctx['target_yaw_rad'], 'target_yaw_rad', -math.pi, math.pi)
        previous = ctx['previous_action']
        if previous is not None and (type(previous) is not int or not 0 <= previous < 9):
            raise ValueError('previous_action: recorded SDK9 action required')
        duration = number(ctx['previous_duration_s'], 'previous_duration_s', 0., 180.)
        remaining = number(ctx['remaining_s'], 'remaining_s', 0., 60.)
        if remaining == 0:
            reasons.append('phase_deadline')
        if ctx['operation_status'] not in ('idle', 'busy', 'unknown', 'error'):
            raise ValueError('unknown operation status')
        if ctx['operation_status'] != 'idle':
            reasons.append('operation_' + ctx['operation_status'])
        rotation = self.profile.rotation()
        pose_ok = not any(r.startswith('pose_') for r in reasons)
        state_ok = not any(r.startswith('state_') for r in reasons)
        room_yaw = None if yaw is None else math.atan2(math.sin(yaw + self.profile.yaw_to_room_rad), math.cos(yaw + self.profile.yaw_to_room_rad))
        sensor = SensorObservation(seq, round(p['captured_ns'] * 120 / 1e9), round(now * 120 / 1e9),
            'room_map', tuple(rotation @ pos + self.profile.translation_m) if pose_ok else None,
            room_yaw if pose_ok else None, tuple(rotation @ vel) if pose_ok else None,
            height if state_ok else None, battery if state_ok else None, ages['pose'], ages['state'])
        observation = sensor.vector(goal, previous, duration, remaining / 60.)
        mask = np.zeros(9, dtype=bool)
        mask[0] = True
        if phase == 'heading':
            if observation[12]:
                error = target_yaw - sensor.yaw_rad
                observation[6:8] = [math.sin(error), math.cos(error)]
            if observation[12] and observation[14]:
                mask[7:9] = True
        elif phase == 'navigation':
            mask = action_mask(observation, 'C1')
        elif observation[12] and observation[14]:
            mask[5] = sensor.position_m[2] < 2.1
            mask[6] = sensor.position_m[2] > .35
        if reasons:
            mask[:] = False
            mask[0] = True
        # Commit chronology only after all structural validation succeeds.
        self.last_seq, self.last_time = seq, now
        self.last_capture = captures
        return Prepared(seq, now, phase, sensor, observation, mask, reasons)
