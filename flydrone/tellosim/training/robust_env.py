"""J2 mild disturbances, added without changing frozen J1 physics or acceptance.

Forces enter MuJoCo before every integration step. No position, velocity,
command target or neural state is overwritten by the disturbance generator.
"""
import math
import numpy as np
from .env import TrainingEnv
from .heading import HeadingEnv
from .joint import JointEnv

PROFILES = {
    'clean': {'noise': 0., 'delay': 0, 'force_n': 0., 'yaw_torque_nm': 0.},
    'pose': {'noise': .002, 'delay': 1, 'force_n': 0., 'yaw_torque_nm': 0.},
    'force': {'noise': 0., 'delay': 0, 'force_n': .006, 'yaw_torque_nm': .00003},
    'combined': {'noise': .002, 'delay': 1, 'force_n': .006, 'yaw_torque_nm': .00003},
}
DISTURBANCE_SPEC = {
    'format': 'tellosim.mild_disturbance/1', 'profiles': PROFILES,
    'pose_delay_unit_s': .1, 'position_noise_distribution': 'independent Gaussian per axis',
    'yaw_noise_sigma_rad': 'position_noise_sigma_m * 0.2 (engineering sensor model)',
    'force_frame': 'world XY; torque world Z', 'pulse_start_s': 5.,
    'pulse_period_s': 12., 'pulse_duration_s': 2.,
    'direction': 'seeded initial angle, rotated by 1.7 radians per pulse',
    'scope': 'mild synthetic disturbances, not aircraft-calibrated wind',
}

def with_profile(case, profile):
    if profile not in PROFILES: raise ValueError('unknown disturbance profile')
    return {**case, **{k: PROFILES[profile][k] for k in ('noise', 'delay')},
            'dropout': 0., 'disturbance_profile': profile}


def disturbance_wrench(seed, elapsed_s, profile):
    if profile not in PROFILES or not math.isfinite(elapsed_s):
        raise ValueError('invalid disturbance input')
    spec = PROFILES[profile]
    phase = elapsed_s - DISTURBANCE_SPEC['pulse_start_s']
    force = np.zeros(3); torque = np.zeros(3)
    if phase >= 0 and phase % 12. < 2.:
        pulse = int(phase // 12.)
        angle = (int(seed) % 997) / 997 * 2 * math.pi + 1.7 * pulse
        force[:2] = spec['force_n'] * np.array([math.cos(angle), math.sin(angle)])
        torque[2] = spec['yaw_torque_nm'] * (1 if (int(seed) + pulse) % 2 else -1)
    return force, torque


class DisturbanceMixin:
    def __init__(self, root, case, agent, **kwargs):
        self.disturbance_profile = case.get('disturbance_profile', 'clean')
        if self.disturbance_profile not in PROFILES: raise ValueError('unknown disturbance profile')
        expected = with_profile(case, self.disturbance_profile)
        if any(case.get(k, 0) != expected[k] for k in ('noise', 'delay', 'dropout')):
            raise ValueError('case does not match fixed disturbance profile')
        self.disturbance_ticks = 0; self.disturbance_impulse_ns = np.zeros(3)
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
                self.disturbance_impulse_ns += force * world.config.dt
                self.disturbance_angular_impulse_nms += torque * world.config.dt
                self.disturbance_abs_impulse_ns += float(np.linalg.norm(force)) * world.config.dt
        world._apply_controller = controller_with_disturbance
        self.publish()

    def disturbance_evidence(self):
        return {'profile': self.disturbance_profile, 'applied_physics_ticks': self.disturbance_ticks,
                'impulse_world_ns': self.disturbance_impulse_ns.tolist(),
                'angular_impulse_world_nms': self.disturbance_angular_impulse_nms.tolist(),
                'absolute_impulse_ns': self.disturbance_abs_impulse_ns}

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


class RobustNavigationEnv(DisturbanceMixin, TrainingEnv): pass
class RobustHeadingEnv(DisturbanceMixin, HeadingEnv): pass
class RobustJointEnv(DisturbanceMixin, JointEnv): pass
