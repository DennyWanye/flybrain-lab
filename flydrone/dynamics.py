from __future__ import annotations

import numpy as np

from .contracts import FlightAction, FlightConfig


ACTION_DELTA = {
    FlightAction.HOLD: np.array([0.0, 0.0], dtype=np.float64),
    FlightAction.POS_X: np.array([1.0, 0.0], dtype=np.float64),
    FlightAction.NEG_X: np.array([-1.0, 0.0], dtype=np.float64),
    FlightAction.POS_Y: np.array([0.0, 1.0], dtype=np.float64),
    FlightAction.NEG_Y: np.array([0.0, -1.0], dtype=np.float64),
}


def clip_norm(value: np.ndarray, limit: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value * min(1.0, limit / max(norm, 1e-12))


def local_waypoint(position_xy: np.ndarray, action: FlightAction) -> np.ndarray:
    return np.asarray(position_xy, dtype=np.float64) + .2 * ACTION_DELTA[action]


def integrate(position_xy: np.ndarray, velocity_xy: np.ndarray, waypoint_xy: np.ndarray,
              config: FlightConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One semi-implicit Euler tick, preserving velocity across actions."""
    v_set = clip_norm(config.position_kp * (waypoint_xy - position_xy), config.speed_max_mps)
    acceleration = clip_norm((v_set - velocity_xy) / config.velocity_tau_s, config.accel_max_mps2)
    v_new = clip_norm(velocity_xy + acceleration * config.physics_dt_s, config.speed_max_mps)
    p_new = position_xy + v_new * config.physics_dt_s
    return p_new, v_new, acceleration
