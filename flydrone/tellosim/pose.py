from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PoseSnapshot:
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    yaw_rad: float
    airborne: bool
    sim_tick: int

    def observation26(self, goal_m=(0.0, 0.0, 1.0), room_half_extent_m=3.0, speed_scale_mps=1.0) -> np.ndarray:
        goal = np.asarray(goal_m, dtype=np.float64)
        pos = np.asarray(self.position_m, dtype=np.float64)
        vel = np.asarray(self.velocity_mps, dtype=np.float64)
        if goal.shape != (3,):
            raise ValueError("goal_m must have length three")
        if room_half_extent_m <= 0 or speed_scale_mps <= 0:
            raise ValueError("observation scales must be positive")
        heading = np.asarray((math.cos(self.yaw_rad), math.sin(self.yaw_rad)), dtype=np.float64)
        right = np.asarray((-heading[1], heading[0]), dtype=np.float64)
        rel = goal - pos
        horizontal = np.asarray((rel[0], rel[1]), dtype=np.float64)
        features = np.asarray((
            rel[0] / room_half_extent_m, rel[1] / room_half_extent_m, rel[2] / room_half_extent_m,
            vel[0] / speed_scale_mps, vel[1] / speed_scale_mps, vel[2] / speed_scale_mps,
            float(np.dot(horizontal, heading)) / room_half_extent_m,
            float(np.dot(horizontal, right)) / room_half_extent_m,
            math.sin(self.yaw_rad), math.cos(self.yaw_rad),
            pos[0] / room_half_extent_m, pos[1] / room_half_extent_m, pos[2] / room_half_extent_m,
            float(self.airborne), float(self.sim_tick),
        ), dtype=np.float32)
        # Keep the contract at 26 channels: the final 11 are reserved for
        # command/lifecycle and sensor extensions in later TS1 stages.
        return np.pad(features, (0, 11), constant_values=0.0)


class PoseProvider:
    def __init__(self, world, airborne_getter=None, tick_getter=None):
        self.world = world
        self.airborne_getter = airborne_getter or (lambda: False)
        self.tick_getter = tick_getter or (lambda: 0)

    def snapshot(self) -> PoseSnapshot:
        return PoseSnapshot(tuple(float(x) for x in self.world.position),
                            tuple(float(x) for x in self.world.velocity),
                            float(self.world.yaw_rad), bool(self.airborne_getter()),
                            int(self.tick_getter()))
