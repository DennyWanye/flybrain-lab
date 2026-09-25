from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum


class FlightAction(IntEnum):
    HOLD = 0
    POS_X = 1
    NEG_X = 2
    POS_Y = 3
    NEG_Y = 4


ACTION_SCHEMA = {action.name: int(action) for action in FlightAction}
OBSERVATION_SCHEMA = [
    "goal_dx_over_4m", "goal_dy_over_4m", "vx_over_0_3mps", "vy_over_0_3mps",
    "x_over_2m", "y_over_2m", "dwell_over_2s", "remaining_over_60s",
]


@dataclass(frozen=True)
class FlightConfig:
    id: str = "FlyToTarget2D-v1"
    curriculum: str = "C1"
    room_half_extent_m: float = 2.0
    body_radius_m: float = 0.15
    sample_half_extent_m: float = 1.2
    start_goal_min_m: float = 0.8
    start_goal_max_m: float = 2.5
    height_m: float = 1.0
    yaw_rad: float = 0.0
    physics_dt_s: float = 0.02
    action_duration_s: float = 1.0
    waypoint_step_m: float = 0.2
    speed_max_mps: float = 0.30
    accel_max_mps2: float = 0.80
    velocity_tau_s: float = 0.20
    position_kp: float = 2.0
    deadline_s: float = 60.0
    goal_radius_m: float = 0.20
    settle_speed_mps: float = 0.08
    dwell_s: float = 2.0

    def __post_init__(self) -> None:
        if self.curriculum not in {"C0", "C1"}:
            raise ValueError("curriculum must be C0 or C1")
        if self.physics_dt_s <= 0 or self.action_duration_s <= 0:
            raise ValueError("time steps must be positive")
        if self.dwell_s <= 0 or self.speed_max_mps <= 0:
            raise ValueError("physical limits must be positive")

    @property
    def safe_half_extent_m(self) -> float:
        return self.room_half_extent_m - self.body_radius_m

    @property
    def action_ticks(self) -> int:
        return round(self.action_duration_s / self.physics_dt_s)

    @property
    def deadline_ticks(self) -> int:
        return round(self.deadline_s / self.physics_dt_s)

    @property
    def dwell_ticks_required(self) -> int:
        return round(self.dwell_s / self.physics_dt_s)

    def to_dict(self) -> dict:
        return asdict(self)
