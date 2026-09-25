from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class WorldConfig:
    physics_hz: int = 120
    room_half_extent_m: float = 3.0
    floor_z_m: float = 0.0
    mass_kg: float = 0.1
    takeoff_height_m: float = 1.0
    max_accel_mps2: float = 0.6
    max_tilt_deg: float = 15.0
    max_thrust_weight_ratio: float = 2.0
    position_kp: float = 0.7
    velocity_kd: float = 1.8
    attitude_kp: float = 0.2
    angular_kd: float = 0.25
    yaw_kp: float = 0.8
    yaw_kd: float = 0.18
    max_yaw_accel_rad_s2: float = 1.0
    max_yaw_rate_rad_s: float = math.radians(45.0)

    @classmethod
    def from_json(cls, path: str | Path) -> "WorldConfig":
        import json

        value = json.loads(Path(path).read_text(encoding="utf-8"))
        room = value.get("room_size_m", [6.0, 6.0, 3.0])
        vehicle = value.get("vehicle", {})
        return cls(
            physics_hz=int(value.get("physics_hz", 120)),
            room_half_extent_m=float(room[0]) / 2.0,
            floor_z_m=float(value.get("floor_z_m", 0.0)),
            mass_kg=float(vehicle.get("mass_kg", 0.1)),
            takeoff_height_m=float(vehicle.get("takeoff_height_m", 1.0)),
            max_accel_mps2=float(vehicle.get("max_accel_mps2", 0.6)),
            max_tilt_deg=float(vehicle.get("max_tilt_deg", 15.0)),
            max_thrust_weight_ratio=float(vehicle.get("max_thrust_weight_ratio", 2.0)),
        )

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz


class SimWorld:
    """Headless MuJoCo surrogate for the first controller spike.

    This is intentionally not a Tello digital twin. It uses a rigid body,
    gravity, body-axis thrust, and a bounded attitude controller. Commands are
    translated into targets; qpos/qvel are only initialized during reset.
    """

    def __init__(self, config: WorldConfig | None = None):
        self.config = config or WorldConfig()
        self.model = mujoco.MjModel.from_xml_string(self._xml())
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tello")
        self.target = np.zeros(3, dtype=np.float64)
        self.target_yaw = 0.0
        self._yaw_state = 0.0
        self.reset()

    def _xml(self) -> str:
        c = self.config
        return f"""
        <mujoco model="tellosim-spike">
          <option timestep="{c.dt:.12g}" gravity="0 0 -9.81" integrator="implicitfast"/>
          <size nconmax="64" njmax="128"/>
          <worldbody>
            <geom name="floor" type="plane" size="{c.room_half_extent_m} {c.room_half_extent_m} 0.05"/>
            <body name="tello" pos="0 0 {c.takeoff_height_m}">
              <freejoint/>
              <geom name="body" type="box" size="0.1 0.1 0.045" mass="{c.mass_kg}"/>
            </body>
          </worldbody>
        </mujoco>
        """

    def reset(self, position: tuple[float, float, float] | None = None) -> None:
        position = position or (0.0, 0.0, self.config.takeoff_height_m)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.asarray(position, dtype=np.float64)
        self.data.qpos[3:7] = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self.data.qvel[:] = 0.0
        self.target = np.asarray(position, dtype=np.float64)
        self.target_yaw = 0.0
        self._yaw_state = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def position(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.data.qvel[:3].copy()

    def set_target(self, target: tuple[float, float, float]) -> None:
        self.target = np.asarray(target, dtype=np.float64)

    def set_yaw(self, yaw_rad: float) -> None:
        """Set a continuous yaw target; physics performs the rotation."""
        self.target_yaw = float(yaw_rad)

    @property
    def yaw_rad(self) -> float:
        return self._yaw_state

    def hold(self) -> None:
        self.target = self.position

    def _apply_controller(self) -> None:
        c = self.config
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        error = self.target - pos
        accel = c.position_kp * error - c.velocity_kd * vel
        horizontal = np.linalg.norm(accel[:2])
        horizontal_limit = 9.81 * math.tan(math.radians(c.max_tilt_deg))
        if horizontal > min(c.max_accel_mps2, horizontal_limit):
            accel[:2] *= min(c.max_accel_mps2, horizontal_limit) / horizontal
        accel[2] = float(np.clip(accel[2], -c.max_accel_mps2, c.max_accel_mps2))
        desired_thrust = np.asarray((accel[0], accel[1], 9.81 + accel[2])) * c.mass_kg
        thrust = float(np.linalg.norm(desired_thrust))
        thrust = float(np.clip(thrust, 0.0, c.max_thrust_weight_ratio * c.mass_kg * 9.81))

        # First-stage surrogate keeps the body level and models yaw as a
        # bounded command state. Translational motion remains MuJoCo-integrated.
        horizontal_force = np.clip(accel[:2] * c.mass_kg, -0.06, 0.06)
        self.data.xfrc_applied[self.body_id, :3] = np.asarray((horizontal_force[0], horizontal_force[1], thrust))
        self.data.xfrc_applied[self.body_id, 3:] = 0.0

    def step(self, ticks: int = 1) -> list[dict[str, float | int]]:
        trajectory: list[dict[str, float | int]] = []
        for _ in range(int(ticks)):
            self.data.xfrc_applied[:] = 0.0
            self._apply_controller()
            mujoco.mj_step(self.model, self.data)
            yaw_error = math.atan2(math.sin(self.target_yaw - self._yaw_state), math.cos(self.target_yaw - self._yaw_state))
            yaw_rate = float(np.clip(yaw_error / max(0.25, 1.0 / self.config.max_yaw_rate_rad_s),
                                     -self.config.max_yaw_rate_rad_s, self.config.max_yaw_rate_rad_s))
            self._yaw_state += yaw_rate * self.config.dt
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                raise FloatingPointError(f"MuJoCo state became non-finite at tick {self.data.time}")
            trajectory.append({
                "sim_tick": int(round(self.data.time / self.config.dt)),
                "time_s": float(self.data.time),
                "x_m": float(self.data.qpos[0]),
                "y_m": float(self.data.qpos[1]),
                "z_m": float(self.data.qpos[2]),
                "speed_mps": float(np.linalg.norm(self.data.qvel[:3])),
            })
        return trajectory
