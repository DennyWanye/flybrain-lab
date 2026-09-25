from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .contracts import FlightAction, FlightConfig
from .dynamics import integrate, local_waypoint


@dataclass
class EnvState:
    position_xy: np.ndarray
    velocity_xy: np.ndarray
    goal_xy: np.ndarray
    physics_tick: int = 0
    dwell_ticks: int = 0
    episode_return: float = 0.0
    episode_seed: int | None = None
    ended: bool = False
    end_reason: str | None = None


class FlyToTargetEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, config: FlightConfig | None = None):
        super().__init__()
        self.config = config or FlightConfig()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.state: EnvState | None = None
        self.case_id: str | None = None

    def _sample_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.config.curriculum == "C0":
            position = np.zeros(2, dtype=np.float64)
            goal = np.array([[.8, 0.], [-.8, 0.], [0., .8], [0., -.8]], dtype=np.float64)[
                int(self.np_random.integers(4))]
            return position, goal
        for _ in range(10000):
            position = self.np_random.uniform(-self.config.sample_half_extent_m,
                                              self.config.sample_half_extent_m, 2)
            goal = self.np_random.uniform(-self.config.sample_half_extent_m,
                                          self.config.sample_half_extent_m, 2)
            distance = float(np.linalg.norm(goal - position))
            if self.config.start_goal_min_m <= distance <= self.config.start_goal_max_m:
                return position, goal
        raise RuntimeError("failed to sample valid C1 start and goal")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}
        initial = options.get("initial_state")
        if initial is None:
            position, goal = self._sample_state()
        else:
            position = np.asarray(initial["position_xy"], dtype=np.float64)
            goal = np.asarray(initial["goal_xy"], dtype=np.float64)
        velocity = np.asarray(options.get("velocity_xy", [0., 0.]), dtype=np.float64)
        if position.shape != (2,) or goal.shape != (2,) or velocity.shape != (2,):
            raise ValueError("position_xy, goal_xy, and velocity_xy must be length two")
        self.state = EnvState(position, velocity, goal, episode_seed=seed)
        self.case_id = options.get("case_id")
        return self._observation(), self._info({}, 0, False)

    def _require_state(self) -> EnvState:
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        return self.state

    def _observation(self) -> np.ndarray:
        state = self._require_state()
        remaining = max(0., self.config.deadline_s - state.physics_tick * self.config.physics_dt_s)
        raw = np.array([
            *(state.goal_xy - state.position_xy) / 4., *state.velocity_xy / self.config.speed_max_mps,
            *state.position_xy / self.config.room_half_extent_m,
            state.dwell_ticks * self.config.physics_dt_s / self.config.dwell_s,
            remaining / self.config.deadline_s,
        ], dtype=np.float64)
        if not np.isfinite(raw).all():
            raise FloatingPointError("nonfinite physical state")
        return np.clip(raw, -1., 1.).astype(np.float32)

    def _info(self, reward_parts: dict[str, float], elapsed_ticks: int, terminated: bool) -> dict:
        state = self._require_state()
        return {
            "case_id": self.case_id, "position_xy_m": state.position_xy.copy(),
            "velocity_xy_mps": state.velocity_xy.copy(), "goal_xy_m": state.goal_xy.copy(),
            "distance_m": float(np.linalg.norm(state.goal_xy - state.position_xy)),
            "speed_mps": float(np.linalg.norm(state.velocity_xy)),
            "dwell_seconds": state.dwell_ticks * self.config.physics_dt_s,
            "elapsed_physics_ticks": elapsed_ticks, "physics_tick": state.physics_tick,
            "end_reason": state.end_reason, "reward_parts": reward_parts, "terminated": terminated,
        }

    def step(self, action: int):
        state = self._require_state()
        if state.ended:
            raise RuntimeError("step called after episode end; call reset")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        action = FlightAction(action)
        waypoint = local_waypoint(state.position_xy, action)
        distance_before = float(np.linalg.norm(state.goal_xy - state.position_xy))
        elapsed = 0
        terminated = False
        for _ in range(self.config.action_ticks):
            state.position_xy, state.velocity_xy, _ = integrate(
                state.position_xy, state.velocity_xy, waypoint, self.config)
            state.physics_tick += 1
            elapsed += 1
            distance = float(np.linalg.norm(state.goal_xy - state.position_xy))
            speed = float(np.linalg.norm(state.velocity_xy))
            if np.max(np.abs(state.position_xy)) >= self.config.safe_half_extent_m:
                state.end_reason, terminated = "boundary", True
            elif distance <= self.config.goal_radius_m and speed <= self.config.settle_speed_mps:
                state.dwell_ticks += 1
                if state.dwell_ticks >= self.config.dwell_ticks_required:
                    state.end_reason, terminated = "success", True
            else:
                state.dwell_ticks = 0
            if not terminated and state.physics_tick >= self.config.deadline_ticks:
                state.end_reason, terminated = "deadline", True
            if terminated:
                state.ended = True
                break
        distance_after = float(np.linalg.norm(state.goal_xy - state.position_xy))
        parts = {
            "progress": 2.0 * (distance_before - distance_after), "step_cost": -0.02,
            "success_bonus": 10.0 if state.end_reason == "success" else 0.0,
            "boundary_cost": -10.0 if state.end_reason == "boundary" else 0.0,
            "deadline_cost": -2.0 if state.end_reason == "deadline" else 0.0,
        }
        reward = float(sum(parts.values()))
        state.episode_return += reward
        return self._observation(), reward, terminated, False, self._info(parts, elapsed, terminated)
