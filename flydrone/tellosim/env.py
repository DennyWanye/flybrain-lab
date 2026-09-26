from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .pose import PoseProvider
from .sdk import TelloCommand


@dataclass(frozen=True)
class VariableDurationConfig:
    action_duration_s: tuple[float, ...] = (0.5, 1.0, 2.0)
    max_episode_steps: int = 120
    start_m: tuple[float, float, float] = (0.0, 0.0, 1.0)
    goal_m: tuple[float, float, float] = (0.8, 0.0, 1.0)
    target_radius_m: float = 0.15
    stable_hold_s: float = 0.0
    stable_speed_mps: float = 0.15


class TelloSimEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, sim, config: VariableDurationConfig | None = None):
        super().__init__()
        self.sim = sim
        self.config = config or VariableDurationConfig()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32)
        self.pose = PoseProvider(sim.world, lambda: sim.airborne, lambda: sim.sim_tick)
        self.steps = 0
        self.stable_hold_s = 0.0
        self.goal = np.asarray(self.config.goal_m, dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.sim.world.reset(position=tuple(self.config.start_m))
        self.sim.sdk_mode = True
        self.sim.airborne = False
        self.sim.sim_tick = 0
        self.steps = 0
        self.stable_hold_s = 0.0
        return self.pose.snapshot().observation26(self.goal), {"goal_m": self.goal.copy()}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        if self.steps >= self.config.max_episode_steps:
            raise RuntimeError("step called after episode end; call reset")
        duration = self.config.action_duration_s[self.steps % len(self.config.action_duration_s)]
        before = self.pose.snapshot()
        commands = (
            TelloCommand("stop"), TelloCommand("forward", (20,)),
            TelloCommand("back", (20,)), TelloCommand("left", (20,)),
            TelloCommand("right", (20,)), TelloCommand("up", (20,)),
            TelloCommand("down", (20,)), TelloCommand("cw", (30,)),
            TelloCommand("ccw", (30,)),
        )
        command = commands[int(action)]
        issued_tick = int(self.sim.sim_tick)
        command_result = {"phase": "completed", "device_execution": "completed", "raw_response": "ok"}
        if not self.sim.airborne:
            self.sim.airborne = True
        command_result = self.sim.submit(" ".join([command.verb, *(str(v) for v in command.args)]), f"env-step-{self.steps}")

        after = self.pose.snapshot()
        self.steps += 1
        distance_before = float(np.linalg.norm(np.asarray(before.position_m) - self.goal))
        distance_after = float(np.linalg.norm(np.asarray(after.position_m) - self.goal))
        progress_reward = distance_before - distance_after
        time_penalty = -0.01 * duration
        speed = float(np.linalg.norm(after.velocity_mps))
        if distance_after <= self.config.target_radius_m and speed <= self.config.stable_speed_mps:
            self.stable_hold_s += duration
        else:
            self.stable_hold_s = 0.0
        if self.config.stable_hold_s > 0:
            terminated = self.stable_hold_s >= self.config.stable_hold_s
        else:
            terminated = distance_after <= self.config.target_radius_m
        target_bonus = 1.0 if terminated else 0.0
        reward = progress_reward + time_penalty + target_bonus
        truncated = self.steps >= self.config.max_episode_steps
        return after.observation26(self.goal), float(reward), terminated, truncated, {
            "duration_s": duration, "distance_m": distance_after, "sim_tick": after.sim_tick,
            "command": command.verb, "command_args": list(command.args),
            "command_id": command_result.get("operation_id", f"env-step-{self.steps - 1}"),
            "issued_sim_tick": issued_tick, "started_sim_tick": issued_tick,
            "completed_sim_tick": int(after.sim_tick), "command_result": command_result,
            "speed_mps": speed, "stable_hold_s": self.stable_hold_s, "success": terminated,
            "reward_components": {"progress_reward": float(progress_reward), "time_penalty": float(time_penalty), "target_bonus": float(target_bonus)},
        }
