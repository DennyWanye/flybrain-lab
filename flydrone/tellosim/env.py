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
    goal_m: tuple[float, float, float] = (0.8, 0.0, 1.0)


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
        self.goal = np.asarray(self.config.goal_m, dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.sim.world.reset()
        self.sim.airborne = False
        self.sim.sim_tick = 0
        self.steps = 0
        return self.pose.snapshot().observation26(self.goal), {"goal_m": self.goal.copy()}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        if self.steps >= self.config.max_episode_steps:
            raise RuntimeError("step called after episode end; call reset")
        duration = self.config.action_duration_s[self.steps % len(self.config.action_duration_s)]
        before = self.pose.snapshot()
        # The action catalog is kept explicit; this stage uses the target
        # waypoint as a deterministic smoke policy while PPO wiring is tested.
        self.sim.world.set_target(tuple(self.goal))
        self.sim._step_for_seconds(duration)
        after = self.pose.snapshot()
        self.steps += 1
        distance_before = float(np.linalg.norm(np.asarray(before.position_m) - self.goal))
        distance_after = float(np.linalg.norm(np.asarray(after.position_m) - self.goal))
        reward = distance_before - distance_after - 0.01 * duration
        terminated = distance_after < 0.15
        truncated = self.steps >= self.config.max_episode_steps
        return after.observation26(self.goal), float(reward), terminated, truncated, {"duration_s": duration, "distance_m": distance_after, "sim_tick": after.sim_tick}
