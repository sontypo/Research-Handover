"""Minimal reset/step environment for reinforcement learning.

``CrowdNavEnv`` has no dependency on gym/gymnasium (the Gymnasium wrapper for
stable-baselines3 is in ``gym_env.py``). The reward is a standard, generic one:

    r = w_progress * (decrease of the distance to the goal)
      + w_goal      on reaching the goal
      + w_collision on a collision (episode ends)
      - w_close * (gap_min - gap) when a person is closer than gap_min
      - w_time per step

It is a baseline to start from, not a tuned reward: designing the reward is part
of the research.

Example
-------
    env = CrowdNavEnv("corridor", seed=0)
    obs = env.reset()
    done = False
    while not done:
        obs, r, done, info = env.step(env.sample_action())
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .controllers import Controller
from .observation import obs_dim, to_vector
from .robot import Robot, RobotLimits
from .scenarios import SCENARIOS, make_scenario
from .sfm import Crowd
from .sim import Simulator


@dataclass
class RewardWeights:
    progress: float = 1.0
    goal: float = 10.0
    collision: float = -10.0
    close: float = 1.0        # penalty per metre below gap_min (per step)
    gap_min: float = 0.3      # m, surface-to-surface gap that starts the penalty
    time: float = 0.01        # penalty per step


@dataclass
class ObsConfig:
    """Everything needed to rebuild the observation vector at deployment."""
    k: int = 8
    n_rays: int = 16
    ray_range: float = 4.0
    max_range: float = 8.0


class _Null(Controller):
    name = "env"

    def act(self, obs):
        return 0.0, 0.0


class CrowdNavEnv:
    """2-D crowd navigation task.

    scenario:  a scenario name, or "mixed" for a random scenario per episode
               (the four benchmark layouts plus random_crowd)
    """

    def __init__(self, scenario: str = "mixed", seed: int = 0, dt: float = 0.1,
                 limits: RobotLimits | None = None, robot_visible: bool = False,
                 obs_cfg: ObsConfig | None = None, reward: RewardWeights | None = None,
                 random_heading: float = 0.3, randomise_scenario: bool | None = None):
        self.name, self.dt = scenario, dt
        self.lim = limits or RobotLimits()
        self.robot_visible = robot_visible
        self.ocfg = obs_cfg or ObsConfig()
        self.rw = reward or RewardWeights()
        self.random_heading = random_heading
        if randomise_scenario:                 # backwards-compatible alias
            self.name = "mixed"
        self.pool = sorted(SCENARIOS) + ["random_crowd", "random_crowd"] \
            if self.name == "mixed" else [self.name]
        self.rng = np.random.default_rng(seed)
        self.obs_dim = obs_dim(self.ocfg.k, self.ocfg.n_rays)
        self.act_dim = 2

    def config(self) -> dict:
        """Settings a deployed policy must reproduce (saved with the model)."""
        return {"dt": self.dt, "obs": asdict(self.ocfg),
                "action": {"v_min": self.lim.v_min, "v_max": self.lim.v_max,
                           "w_max": self.lim.w_max},
                "robot": asdict(self.lim), "reward": asdict(self.rw),
                "robot_visible": self.robot_visible, "scenario": self.name}

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        name = str(self.rng.choice(self.pool))
        self.sc = make_scenario(name, int(self.rng.integers(1_000_000)))
        self.sim = Simulator(self.sc, _Null(self.lim), dt=self.dt, limits=self.lim,
                             robot_visible=self.robot_visible)
        x, y, th = self.sc.robot_start
        th += self.rng.uniform(-self.random_heading, self.random_heading)
        self.robot = Robot(x, y, th, limits=self.lim)
        self.crowd = Crowd(self.sc, self.sim.sfm)
        self.t = 0.0
        self.goal = np.asarray(self.sc.robot_goal, float)
        self.prev_dist = np.linalg.norm(self.goal - self.robot.pos)
        return self._vec()

    def observe(self):
        return self.sim.observe(self.t, self.robot, self.crowd)

    def _vec(self):
        c = self.ocfg
        return to_vector(self.observe(), c.k, c.max_range, self.lim.v_max, self.lim.w_max,
                         c.n_rays, c.ray_range)

    def action_to_command(self, a):
        a = np.clip(np.asarray(a, float).ravel(), -1, 1)
        v = self.lim.v_min + (a[0] + 1) / 2 * (self.lim.v_max - self.lim.v_min)
        return v, a[1] * self.lim.w_max

    def sample_action(self):
        return self.rng.uniform(-1, 1, size=2)

    def step(self, action):
        v, w = self.action_to_command(action)
        self.robot.step(v, w, self.dt)
        rob = (self.robot.x, self.robot.y, *self.robot.vel) if self.robot_visible else None
        self.crowd.step(self.t, self.dt, rob, self.lim.radius)
        self.t += self.dt

        rw = self.rw
        dist = np.linalg.norm(self.goal - self.robot.pos)
        reward = rw.progress * (self.prev_dist - dist) - rw.time
        self.prev_dist = dist
        if self.crowd.n:
            gap = np.min(np.linalg.norm(self.crowd.pos - self.robot.pos, axis=1)) \
                - self.lim.radius - self.sim.sfm.radius
            if gap < rw.gap_min:
                reward -= rw.close * (rw.gap_min - max(gap, 0.0))
        info = {"outcome": None, "scenario": self.sc.name}
        terminated = truncated = False
        if dist < self.sim.goal_tol:
            reward += rw.goal
            terminated, info["outcome"] = True, "success"
        elif self.sim._collided(self.robot, self.crowd):
            reward += rw.collision
            terminated, info["outcome"] = True, "collision"
        elif self.t >= self.sc.time_limit:
            truncated, info["outcome"] = True, "timeout"
        info["truncated"] = truncated
        return self._vec(), float(reward), terminated or truncated, info
