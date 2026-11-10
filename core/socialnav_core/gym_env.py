"""Gymnasium wrapper of ``CrowdNavEnv`` (for stable-baselines3 and similar).

    from socialnav_core.gym_env import SocialNavGymEnv
    env = SocialNavGymEnv(scenario="mixed")

Requires ``gymnasium``; the rest of socialnav_core does not.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .env import CrowdNavEnv


class SocialNavGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.core = CrowdNavEnv(**kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.core.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.core.rng = np.random.default_rng(seed)
        return self.core.reset(), {}

    def step(self, action):
        obs, reward, done, info = self.core.step(action)
        truncated = bool(info.pop("truncated", False))
        terminated = bool(done and not truncated)
        return obs, reward, terminated, truncated, info
