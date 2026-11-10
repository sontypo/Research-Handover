"""Unicycle robot with speed and acceleration limits.

The command is a velocity target (v, w). As on most real mobile bases, the
achieved velocities approach the target at bounded accelerations, so a policy
that is trained here already experiences actuator lag.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobotLimits:
    radius: float = 0.35      # footprint radius (m)
    v_max: float = 1.0        # forward speed (m/s)
    v_min: float = 0.0        # set < 0 to allow reversing
    w_max: float = 1.2        # yaw rate (rad/s)
    a_max: float = 1.5        # linear acceleration (m/s^2)
    alpha_max: float = 2.5    # angular acceleration (rad/s^2)


class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0, limits: RobotLimits | None = None):
        self.lim = limits or RobotLimits()
        self.x, self.y, self.theta = float(x), float(y), float(theta)
        self.v = 0.0
        self.w = 0.0

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def vel(self) -> np.ndarray:
        return self.v * np.array([np.cos(self.theta), np.sin(self.theta)])

    def step(self, v_cmd: float, w_cmd: float, dt: float):
        lim = self.lim
        v_cmd = float(np.clip(v_cmd, lim.v_min, lim.v_max))
        w_cmd = float(np.clip(w_cmd, -lim.w_max, lim.w_max))
        self.v += float(np.clip(v_cmd - self.v, -lim.a_max * dt, lim.a_max * dt))
        self.w += float(np.clip(w_cmd - self.w, -lim.alpha_max * dt, lim.alpha_max * dt))
        self.theta = float(np.arctan2(np.sin(self.theta + self.w * dt),
                                      np.cos(self.theta + self.w * dt)))
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
