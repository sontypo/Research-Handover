"""What a controller sees.

``Obs`` is the common input of every controller, in simulation and on the robot.
The ROS nodes (stages 2 and 3) and the MuJoCo runner fill the same structure, so
a controller that works here runs unchanged there.

``to_vector`` turns an ``Obs`` into a fixed-size vector for neural policies:

  robot      5 values   [goal distance, sin/cos goal bearing, v, w]
  people     6 x k      k nearest people in the robot's body frame
  rays       n_rays     optional range readings (a coarse 2-D LiDAR)

The rays let a learned policy see walls. In the 2-D simulator they are ray-cast
against the scenario walls; on a robot they come from /scan (``Obs.ranges``).
This is the usual CADRL/SARL-style encoding, kept simple on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Obs:
    t: float
    dt: float
    robot_xy: np.ndarray            # (2,) world frame
    robot_theta: float
    robot_v: float
    robot_w: float
    goal: np.ndarray                # (2,) world frame
    ped_pos: np.ndarray             # (N, 2) world frame, only perceived pedestrians
    ped_vel: np.ndarray             # (N, 2)
    walls: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    robot_radius: float = 0.35
    obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    # obstacles: (M, 2) static obstacle points in the world frame, e.g. from a LiDAR
    ranges: Optional[np.ndarray] = None
    # ranges: (n,) distances of a 360 deg range sensor in the body frame; ray i points
    # at angle -pi + (i + 0.5) * 2 pi / n. None -> ray-cast against ``walls``.


ROBOT_FEATURES = 5
PED_FEATURES = 6


def ray_angles(n: int) -> np.ndarray:
    return -np.pi + (np.arange(n) + 0.5) * 2 * np.pi / n


def ray_cast(xy, theta, walls, n: int, max_range: float) -> np.ndarray:
    """Distances from xy along n body-frame rays to the nearest wall segment."""
    out = np.full(n, max_range)
    if not walls:
        return out
    ang = theta + ray_angles(n)
    d = np.stack([np.cos(ang), np.sin(ang)], axis=1)                 # (n, 2)
    p = np.asarray(xy, float)
    for a, b in walls:
        a, b = np.asarray(a, float), np.asarray(b, float)
        e = b - a
        denom = d[:, 0] * e[1] - d[:, 1] * e[0]                       # cross(d, e)
        ok = np.abs(denom) > 1e-9
        ap = a - p
        t = np.where(ok, (ap[0] * e[1] - ap[1] * e[0]) / np.where(ok, denom, 1), np.inf)
        u = np.where(ok, (ap[0] * d[:, 1] - ap[1] * d[:, 0]) / np.where(ok, denom, 1), -1)
        hit = ok & (t > 0) & (u >= 0) & (u <= 1)
        out = np.where(hit, np.minimum(out, t), out)
    return out


def scan_to_ranges(ranges, angle_min, angle_inc, n: int, max_range: float,
                   range_min: float = 0.0) -> np.ndarray:
    """Down-sample a LaserScan to n sectors (minimum per sector), body frame."""
    r = np.asarray(ranges, float)
    a = angle_min + np.arange(len(r)) * angle_inc
    a = (a + np.pi) % (2 * np.pi) - np.pi
    valid = np.isfinite(r) & (r > range_min)
    sector = np.clip(((a + np.pi) / (2 * np.pi) * n).astype(int), 0, n - 1)
    out = np.full(n, max_range)
    np.minimum.at(out, sector[valid], r[valid])
    return np.minimum(out, max_range)


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])        # world -> body


def obs_dim(k: int = 8, n_rays: int = 0) -> int:
    return ROBOT_FEATURES + PED_FEATURES * k + n_rays


def to_vector(obs: Obs, k: int = 8, max_range: float = 8.0, v_max: float = 1.0,
              w_max: float = 1.2, n_rays: int = 0, ray_range: float = 4.0) -> np.ndarray:
    """Vector of size 5 + 6k + n_rays (see module docstring)."""
    R = _rot(obs.robot_theta)
    g = R @ (obs.goal - obs.robot_xy)
    bearing = np.arctan2(g[1], g[0])
    robot = [min(np.linalg.norm(g), 20.0) / 10.0, np.sin(bearing), np.cos(bearing),
             obs.robot_v / v_max, obs.robot_w / w_max]
    peds = np.zeros((k, PED_FEATURES))
    if len(obs.ped_pos):
        rel = obs.ped_pos - obs.robot_xy
        dist = np.linalg.norm(rel, axis=1)
        order = [i for i in np.argsort(dist) if dist[i] <= max_range][:k]
        v_robot = obs.robot_v * np.array([np.cos(obs.robot_theta), np.sin(obs.robot_theta)])
        for slot, i in enumerate(order):
            p = R @ rel[i]
            v = R @ (obs.ped_vel[i] - v_robot)
            peds[slot] = [p[0] / 10, p[1] / 10, v[0] / 2, v[1] / 2, dist[i] / max_range, 1.0]
    parts = [robot, peds.ravel()]
    if n_rays:
        r = obs.ranges if obs.ranges is not None else \
            ray_cast(obs.robot_xy, obs.robot_theta, obs.walls, n_rays, ray_range)
        parts.append(np.clip(np.asarray(r, float), 0.0, ray_range) / ray_range)
    return np.concatenate(parts).astype(np.float32)
