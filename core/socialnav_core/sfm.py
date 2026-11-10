"""Social Force Model (Helbing and Molnar, 1995) for the simulated pedestrians.

Each pedestrian i is accelerated towards its goal and repelled by the other
pedestrians, by walls, and (optionally) by the robot:

    dv_i/dt = (v0_i e_i - v_i) / tau
              + sum_j A exp((r_ij - d_ij) / B) w(phi_ij) n_ij      (pedestrians)
              + sum_W A_w exp((r_i - d_iW) / B_w) n_iW              (walls)
              + group cohesion towards the centroid of the own group

w(phi) = lam + (1 - lam)(1 + cos phi) / 2 makes people react more strongly to
what is in front of them. The parameters are textbook values, not fitted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SFMParams:
    tau: float = 0.5          # relaxation time (s)
    A: float = 2.0            # pedestrian repulsion strength (m/s^2)
    B: float = 0.3            # pedestrian repulsion range (m)
    lam: float = 0.35         # anisotropy
    A_wall: float = 5.0
    B_wall: float = 0.2
    k_group: float = 0.6      # cohesion gain towards the group centroid
    group_dist: float = 0.9   # cohesion only beyond this distance (m)
    radius: float = 0.25      # body radius of a pedestrian (m)
    max_speed_factor: float = 1.3


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Closest point on segment ab to each point in p (N, 2); returns (dist, normal)."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0.0, 1.0)
    q = a + t[:, None] * ab
    d = p - q
    dist = np.linalg.norm(d, axis=1)
    n = d / np.maximum(dist, 1e-9)[:, None]
    return dist, n


def social_forces(pos, vel, goals, speeds, groups, walls, prm: SFMParams,
                  robot=None, robot_radius=0.35, final=None):
    """Total acceleration of every pedestrian.

    pos, vel, goals: (N, 2); speeds, groups: (N,); walls: list of (a, b) arrays;
    robot: optional (x, y, vx, vy) the pedestrians react to.
    """
    n = len(pos)
    acc = np.zeros((n, 2))
    if n == 0:
        return acc

    # 1) driving force towards the goal
    to_goal = goals - pos
    dist_goal = np.linalg.norm(to_goal, axis=1)
    e = to_goal / np.maximum(dist_goal, 1e-9)[:, None]
    slow = np.clip(dist_goal / 0.5, 0.0, 1.0)
    if final is not None:                      # only slow down at the last waypoint
        slow = np.where(final, slow, 1.0)
    v_des = e * speeds[:, None] * slow[:, None]
    acc += (v_des - vel) / prm.tau

    # heading used for the anisotropy (velocity if moving, otherwise goal direction)
    spd = np.linalg.norm(vel, axis=1)
    head = np.where(spd[:, None] > 0.05, vel / np.maximum(spd, 1e-9)[:, None], e)

    def repulse(others, r_sum, A, B):
        d = pos[:, None, :] - others[None, :, :]               # (N, M, 2)
        dist = np.linalg.norm(d, axis=2)
        nrm = d / np.maximum(dist, 1e-9)[:, :, None]
        mag = A * np.exp((r_sum - dist) / B)
        cos_phi = -(nrm * head[:, None, :]).sum(axis=2)        # other in front -> cos > 0
        w = prm.lam + (1 - prm.lam) * (1 + cos_phi) / 2
        return mag, w, nrm

    # 2) pedestrian-pedestrian repulsion
    if n > 1:
        mag, w, nrm = repulse(pos, 2 * prm.radius, prm.A, prm.B)
        np.fill_diagonal(mag, 0.0)
        same = (groups[:, None] == groups[None, :]) & (groups[:, None] >= 0)
        mag = np.where(same, 0.3 * mag, mag)                   # group members tolerate proximity
        acc += ((mag * w)[:, :, None] * nrm).sum(axis=1)

    # 3) robot repulsion (only when pedestrians can see the robot)
    if robot is not None:
        mag, w, nrm = repulse(np.asarray(robot[:2])[None, :], prm.radius + robot_radius,
                              prm.A * 1.5, prm.B * 1.5)
        acc += ((mag * w)[:, :, None] * nrm).sum(axis=1)

    # 4) walls
    for a, b in walls:
        dist, nrm = point_segment(pos, np.asarray(a, float), np.asarray(b, float))
        acc += (prm.A_wall * np.exp((prm.radius - dist) / prm.B_wall))[:, None] * nrm

    # 5) group cohesion
    for g in np.unique(groups[groups >= 0]):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        c = pos[idx].mean(axis=0)
        d = c - pos[idx]
        dist = np.linalg.norm(d, axis=1)
        pull = np.where(dist > prm.group_dist, prm.k_group * (dist - prm.group_dist), 0.0)
        acc[idx] += pull[:, None] * d / np.maximum(dist, 1e-9)[:, None]
    return acc


class Crowd:
    """State and integration of all pedestrians of a scenario."""

    def __init__(self, scenario, params: SFMParams | None = None):
        self.prm = params or SFMParams()
        specs = scenario.peds
        self.n = len(specs)
        self.pos = np.array([s.start for s in specs], float).reshape(-1, 2)
        self.vel = np.zeros_like(self.pos)
        self.routes = [np.array(s.route(), float) for s in specs]
        self.wp = np.ones(self.n, int)           # index of the current target waypoint
        self.speed = np.array([s.speed for s in specs], float)
        self.group = np.array([s.group for s in specs], int)
        self.delay = np.array([s.delay for s in specs], float)
        self.loop = np.array([s.loop for s in specs], bool)
        self.static = self.speed <= 0.0
        self.done = np.zeros(self.n, bool)
        self.walls = [(np.asarray(w.a, float), np.asarray(w.b, float)) for w in scenario.walls]

    def step(self, t: float, dt: float, robot=None, robot_radius=0.35):
        if self.n == 0:
            return
        active = (t >= self.delay) & ~self.static & ~self.done
        target = np.array([r[min(k, len(r) - 1)] for r, k in zip(self.routes, self.wp)])
        final = np.array([k >= len(r) - 1 for r, k in zip(self.routes, self.wp)])
        acc = social_forces(self.pos, self.vel, target, self.speed, self.group, self.walls,
                            self.prm, robot, robot_radius, final)
        vel = self.vel + acc * dt
        vmax = self.speed * self.prm.max_speed_factor
        spd = np.linalg.norm(vel, axis=1)
        vel *= np.minimum(1.0, vmax / np.maximum(spd, 1e-9))[:, None]
        vel[~active] = 0.0
        self.vel = vel
        self.pos = self.pos + vel * dt
        # waypoint switching; at the end loopers turn around, the others stop
        reached = np.linalg.norm(target - self.pos, axis=1) < np.where(final, 0.4, 0.6)
        for i in np.where(reached & active)[0]:
            if not final[i]:
                self.wp[i] += 1
            elif self.loop[i]:
                self.routes[i] = self.routes[i][::-1].copy()
                self.wp[i] = 1
            else:
                self.done[i] = True
                self.vel[i] = 0.0
