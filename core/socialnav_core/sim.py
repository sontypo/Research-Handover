"""Episode loop: scenario + crowd + robot + controller -> log."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controllers import Controller
from .observation import Obs
from .robot import Robot, RobotLimits
from .scenarios import Scenario
from .sfm import Crowd, SFMParams, point_segment


@dataclass
class EpisodeLog:
    t: np.ndarray          # (T,)
    robot: np.ndarray      # (T, 5) x, y, theta, v, w
    cmd: np.ndarray        # (T, 2) commanded v, w
    peds: np.ndarray       # (T, N, 2)
    ped_vel: np.ndarray    # (T, N, 2)
    goal: np.ndarray       # (2,)
    outcome: str           # success | collision | timeout
    scenario: str = ""
    controller: str = ""

    def save(self, path):
        np.savez_compressed(path, t=self.t, robot=self.robot, cmd=self.cmd, peds=self.peds,
                            ped_vel=self.ped_vel, goal=self.goal, outcome=self.outcome,
                            scenario=self.scenario, controller=self.controller)


class Simulator:
    """Steps a scenario with a controller.

    robot_visible=False means pedestrians ignore the robot (they never yield),
    which is the harder and more honest test: a controller cannot succeed by
    walking into people and letting them step aside.
    """

    def __init__(self, scenario: Scenario, controller: Controller, dt: float = 0.05,
                 limits: RobotLimits | None = None, robot_visible: bool = False,
                 sensor_range: float = 8.0, goal_tol: float = 0.4,
                 sfm: SFMParams | None = None):
        self.sc = scenario
        self.ctrl = controller
        self.dt = dt
        self.lim = limits or controller.lim
        self.robot_visible = robot_visible
        self.sensor_range = sensor_range
        self.goal_tol = goal_tol
        self.sfm = sfm or SFMParams()

    def observe(self, t, robot, crowd) -> Obs:
        rel = np.linalg.norm(crowd.pos - robot.pos, axis=1) if crowd.n else np.zeros(0)
        vis = rel <= self.sensor_range
        return Obs(t=t, dt=self.dt, robot_xy=robot.pos, robot_theta=robot.theta,
                   robot_v=robot.v, robot_w=robot.w, goal=np.asarray(self.sc.robot_goal, float),
                   ped_pos=crowd.pos[vis], ped_vel=crowd.vel[vis],
                   walls=crowd.walls, robot_radius=self.lim.radius)

    def run(self) -> EpisodeLog:
        sc = self.sc
        robot = Robot(*sc.robot_start, limits=self.lim)
        crowd = Crowd(sc, self.sfm)
        self.ctrl.reset()
        goal = np.asarray(sc.robot_goal, float)
        T, R, C, P, PV = [], [], [], [], []
        outcome, t = "timeout", 0.0
        while t <= sc.time_limit:
            obs = self.observe(t, robot, crowd)
            v, w = self.ctrl.act(obs)
            T.append(t)
            R.append([robot.x, robot.y, robot.theta, robot.v, robot.w])
            C.append([v, w])
            P.append(crowd.pos.copy())
            PV.append(crowd.vel.copy())
            if np.linalg.norm(goal - robot.pos) < self.goal_tol:
                outcome = "success"
                break
            if self._collided(robot, crowd):
                outcome = "collision"
                break
            robot.step(v, w, self.dt)
            rob = (robot.x, robot.y, *robot.vel) if self.robot_visible else None
            crowd.step(t, self.dt, rob, self.lim.radius)
            t += self.dt
        n = crowd.n
        return EpisodeLog(np.array(T), np.array(R), np.array(C),
                          np.array(P).reshape(len(T), n, 2), np.array(PV).reshape(len(T), n, 2),
                          goal, outcome, sc.name, self.ctrl.name)

    def _collided(self, robot, crowd) -> bool:
        if crowd.n and np.min(np.linalg.norm(crowd.pos - robot.pos, axis=1)) < \
                self.lim.radius + self.sfm.radius:
            return True
        for a, b in crowd.walls:
            dist, _ = point_segment(robot.pos[None], a, b)
            if dist[0] < self.lim.radius:
                return True
        return False


def run_episode(scenario: Scenario, controller: Controller, **kw) -> EpisodeLog:
    return Simulator(scenario, controller, **kw).run()
