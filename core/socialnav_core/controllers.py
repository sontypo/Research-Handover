"""Reference robot controllers and the plug-in interface for your own.

All controllers implement ``act(obs) -> (v, w)``. Three baselines are provided:

GoalSeeker             drives straight to the goal and ignores people (lower bound)
SocialForceController  treats the robot as one more Social Force agent (reactive)
PredictiveSocialForce  reacts to where people *will be*, using any ``Predictor``
                       (prediction-then-planning; CV prediction by default)

``LearnedPolicy`` is the slot for a learned controller (e.g. a policy trained with
reinforcement learning in ``env.CrowdNavEnv`` and saved as a .pt model); give it the
model with ``LearnedPolicy(model=...)``.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .observation import Obs, to_vector
from .prediction import ConstantVelocityPredictor, Predictor
from .robot import RobotLimits
from .sfm import point_segment


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    name = "base"

    def __init__(self, limits: Optional[RobotLimits] = None):
        self.lim = limits or RobotLimits()

    def reset(self):
        pass

    def act(self, obs: Obs):
        raise NotImplementedError

    # helper: turn a desired planar velocity into unicycle commands
    def track_velocity(self, obs: Obs, v_des: np.ndarray, k_w: float = 2.5):
        speed = float(np.linalg.norm(v_des))
        if speed < 1e-3:
            return 0.0, 0.0
        err = wrap(np.arctan2(v_des[1], v_des[0]) - obs.robot_theta)
        v = speed * max(np.cos(err), 0.0)
        w = k_w * err
        return float(np.clip(v, self.lim.v_min, self.lim.v_max)), \
            float(np.clip(w, -self.lim.w_max, self.lim.w_max))


class GoalSeeker(Controller):
    name = "goal_seeker"

    def act(self, obs: Obs):
        to_goal = obs.goal - obs.robot_xy
        dist = np.linalg.norm(to_goal)
        v_des = to_goal / max(dist, 1e-9) * self.lim.v_max * min(1.0, dist / 1.0)
        return self.track_velocity(obs, v_des)


class SocialForceController(Controller):
    """Reactive Social Force controller for the robot.

    The robot is attracted by its goal and repelled by pedestrians and walls; the
    resulting force is integrated over a short look-ahead to obtain a desired
    velocity, which a heading controller tracks.
    """
    name = "social_force"

    def __init__(self, limits=None, A=2.5, B=0.8, lam=0.3, A_wall=3.0, B_wall=0.2,
                 tau=0.6, lookahead=0.6, ped_radius=0.25):
        super().__init__(limits)
        self.A, self.B, self.lam = A, B, lam
        self.A_wall, self.B_wall = A_wall, B_wall
        self.tau, self.lookahead, self.ped_radius = tau, lookahead, ped_radius

    def _goal_force(self, obs):
        to_goal = obs.goal - obs.robot_xy
        dist = np.linalg.norm(to_goal)
        v_pref = to_goal / max(dist, 1e-9) * self.lim.v_max * min(1.0, dist / 1.0)
        v_robot = obs.robot_v * np.array([np.cos(obs.robot_theta), np.sin(obs.robot_theta)])
        return (v_pref - v_robot) / self.tau, v_robot

    def _repulsion(self, obs, points, weight=1.0):
        if len(points) == 0:
            return np.zeros(2)
        d = obs.robot_xy - points
        dist = np.linalg.norm(d, axis=1)
        n = d / np.maximum(dist, 1e-9)[:, None]
        heading = np.array([np.cos(obs.robot_theta), np.sin(obs.robot_theta)])
        cos_phi = -(n @ heading)
        w = self.lam + (1 - self.lam) * (1 + cos_phi) / 2
        mag = self.A * np.exp((obs.robot_radius + self.ped_radius - dist) / self.B)
        return weight * ((mag * w)[:, None] * n).sum(axis=0)

    def _walls(self, obs):
        f = np.zeros(2)
        for a, b in obs.walls:
            dist, n = point_segment(obs.robot_xy[None], np.asarray(a), np.asarray(b))
            f += self.A_wall * np.exp((obs.robot_radius - dist[0]) / self.B_wall) * n[0]
        if len(obs.obstacles):                      # point obstacles, e.g. LiDAR returns
            d = obs.robot_xy - obs.obstacles
            dist = np.linalg.norm(d, axis=1)
            n = d / np.maximum(dist, 1e-9)[:, None]
            mag = self.A_wall * np.exp((obs.robot_radius - dist) / self.B_wall)
            f += (mag[:, None] * n).sum(axis=0)
        return f

    def pedestrian_force(self, obs):
        return self._repulsion(obs, obs.ped_pos)

    def act(self, obs: Obs):
        f_goal, v_robot = self._goal_force(obs)
        f = f_goal + self.pedestrian_force(obs) + self._walls(obs)
        v_des = v_robot + f * self.lookahead
        n = np.linalg.norm(v_des)
        if n > self.lim.v_max:
            v_des *= self.lim.v_max / n
        return self.track_velocity(obs, v_des)


class PredictiveSocialForce(SocialForceController):
    """Social Force controller that is repelled by predicted pedestrian positions.

    The repulsion is a weighted sum over the predicted positions at several
    future times (and over the K hypotheses of a multimodal predictor), so the
    robot starts to give way before a pedestrian is close.
    """
    name = "predictive_sf"

    def __init__(self, limits=None, predictor: Optional[Predictor] = None,
                 horizon=1.5, dt=0.5, decay=0.6, B=0.45, **kw):
        super().__init__(limits, B=B, **kw)
        self.predictor = predictor or ConstantVelocityPredictor()
        self.horizon, self.dt, self.decay = horizon, dt, decay

    def pedestrian_force(self, obs):
        f = self._repulsion(obs, obs.ped_pos)
        if len(obs.ped_pos) == 0:
            return f
        traj = self.predictor.predict(obs.ped_pos, obs.ped_vel, self.horizon, self.dt)
        wk = self.predictor.weights()
        for k in range(traj.shape[0]):
            for t in range(traj.shape[1]):
                f += self._repulsion(obs, traj[k, t], weight=wk[k] * self.decay ** (t + 1))
        return f


class LearnedPolicy(Controller):
    """A learned policy: observation vector -> normalised action in [-1, 1]^2.

    ``model`` is any callable on the vector of ``observation.to_vector`` (e.g. an
    exported policy, see ``policy_io.load_policy``). ``meta`` holds the settings
    used in training (``policy_meta.json``): the observation layout and the action
    scaling. The action is mapped with the *training* speed range and then clipped
    to the limits of the robot it runs on, so stricter deployment limits do not
    change the meaning of the policy's output.
    """
    name = "learned"

    def __init__(self, limits=None, model: Optional[Callable] = None, meta: Optional[dict] = None,
                 policy_path: Optional[str] = None):
        super().__init__(limits)
        if policy_path:
            from .policy_io import load_policy
            model, meta = load_policy(policy_path)
        self.model = model
        meta = meta or {}
        o = meta.get("obs", {})
        self.k, self.n_rays = int(o.get("k", 8)), int(o.get("n_rays", 0))
        self.ray_range, self.max_range = float(o.get("ray_range", 4.0)), float(o.get("max_range", 8.0))
        a = meta.get("action", {})
        self.a_vmin = float(a.get("v_min", self.lim.v_min))
        self.a_vmax = float(a.get("v_max", self.lim.v_max))
        self.a_wmax = float(a.get("w_max", self.lim.w_max))
        self.dt = meta.get("dt")

    def vector(self, obs: Obs):
        return to_vector(obs, self.k, self.max_range, self.a_vmax, self.a_wmax,
                         self.n_rays, self.ray_range)

    def act(self, obs: Obs):
        if self.model is None:
            raise NotImplementedError(
                "LearnedPolicy has no model. Train one (stage1_python_sim/rl/train_sb3.py) "
                "and pass policy_path=... or model=callable(obs_vector) -> action.")
        a = np.clip(np.asarray(self.model(self.vector(obs)), float).ravel(), -1, 1)
        v = self.a_vmin + (a[0] + 1) / 2 * (self.a_vmax - self.a_vmin)
        w = a[1] * self.a_wmax
        return (float(np.clip(v, self.lim.v_min, self.lim.v_max)),
                float(np.clip(w, -self.lim.w_max, self.lim.w_max)))


CONTROLLERS: Dict[str, type] = {
    c.name: c for c in (GoalSeeker, SocialForceController, PredictiveSocialForce, LearnedPolicy)
}


def make_controller(name: str, limits: Optional[RobotLimits] = None, **kw) -> Controller:
    if name not in CONTROLLERS:
        raise KeyError(f"unknown controller '{name}', choose from {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](limits, **kw)
