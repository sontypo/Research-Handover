#!/usr/bin/env python3
"""Run a controller or an exported policy in MuJoCo.

The MuJoCo world is generated from the socialnav_core scenarios:
  * walls are static boxes; the robot touching one is a collision (MuJoCo contacts),
  * the robot is a cylinder on planar joints (x, y, yaw) driven by velocity
    actuators; unicycle commands (v, w) are rate-limited and turned into joint
    velocity targets, so the robot is a physical body with inertia and lag,
  * pedestrians are mocap bodies moved by the Social Force crowd (they never react
    to the robot, as in stages 1 and 2),
  * a 2-D LiDAR is simulated with mj_ray against the walls (like Gazebo, where
    actors are invisible to the LiDAR); its rays feed learned policies and give
    the other controllers obstacle points (the robot has no wall map).

Usage:
  python3 mujoco_crowd.py --scenario corridor --controller predictive_sf --gif out.gif
  python3 mujoco_crowd.py --scenario crossing --policy ../stage1_python_sim/rl/runs/ppo_mixed/policy.npz
  python3 mujoco_crowd.py --all --policy .../policy.pt --episodes 10     # metrics only
  python3 mujoco_crowd.py --scenario chokepoint --controller social_force --viewer   # needs a display
"""
import argparse
import os
import sys
import time

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")      # off-screen rendering on headless machines

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))

import mujoco  # noqa: E402

from socialnav_core import SCENARIOS, RobotLimits, episode_metrics, make_controller, make_scenario  # noqa: E402
from socialnav_core.controllers import LearnedPolicy  # noqa: E402
from socialnav_core.observation import Obs, ray_angles  # noqa: E402
from socialnav_core.sfm import Crowd  # noqa: E402
from socialnav_core.sim import EpisodeLog  # noqa: E402

GROUP_RGBA = ["0.12 0.47 0.71 1", "1.0 0.5 0.05 1", "0.17 0.63 0.17 1", "0.84 0.15 0.16 1",
              "0.58 0.4 0.74 1", "0.55 0.34 0.29 1"]


def build_xml(sc, radius=0.35, ped_radius=0.25):
    walls = []
    for k, w in enumerate(sc.walls):
        (ax, ay), (bx, by) = w.a, w.b
        L = np.hypot(bx - ax, by - ay) / 2 + 0.05
        yaw = np.degrees(np.arctan2(by - ay, bx - ax))
        walls.append(f'<geom name="wall{k}" type="box" group="1" size="{L:.3f} 0.05 0.5" '
                     f'pos="{(ax + bx) / 2:.3f} {(ay + by) / 2:.3f} 0.5" euler="0 0 {yaw:.2f}" '
                     f'rgba="0.55 0.58 0.65 1"/>')
    peds = []
    for i, p in enumerate(sc.peds):
        rgba = "0.23 0.43 0.66 1" if p.group < 0 else GROUP_RGBA[p.group % len(GROUP_RGBA)]
        peds.append(f'<body name="ped{i}" mocap="true" pos="{p.start[0]:.3f} {p.start[1]:.3f} 0">'
                    f'<geom type="capsule" group="2" size="{ped_radius} 0.6" pos="0 0 0.85" '
                    f'contype="0" conaffinity="0" rgba="{rgba}"/></body>')
    x, y, th = sc.robot_start
    gx, gy = sc.robot_goal
    cx = (sc.bounds[0] + sc.bounds[1]) / 2
    span = max(sc.bounds[1] - sc.bounds[0], 1.6 * (sc.bounds[3] - sc.bounds[2]))
    return f"""
<mujoco model="crowd_{sc.name}">
  <option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.93 0.95" rgb2="0.86 0.87 0.9"
             width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="15 15"/>
  </asset>
  <worldbody>
    <light pos="{cx} 0 12" dir="0 0 -1" directional="true"/>
    <camera name="overview" pos="{cx} {-0.55 * span:.2f} {0.62 * span:.2f}" xyaxes="1 0 0 0 0.75 0.66"/>
    <geom name="floor" type="plane" size="40 40 0.1" material="grid" contype="0" conaffinity="0"/>
    <site name="goal" type="cylinder" size="0.3 0.01" pos="{gx} {gy} 0.01" rgba="0.89 0.34 0.18 1"/>
    {''.join(walls)}
    <body name="robot" pos="0 0 0.2">
      <joint name="x" type="slide" axis="1 0 0" damping="1"/>
      <joint name="y" type="slide" axis="0 1 0" damping="1"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0.5"/>
      <geom name="robot" type="cylinder" size="{radius} 0.15" mass="25" rgba="0.89 0.34 0.18 1"/>
      <geom type="box" size="0.15 0.04 0.02" pos="0.2 0 0.17" contype="0" conaffinity="0"
            rgba="1 1 1 1"/>
    </body>
    {''.join(peds)}
  </worldbody>
  <actuator>
    <velocity joint="x" kv="600" ctrlrange="-3 3"/>
    <velocity joint="y" kv="600" ctrlrange="-3 3"/>
    <velocity joint="yaw" kv="60" ctrlrange="-4 4"/>
  </actuator>
</mujoco>"""


class MujocoCrowdRunner:
    def __init__(self, sc, controller, dt=0.05, limits=None, render=False, size=(640, 360)):
        self.sc, self.ctrl, self.dt = sc, controller, dt
        self.lim = limits or controller.lim
        self.m = mujoco.MjModel.from_xml_string(build_xml(sc, self.lim.radius))
        self.d = mujoco.MjData(self.m)
        x, y, th = sc.robot_start
        self.d.qpos[:3] = [x, y, th]
        mujoco.mj_forward(self.m, self.d)
        self.crowd = Crowd(sc)
        self.robot_body = self.m.body("robot").id
        self.robot_geom = self.m.geom("robot").id
        self.wall_geoms = {self.m.geom(f"wall{k}").id for k in range(len(sc.walls))}
        self.n_rays = getattr(controller, "n_rays", 0) or 36
        self.ray_range = getattr(controller, "ray_range", 4.0)
        self.renderer = mujoco.Renderer(self.m, size[1], size[0]) if render else None
        self.frames = []
        self.cmd = np.zeros(2)

    def rays(self):
        x, y, th = self.d.qpos[:3]
        out = np.full(self.n_rays, self.ray_range)
        gid = np.zeros(1, np.int32)
        for i, a in enumerate(ray_angles(self.n_rays)):
            vec = np.array([np.cos(th + a), np.sin(th + a), 0.0])
            dist = mujoco.mj_ray(self.m, self.d, np.array([x, y, 0.3]), vec,
                                 np.array([0, 1, 0, 0, 0, 0], np.uint8), 1, self.robot_body, gid)
            if 0 <= dist < out[i]:
                out[i] = dist
        return out

    def observe(self, t):
        x, y, th = self.d.qpos[:3]
        vx, vy, w = self.d.qvel[:3]
        v = vx * np.cos(th) + vy * np.sin(th)
        r = self.rays()
        ang = th + ray_angles(len(r))
        hit = r < self.ray_range - 1e-6
        pts = np.stack([x + r * np.cos(ang), y + r * np.sin(ang)], axis=1)[hit]
        return Obs(t=t, dt=self.dt, robot_xy=np.array([x, y]), robot_theta=float(th),
                   robot_v=float(v), robot_w=float(w), goal=np.asarray(self.sc.robot_goal, float),
                   ped_pos=self.crowd.pos.copy(), ped_vel=self.crowd.vel.copy(), walls=[],
                   robot_radius=self.lim.radius, obstacles=pts,
                   ranges=r if getattr(self.ctrl, "n_rays", 0) else None)

    def wall_contact(self):
        for c in self.d.contact[:self.d.ncon]:
            if self.robot_geom in (c.geom1, c.geom2) and \
                    (c.geom1 in self.wall_geoms or c.geom2 in self.wall_geoms):
                return True
        return False

    def run(self, frame_every=2):
        sc, lim, dt = self.sc, self.lim, self.dt
        goal = np.asarray(sc.robot_goal, float)
        n_sub = max(1, int(round(dt / self.m.opt.timestep)))
        T, R, C, P, PV = [], [], [], [], []
        t, k, outcome = 0.0, 0, "timeout"
        while t <= sc.time_limit:
            obs = self.observe(t)
            v, w = self.ctrl.act(obs)
            # kinematic filter, as on the real robot
            self.cmd[0] += np.clip(np.clip(v, lim.v_min, lim.v_max) - self.cmd[0], -lim.a_max * dt, lim.a_max * dt)
            self.cmd[1] += np.clip(np.clip(w, -lim.w_max, lim.w_max) - self.cmd[1], -lim.alpha_max * dt, lim.alpha_max * dt)
            th = self.d.qpos[2]
            T.append(t)
            R.append([*self.d.qpos[:3], obs.robot_v, obs.robot_w])
            C.append(self.cmd.copy())
            P.append(self.crowd.pos.copy())
            PV.append(self.crowd.vel.copy())
            if np.linalg.norm(goal - obs.robot_xy) < 0.4:
                outcome = "success"
                break
            near = self.crowd.n and np.min(np.linalg.norm(self.crowd.pos - obs.robot_xy, axis=1)) \
                < lim.radius + self.crowd.prm.radius
            if near or self.wall_contact():
                outcome = "collision"
                break
            for _ in range(n_sub):
                th = self.d.qpos[2]
                self.d.ctrl[:] = [self.cmd[0] * np.cos(th), self.cmd[0] * np.sin(th), self.cmd[1]]
                mujoco.mj_step(self.m, self.d)
            self.crowd.step(t, dt)
            for i in range(self.crowd.n):
                self.d.mocap_pos[i] = [*self.crowd.pos[i], 0.0]
            if self.renderer is not None and k % frame_every == 0:
                self.renderer.update_scene(self.d, camera="overview")
                self.frames.append(self.renderer.render().copy())
            t += dt
            k += 1
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        n = self.crowd.n
        return EpisodeLog(np.array(T), np.array(R), np.array(C), np.array(P).reshape(len(T), n, 2),
                          np.array(PV).reshape(len(T), n, 2), goal, outcome, sc.name,
                          getattr(self.ctrl, "name", "controller"))


def save_gif(frames, path, fps):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)


def make_ctrl(args, limits):
    if args.policy:
        ctrl = LearnedPolicy(limits, policy_path=args.policy)
        return ctrl, float(ctrl.dt or 0.1)
    return make_controller(args.controller, limits), 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenario", default="corridor", choices=sorted(SCENARIOS))
    ap.add_argument("--all", action="store_true", help="every benchmark scenario")
    ap.add_argument("--controller", default="predictive_sf")
    ap.add_argument("--policy", default=None, help="exported policy (.pt/.npz/.zip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--gif", default=None, help="write a GIF of the first episode")
    ap.add_argument("--viewer", action="store_true", help="interactive MuJoCo viewer (display)")
    a = ap.parse_args()
    limits = RobotLimits()
    scns = sorted(SCENARIOS) if a.all else [a.scenario]

    if a.viewer:
        import mujoco.viewer
        ctrl, dt = make_ctrl(a, limits)
        runner = MujocoCrowdRunner(make_scenario(scns[0], a.seed), ctrl, dt, limits)
        with mujoco.viewer.launch_passive(runner.m, runner.d) as viewer:
            orig = mujoco.mj_step

            def step(m, d):
                orig(m, d)
                viewer.sync()
                time.sleep(m.opt.timestep)
            mujoco.mj_step = step
            log = runner.run()
            mujoco.mj_step = orig
        print(f"{log.scenario}: {log.outcome} in {log.t[-1]:.1f} s")
        return

    print(f"{'scenario':13s} {'outcome':>10s} {'time (s)':>9s} {'min dist':>9s}  "
          f"({'policy ' + a.policy if a.policy else a.controller})")
    for scn in scns:
        res = []
        for e in range(a.episodes):
            ctrl, dt = make_ctrl(a, limits)
            render = bool(a.gif) and e == 0
            runner = MujocoCrowdRunner(make_scenario(scn, a.seed + e), ctrl, dt, limits, render)
            log = runner.run()
            m = episode_metrics(log)
            res.append(m)
            print(f"{scn:13s} {m['outcome']:>10s} {m['time']:9.1f} {m['min_dist']:9.2f}")
            if render and runner.frames:
                path = a.gif if len(scns) == 1 else a.gif.replace(".gif", f"_{scn}.gif")
                save_gif(runner.frames, path, fps=1.0 / (2 * dt))
                print(f"  gif: {path}")
        if a.episodes > 1:
            print(f"{scn:13s} success {100 * np.mean([r['success'] for r in res]):.0f}% over "
                  f"{a.episodes} episodes")


if __name__ == "__main__":
    main()
