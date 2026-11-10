#!/usr/bin/env python3
"""Run a socialnav_core controller on a ROS 2 robot (Gazebo or real).

Inputs
  /odom           nav_msgs/Odometry        robot pose and velocity (in "odom")
  /crowd/people   std_msgs/Float32MultiArray  rows [x, y, vx, vy] in "odom"
  /scan           sensor_msgs/LaserScan    static obstacles + emergency stop
  /goal_pose      geometry_msgs/PoseStamped  (optional) new goal -> new trial
Outputs
  /cmd_vel        geometry_msgs/Twist
  /trial/status   std_msgs/String          running | success | timeout | aborted

Static obstacles: the robot has no wall map, so the nearest LiDAR return in
every ``obstacle_sector_deg`` sector (within ``obstacle_range``) is given to the
controller as an obstacle point; returns close to a tracked person are dropped
so that people are not counted twice. The LiDAR is assumed at ``laser_x``
metres in front of the robot centre, aligned with it.

Safety layers, applied after the controller, in this order:
  1. stale odometry (> pose_timeout)        -> zero command
  2. LiDAR emergency stop: an obstacle closer than estop_distance inside
     +-estop_half_angle around the direction of travel -> no translation
  3. speed and acceleration limits (kinematic filter)
  4. yaw_rate_scale: compensates a base that turns slower than commanded
     (calibrate it on your robot; 1.0 = no compensation)
These are basic safeguards for a supervised experiment. They do not replace an
emergency stop button and a human operator. Also enable the command timeout of
your base driver, so that the robot stops when this node dies.
"""
import math
import signal

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, String

from socialnav_core import RobotLimits, make_controller
from socialnav_core.controllers import LearnedPolicy
from socialnav_core.observation import Obs, scan_to_ranges


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class NavController(Node):
    def __init__(self):
        super().__init__("nav_controller")
        p = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.ctrl_name = p("controller", "predictive_sf")
        self.goal = np.array([p("goal_x", 11.0), p("goal_y", 0.0)])
        self.rate = p("rate", 20.0)
        self.goal_tol = p("goal_tolerance", 0.4)
        self.time_limit = p("time_limit", 60.0)
        self.pose_timeout = p("pose_timeout", 0.5)
        self.people_timeout = p("people_timeout", 1.0)
        self.estop_dist = p("estop_distance", 0.45)
        self.estop_angle = math.radians(p("estop_half_angle_deg", 60.0))
        self.yaw_scale = p("yaw_rate_scale", 1.0)
        self.use_scan_obs = p("use_scan_obstacles", True)
        self.obs_range = p("obstacle_range", 2.5)
        self.obs_sector = math.radians(p("obstacle_sector_deg", 10.0))
        self.person_excl = p("person_exclusion", 0.5)
        self.laser_x = p("laser_x", 0.0)
        self.lim = RobotLimits(radius=p("robot_radius", 0.35), v_max=p("max_v", 1.0),
                               v_min=p("min_v", 0.0), w_max=p("max_w", 1.2),
                               a_max=p("max_acc", 1.5), alpha_max=p("max_alpha", 2.5))
        self.ctrl = self._make_controller(p("policy", ""))

        self.odom, self.odom_t = None, -1e9
        self.people, self.people_t = np.zeros((0, 4)), -1e9
        self.scan = None
        self.cmd = np.zeros(2)
        self.state, self.t_start = "running", None

        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(Float32MultiArray, "/crowd/people", self.on_people, 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_status = self.create_publisher(String, "/trial/status", 10)
        self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(
            f"controller={self.ctrl_name} goal=({self.goal[0]:.2f}, {self.goal[1]:.2f}) "
            f"v_max={self.lim.v_max} w_max={self.lim.w_max} rate={self.rate} Hz")

    def _make_controller(self, policy_path):
        if self.ctrl_name != "learned":
            return make_controller(self.ctrl_name, self.lim)
        if not policy_path:
            raise RuntimeError("controller=learned needs the parameter policy "
                               "(policy.pt, policy.npz or a stable-baselines3 .zip)")
        ctrl = LearnedPolicy(self.lim, policy_path=policy_path)
        if ctrl.dt:                       # run at the control period used in training
            self.rate = 1.0 / float(ctrl.dt)
        self.get_logger().info(f"learned policy {policy_path}: {ctrl.k} people slots, "
                               f"{ctrl.n_rays} rays, control at {self.rate:.1f} Hz")
        return ctrl

    # ---------------- callbacks ----------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        self.odom, self.odom_t = msg, self.now()

    def on_people(self, msg):
        self.people = np.asarray(msg.data, float).reshape(-1, 4)
        self.people_t = self.now()

    def on_scan(self, msg):
        self.scan = msg

    def on_goal(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.state, self.t_start = "running", None
        self.ctrl.reset()
        self.get_logger().info(f"new goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}), new trial")

    # ---------------- control loop ----------------
    def publish(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w * self.yaw_scale)
        self.pub_cmd.publish(t)

    def set_state(self, s):
        if s != self.state:
            self.state = s
            self.get_logger().info(f"trial {s}")
        self.pub_status.publish(String(data=self.state))

    def scan_obstacles(self, xy, yaw, people):
        if not self.use_scan_obs or self.scan is None:
            return np.zeros((0, 2))
        r = np.asarray(self.scan.ranges, float)
        a = self.scan.angle_min + np.arange(len(r)) * self.scan.angle_increment
        ok = np.isfinite(r) & (r > self.scan.range_min) & (r < self.obs_range)
        r, a = r[ok], a[ok]
        if not len(r):
            return np.zeros((0, 2))
        sector = np.floor((a + math.pi) / self.obs_sector).astype(int)
        keep = [np.flatnonzero(sector == k)[np.argmin(r[sector == k])] for k in np.unique(sector)]
        r, a = r[keep], a[keep]
        body = np.stack([self.laser_x + r * np.cos(a), r * np.sin(a)], axis=1)
        c, s_ = math.cos(yaw), math.sin(yaw)
        pts = body @ np.array([[c, s_], [-s_, c]]) + xy
        if len(people):
            d = np.linalg.norm(pts[:, None] - people[None, :, :2], axis=2).min(axis=1)
            pts = pts[d > self.person_excl]
        return pts

    def scan_ranges(self):
        """LiDAR down-sampled to the rays a learned policy was trained with."""
        n = getattr(self.ctrl, "n_rays", 0)
        if not n or self.scan is None:
            return None
        sc = self.scan
        return scan_to_ranges(sc.ranges, sc.angle_min, sc.angle_increment, n,
                              self.ctrl.ray_range, sc.range_min)

    def estop(self, v):
        if self.scan is None or abs(v) < 1e-3:
            return False
        r = np.asarray(self.scan.ranges, float)
        a = self.scan.angle_min + np.arange(len(r)) * self.scan.angle_increment
        heading = 0.0 if v > 0 else math.pi                # sector around the travel direction
        diff = np.abs((a - heading + math.pi) % (2 * math.pi) - math.pi)
        ok = np.isfinite(r) & (r > self.scan.range_min) & (diff <= self.estop_angle)
        return bool(ok.any() and r[ok].min() < self.estop_dist)

    def tick(self):
        now = self.now()
        dt = 1.0 / self.rate
        if self.odom is None or now - self.odom_t > self.pose_timeout:
            self.cmd[:] = 0.0
            self.publish(0.0, 0.0)
            self.get_logger().warn("no fresh odometry: stopping", throttle_duration_sec=2.0)
            return
        if self.state != "running":
            self.publish(0.0, 0.0)
            self.pub_status.publish(String(data=self.state))
            return
        if self.t_start is None:
            self.t_start = now
        o = self.odom
        xy = np.array([o.pose.pose.position.x, o.pose.pose.position.y])
        if np.linalg.norm(self.goal - xy) < self.goal_tol:
            self.set_state("success")
            self.publish(0.0, 0.0)
            return
        if now - self.t_start > self.time_limit:
            self.set_state("timeout")
            self.publish(0.0, 0.0)
            return
        people = self.people if now - self.people_t < self.people_timeout else np.zeros((0, 4))
        yaw = yaw_of(o.pose.pose.orientation)
        obs = Obs(t=now - self.t_start, dt=dt, robot_xy=xy,
                  robot_theta=yaw,
                  robot_v=o.twist.twist.linear.x, robot_w=o.twist.twist.angular.z,
                  goal=self.goal, ped_pos=people[:, :2], ped_vel=people[:, 2:4],
                  walls=[], robot_radius=self.lim.radius,
                  obstacles=self.scan_obstacles(xy, yaw, people),
                  ranges=self.scan_ranges())
        v, w = self.ctrl.act(obs)
        v = float(np.clip(v, self.lim.v_min, self.lim.v_max))
        w = float(np.clip(w, -self.lim.w_max, self.lim.w_max))
        if self.estop(v):
            v = 0.0
            self.get_logger().warn("obstacle in the travel direction: translation stopped",
                                   throttle_duration_sec=1.0)
        # kinematic filter: bounded change of the command per control step
        self.cmd[0] += np.clip(v - self.cmd[0], -self.lim.a_max * dt, self.lim.a_max * dt)
        self.cmd[1] += np.clip(w - self.cmd[1], -self.lim.alpha_max * dt, self.lim.alpha_max * dt)
        self.publish(*self.cmd)
        self.pub_status.publish(String(data="running"))

    def stop(self):
        for _ in range(3):
            self.publish(0.0, 0.0)


def main():
    rclpy.init()
    node = NavController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # finish cleanly on a second Ctrl-C
        if rclpy.ok():
            node.stop()          # last zero command while the context is alive
        try:
            node.destroy_node()
        except BaseException:  # noqa: BLE001 - shutting down anyway
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
