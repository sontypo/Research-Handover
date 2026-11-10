#!/usr/bin/env python3
"""Record one trial and compute its metrics.

Records odometry, the commanded velocity and /crowd/people at ``rate`` Hz. The
trial ends when /trial/status reports success, timeout or aborted, or when the
node is stopped (Ctrl-C). It then writes

  <output_dir>/<trial_name>_<timestamp>.npz   raw data (people padded with NaN)
  <output_dir>/trials.csv                     one line of metrics per trial

Metrics follow stage 1: outcome, time, path length, mean speed, minimum
centre distance to any tracked person, and exposure (share of time with a
tracked person closer than ``comfort_distance``). On a real robot, these
distances come from your perception, not from ground truth: report them as such.
"""
import csv
import os
import time

import signal

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


class TrialLogger(Node):
    def __init__(self):
        super().__init__("trial_logger")
        p = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.rate = p("log_rate", 20.0)
        self.name = p("trial_name", "trial")
        self.out = os.path.expanduser(p("output_dir", "~/handover_trials"))
        self.comfort = p("comfort_distance", 1.0)
        os.makedirs(self.out, exist_ok=True)
        self.odom = None
        self.cmd = np.zeros(2)
        self.people = np.zeros((0, 4))
        self.rows, self.peds = [], []
        self.t0 = None
        self.saved = False
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "odom", m), 20)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 20)
        self.create_subscription(Float32MultiArray, "/crowd/people", self.on_people, 10)
        self.create_subscription(String, "/trial/status", self.on_status, 10)
        self.create_timer(1.0 / self.rate, self.tick)

    def on_cmd(self, m):
        self.cmd = np.array([m.linear.x, m.angular.z])

    def on_people(self, m):
        self.people = np.asarray(m.data, float).reshape(-1, 4)

    def on_status(self, m):
        if m.data in ("success", "timeout", "aborted") and self.rows:
            self.save(m.data)

    def tick(self):
        if self.odom is None or self.saved:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        self.t0 = now if self.t0 is None else self.t0
        o = self.odom
        q = o.pose.pose.orientation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.rows.append([now - self.t0, o.pose.pose.position.x, o.pose.pose.position.y, yaw,
                          o.twist.twist.linear.x, o.twist.twist.angular.z, *self.cmd])
        self.peds.append(self.people.copy())

    def metrics(self, outcome):
        r = np.array(self.rows)
        xy = r[:, 1:3]
        dmin_t = np.array([np.min(np.linalg.norm(p[:, :2] - c, axis=1)) if len(p) else np.inf
                           for p, c in zip(self.peds, xy)])
        finite = np.isfinite(dmin_t)
        return {
            "trial": self.name, "outcome": outcome, "time": round(float(r[-1, 0]), 2),
            "path_length": round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 2),
            "mean_speed": round(float(r[:, 4].mean()), 3),
            "min_dist": round(float(dmin_t[finite].min()), 3) if finite.any() else "",
            "exposure_pct": round(float((dmin_t < self.comfort).mean() * 100), 2),
            "mean_people": round(float(np.mean([len(p) for p in self.peds])), 2),
        }

    def save(self, outcome):
        if self.saved or not self.rows:
            return
        self.saved = True
        stamp = time.strftime("%Y%m%d_%H%M%S")
        n = max([len(p) for p in self.peds] + [0])
        peds = np.full((len(self.peds), n, 4), np.nan)
        for k, p in enumerate(self.peds):
            peds[k, :len(p)] = p
        path = os.path.join(self.out, f"{self.name}_{stamp}.npz")
        np.savez_compressed(path, data=np.array(self.rows), people=peds, outcome=outcome,
                            columns="t,x,y,yaw,v,w,cmd_v,cmd_w")
        m = self.metrics(outcome)
        csv_path = os.path.join(self.out, "trials.csv")
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stamp", *m])
            if new:
                w.writeheader()
            w.writerow({"stamp": stamp, **m})
        self.get_logger().info(f"saved {path}: {m}")


def main():
    rclpy.init()
    node = TrialLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # finish cleanly on a second Ctrl-C
        node.save("aborted")    # no-op if the trial was already saved
        try:
            node.destroy_node()
        except BaseException:  # noqa: BLE001 - shutting down anyway
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
