#!/usr/bin/env python3
"""Ground-truth "tracker" for Gazebo: actor poses -> /crowd/people.

Gazebo Classic publishes every model, including actors, on /gazebo/model_states.
This node keeps the actors whose name starts with ``prefix`` (default "ped_"),
estimates their velocities by filtered finite differences, and publishes them
in the same format that the real-robot tracker adapter produces (stage 3):

  /crowd/people        std_msgs/Float32MultiArray, rows [x, y, vx, vy] in "odom"
  /crowd/people_poses  geometry_msgs/PoseArray (for RViz)

Because it uses ground truth, this node is the ideal perception case. To test
robustness, set ``noise_std`` (m) and ``drop_prob`` (probability of a missed
detection per message).
"""
import signal

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


def people_msg(rows):
    msg = Float32MultiArray()
    rows = np.asarray(rows, np.float32).reshape(-1, 4)
    msg.layout.dim = [MultiArrayDimension(label="people", size=len(rows), stride=4 * len(rows)),
                      MultiArrayDimension(label="x_y_vx_vy", size=4, stride=4)]
    msg.data = rows.ravel().tolist()
    return msg


class ActorTracker(Node):
    def __init__(self):
        super().__init__("actor_tracker")
        self.prefix = self.declare_parameter("prefix", "ped_").value
        self.frame = self.declare_parameter("frame_id", "odom").value
        self.rate = self.declare_parameter("rate", 15.0).value
        self.alpha = self.declare_parameter("velocity_smoothing", 0.4).value
        self.noise = self.declare_parameter("noise_std", 0.0).value
        self.drop = self.declare_parameter("drop_prob", 0.0).value
        self.rng = np.random.default_rng(0)
        self.latest = {}
        self.prev = {}
        self.vel = {}
        self.create_subscription(ModelStates, "/gazebo/model_states", self.on_states, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/crowd/people", 10)
        self.pub_pose = self.create_publisher(PoseArray, "/crowd/people_poses", 10)
        self.create_timer(1.0 / self.rate, self.tick)
        self.get_logger().info(f"tracking models named '{self.prefix}*' at {self.rate} Hz")

    def on_states(self, msg):
        for name, pose in zip(msg.name, msg.pose):
            if name.startswith(self.prefix):
                self.latest[name] = np.array([pose.position.x, pose.position.y])

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        rows = []
        for name, p in self.latest.items():
            if name in self.prev:
                t0, p0 = self.prev[name]
                dt = now - t0
                if dt > 1e-3:
                    v = (p - p0) / dt
                    self.vel[name] = self.alpha * v + (1 - self.alpha) * self.vel.get(name, v)
            self.prev[name] = (now, p.copy())
            if self.rng.random() < self.drop:
                continue
            q = p + self.rng.normal(0, self.noise, 2) if self.noise > 0 else p
            rows.append([*q, *self.vel.get(name, np.zeros(2))])
        self.pub.publish(people_msg(rows))
        pa = PoseArray()
        pa.header.frame_id = self.frame
        pa.header.stamp = self.get_clock().now().to_msg()
        for r in rows:
            ps = Pose()
            ps.position.x, ps.position.y = float(r[0]), float(r[1])
            yaw = float(np.arctan2(r[3], r[2]))
            ps.orientation.z, ps.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
            pa.poses.append(ps)
        self.pub_pose.publish(pa)


def main():
    rclpy.init()
    node = ActorTracker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # finish cleanly on a second Ctrl-C
        try:
            node.destroy_node()
        except BaseException:  # noqa: BLE001 - shutting down anyway
            pass
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
