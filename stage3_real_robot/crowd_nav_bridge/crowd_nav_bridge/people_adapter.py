#!/usr/bin/env python3
"""Adapter: your people detector/tracker -> /crowd/people.

The controller only needs the people around the robot as rows [x, y, vx, vy] in
the odometry frame. Real trackers publish many different messages, so this node
converts the two most common generic ones:

  input_type: pose_array    geometry_msgs/PoseArray (positions only)
  input_type: detection3d   vision_msgs/Detection3DArray (3-D boxes, e.g. from a
                            LiDAR or RGB-D detector); boxes outside the size gate
                            [min_height, max_height] x [.., max_width] are dropped

Points are transformed into ``target_frame`` with TF. Velocities are estimated
by greedy nearest-neighbour association between frames and smoothed
finite differences (no track IDs are assumed). If your tracker already provides
IDs and velocities, write a small node that publishes /crowd/people directly.
"""
import signal

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from tf2_ros import Buffer, TransformListener


def quat_yaw(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class PeopleAdapter(Node):
    def __init__(self):
        super().__init__("people_adapter")
        p = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731
        self.input_type = p("input_type", "pose_array")
        topic = p("input_topic", "/detections")
        self.target = p("target_frame", "odom")
        self.max_range = p("max_range", 8.0)
        self.min_h, self.max_h, self.max_w = p("min_height", 0.2), p("max_height", 2.0), p("max_width", 1.2)
        self.assoc = p("association_distance", 0.6)
        self.alpha = p("velocity_smoothing", 0.4)
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.prev = np.zeros((0, 4))
        self.prev_t = None
        if self.input_type == "pose_array":
            self.create_subscription(PoseArray, topic, self.on_pose_array, 10)
        elif self.input_type == "detection3d":
            from vision_msgs.msg import Detection3DArray
            self.create_subscription(Detection3DArray, topic, self.on_detections, 10)
        else:
            raise ValueError("input_type must be pose_array or detection3d")
        self.pub = self.create_publisher(Float32MultiArray, "/crowd/people", 10)
        self.get_logger().info(f"{self.input_type} on {topic} -> /crowd/people ({self.target})")

    def to_target(self, pts, frame, stamp):
        if not len(pts) or frame in ("", self.target):
            return pts
        try:
            tr = self.tf.lookup_transform(self.target, frame, Time.from_msg(stamp))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"TF {frame}->{self.target} unavailable: {e}",
                                   throttle_duration_sec=2.0)
            return None
        t, q = tr.transform.translation, tr.transform.rotation
        yaw = quat_yaw(q)
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        return pts @ R.T + np.array([t.x, t.y])

    def on_pose_array(self, msg):
        pts = np.array([[ps.position.x, ps.position.y] for ps in msg.poses]).reshape(-1, 2)
        self.update(pts, msg.header)

    def on_detections(self, msg):
        pts = []
        for d in msg.detections:
            s = d.bbox.size
            if self.min_h <= s.z <= self.max_h and max(s.x, s.y) <= self.max_w:
                c = d.bbox.center.position
                pts.append([c.x, c.y])
        self.update(np.array(pts).reshape(-1, 2), msg.header)

    def update(self, pts, header):
        pts = self.to_target(pts, header.frame_id, header.stamp)
        if pts is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        rows = np.zeros((len(pts), 4))
        rows[:, :2] = pts
        if self.prev_t is not None and len(self.prev) and len(pts):
            dt = max(t - self.prev_t, 1e-3)
            d = np.linalg.norm(pts[:, None] - self.prev[None, :, :2], axis=2)
            used = set()
            for i in np.argsort(d.min(axis=1)):         # closest pairs first
                cand = d[i].copy()
                cand[list(used)] = np.inf
                j = int(np.argmin(cand))
                if cand[j] < self.assoc:
                    used.add(j)
                    v = (pts[i] - self.prev[j, :2]) / dt
                    rows[i, 2:] = self.alpha * v + (1 - self.alpha) * self.prev[j, 2:]
        self.prev, self.prev_t = rows, t
        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label="people", size=len(rows), stride=4 * len(rows)),
                          MultiArrayDimension(label="x_y_vx_vy", size=4, stride=4)]
        msg.data = rows.astype(np.float32).ravel().tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = PeopleAdapter()
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
