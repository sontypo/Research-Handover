#!/usr/bin/env python3
"""Generate Gazebo Classic worlds from the socialnav_core scenarios.

Every scenario of the 2-D simulator becomes a .world file with
  * the walls of the scenario (static boxes),
  * one animated actor per pedestrian, walking its route (start, via points,
    goal) at its preferred speed; standing pedestrians use the standing pose,
  * a simple differential-drive robot with a 360 deg 2-D LiDAR at the origin,
  * the gazebo_ros_state plugin, which publishes /gazebo/model_states.

Actors in Gazebo Classic follow scripted trajectories: they do not react to the
robot and have no collision, which matches the non-yielding test protocol of
stage 1. Usage:

  python3 generate_worlds.py --out ../worlds            # all scenarios, seed 0
  python3 generate_worlds.py --scenario crossing --seed 3 --out /tmp
"""
import argparse
import math
import os

import numpy as np

from socialnav_core import SCENARIOS, make_scenario

ROBOT_SDF = """
    <model name="robot">
      <pose>{x} {y} 0.1 0 0 {yaw}</pose>
      <link name="base_link">
        <inertial><mass>12.0</mass>
          <inertia><ixx>0.2</ixx><iyy>0.3</iyy><izz>0.4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="collision"><geometry><box><size>0.56 0.42 0.18</size></box></geometry></collision>
        <visual name="visual"><geometry><box><size>0.56 0.42 0.18</size></box></geometry>
          <material><ambient>0.85 0.35 0.18 1</ambient><diffuse>0.89 0.34 0.18 1</diffuse></material></visual>
        <visual name="mast"><pose>0 0 0.35 0 0 0</pose><geometry><cylinder><radius>0.04</radius><length>0.5</length></cylinder></geometry></visual>
        <collision name="caster_front"><pose>0.2 0 -0.07 0 0 0</pose><geometry><sphere><radius>0.03</radius></sphere></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface></collision>
        <collision name="caster_back"><pose>-0.2 0 -0.07 0 0 0</pose><geometry><sphere><radius>0.03</radius></sphere></geometry>
          <surface><friction><ode><mu>0.0</mu><mu2>0.0</mu2></ode></friction></surface></collision>
      </link>
      <link name="left_wheel">
        <pose>0 0.24 -0.0 -1.5707 0 0</pose>
        <inertial><mass>0.8</mass><inertia><ixx>0.002</ixx><iyy>0.002</iyy><izz>0.004</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="collision"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></collision>
        <visual name="visual"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></visual>
      </link>
      <link name="right_wheel">
        <pose>0 -0.24 -0.0 -1.5707 0 0</pose>
        <inertial><mass>0.8</mass><inertia><ixx>0.002</ixx><iyy>0.002</iyy><izz>0.004</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="collision"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></collision>
        <visual name="visual"><geometry><cylinder><radius>0.1</radius><length>0.05</length></cylinder></geometry></visual>
      </link>
      <joint name="left_wheel_joint" type="revolute"><parent>base_link</parent><child>left_wheel</child><axis><xyz>0 0 1</xyz></axis></joint>
      <joint name="right_wheel_joint" type="revolute"><parent>base_link</parent><child>right_wheel</child><axis><xyz>0 0 1</xyz></axis></joint>
      <link name="laser">
        <pose>0 0 0.25 0 0 0</pose>
        <inertial><mass>0.1</mass><inertia><ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <visual name="visual"><geometry><cylinder><radius>0.05</radius><length>0.06</length></cylinder></geometry></visual>
        <sensor name="lidar" type="ray">
          <always_on>true</always_on><update_rate>10</update_rate><visualize>false</visualize>
          <ray>
            <scan><horizontal><samples>720</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal></scan>
            <range><min>0.25</min><max>12.0</max><resolution>0.01</resolution></range>
          </ray>
          <plugin name="lidar_plugin" filename="libgazebo_ros_ray_sensor.so">
            <ros><remapping>~/out:=scan</remapping></ros>
            <output_type>sensor_msgs/LaserScan</output_type>
            <frame_name>laser</frame_name>
          </plugin>
        </sensor>
      </link>
      <joint name="laser_joint" type="fixed"><parent>base_link</parent><child>laser</child></joint>
      <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
        <update_rate>50</update_rate>
        <left_joint>left_wheel_joint</left_joint>
        <right_joint>right_wheel_joint</right_joint>
        <wheel_separation>0.48</wheel_separation>
        <wheel_diameter>0.2</wheel_diameter>
        <max_wheel_torque>30</max_wheel_torque>
        <max_wheel_acceleration>3.0</max_wheel_acceleration>
        <publish_odom>true</publish_odom>
        <publish_odom_tf>true</publish_odom_tf>
        <publish_wheel_tf>false</publish_wheel_tf>
        <odometry_frame>odom</odometry_frame>
        <robot_base_frame>base_link</robot_base_frame>
      </plugin>
    </model>"""

WALL_SDF = """
    <model name="{name}">
      <static>true</static>
      <pose>{cx:.3f} {cy:.3f} 0.5 0 0 {yaw:.4f}</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{length:.3f} 0.1 1.0</size></box></geometry></collision>
        <visual name="visual"><geometry><box><size>{length:.3f} 0.1 1.0</size></box></geometry>
          <material><ambient>0.35 0.38 0.45 1</ambient><diffuse>0.55 0.58 0.65 1</diffuse></material></visual>
      </link>
    </model>"""


def _heading(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _waypoints(route, speed, delay, round_trip=True):
    """Waypoints (time, x, y, yaw) along the route, and back if ``round_trip``.

    The actor waits ``delay`` seconds at the start, walks each segment at
    ``speed`` and turns on the spot (0.3 s) at intermediate waypoints. One-way
    actors hold their final pose, like the pedestrians of the 2-D simulator.
    """
    pts = [tuple(p) for p in route]
    if round_trip:
        pts += pts[-2::-1]
    speed = max(speed, 0.1)
    t = 0.0
    out = []
    if delay > 0:
        out.append((0.0, *pts[0], _heading(pts[0], pts[1])))
        t = delay
    out.append((t, *pts[0], _heading(pts[0], pts[1])))
    for k in range(1, len(pts)):
        yaw = _heading(pts[k - 1], pts[k])
        t += math.dist(pts[k - 1], pts[k]) / speed
        out.append((t, *pts[k], yaw))
        if k < len(pts) - 1:
            t += 0.3
            out.append((t, *pts[k], _heading(pts[k], pts[k + 1])))
    if not round_trip:
        out.append((t + 3600.0, *pts[-1], out[-1][3]))   # stay at the goal
    return out


def actor_sdf(name, spec):
    if spec.speed <= 0.0:                               # standing pedestrian, talking
        x, y = spec.start
        anim, interp = "talk_b.dae", "false"            # no root motion when standing still
        yaw = math.atan2(-y, -x) if (x or y) else 0.0
        wps = [(0.0, x, y, yaw), (10.0, x, y, yaw)]
    else:
        anim, interp = "walk.dae", "true"
        wps = _waypoints(spec.route(), spec.speed, spec.delay, round_trip=spec.loop)
    lines = "\n".join(
        f"            <waypoint><time>{t:.2f}</time><pose>{x:.3f} {y:.3f} 0 0 0 {yaw:.4f}</pose></waypoint>"
        for t, x, y, yaw in wps)
    return f"""
    <actor name="{name}">
      <skin><filename>file://media/models/{anim}</filename><scale>1.0</scale></skin>
      <animation name="anim"><filename>file://media/models/{anim}</filename><scale>1.0</scale><interpolate_x>{interp}</interpolate_x></animation>
      <script>
        <loop>true</loop><delay_start>0.0</delay_start><auto_start>true</auto_start>
        <trajectory id="0" type="anim">
{lines}
        </trajectory>
      </script>
    </actor>"""


def world_sdf(scn):
    walls = []
    for k, w in enumerate(scn.walls):
        (ax, ay), (bx, by) = w.a, w.b
        walls.append(WALL_SDF.format(name=f"wall_{k}", cx=(ax + bx) / 2, cy=(ay + by) / 2,
                                     yaw=math.atan2(by - ay, bx - ax),
                                     length=math.hypot(bx - ax, by - ay) + 0.1))
    actors = [actor_sdf(f"ped_{i:02d}", p) for i, p in enumerate(scn.peds)]
    gx, gy = scn.robot_goal
    x, y, yaw = scn.robot_start
    return f"""<?xml version="1.0"?>
<!-- Generated by crowd_gazebo/generate_worlds.py from socialnav_core scenario '{scn.name}'.
     {scn.description}. Robot start ({x}, {y}), goal ({gx}, {gy}). Do not edit by hand. -->
<sdf version="1.6">
  <world name="crowd_{scn.name}">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    <physics type="ode"><max_step_size>0.002</max_step_size><real_time_update_rate>500</real_time_update_rate></physics>
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>20.0</update_rate>
    </plugin>
    <model name="goal_marker">
      <static>true</static><pose>{gx} {gy} 0.01 0 0 0</pose>
      <link name="link"><visual name="visual"><geometry><cylinder><radius>0.3</radius><length>0.02</length></cylinder></geometry>
        <material><ambient>0.9 0.35 0.2 1</ambient><diffuse>0.9 0.35 0.2 1</diffuse></material></visual></link>
    </model>
{ROBOT_SDF.format(x=x, y=y, yaw=yaw)}
{''.join(walls)}
{''.join(actors)}
  </world>
</sdf>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scenario", nargs="*", default=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "worlds"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in a.scenario:
        scn = make_scenario(name, a.seed)
        path = os.path.join(a.out, f"{name}.world")
        with open(path, "w") as f:
            f.write(world_sdf(scn))
        print(f"wrote {path}  ({len(scn.peds)} actors, {len(scn.walls)} walls)")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
