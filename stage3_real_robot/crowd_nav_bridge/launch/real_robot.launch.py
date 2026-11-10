"""Stage 3: run a socialnav_core controller on the real robot.

Start the robot drivers, localisation (odom -> base_link) and your people
detector first. Then:

  ros2 launch crowd_nav_bridge real_robot.launch.py controller:=social_force \
       input_topic:=/my_tracker/people input_type:=pose_array trial_name:=corridor_01

Keep a hand on the emergency stop. The parameters are in config/real_robot.yaml.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg_default = os.path.join(get_package_share_directory("crowd_nav_bridge"), "config",
                               "real_robot.yaml")
    cfg = LaunchConfiguration("config")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=cfg_default),
        DeclareLaunchArgument("controller", default_value="social_force"),
        DeclareLaunchArgument("input_topic", default_value="/detections"),
        DeclareLaunchArgument("input_type", default_value="pose_array"),
        DeclareLaunchArgument("trial_name", default_value="trial"),
        DeclareLaunchArgument("policy", default_value="",
                              description="exported policy for controller:=learned"),
        Node(package="crowd_nav_bridge", executable="people_adapter", output="screen",
             parameters=[cfg, {"input_topic": LaunchConfiguration("input_topic"),
                               "input_type": LaunchConfiguration("input_type")}]),
        Node(package="crowd_nav_bridge", executable="nav_controller", output="screen",
             parameters=[cfg, {"controller": LaunchConfiguration("controller"),
                               "policy": LaunchConfiguration("policy")}]),
        Node(package="crowd_nav_bridge", executable="trial_logger", output="screen",
             parameters=[cfg, {"trial_name": LaunchConfiguration("trial_name")}]),
    ])
