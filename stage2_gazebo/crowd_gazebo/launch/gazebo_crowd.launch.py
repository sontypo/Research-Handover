"""Stage 2: robot + crowd in Gazebo Classic, driven by a socialnav_core controller.

  ros2 launch crowd_gazebo gazebo_crowd.launch.py scenario:=corridor
  ros2 launch crowd_gazebo gazebo_crowd.launch.py scenario:=crossing controller:=social_force gui:=false
  ros2 launch crowd_gazebo gazebo_crowd.launch.py controller:=none     # drive it yourself (teleop)
  ros2 launch crowd_gazebo gazebo_crowd.launch.py controller:=learned policy:=/path/to/policy.pt
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    scenario = LaunchConfiguration("scenario").perform(context)
    controller = LaunchConfiguration("controller").perform(context)
    gui = LaunchConfiguration("gui").perform(context)
    policy = LaunchConfiguration("policy").perform(context)
    world = os.path.join(get_package_share_directory("crowd_gazebo"), "worlds", f"{scenario}.world")
    if not os.path.isfile(world):
        raise RuntimeError(f"world not found: {world} (run `make worlds` and rebuild)")
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("gazebo_ros"), "launch", "gazebo.launch.py")),
        launch_arguments={"world": world, "gui": gui, "verbose": "false"}.items())
    actions = [gazebo,
               Node(package="crowd_gazebo", executable="actor_tracker", output="screen",
                    parameters=[{"use_sim_time": True}])]
    if controller != "none":
        cfg = os.path.join(get_package_share_directory("crowd_nav_bridge"), "config", "sim.yaml")
        actions += [
            Node(package="crowd_nav_bridge", executable="nav_controller", output="screen",
                 parameters=[cfg, {"use_sim_time": True, "controller": controller,
                                   "policy": os.path.abspath(policy) if policy else ""}]),
            Node(package="crowd_nav_bridge", executable="trial_logger", output="screen",
                 parameters=[cfg, {"use_sim_time": True, "trial_name": f"gazebo_{scenario}_{controller}"}]),
        ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="corridor",
                              description="corridor | crossing | chokepoint | dense_social"),
        DeclareLaunchArgument("controller", default_value="predictive_sf",
                              description="goal_seeker | social_force | predictive_sf | learned | none"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("policy", default_value="",
                              description="exported policy (.pt/.npz/.zip) for controller:=learned"),
        OpaqueFunction(function=setup),
    ])
