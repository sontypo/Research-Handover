# Stage 3 — Real robot

ROS 2 package `crowd_nav_bridge` (also used by stage 2):

* `nav_controller` — runs any `socialnav_core` controller (or a trained `.pt` policy) with
  safety layers: stale-odometry stop, LiDAR stop, kinematic filter, yaw-rate scale.
* `people_adapter` — your detector (`PoseArray` or `vision_msgs/Detection3DArray`) →
  `/crowd/people` in the odometry frame, with velocity estimation.
* `trial_logger` — one `.npz` per trial and a `trials.csv` with metrics.
* `config/real_robot.yaml` — conservative limits for the first runs.

```bash
source scripts/env.sh        # NOT sim_env.sh: use the robot's ROS domain
ros2 launch crowd_nav_bridge real_robot.launch.py controller:=social_force \
     input_topic:=/my_detector/people input_type:=pose_array trial_name:=corridor_01
```

You provide: the base driver (`/cmd_vel`, `/odom` with a command timeout), localisation,
the LiDAR (`/scan`) and a people detector. Follow `docs/REAL_ROBOT_CHECKLIST.md`.
