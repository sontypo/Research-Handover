# Interfaces

## Python (`core/socialnav_core`)

| Object | Contract |
|---|---|
| `Obs` | `robot_xy (2,)`, `robot_theta`, `robot_v`, `robot_w`, `goal (2,)`, `ped_pos (N,2)`, `ped_vel (N,2)`, `walls [(a,b)]`, `obstacles (M,2)`, `ranges (n,)` (optional LiDAR rays), `t`, `dt`, `robot_radius`; world/odom frame, metres, seconds |
| `Controller.act(obs)` | returns `(v, w)` in m/s and rad/s |
| `Predictor.predict(pos, vel, horizon, dt)` | returns `(K, T, N, 2)`; `weights()` returns `(K,)` |
| `observation.to_vector(obs, k=8, n_rays=16)` | `5 + 6k + n_rays` floats for neural policies (goal, k nearest people in the body frame, range rays) |
| `env.CrowdNavEnv` | `reset() -> obs_vec`, `step(a in [-1,1]^2) -> obs_vec, reward, done, info`; `config()` = settings to deploy |
| `gym_env.SocialNavGymEnv` | Gymnasium wrapper (stable-baselines3) |
| `sim.EpisodeLog` | `t, robot (T,5), cmd (T,2), peds (T,N,2), ped_vel, goal, outcome`; `.save(path)` |

## Exported policies

| File | Content |
|---|---|
| `policy.pt` | the actor as a TorchScript model: input (batch, obs_dim), output action (batch, 2); `torch.jit.load` |
| `policy.npz` | same actor's MLP weights (`W0, b0, ...`, activation, output squashing); runs with numpy (`policy_io.numpy_mlp`) |
| `policy_meta.json` | `dt`, `obs` (k, n_rays, ray_range, max_range), `action` (v_min, v_max, w_max), robot limits, reward weights |

`policy_io.load_policy(path)` loads `.pt`, `.npz` or an SB3 `.zip`;
`LearnedPolicy(policy_path=...)` maps the action with the training speed range and then
clips it to the limits of the robot it runs on.

## ROS 2 topics (stages 2 and 3)

| Topic | Type | Producer → consumer | Notes |
|---|---|---|---|
| `/odom` | `nav_msgs/Odometry` | robot driver / Gazebo → `nav_controller`, `trial_logger` | frame `odom`, child `base_link` |
| `/crowd/people` | `std_msgs/Float32MultiArray` | `actor_tracker` or `people_adapter` → `nav_controller`, `trial_logger` | rows `[x, y, vx, vy]` in `odom` |
| `/crowd/people_poses` | `geometry_msgs/PoseArray` | `actor_tracker` → RViz | visualisation |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR → `nav_controller` | obstacle points + emergency stop |
| `/goal_pose` | `geometry_msgs/PoseStamped` | you / RViz → `nav_controller` | starts a new trial |
| `/cmd_vel` | `geometry_msgs/Twist` | `nav_controller` → base | `linear.x`, `angular.z` only |
| `/trial/status` | `std_msgs/String` | `nav_controller` → `trial_logger` | running, success, timeout |
| `/gazebo/model_states` | `gazebo_msgs/ModelStates` | Gazebo → `actor_tracker` | actors named `ped_*` |

`/crowd/people` uses a generic message on purpose. If your detector publishes IDs and
velocities, publish `/crowd/people` directly; otherwise use `people_adapter`.

## Main parameters (`crowd_nav_bridge/config/*.yaml`)

| Parameter | Meaning |
|---|---|
| `controller` | `goal_seeker`, `social_force`, `predictive_sf`, `learned` |
| `policy` | exported policy for `learned` (`.pt`, `.npz`, or SB3 `.zip`); the control rate follows `policy_meta.json` |
| `goal_x`, `goal_y`, `goal_tolerance`, `time_limit` | trial definition |
| `max_v`, `min_v`, `max_w`, `max_acc`, `max_alpha` | kinematic filter |
| `estop_distance`, `estop_half_angle_deg` | LiDAR stop around the travel direction |
| `use_scan_obstacles`, `obstacle_range`, `obstacle_sector_deg`, `person_exclusion`, `laser_x` | LiDAR obstacles given to the controller |
| `yaw_rate_scale` | multiplies the yaw-rate command (calibrate per robot) |
| `input_type`, `input_topic`, `target_frame`, size gates | `people_adapter` |
| `output_dir`, `comfort_distance` | `trial_logger` |
