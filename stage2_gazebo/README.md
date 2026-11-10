# Stage 2 — Gazebo Classic

ROS 2 package `crowd_gazebo`:

* `crowd_gazebo/generate_worlds.py` — scenarios of `socialnav_core` → `worlds/*.world`
  (walls, a diff-drive robot with a 360° LiDAR, one actor per pedestrian).
  Regenerate with `make worlds` (or `--seed N` for other crowds).
* `actor_tracker` — `/gazebo/model_states` → `/crowd/people` (+ optional noise/dropouts).
* `launch/gazebo_crowd.launch.py` — Gazebo + tracker + `nav_controller` + `trial_logger`.

```bash
make ros && source scripts/sim_env.sh       # isolated ROS domain for simulation
ros2 launch crowd_gazebo gazebo_crowd.launch.py scenario:=crossing controller:=predictive_sf
ros2 launch crowd_gazebo gazebo_crowd.launch.py controller:=none   # teleop yourself
bash scripts/gazebo_smoke_test.sh dense_social social_force          # headless check
```

Actors follow scripted routes: they do not react to the robot and have no collision.
Trials are logged to `~/handover_trials/gazebo/`.
