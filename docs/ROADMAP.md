# Roadmap: from 2-D simulation to the real robot

Indicative durations assume part-time work alongside courses. Do not skip the exit
criteria: each stage removes a class of bugs before the next, more expensive one.

## Stage 0 — orientation (week 1)

* Read `docs/CONCEPTS.md` and two of the suggested papers.
* `make test && make sim2d && make gifs`; open `stage1_python_sim/outputs/`.
* Exercise: explain why `goal_seeker` collides in every scenario and why
  `predictive_sf` keeps larger distances in the corridor.

## Stage 1 — 2-D simulation (weeks 2–10)

Code: `core/socialnav_core/`, `stage1_python_sim/`.

1. **Understand the baselines.** Change `SocialForceController` gains; plot the
   success/collision/timeout trade-off. Re-run with `--robot-visible`.
2. **Extend the benchmark.** Add a scenario (e.g. a T-junction or a queue) in
   `scenarios.py`, with seeds and a description.
3. **Prediction.** Implement a `Predictor` subclass; evaluate it with `ade_fde` on logs
   saved with `--save-logs`; plug it into `PredictiveSocialForce`.
4. **Your method.** Start from the working RL pipeline (`rl/`: `make train`,
   `make play`, `run_benchmark.py --policy ...`), then design your own controller or
   policy. Typical choices to make yourself: the encoding of a
   variable number of people, the reward, the training algorithm, the curriculum.
5. **Evaluate fairly.** Same seeds for every method, ≥ 50 seeds per cell, all four
   scenarios, non-yielding people.

Exit: your method beats `predictive_sf` on success **and** minimum distance on all four
scenarios, and you can explain its failures with `animate.py`.

## Stage 1b — MuJoCo (week 11, optional)

`make mujoco` runs the same policy with inertia, contacts and a simulated LiDAR.
Exit: the policy still reaches the goal in all four scenarios.

## Stage 2 — Gazebo (weeks 11–14)

Code: `stage2_gazebo/crowd_gazebo/`, `stage3_real_robot/crowd_nav_bridge/`.

1. `make ros`, then `source scripts/sim_env.sh` in every terminal.
2. `make gazebo SCENARIO=corridor CONTROLLER=social_force`; inspect `/odom`,
   `/crowd/people`, `/scan`, `/cmd_vel` with `ros2 topic echo` and RViz.
3. Add perception noise: `ros2 param set /actor_tracker noise_std 0.1` and `drop_prob 0.1`.
4. Run your controller (`controller:=learned policy:=.../policy.pt` or a new class).
5. `make smoke` for every scenario; compare `~/handover_trials/gazebo/trials.csv` with
   stage 1.

Exit: all four worlds pass the smoke test with your controller, with noise enabled.

Notes: Gazebo actors follow scripted paths, do not react and have no collision (the
LiDAR does not see them). The world generator (`generate_worlds.py`) rebuilds worlds
from the same scenarios, so changes in stage 1 propagate with `make worlds`.

## Stage 3 — real robot (weeks 15+)

Code: `stage3_real_robot/crowd_nav_bridge/`, `docs/REAL_ROBOT_CHECKLIST.md`.

1. Bring up the robot driver and localisation so that `/odom` (odom → base_link) is
   correct; drive by teleop and compare odometry with a tape measure.
2. Bring up a people detector; connect it with `people_adapter`
   (`input_type:=pose_array` or `detection3d`) and check `/crowd/people` in RViz.
3. Calibrate `yaw_rate_scale` (command a constant yaw rate, measure the real one).
4. Empty-room runs, then one person, then small groups, with the conservative limits.
5. Log every trial (`trial_logger`), keep a lab notebook, and report the protocol.

Exit: supervised trials with volunteers, logged, with metrics comparable to stage 2.
