# Real-robot checklist

Print this page and tick every item before each session. If one item fails, do not run.

## Before the session

- [ ] Experiments with people are approved by the lab / supervisor (and the IRB, when
      required); volunteers are informed and can stop at any time.
- [ ] A second person is present as safety operator, holding the **hardware emergency
      stop** for the whole run.
- [ ] The test area is closed to uninvolved people; the path is free of fragile objects.
- [ ] Batteries charged; wheels and sensors mounted firmly.

## Software bring-up

- [ ] Real-robot terminals use the robot's ROS domain, **not** `scripts/sim_env.sh`;
      no simulation runs on that domain.
- [ ] The base driver's **command timeout** is enabled (the robot stops if `/cmd_vel`
      stops arriving).
- [ ] `/odom` is correct: drive 3 m straight and turn 360° by teleop; error < 5 %.
- [ ] `/scan` is in the correct frame; `laser_x` matches the LiDAR position.
- [ ] `/crowd/people` shows a person standing 2 m in front at about (2, 0) in RViz,
      and moves correctly when the person walks.
- [ ] `yaw_rate_scale` calibrated: command 0.5 rad/s for 10 s, measure the turned angle.

## First runs (in this order)

1. Empty area, goal 3–5 m, `max_v 0.3`: the robot stops within `goal_tolerance`.
2. Emergency stop test: a person steps in front at 1 m; translation stops.
3. One volunteer walking head-on, then crossing; then small groups.
4. Raise the limits step by step (never above the training limits of your policy).

## During every trial

- [ ] Unique `trial_name`; the logger prints `saved ...` at the end.
- [ ] Note in the lab notebook: scenario, people, instructions, anything unusual.
- [ ] Stop immediately on unexpected behaviour; never "let it try once more" unsupervised.

## After the session

- [ ] Copy `~/handover_trials/real/` to the lab storage.
- [ ] Report outcomes, minimum distances (from perception, not ground truth), number of
      trials, and limitations.
