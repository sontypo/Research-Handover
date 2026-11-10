# MuJoCo runner

`mujoco_crowd.py` runs any socialnav_core controller or an exported policy in MuJoCo,
without ROS. The world is generated from the same scenarios as stages 1 and 2:

* walls are static boxes; touching one is a collision (MuJoCo contacts);
* the robot is a 25 kg cylinder on planar joints (x, y, yaw) driven by velocity
  actuators, so it has inertia and lag; unicycle commands are rate-limited first;
* people are mocap bodies moved by the Social Force crowd (they do not react);
* a 360° LiDAR is simulated with `mj_ray` (walls only, as in Gazebo); it feeds the
  rays of learned policies and the obstacle points of the baselines.

```bash
python3 mujoco_crowd.py --scenario corridor --controller predictive_sf --gif out.gif
python3 mujoco_crowd.py --all --policy ../stage1_python_sim/rl/runs/ppo_mixed/policy.npz --episodes 10
python3 mujoco_crowd.py --scenario crossing --policy .../policy.pt --viewer     # needs a display
```

Next steps if you need more physics: replace the planar joints with two wheel hinges
and a caster, add sensor noise to `rays()`, or make people react to the robot.
