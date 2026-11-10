# NRSL Social-Navigation Handover

A starter kit for the next student working on **robot navigation among people** at the
Networked Robotic Systems Laboratory (NRSL), NCKU. It takes you from a 2-D Python
simulation and a reinforcement-learning pipeline (stable-baselines3), through MuJoCo and
Gazebo crowd simulations, to a ROS 2 node that can drive a real mobile robot among
pedestrians, with one shared controller library and one command per step.

**New here? Start with `docs/GETTING_STARTED.md`** — it sets everything up on your own
Ubuntu 22.04 machine.

> **Scope.** This kit is an *idea-level, minimal* handover written from scratch. It
> contains baselines (Social Force, constant-velocity prediction, a generic PPO
> policy), tools and interfaces. It does **not** contain the forecaster, the learned policy, the rewards,
> the training code, the checkpoints or the hardware stack of the original thesis.
> Designing and training your own methods is your research.

## Quick start

```bash
cd Handover
bash scripts/setup_python.sh   # Python deps into .venv, system Python untouched
make doctor    # what can this machine run?
make test      # unit tests of the core library            (seconds)
make sim2d     # stage 1: 2-D benchmark -> stage1_python_sim/outputs/   (~15 s)
make gifs      # stage 1: animations of every scenario
make train     # RL: PPO with stable-baselines3, saved as a .pt model (~10-15 min)
make play      # RL: metrics and GIFs of the trained policy
make mujoco    # MuJoCo: the trained policy (or a baseline) -> GIFs
make ros       # stage 2+3: generate Gazebo worlds, build the ROS 2 packages
make smoke     # stage 2: headless Gazebo end-to-end test   (~30-90 s)
make gazebo SCENARIO=crossing CONTROLLER=learned           # Gazebo with GUI
make slides    # rebuild slides/Handover_Guide.pptx
```

Requirements: Ubuntu 22.04 with Python 3.10 (`requirements*.txt`); for stages 2–3
ROS 2 Humble, Gazebo Classic 11, `ros-humble-gazebo-ros-pkgs`, `ros-humble-vision-msgs`.
Details: `docs/GETTING_STARTED.md`.

## Roadmap

| Stage | Folder | You learn / do | Exit criterion |
|---|---|---|---|
| 0 | `docs/` | the problem, metrics, safety | read `docs/CONCEPTS.md` |
| 1 | `stage1_python_sim/` | crowd model, baselines, metrics, 2-D visualisation; RL with stable-baselines3 (`rl/`); design and train **your** method | your method beats the baselines on all 4 scenarios with non-yielding pedestrians |
| 1b | `stage2_mujoco/` | the same controller/policy in a physics engine (inertia, contacts, simulated LiDAR) | policy still succeeds in MuJoCo |
| 2 | `stage2_gazebo/` | ROS 2, Gazebo, topics, timing, sensor noise | `make smoke` passes for your controller on all 4 worlds |
| 3 | `stage3_real_robot/` | perception adapter, safety layers, calibration, experiments with people | supervised runs following `docs/REAL_ROBOT_CHECKLIST.md` |

Details and suggested milestones: `docs/ROADMAP.md`.

## Folder map

```
Handover/
├── README.md                   this file
├── Makefile                    one command per step
├── requirements*.txt           Python dependencies (core / RL / MuJoCo), constraints.txt
├── docs/                       GETTING_STARTED, CONCEPTS, ROADMAP, INTERFACES,
│                               REAL_ROBOT_CHECKLIST
├── slides/                     Handover_Guide.pptx (+ build_slides.py)
├── core/socialnav_core/        shared library (scenarios, Social Force crowd, robot,
│                               controllers, predictor, observation, metrics, env,
│                               gym_env, policy_io, viz)
├── stage1_python_sim/          run_benchmark.py, animate.py, train_policy_template.py
│   └── rl/                     train_sb3.py, play.py, export_policy.py
├── stage2_mujoco/              mujoco_crowd.py (policy or controller in MuJoCo)
├── stage2_gazebo/crowd_gazebo/ ROS 2 package: world generator, worlds, actor tracker, launch
├── stage3_real_robot/crowd_nav_bridge/
│                               ROS 2 package: nav_controller, people_adapter,
│                               trial_logger, configs, launch
└── scripts/                    setup_python.sh, doctor.py, env.sh, sim_env.sh,
                                build_ros.sh, gazebo_smoke_test.sh
```

## One controller, three stages

Every controller implements `act(obs) -> (v, w)` on the same `Obs` structure. In stage 1
the simulator fills `Obs`; in stages 2 and 3 `nav_controller` fills it from `/odom`,
`/crowd/people` and `/scan`, and `mujoco_crowd.py` fills it from MuJoCo. A controller
you write and test in 2-D therefore runs in MuJoCo, Gazebo and on the robot without
changes. Learned policies are exported once (`policy.pt` / `policy.npz` +
`policy_meta.json`) and loaded by every runner through `LearnedPolicy`.

## Safety

* Simulation runs in an **isolated ROS domain** (`scripts/sim_env.sh`: localhost only,
  `ROS_DOMAIN_ID=77`). Never run a simulation on the domain of a real robot.
* Real-robot runs start with the conservative limits of
  `stage3_real_robot/crowd_nav_bridge/config/real_robot.yaml` and follow
  `docs/REAL_ROBOT_CHECKLIST.md`: an operator with an emergency stop, the base driver's
  command timeout enabled, and lab approval for experiments with people.

## Where the original research is described

The ideas behind this kit are discussed in the MSc thesis *Safe Navigation for
Robot-Crowd Interaction via Learning-Based Approaches* (H.-S. Nguyen, NCKU) and in the
public preprint *Social Graph Mamba* (arXiv:2608.21411). Read them for the motivation
and the evaluation methodology; the implementations are not part of this handover.
