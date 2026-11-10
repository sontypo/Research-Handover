# Getting started on your own machine

This page assumes a fresh **Ubuntu 22.04** computer. Every step says what it enables,
so you can stop after the stages you need. Nothing here depends on another person's
computer: the kit is self-contained.

## 0. Get the folder

Copy the `Handover/` folder (USB, lab server or git) anywhere, e.g. `~/Handover`, and
open a terminal in it. All commands below run from that folder.

## 1. Stage 1 and RL: Python only (≈ 10 min)

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git make
bash scripts/setup_python.sh          # numpy, matplotlib, SB3, torch, MuJoCo
make doctor                           # shows what is ready
make test                             # unit tests, a few seconds
make sim2d && make gifs               # baselines -> stage1_python_sim/outputs/
```

`setup_python.sh` creates a virtual environment `.venv` (it can still see ROS 2 and
packages you already have), so your system Python is not modified; `make` and
`scripts/env.sh` use it automatically. Only want stage 1?
`bash scripts/setup_python.sh core` skips torch. No `sudo`? Use
`bash scripts/setup_python.sh all --user`. numpy is kept below 2 (`constraints.txt`)
because ROS 2 Humble needs numpy 1.x.

No GPU? Fine: the default RL training runs on the CPU (small MLPs train faster there).
To keep the download small on a CPU-only machine, install the CPU build of torch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## 2. Reinforcement learning (≈ 15 min on 16 cores)

```bash
make train-quick                     # 200k steps: checks the pipeline (a few minutes)
make train                           # 1.5 M steps, PPO, 16 parallel environments
make play                            # metrics + GIFs of the trained policy
tensorboard --logdir stage1_python_sim/rl/runs     # optional learning curves
```

Fewer cores? `make train` uses 16 worker processes; pass fewer with
`python3 stage1_python_sim/rl/train_sb3.py --n-envs 4`.

## 3. MuJoCo (no ROS needed)

```bash
make mujoco                           # trained policy (or a baseline) in MuJoCo -> GIFs
python3 stage2_mujoco/mujoco_crowd.py --scenario crossing --controller social_force --viewer
```

The viewer needs a display; on a remote machine use the GIFs.

## 4. Stages 2–3: ROS 2 Humble + Gazebo Classic (≈ 30 min)

Install ROS 2 Humble (official instructions: docs.ros.org/en/humble/Installation/
Ubuntu-Install-Debians.html), then:

```bash
sudo apt install -y ros-humble-desktop ros-humble-gazebo-ros-pkgs \
     ros-humble-vision-msgs python3-colcon-common-extensions
make doctor                           # ROS 2 / Gazebo should now be "ready"
make ros                              # generate worlds, build the two packages
make smoke                            # headless Gazebo run, prints SMOKE TEST PASSED
make gazebo SCENARIO=crossing CONTROLLER=social_force          # with the GUI
make gazebo SCENARIO=corridor CONTROLLER=learned                # your trained policy
```

Simulation always runs in an isolated ROS domain (`scripts/sim_env.sh`), so it cannot
interfere with robots or other computers on your network.

## 5. Stage 3: the real robot

Only after stages 1–2 work and with your supervisor's approval: read
`docs/REAL_ROBOT_CHECKLIST.md` and `stage3_real_robot/README.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | run `make doctor`; then `bash scripts/setup_python.sh` |
| numpy 2.x breaks ROS 2 | `requirements.txt` pins `numpy<2`; reinstall with the script |
| Gazebo hangs or a smoke test fails | another `gzserver` may be running: `pkill -f gzserver`, retry |
| Training is slow | lower `--n-envs` to your number of cores; keep `--device cpu` |
| MuJoCo cannot render | headless machines need EGL (NVIDIA driver) or `MUJOCO_GL=osmesa` |
