#!/usr/bin/env python3
"""Check that this machine can run each stage of the handover (changes nothing)."""
import importlib
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK, BAD = "\033[32mOK \033[0m", "\033[31m-- \033[0m"


def mod(name, stage, hint):
    try:
        m = importlib.import_module(name)
        print(f"  {OK} {name:18s} {getattr(m, '__version__', '')}")
        return True
    except Exception:  # noqa: BLE001
        print(f"  {BAD} {name:18s} missing  -> needed for {stage}; {hint}")
        return False


def cmd(name, stage, hint):
    path = shutil.which(name)
    print(f"  {OK if path else BAD} {name:18s} {path or 'missing  -> needed for ' + stage + '; ' + hint}")
    return bool(path)


print(f"Python {platform.python_version()} on {platform.platform()}")
print("\nStage 1 (2-D simulation)")
s1 = all([mod(m, "stage 1", "bash scripts/setup_python.sh core") for m in ("numpy", "matplotlib", "PIL", "pytest")])
print("\nReinforcement learning")
rl = all([mod(m, "RL training", "bash scripts/setup_python.sh") for m in
          ("gymnasium", "stable_baselines3", "torch")])
try:
    import torch
    print(f"  {'OK ' if torch.cuda.is_available() else '-- '} CUDA GPU           "
          f"{'available' if torch.cuda.is_available() else 'not found (CPU training works, it is the default)'}")
except ImportError:
    pass
print("\nMuJoCo")
mj = mod("mujoco", "stage2_mujoco", "bash scripts/setup_python.sh")
print("\nStage 2-3 (ROS 2 Humble + Gazebo Classic)")
ros = cmd("ros2", "stages 2-3", "see docs/GETTING_STARTED.md (ROS 2 Humble)")
gz = cmd("gzserver", "stage 2", "sudo apt install ros-humble-gazebo-ros-pkgs")
colcon = cmd("colcon", "stages 2-3", "sudo apt install python3-colcon-common-extensions")
if ros:
    for pkg in ("gazebo_ros", "vision_msgs"):
        r = subprocess.run(["bash", "-c", f"source /opt/ros/humble/setup.bash && ros2 pkg prefix {pkg}"],
                           capture_output=True, text=True)
        print(f"  {OK if r.returncode == 0 else BAD} {pkg:18s} "
              f"{r.stdout.strip() if r.returncode == 0 else 'missing -> sudo apt install ros-humble-' + pkg.replace('_', '-')}")
print("\nSummary: stage 1 %s | RL %s | MuJoCo %s | ROS 2/Gazebo %s" % tuple(
    "ready" if x else "not ready" for x in (s1, rl, mj, ros and gz and colcon)))
