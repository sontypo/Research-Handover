#!/usr/bin/env bash
# Source this instead of env.sh for SIMULATION (stage 2).
# It isolates the simulated robot from any real robot on the lab network:
# a simulated /cmd_vel must never reach real hardware.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${HANDOVER_SIM_DOMAIN:-77}"
export GAZEBO_MASTER_URI="http://localhost:${HANDOVER_GAZEBO_PORT:-11399}"   # own Gazebo master
echo "[sim_env] ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=$ROS_DOMAIN_ID (isolated simulation)"
