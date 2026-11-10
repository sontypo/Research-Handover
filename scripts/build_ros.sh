#!/usr/bin/env bash
# Build the two ROS 2 packages (stage 2 and stage 3) in Handover/ros_ws.
set -e
HANDOVER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
export PYTHONPATH="$HANDOVER/core:${PYTHONPATH}"
mkdir -p "$HANDOVER/ros_ws/src"
ln -sfn "$HANDOVER/stage2_gazebo/crowd_gazebo" "$HANDOVER/ros_ws/src/crowd_gazebo"
ln -sfn "$HANDOVER/stage3_real_robot/crowd_nav_bridge" "$HANDOVER/ros_ws/src/crowd_nav_bridge"
cd "$HANDOVER/ros_ws"
colcon build --symlink-install
echo "Built. In every new terminal: source $HANDOVER/scripts/env.sh"
