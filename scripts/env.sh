#!/usr/bin/env bash
# Source this file in every terminal that runs stage 2 or 3:
#   source scripts/env.sh
HANDOVER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
[ -f "$HANDOVER/ros_ws/install/setup.bash" ] && source "$HANDOVER/ros_ws/install/setup.bash"
export PYTHONPATH="$HANDOVER/core:${PYTHONPATH}"
# Python dependencies installed by scripts/setup_python.sh
[ -d "$HANDOVER/.venv" ] && export PATH="$HANDOVER/.venv/bin:${PATH}"
export HANDOVER
