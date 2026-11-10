#!/usr/bin/env bash
# Install the Python dependencies of the handover.
#
#   bash scripts/setup_python.sh            # core + RL + MuJoCo, into .venv (recommended)
#   bash scripts/setup_python.sh core       # only stage 1 (no torch)
#   bash scripts/setup_python.sh all --user # no venv available: install with pip --user
#
# The virtual environment .venv is created with --system-site-packages, so it also sees
# ROS 2 (rclpy) and anything already installed; pip then installs only what is missing.
# constraints.txt keeps numpy < 2, which ROS 2 Humble needs. Delete .venv to undo.
set -e
HANDOVER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HANDOVER"
WHAT="${1:-all}"
REQS="-r requirements.txt"
[ "$WHAT" = "all" ] && REQS="$REQS -r requirements-rl.txt -r requirements-mujoco.txt"

if [ "${2:-}" = "--user" ]; then
    python3 -m pip install --user -c constraints.txt $REQS
    echo "Installed with pip --user."
    exit 0
fi

if ! python3 -m venv --system-site-packages .venv 2>/dev/null; then
    rm -rf .venv
    echo "Cannot create a virtual environment: python3-venv is missing."
    echo "  Recommended:  sudo apt install python3-venv && bash scripts/setup_python.sh $WHAT"
    echo "  Without sudo: bash scripts/setup_python.sh $WHAT --user"
    exit 1
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -c constraints.txt $REQS
echo "Installed into $HANDOVER/.venv (used automatically by 'make' and scripts/env.sh)."
