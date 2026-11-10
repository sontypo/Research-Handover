#!/usr/bin/env bash
# Headless end-to-end check of stage 2: Gazebo + actors + tracker + controller + logger.
#   bash scripts/gazebo_smoke_test.sh [scenario] [controller] [max_seconds] [policy_file]
# Passes when a trial line is appended to ~/handover_trials/gazebo/trials.csv.
# Runs in an isolated ROS domain (scripts/sim_env.sh): nothing reaches a real robot.
SCENARIO="${1:-crossing}"
CONTROLLER="${2:-predictive_sf}"
SECONDS_MAX="${3:-90}"
POLICY="${4:-}"
HANDOVER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HANDOVER/scripts/sim_env.sh"
CSV="$HOME/handover_trials/gazebo/trials.csv"
LOG="$HANDOVER/ros_ws/log/smoke_${SCENARIO}_${CONTROLLER}.log"
mkdir -p "$(dirname "$LOG")"
lines() { [ -f "$CSV" ] && wc -l < "$CSV" || echo 0; }
BEFORE=$(lines)

# the launch runs in its own process group, so that exactly this run is stopped
PIDFILE="$(mktemp)"
setsid bash -c 'echo $$ > "$0"; exec ros2 launch crowd_gazebo gazebo_crowd.launch.py \
    scenario:="$1" controller:="$2" policy:="$3" gui:=false' \
    "$PIDFILE" "$SCENARIO" "$CONTROLLER" "$POLICY" > "$LOG" 2>&1 &
for ((i = 0; i < 50; i++)); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
PGID="$(cat "$PIDFILE")"; rm -f "$PIDFILE"
for ((i = 0; i < SECONDS_MAX; i++)); do
    sleep 1
    [ "$(lines)" -gt "$BEFORE" ] && break
    kill -0 "$PGID" 2>/dev/null || break
done
kill -INT -- -"$PGID" 2>/dev/null
for ((i = 0; i < 15; i++)); do kill -0 -- -"$PGID" 2>/dev/null || break; sleep 1; done
kill -9 -- -"$PGID" 2>/dev/null
sleep 1

if [ "$(lines)" -gt "$BEFORE" ]; then
    echo "SMOKE TEST PASSED [$SCENARIO/$CONTROLLER]: $(tail -n 1 "$CSV")"
else
    echo "SMOKE TEST FAILED [$SCENARIO/$CONTROLLER]: no trial logged (see $LOG)"
    exit 1
fi
