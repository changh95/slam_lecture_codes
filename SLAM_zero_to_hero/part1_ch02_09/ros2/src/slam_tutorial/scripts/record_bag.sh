#!/usr/bin/env bash
# Record the slam_tutorial demo topics to a ROS2 bag.
#
# Usage:
#   scripts/record_bag.sh                # writes ./slam_demo_bag/
#   scripts/record_bag.sh my_run         # writes ./my_run/
#   scripts/record_bag.sh my_run mcap    # writes ./my_run/ in MCAP format
#
# In another terminal, start the demo first:
#   ros2 launch slam_tutorial demo.launch.py
set -euo pipefail

OUT="${1:-slam_demo_bag}"
FORMAT="${2:-sqlite3}"   # sqlite3 (default) or mcap

ros2 bag record \
    -o "${OUT}" \
    -s "${FORMAT}" \
    /robot/pose \
    /imu/data \
    /tf \
    /tf_static
