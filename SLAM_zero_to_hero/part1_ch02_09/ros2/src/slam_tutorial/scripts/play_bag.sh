#!/usr/bin/env bash
# Play back a ROS2 bag with simulated time on /clock.
#
# Usage:
#   scripts/play_bag.sh slam_demo_bag           # 1.0x rate
#   scripts/play_bag.sh slam_demo_bag 0.5       # 0.5x rate (slow debug)
#
# Subscribers that need /clock must set use_sim_time:=true.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <bag_dir> [rate]" >&2
    exit 1
fi

BAG="$1"
RATE="${2:-1.0}"

ros2 bag play "${BAG}" \
    --clock \
    --rate "${RATE}"
