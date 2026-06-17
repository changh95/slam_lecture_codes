#!/usr/bin/env bash
# Run FAST-LIVO2 on the Hilti SLAM Challenge 2022 exp14_basement_2.bag
# Hilti uses Hesai PandarXT-32 + Alphasense IMU + Alphasense fisheye camera —
# FAST-LIVO2 ships a config tuned for exactly this hardware
# (mapping_hesaixt32_hilti22.launch + HILTI22.yaml + camera_fisheye_HILTI22.yaml).

set -euo pipefail

BAG="${1:-/data/exp14_basement_2.bag}"
SEQ_NAME="${2:-hilti_exp14_basement_2}"

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

# Bag duration is ~74s; leave headroom for FAST-LIVO2 to flush the trajectory
BAG_DURATION_BUFFER=15

echo "[run] starting roscore"
roscore &
ROSCORE_PID=$!
sleep 3

echo "[run] launching fastlivo_mapping (headless, rviz disabled)"
roslaunch fast_livo mapping_hesaixt32_hilti22.launch rviz:=false \
  --wait \
  evo/seq_name:="$SEQ_NAME" \
  > /out/fastlivo.log 2>&1 &
LAUNCH_PID=$!
sleep 8

echo "[run] playing bag $BAG"
rosbag play --clock --quiet "$BAG"
echo "[run] bag finished, waiting ${BAG_DURATION_BUFFER}s for trajectory flush"
sleep "$BAG_DURATION_BUFFER"

echo "[run] stopping fastlivo_mapping"
kill -INT $LAUNCH_PID 2>/dev/null || true
sleep 5
kill -KILL $LAUNCH_PID 2>/dev/null || true

kill -INT $ROSCORE_PID 2>/dev/null || true
wait 2>/dev/null || true

# Trajectory was written to ROOT_DIR/Log/result/<seq>.txt; that path is bind-mounted
TRAJ="/out/${SEQ_NAME}.txt"
if [ -f "$TRAJ" ]; then
  echo "[run] OK: trajectory written -> $TRAJ"
  wc -l "$TRAJ"
  head -3 "$TRAJ"
else
  echo "[run] WARN: $TRAJ missing — checking Log/result/" >&2
  ls -la /catkin_ws/src/FAST-LIVO2/Log/result/ 2>&1 || true
fi
echo "[run] done"
