#!/bin/bash
# Entrypoint for FAST-LIO2 + Rerun bridge container.
# Expects rosbag file at /data/input.bag (mount it read-only).
set -e
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

# 1. roscore
roscore &
ROSCORE_PID=$!
sleep 3

# 2. FAST-LIO2 launch (rviz disabled). LAUNCH_FILE env var overrides default.
LAUNCH_FILE="${LAUNCH_FILE:-mapping_hilti.launch}"
roslaunch fast_lio "$LAUNCH_FILE" rviz:=false &
FASTLIO_PID=$!
sleep 5

# 3. rerun bridge (subscribes to FAST-LIO2 topics + serves web viewer)
python3 /app/ros_rerun_bridge.py &
BRIDGE_PID=$!
sleep 2

echo ""
echo "=== Open http://localhost:9090/?url=ws://localhost:9877 ==="
echo ""

# 4. Play the rosbag (blocks until done)
BAG="${BAG_PATH:-/data/input.bag}"
if [ -f "$BAG" ]; then
  echo "Playing rosbag: $BAG"
  rosbag play "$BAG" --clock -r 1.0
else
  echo "No rosbag at $BAG. Set BAG_PATH env var or mount a bag at /data/input.bag."
  echo "Keeping bridge alive for manual topic publishing..."
fi

echo "Bag finished. Bridge still serving on :9090. Ctrl+C to exit."
wait $BRIDGE_PID
