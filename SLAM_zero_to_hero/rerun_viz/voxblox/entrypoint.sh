#!/bin/bash
# Entrypoint for Voxblox + Rerun bridge container.
# Expects rosbag file at /data/input.bag (mount it read-only).
# Set LAUNCH_FILE env var to override the default voxblox launch file.
set -e
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

# 1. roscore
roscore &
ROSCORE_PID=$!
sleep 3

# 2. Voxblox node (rviz disabled). Default launch: euroc.launch
#    Override via: -e LAUNCH_FILE=cow_and_lady.launch
LAUNCH_FILE="${LAUNCH_FILE:-euroc.launch}"
roslaunch voxblox_ros "$LAUNCH_FILE" \
  enable_mesh_pub:=true \
  publish_tsdf_map:=true \
  publish_esdf_map:=true \
  &
VOXBLOX_PID=$!
sleep 5

# 3. Rerun bridge (subscribes to Voxblox topics + serves web viewer)
python3 /app/ros_rerun_bridge.py &
BRIDGE_PID=$!
sleep 2

echo ""
echo "=== Voxblox + Rerun ==="
echo "=== Open http://localhost:9090/?url=ws://localhost:9877 ==="
echo ""

# 4. Play the rosbag (blocks until done)
BAG="${BAG_PATH:-/data/input.bag}"
if [ -f "$BAG" ]; then
  echo "Playing rosbag: $BAG"
  rosbag play "$BAG" --clock -r 1.0
else
  echo "No rosbag at $BAG."
  echo "Set BAG_PATH env var or mount a bag at /data/input.bag."
  echo "  e.g.: docker run -v /path/to/cow_and_lady_dataset.bag:/data/input.bag ..."
  echo "Keeping bridge alive for manual topic publishing..."
fi

echo "Bag finished. Bridge still serving on :9090. Ctrl+C to exit."
wait $BRIDGE_PID
