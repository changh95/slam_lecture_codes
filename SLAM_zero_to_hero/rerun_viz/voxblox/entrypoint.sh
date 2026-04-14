#!/bin/bash
# Entrypoint for Voxblox + Rerun bridge container.
# Mount the rosbag at /data/input.bag (read-only).
# Set LAUNCH_FILE env var to override the default voxblox launch file.
set -e
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

# (tcmalloc LD_PRELOAD intentionally NOT enabled: on aarch64 it deadlocks
# rosbag play's boost-thread wait loop. Voxblox still benefits from
# integrator_threads=20, LTO, -march=native, and voxel_carving=false.)

# 1. roscore
roscore &
sleep 3

# 2. Rerun viewer as a SEPARATE process (acts as TCP->WebSocket proxy). The
#    Python bridge connects via rr.connect_tcp() so log calls are just socket
#    sends and can never back-pressure the ROS callback thread.
#    --drop-at-latency ensures the viewer drops old packets instead of
#    letting its own queue grow unbounded.
rerun --serve-web \
      --port 9876 \
      --web-viewer-port 9090 \
      --ws-server-port 9877 \
      --bind 0.0.0.0 \
      --drop-at-latency 200ms &
RERUN_PID=$!
sleep 2

# 3. Rerun bridge (subscribes + forwards to rerun TCP sink on :9876)
python3 /app/ros_rerun_bridge.py &
BRIDGE_PID=$!
sleep 2

echo ""
echo "=== Voxblox + Rerun ==="
echo "=== Open http://localhost:9090/?url=ws://localhost:9877 ==="
echo ""

# 3. Launch voxblox. The cow_and_lady launch file plays its own bag, so we
#    pass the mounted bag path via the bag_file arg. For launches without
#    that arg, set LAUNCH_FILE and play the bag yourself.
LAUNCH_FILE="${LAUNCH_FILE:-cow_and_lady_dataset.launch}"
BAG="${BAG_PATH:-/data/input.bag}"
if [ ! -f "$BAG" ]; then
  echo "ERROR: no rosbag at $BAG. Mount one with -v /host/bag:/data/input.bag"
  exit 1
fi
echo "Launching $LAUNCH_FILE (play_bag:=false)"
roslaunch voxblox_ros "$LAUNCH_FILE" play_bag:=false &
LAUNCH_PID=$!
sleep 5  # let voxblox_node come up and advertise
# NOTE: no --clock! cow_and_lady bag timestamps are from 2016 and rosbag's
# --clock path on noetic+aarch64 hangs with a bogus ~9.6-year delay. Voxblox
# matches pointcloud<->transform via msg header timestamps (not ros::Time::now),
# so wall-clock playback works fine and is dramatically faster.
echo "Playing $BAG via custom Python bag player..."
# The ROS1 noetic C++ rosbag player takes ~3 hours to play the 142-second
# cow_and_lady bag on aarch64 (some timing/busy-loop pathology — 100% CPU
# but ~0.2 Hz publish rate). Our bag_player.py uses the rosbag library to
# iterate messages and publishes them via rospy at real-time rate,
# bypassing the bug entirely.
python3 /app/bag_player.py "$BAG" 1.0
kill $LAUNCH_PID 2>/dev/null || true

echo "Bag finished. Bridge still serving on :9090. Ctrl+C to exit."
wait $BRIDGE_PID
